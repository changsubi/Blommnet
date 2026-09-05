"""BMEF — Bayesian Modality Evidence Fusion (03 §5, 06 §3.3.6).

Gaussian product-of-experts 를 **공유 잠재 특징** 위에서 계산한다.
각 path `m ∈ ("rgb","spec","phys")` 가 `(mu_m, log tau_m)` 을 내고,
present set `S` 위의 정밀도 가중 **볼록결합** `mu_f = Σ_{m∈S} w_m mu_m`,
`w_m = tau_m / Σ_{k∈S} tau_k` 로 융합한다 (03 §3.2 정리 1).

핵심 계약 (03 §13.2):
  1. 어떤 `present` 조합(공집합 포함)에서도 shape 오류·NaN·Inf 없음.
  2. `|S| = 1` 이면 그 path 의 정확한 passthrough.
  3. 세 path 가 동일 신호를 담으면 출력은 `present` 와 무관하게 동일.
  4. `Σ_m w_m = 1` 항등. absent path 의 `w_m` 은 정확히 0.0.
  5. `C_in == C_out`.
  ※ 보장하지 **않는** 것 (정정 B-11 / 06 M-12): path 가 서로 **독립**인 신호를
     담을 때 2차 모멘트는 보존되지 않는다 — `rms(fused) ∝ ||w||_2 ≈ 1/sqrt(|S|)`.
     이는 문서화된 동작이며 디코더 BN 정합은 modality dropout(학습 레시피) 책임이다.

구현상 반드시 지켜야 하는 함정 (06 §3.3.6):
  (1) `exp((A_m - amax).clamp(max=0.0)) * d` — clamp 가 없으면 전부-결측에서
      `exp(+1e4) * 0 = NaN`.
  (2) `(B,3,C,H,W)` softmax 텐서를 절대 materialize 하지 않는다.
      `e:(B,1,H,W)` × `q:(1,C,1,1)` 분리형 누산.
  (3) 융합 구간(§5.3~§5.4)은 AMP 하에서도 **fp32 강제** — 최악 `w_min = 3.07e-6`
      은 fp16 subnormal(6e-8) 근처까지 내려간다. 따라서 `fused/vacuity/log_tau/
      feedback` 의 dtype 은 **입력 dtype 과 무관하게 항상 float32** 다 (03 §10 T15 계약).

dtype 계약: 입력 feats 는 임의 부동소수(autocast bf16/fp16 포함), 출력은 전부 float32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import PATHS  # ("rgb", "spec", "phys") — 리터럴 단일 출처
from bloomnet.modules.common import build_norm  # (L1) gn8 = GroupNorm(gcd(8,C), C), 정정 B-1

__all__ = ["PATHS", "NEG", "EPS", "FEEDBACK_PATHS", "BMEFOutput", "BMEF", "identity_fuse"]

# 06 §3.3.6 동결 상수. -inf 대신 유한값(fp16 max 65504 이내).
NEG: float = -1e4
EPS: float = 1e-12

# rgb 는 피드백을 받지 않는다 (CMNeXt 가 x_cam 을 pre-fusion 으로 잇는 것과 동일, 03 §5.5).
FEEDBACK_PATHS: Tuple[str, str] = ("spec", "phys")


@dataclass
class BMEFOutput:
    """03 §5.1 / 06 §3.3.6 출력 계약."""

    fused: Tensor  # (B, C_i, H_i, W_i)        -> PIDDecoder tap
    vacuity: Tensor  # (B, 1,   H_i, W_i) ∈(0,1) -> 진단 전용 (X-12: 어떤 손실도 소비 안 함)
    log_tau: Tensor  # (B, C_i, H_i, W_i) clamp[-(R+3), R+3+log3]
    feedback: Dict[str, Tensor]  # {"spec":…, "phys":…}; stage 4 또는 비활성 시 {}
    weights: Optional[Tensor] = None  # (B, 3, H_i, W_i) 채널평균, return_weights=True 시


# 06 §3.3.6 이 동결한 `norm_layer` 어휘는 {"bn","gn"} 이고, 02 §1.2/정정 B-1 이 동결한
# `build_norm` 의 어휘는 {"bn","syncbn","gn8"} 이다. 여기서 매핑만 하고 실제 생성은
# modules/common.py(L1) 단일 빌더에 위임한다 — gn8 = GroupNorm(gcd(8,C), C).
# BMEF 의 norm 폭은 C ∈ {32,64,160,320} 이라 gcd(8,C) == min(8,C) == 8 로
# 06 §3.3.6 의 `GroupNorm(min(8,C), C)` 표기와 결과가 같다. 파라미터 수는 어느 쪽도 2C.
_NORM_ALIAS: Dict[str, str] = {"bn": "bn", "gn": "gn8", "gn8": "gn8", "syncbn": "syncbn"}


def _build_norm(kind: str, c: int) -> nn.Module:
    try:
        resolved = _NORM_ALIAS[kind]
    except KeyError:
        raise ValueError(
            f"norm_layer 는 {sorted(_NORM_ALIAS)} 중 하나여야 한다: {kind!r}"
        ) from None
    return build_norm(resolved, c)


def _inv_softplus(y: float) -> float:
    """softplus^{-1}(y) = log(exp(y) - 1),  y > 0.

    `torch.expm1` 은 ONNX 에 대응 연산자가 없어(정정 B-13) 쓰지 않는다.
    본 함수는 `__init__` 시점의 순수 python 산술이라 그래프에 남지 않는다.
    """
    if not (y > 0.0):
        raise ValueError(f"kappa_init 은 모두 양수여야 한다 (softplus 치역): {y}")
    if y > 20.0:  # exp overflow 방지. 이 영역에서 softplus 는 항등에 가깝다.
        return y
    return math.log(math.exp(y) - 1.0)


class BMEF(nn.Module):
    """stage `i` 하나를 담당하는 BMEF 모듈 (03 §5).

    Args:
        channels: `C_i`. BMEF 는 채널을 바꾸지 않는다 (`C_in == C_out`).
        stage_idx: 1..4. `g_pol` 축소 배율 `s_i = 2**(stage_idx+1)` 과
            피드백 생성 여부(stage 4 는 다음 stage 가 없어 미생성)를 결정한다.
        se_reduction / se_min_channels: `Cr = max(se_min_channels, C // se_reduction)`.
        logtau_range: `R`. 공간 log-precision 을 `R·tanh(·/R)` 로 유계화.
        chan_logtau_range: `Rp`. 채널 log-precision 유계화.
        r_floor: `eps_r`. reliability 하한 (dead-path 방지).
        kappa_init: path 별 glint 민감도 초기값 `(rgb, spec, phys)`.
            내부적으로 `softplus^{-1}` 을 저장한다.
        gpol_pool: `"area"`(기본) | `"max"`. **`adaptive_*_pool2d` 를 쓰지 않는다**
            (정정 B-12: ONNX `Shape→Gather` 제거).
        norm_mode: `"l1"`(볼록결합, Bayes 해) | `"l2"`(2차 모멘트 보존, ablation).
        norm_layer: `"bn"` | `"gn"`.
        enable_feedback: aux 스트림 가산 피드백 생성 여부.
    """

    def __init__(
        self,
        channels: int,
        stage_idx: int,
        *,
        se_reduction: int = 8,
        se_min_channels: int = 8,
        logtau_range: float = 4.0,
        chan_logtau_range: float = 2.0,
        r_floor: float = 0.05,
        kappa_init: Tuple[float, float, float] = (1.0, 0.5, 0.018),
        gpol_pool: str = "area",
        norm_mode: str = "l1",
        norm_layer: str = "bn",
        enable_feedback: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels > 0 이어야 한다: {channels}")
        if stage_idx not in (1, 2, 3, 4):
            raise ValueError(f"stage_idx 는 1..4: {stage_idx}")
        if gpol_pool not in ("area", "max"):
            raise ValueError(f"gpol_pool 은 {{'area','max'}}: {gpol_pool!r}")
        if norm_mode not in ("l1", "l2"):
            raise ValueError(f"norm_mode 는 {{'l1','l2'}}: {norm_mode!r}")
        if norm_layer not in _NORM_ALIAS:
            raise ValueError(f"norm_layer 는 {sorted(_NORM_ALIAS)}: {norm_layer!r}")
        if len(kappa_init) != 3:
            raise ValueError(f"kappa_init 은 (rgb, spec, phys) 3원소: {kappa_init!r}")
        if not (0.0 < r_floor <= 1.0):
            raise ValueError(f"r_floor ∈ (0,1]: {r_floor}")

        self.channels = int(channels)
        self.stage_idx = int(stage_idx)
        self.logtau_range = float(logtau_range)
        self.chan_logtau_range = float(chan_logtau_range)
        self.r_floor = float(r_floor)
        self.gpol_pool = gpol_pool
        self.norm_mode = norm_mode
        self.norm_layer = norm_layer
        # stage 4 는 다음 stage 가 없으므로 피드백을 만들지 않는다 (03 §5.5).
        self.enable_feedback = bool(enable_feedback) and self.stage_idx < 4
        # g_pol(full-res) -> stage 해상도 축소 배율. stage 1..4 -> 4/8/16/32.
        self.gpol_stride = 2 ** (self.stage_idx + 1)

        c = self.channels
        cr = max(int(se_min_channels), c // int(se_reduction))
        self.se_channels = cr

        # --- (a) mean head: "값" ---------------------------------------------
        self.mean_conv = nn.ModuleDict(
            {m: nn.Conv2d(c, c, 1, bias=False) for m in PATHS}
        )
        self.mean_norm = nn.ModuleDict({m: _build_norm(norm_layer, c) for m in PATHS})

        # --- (b) precision head: "그 값을 얼마나 믿는가" (공간항) ------------
        self.prec_dw = nn.ModuleDict(
            {m: nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False) for m in PATHS}
        )
        self.prec_norm = nn.ModuleDict({m: _build_norm(norm_layer, c) for m in PATHS})
        self.prec_pw = nn.ModuleDict({m: nn.Conv2d(c, 1, 1, bias=True) for m in PATHS})

        # --- (e) 채널 log-precision (rank-1 분리형의 채널항) -----------------
        self.p_raw = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(c)) for m in PATHS}
        )

        # --- post-fusion SE (03 §5.4) ----------------------------------------
        self.se1 = nn.Conv2d(c, cr, 1, bias=True)
        self.se2 = nn.Conv2d(cr, c, 1, bias=True)

        # --- 스칼라들 ---------------------------------------------------------
        self.log_tau0 = nn.Parameter(torch.zeros(1))
        self.kappa_raw = nn.Parameter(
            torch.tensor([_inv_softplus(float(k)) for k in kappa_init], dtype=torch.float32)
        )

        # --- aux 스트림 피드백 LayerScale (zero-init) -------------------------
        if self.enable_feedback:
            self.fb_gamma = nn.ParameterDict(
                {m: nn.Parameter(torch.zeros(c)) for m in FEEDBACK_PATHS}
            )
        else:
            self.fb_gamma = None  # type: ignore[assignment]

        self.reset_parameters()

    # ------------------------------------------------------------------ init
    def reset_parameters(self) -> None:
        """03 §8.2 초기화표. init 상태의 BMEF = `mean_m(BN_m(F_m))` (균등 평균)."""
        c = self.channels
        eye = torch.eye(c).view(c, c, 1, 1)
        for m in PATHS:
            # mean 1x1 은 **항등 초기화** — 주 경로는 처음부터 신호를 그대로 흘린다.
            with torch.no_grad():
                self.mean_conv[m].weight.copy_(eye)
            nn.init.kaiming_normal_(
                self.prec_dw[m].weight, mode="fan_out", nonlinearity="relu"
            )
            # prec 1x1 zero-init -> s_m ≡ 0 -> w_m = 1/|S| 정확히. **필수**
            nn.init.zeros_(self.prec_pw[m].weight)
            nn.init.zeros_(self.prec_pw[m].bias)
            nn.init.zeros_(self.p_raw[m])
        nn.init.kaiming_normal_(self.se1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.se1.bias)
        # SE 2nd conv zero-init -> gain = 2·sigmoid(0) = 1 -> init 시 SE 는 정확한 항등. **필수**
        nn.init.zeros_(self.se2.weight)
        nn.init.zeros_(self.se2.bias)
        nn.init.zeros_(self.log_tau0)
        if self.fb_gamma is not None:
            for m in FEEDBACK_PATHS:
                nn.init.zeros_(self.fb_gamma[m])

    # ------------------------------------------------------------- properties
    @property
    def kappa(self) -> Tensor:
        """`softplus(kappa_raw) >= 0` — 부호 반전(물리 역전) 불가."""
        return F.softplus(self.kappa_raw)

    # --------------------------------------------------------------- helpers
    def _reduce_gpol(self, g_pol: Tensor, h: int, w: int) -> Tensor:
        """g_pol (B,1,H,W) -> log_r (B,1,H_i,W_i).  03 §4.1 Step R1/R2."""
        s = self.gpol_stride
        if g_pol.dim() != 4 or g_pol.shape[1] != 1:
            raise ValueError(f"g_pol 은 (B,1,H,W) 여야 한다: {tuple(g_pol.shape)}")
        if g_pol.shape[-2] != h * s or g_pol.shape[-1] != w * s:
            raise ValueError(
                f"g_pol 은 **full resolution** 이어야 한다 (03 §13.1). stage {self.stage_idx} "
                f"기대 {(h * s, w * s)}, 실제 {tuple(g_pol.shape[-2:])}. "
                "H,W 는 32 의 배수여야 한다 (헌법 C-4)."
            )
        g = g_pol.float()
        # ★ adaptive_*_pool2d 금지 (정정 B-12): 출력 크기가 feature shape 에서 오면
        #   ONNX 에 Shape→Gather 가 생긴다. 축소비가 정확히 정수이므로 등가.
        if self.gpol_pool == "area":
            g_bar = F.avg_pool2d(g, kernel_size=s, stride=s)
        else:
            g_bar = F.max_pool2d(g, kernel_size=s, stride=s)
        g_bar = g_bar.clamp(0.0, 1.0)
        r_base = self.r_floor + (1.0 - self.r_floor) * (1.0 - g_bar)
        return torch.log(r_base)  # <= 0

    def _evidence(self, m: str, x: Tensor, prec_ramp: float) -> Tuple[Tensor, Tensor]:
        """path `m` 의 (mu_m, s_m). 03 §5.2 (a)~(c)."""
        mu = self.mean_norm[m](self.mean_conv[m](x))  # (B,C,H,W)
        h = self.prec_dw[m](x)
        h = self.prec_norm[m](h)
        h = F.gelu(h)
        s = self.prec_pw[m](h)  # (B,1,H,W)
        r = self.logtau_range
        # tanh 유계화(hard clamp 금지: 경계에서 gradient 소멸 -> 영구 고착).
        s = r * torch.tanh((float(prec_ramp) * s) / r)
        return mu, s

    # --------------------------------------------------------------- forward
    def forward(
        self,
        feats: Dict[str, Optional[Tensor]],
        present: Tensor,
        g_pol: Optional[Tensor] = None,
        *,
        prec_ramp: float = 1.0,
        return_weights: bool = False,
    ) -> BMEFOutput:
        unknown = set(feats) - set(PATHS)
        if unknown:
            raise ValueError(f"feats 키는 {PATHS} 만 허용: {sorted(unknown)}")

        live: List[str] = [m for m in PATHS if feats.get(m) is not None]
        if not live:
            raise ValueError(
                "feats 가 전부 None 이면 출력 shape 을 결정할 수 없다. "
                "전부-결측을 표현하려면 텐서는 주고 present=zeros(B,3) 를 쓴다 (03 §10 T5)."
            )
        ref = feats[live[0]]
        assert ref is not None
        b, c, h, w = ref.shape
        if c != self.channels:
            raise ValueError(f"BMEF(channels={self.channels}) 에 C={c} 입력")
        for m in live:
            t = feats[m]
            assert t is not None
            if tuple(t.shape) != (b, c, h, w):
                raise ValueError(
                    f"path 별 feature shape 불일치: {m} {tuple(t.shape)} vs {(b, c, h, w)}"
                )

        if present.dim() != 2 or present.shape[0] != b or present.shape[1] != len(PATHS):
            raise ValueError(
                f"present 는 (B,{len(PATHS)}) 여야 한다: {tuple(present.shape)} (B={b})"
            )
        pres = present.to(device=ref.device).bool()

        log_r = None if g_pol is None else self._reduce_gpol(g_pol, h, w)
        kappa = self.kappa  # (3,)

        # --- path 별 증거 (§5.2 a~d) -----------------------------------------
        mu_map: Dict[str, Tensor] = {}
        a_map: Dict[str, Tensor] = {}
        for m in live:
            x = feats[m]
            assert x is not None
            mu, s = self._evidence(m, x, prec_ramp)
            if log_r is not None:
                # 물리 reliability 는 ramp 대상이 아니다 (§8.3): 학습되지 않은 물리량.
                s = s + kappa[PATHS.index(m)] * log_r
            mu_map[m] = mu
            a_map[m] = s

        # --- 융합 (§5.3~§5.5) — AMP 하에서도 fp32 강제 -------------------------
        with torch.autocast(device_type=ref.device.type, enabled=False):
            r_hi = self.logtau_range + 3.0
            rp = self.chan_logtau_range

            # (e) 채널항
            p_map = {m: rp * torch.tanh(self.p_raw[m].float() / rp) for m in live}
            pmax = torch.stack([p_map[m] for m in live], dim=0).max(dim=0).values
            pmax_v = pmax.view(1, c, 1, 1)

            # 1) 공간 최대값 (present 만)
            amax: Optional[Tensor] = None
            for k, m in enumerate(PATHS):
                if m not in a_map:
                    continue
                d = pres[:, k].view(b, 1, 1, 1)
                cand = a_map[m].float().masked_fill(~d, NEG)
                amax = cand if amax is None else torch.maximum(amax, cand)
            assert amax is not None

            # 3) 분리형 누산 — (B,3,C,H,W) 를 절대 만들지 않는다
            num: Optional[Tensor] = None
            den: Optional[Tensor] = None
            sumsq: Optional[Tensor] = None
            e_map: Dict[str, Tensor] = {}
            for k, m in enumerate(PATHS):
                if m not in a_map:
                    continue
                d = pres[:, k].view(b, 1, 1, 1).float()
                # ★ clamp(max=0.0) 없으면 전부-결측에서 exp(+1e4)*0 = NaN
                e = torch.exp((a_map[m].float() - amax).clamp(max=0.0)) * d  # (B,1,H,W)
                q = torch.exp(p_map[m] - pmax).view(1, c, 1, 1)  # (1,C,1,1)
                t = e * q  # (B,C,H,W) broadcast
                e_map[m] = e
                den = t if den is None else den + t
                nu = t * mu_map[m].float()
                num = nu if num is None else num + nu
                if self.norm_mode == "l2":
                    tt = t * t
                    sumsq = tt if sumsq is None else sumsq + tt
            assert num is not None and den is not None

            mu_f = num / (den + EPS)

            # 5) l2 모드 (옵션): 2차 모멘트 보존. Bayes 해가 아니며 sup-norm 유계가 느슨해진다.
            if self.norm_mode == "l2":
                assert sumsq is not None
                w2 = sumsq / (den + EPS).pow(2)
                mu_f = mu_f / (torch.sqrt(w2) + EPS)

            # 4) 총 정밀도와 vacuity
            log_tau = torch.clamp(
                amax + pmax_v + torch.log(den + EPS),
                min=-r_hi,
                max=r_hi + math.log(3.0),
            )
            vacuity = torch.sigmoid(self.log_tau0.float().view(1, 1, 1, 1) - log_tau)
            vacuity = vacuity.mean(dim=1, keepdim=True)

            # post-fusion SE (§5.4): init 시 gain = 1 (정확한 항등)
            gap = mu_f.mean(dim=(2, 3), keepdim=True)  # AdaptiveAvgPool2d 금지 (04 §9.3-A)
            z = self.se2(F.relu(self.se1(gap)))
            gain = 2.0 * torch.sigmoid(z)  # ∈ (0,2), 무계 곱셈 게이트 금지
            fused = mu_f * gain

            # aux 스트림 피드백 (§5.5): gamma zero-init -> 초기에는 완전 private stream
            feedback: Dict[str, Tensor] = {}
            if self.fb_gamma is not None:
                for m in FEEDBACK_PATHS:
                    feedback[m] = self.fb_gamma[m].float().view(1, c, 1, 1) * fused

            weights: Optional[Tensor] = None
            if return_weights:
                cols: List[Tensor] = []
                for m in PATHS:
                    if m not in e_map:
                        cols.append(torch.zeros(b, 1, h, w, device=fused.device, dtype=fused.dtype))
                    else:
                        q = torch.exp(p_map[m] - pmax).view(1, c, 1, 1)
                        cols.append(((e_map[m] * q) / (den + EPS)).mean(dim=1, keepdim=True))
                weights = torch.cat(cols, dim=1)  # (B,3,H,W)

        return BMEFOutput(
            fused=fused,
            vacuity=vacuity,
            log_tau=log_tau,
            feedback=feedback,
            weights=weights,
        )

    def extra_repr(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"channels={self.channels}, stage_idx={self.stage_idx}, "
            f"se_channels={self.se_channels}, norm_mode={self.norm_mode!r}, "
            f"norm_layer={self.norm_layer!r}, gpol_pool={self.gpol_pool!r}, "
            f"enable_feedback={self.enable_feedback}"
        )


def identity_fuse(
    feats: Dict[str, Optional[Tensor]],
    present: Tensor,
    *,
    stage_idx: int = 1,
    enable_feedback: bool = True,
    return_weights: bool = False,
) -> BMEFOutput:
    """항등(mean-only) 모드 — 정정 A-26.

    `model.bmef.stages` 에 없는 stage 및 `model.bmef.stage1_identity=true`(03 U-8
    ablation) 에서 `BloomNetBackbone`(L4) 이 BMEF 대신 호출한다. 파라미터가 없다.

    `fused = mean_{m ∈ present} F_m`,  `vacuity = 1`,  `log_tau = 0`,  `feedback = 0`.
    present 가 공집합인 샘플은 `fused = 0` (NaN 금지).
    """
    live = [m for m in PATHS if feats.get(m) is not None]
    if not live:
        raise ValueError("feats 가 전부 None 이면 출력 shape 을 결정할 수 없다.")
    ref = feats[live[0]]
    assert ref is not None
    b, c, h, w = ref.shape
    if present.dim() != 2 or present.shape[0] != b or present.shape[1] != len(PATHS):
        raise ValueError(f"present 는 (B,{len(PATHS)}) 여야 한다: {tuple(present.shape)}")
    pres = present.to(device=ref.device).bool()

    acc: Optional[Tensor] = None
    cnt: Optional[Tensor] = None
    for k, m in enumerate(PATHS):
        if m not in live:
            continue
        d = pres[:, k].view(b, 1, 1, 1).float()
        x = feats[m]
        assert x is not None
        term = x.float() * d
        acc = term if acc is None else acc + term
        cnt = d if cnt is None else cnt + d
    assert acc is not None and cnt is not None
    fused = acc / cnt.clamp(min=1.0)

    vacuity = torch.ones(b, 1, h, w, device=fused.device, dtype=fused.dtype)
    log_tau = torch.zeros(b, c, h, w, device=fused.device, dtype=fused.dtype)
    feedback: Dict[str, Tensor] = {}
    if enable_feedback and stage_idx < 4:
        for m in FEEDBACK_PATHS:
            feedback[m] = torch.zeros_like(fused)
    weights: Optional[Tensor] = None
    if return_weights:
        cols = []
        for k, m in enumerate(PATHS):
            d = pres[:, k].view(b, 1, 1, 1).float() if m in live else torch.zeros(
                b, 1, 1, 1, device=fused.device, dtype=fused.dtype
            )
            cols.append((d / cnt.clamp(min=1.0)).expand(b, 1, h, w))
        weights = torch.cat(cols, dim=1)
    return BMEFOutput(
        fused=fused, vacuity=vacuity, log_tau=log_tau, feedback=feedback, weights=weights
    )
