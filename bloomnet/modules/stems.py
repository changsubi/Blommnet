"""Stem 3종 — PPN / SPS / TPS + CanonicalScatter (02 §2, 06 §3.3.2, 레벨 L2).

세 stem 은 모두 ``(B, 32, H/4, W/4)`` 를 낸다.

동결 사항
    * **canonical slot** (02 §2.1): 센서가 4/5/6밴드 무엇이든 **단일 weight** 로 처리한다.
      결측 slot 은 zero-fill + 학습형 보상 ``C_abs`` (init 0 → 초기 동작은 순수 zero-fill).
    * **A7 결측 전파** (01 §2.4, 정정 A-25/B-2): ``band_ids`` 는 배치 상수라
      샘플 수준 결측을 표현할 수 없다. ``avail_k`` 를 받아
      ``m = m_from_band_ids × avail_k`` 로 만들고 **stem 출력도 0 마스킹**한다.
      이게 없으면 "0 으로 채워진 정상 slot" 과 "진짜 결측 slot" 이 구분되지 않아
      SPS/TPS 의 BN 이 전부-0 샘플 통계를 흡수한다.
    * **x_bio = [NDCI, MCI_norm]** (X-01/X-27, 정정 B-3). GNDVI 는 ablation 대조군.
      MCI 는 681 nm 를 요구하지 않는다 — 02 초판의 기각 논거는 사실 오류였다.
    * **폴백 사다리 F0~F3** (정정 B-3) 와 ``bio_valid (B,2)`` 출력.
    * **glint gate** ``a = softplus(a_hat)``, init ``a≈20 / b=−5`` (헌법 R3).
      원설계 ``a=10, b=−3`` 은 깨끗한 수체를 18% 오검출한다.
    * ``use_pol=False`` 이면 gate·inpainter 를 **생성하지 않고** ``g_pol=None`` (X-05, 정정 B-4).
      ``None`` 은 ``zeros`` 와 수치적으로 동일하다(03 §4.1 에서 ``log_r = 0``).

레벨 L2. 아래로만 의존한다: ``bloomnet.constants`` (L−1),
``bloomnet.modules.common`` (L1), ``bloomnet.data.indices`` (L1, 단방향 — 06 §2.2).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import MSI_SLOT_ID, MSI_SLOTS, PHYS_SLOTS
from bloomnet.data.indices import compute_bio_canonical, specular_proxy
from bloomnet.modules.common import ConvBN, build_act, build_norm, init_encoder

__all__ = ["CanonicalScatter", "PPN", "SPS", "SPSOut", "TPS"]

_SLOT_RED = MSI_SLOT_ID["red"]
_SLOT_RE1 = MSI_SLOT_ID["rededge1"]
_SLOT_RE2 = MSI_SLOT_ID["rededge2"]
_SLOT_NIR = MSI_SLOT_ID["nir"]
_SLOT_GREEN = MSI_SLOT_ID["green"]


class SPSOut(NamedTuple):
    """``SPS.forward`` 반환 (02 §2.3 계약, 정정 B-3).

    06 §3.3.2 초판은 2-tuple ``(spec_stem, x_bio_full)`` 로 적혀 있으나
    정정 B-3 이 ``bio_valid (B,2)`` 를 **필수 출력**으로 추가했고
    02 §9 테스트 계약(`bio_valid==[[1,1],…]`)도 그것을 검사한다.
    07 정정이 06 에 우선하므로 3-tuple 이 정본이다. NamedTuple 이라
    2개로 언패킹하면 **조용히 통과하지 않고** 즉시 ValueError 를 낸다.
    """

    spec_stem: Tensor  # (B, embed_dim, H/4, W/4)
    x_bio_full: Tensor  # (B, 2, H, W)
    bio_valid: Tensor  # (B, 2) float {0,1}


def _check_band_ids(band_ids: Sequence[int], num_slots: int, k_in: int) -> Tuple[int, ...]:
    ids = tuple(int(i) for i in band_ids)
    if len(ids) != k_in:
        raise ValueError(f"len(band_ids)={len(ids)} != K_in={k_in}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"band_ids must be unique, got {ids}")
    if any(i < 0 or i >= num_slots for i in ids):
        raise ValueError(f"band_ids out of range [0,{num_slots}): {ids}")
    return ids


def _as_avail(avail: Optional[Tensor], b: int, ref: Tensor) -> Optional[Tensor]:
    """``avail`` 을 ``(B,)`` float 로 정규화한다. ``None`` 은 그대로 통과."""
    if avail is None:
        return None
    a = avail.to(device=ref.device, dtype=ref.dtype)
    if a.dim() == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.dim() != 1 or a.shape[0] != b:
        raise ValueError(f"avail must be (B,), got {tuple(avail.shape)} (B={b})")
    return a


# ─────────────────────────────────────────────────────────────────────────────
# Canonical slot scatter
# ─────────────────────────────────────────────────────────────────────────────
class CanonicalScatter(nn.Module):
    """센서 밴드를 기능 기반 canonical slot 으로 흩뿌린다 (02 §2.1). 파라미터 0.

    Args:
        num_slots: MSI 6 / PHYS 4 (★X-04).

    Forward:
        ``x_raw (B,K_in,H,W)``, ``band_ids`` (길이 K_in), ``avail_k (B,)|None``
        → ``(S (B,K,H,W), m (B,K) float32)``.

    ``avail_k`` 가 주어지면 ``S *= avail_k.view(B,1,1,1)``, ``m *= avail_k.view(B,1)``.
    ``avail_k=None`` 은 초판 동작과 **bit-identical** 이다.
    """

    def __init__(self, num_slots: int) -> None:
        super().__init__()
        if int(num_slots) <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        self.num_slots = int(num_slots)

    def forward(
        self,
        x_raw: Tensor,
        band_ids: Sequence[int],
        avail_k: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if x_raw.dim() != 4:
            raise ValueError(f"x_raw must be (B,K_in,H,W), got {tuple(x_raw.shape)}")
        b, k_in = x_raw.shape[0], x_raw.shape[1]
        ids = _check_band_ids(band_ids, self.num_slots, k_in)

        idx = torch.as_tensor(ids, device=x_raw.device, dtype=torch.long)
        s = x_raw.new_zeros((b, self.num_slots, x_raw.shape[2], x_raw.shape[3]))
        s = s.index_copy(1, idx, x_raw)  # autograd-safe (out-of-place)
        m = x_raw.new_zeros((b, self.num_slots)).index_fill(1, idx, 1.0)

        a = _as_avail(avail_k, b, x_raw)
        if a is not None:  # ★ A7(i) — 샘플 수준 결측을 slot presence 까지 전파
            s = s * a.view(b, 1, 1, 1)
            m = m * a.view(b, 1)
        return s, m

    def extra_repr(self) -> str:
        return f"num_slots={self.num_slots}"


# ─────────────────────────────────────────────────────────────────────────────
# PPN
# ─────────────────────────────────────────────────────────────────────────────
class PPN(nn.Module):
    """Polarimetric Pre-conditioning Network (02 §2.2, ★X-04/X-05, 헌법 R3).

    ``forward(x_rgb, x_pol=None) -> (rgb_stem (B,32,H/4,W/4), g_pol (B,1,H,W)|None)``

    gate 모드 3종 (01 §5.2 의 ``ppn.glint_gate`` 와 1:1):

    ==============  =========================================  ==========
    mode            조건                                        params
    ==============  =========================================  ==========
    ``"none"``      기본. gate·inpainter **미생성**              4,768
    ``"pol"``       ``use_pol=True``. DoLP 기반 Fresnel gate     10,101
    ``"rgb_proxy"`` ``rgb_proxy_gate=True``. 밝기·채도 heuristic 10,102
    ==============  =========================================  ==========

    ``"rgb_proxy"`` 는 **물리적 glint 가 아니다** (01 §5.2.3): 흰 거품·모래톱·대발생
    표면 scum 이 모두 밝고 무채색이라 탐지 대상을 지울 수 있고, 융합 스케일을 바꾼다.
    기본 비활성이며, 켜면 보고서에 "밝기·채도 기반 heuristic" 임을 명시해야 한다.

    ``use_pol=True`` 인데 런타임에 ``x_pol=None`` 이 오면(모달 dropout) gate 를 건너뛰고
    ``g_pol=None`` 을 낸다 — 파라미터는 생성돼 있으나 그 배치에서 gradient 가 없다.
    """

    def __init__(
        self,
        *,
        in_chans: int = 3,
        embed_dim: int = 32,
        use_pol: bool = False,
        inpaint_ch: int = 16,
        dilations: Tuple[int, int, int] = (1, 2, 4),
        gate_a_init: float = 20.0,
        gate_b_init: float = -5.0,
        rgb_proxy_gate: bool = False,
        always_build: bool = False,
        norm: str = "bn",
        proxy_t_init: float = 0.55,
        proxy_c_init: float = -3.0,
    ) -> None:
        super().__init__()
        if use_pol and rgb_proxy_gate:
            raise ValueError(
                "use_pol 과 rgb_proxy_gate 는 동시에 켤 수 없다 "
                "(01 §5.2 glint_gate ∈ {none, pol, rgb_proxy})"
            )
        if len(dilations) != 3:
            raise ValueError(f"dilations must have 3 entries, got {tuple(dilations)}")

        self.in_chans = int(in_chans)
        self.embed_dim = int(embed_dim)
        self.use_pol = bool(use_pol)
        self.rgb_proxy_gate = bool(rgb_proxy_gate)
        self.gate_mode = "pol" if use_pol else ("rgb_proxy" if rgb_proxy_gate else "none")

        # ---- Step 5. Patch embed (항상 생성) --------------------------------
        # 02 §2.5: PPN 만 7×7 (RGB 는 3채널로 정보 밀도가 가장 낮고,
        #          MiT/SegFormer stage-1 OverlapPatchEmbed 와 기하가 동일하다).
        self.patch_embed = ConvBN(7, self.in_chans, self.embed_dim, stride=4, norm=norm)

        # ---- Step 1. Glint soft-gate ---------------------------------------
        if self.gate_mode == "pol":
            self.a_hat = nn.Parameter(torch.tensor(float(gate_a_init)))
            self.b = nn.Parameter(torch.tensor(float(gate_b_init)))
        elif self.gate_mode == "rgb_proxy":
            # 01 §5.2.2 [M11] 캘리브레이션: a=40, t=0.55, c_hat=−3 (c≈0.049)
            self.a_hat = nn.Parameter(torch.tensor(40.0))
            self.b = nn.Parameter(torch.tensor(float(proxy_t_init)))
            self.c_hat = nn.Parameter(torch.tensor(float(proxy_c_init)))

        # ---- Step 3. Learned inpainter (H/2) --------------------------------
        self.has_inpainter = self.gate_mode != "none" or bool(always_build)
        if self.has_inpainter:
            c = int(inpaint_ch)
            d1, d2, d3 = (int(d) for d in dilations)
            self.inp1 = ConvBN(3, self.in_chans + 1, c, stride=2, dilation=d1, norm=norm)
            self.inp2 = ConvBN(3, c, c, dilation=d2, norm=norm)
            self.inp3 = ConvBN(3, c, c, dilation=d3, norm=norm)
            self.inp_head = nn.Conv2d(c, self.in_chans, 1, bias=True)
            self.act = build_act("gelu")

        self.apply(init_encoder)
        self._reset_specials()

    @torch.no_grad()
    def _reset_specials(self) -> None:
        """02 §1.5 초기화 특칙 — ``apply(init_encoder)`` **뒤에** 덮어쓴다.

        ``a_hat``/``b``/``c_hat`` 은 ``init_encoder`` 가 건드리지 않으므로
        (Conv2d/norm 이 아니다) ``__init__`` 의 값이 그대로 유지된다.
        """
        if self.has_inpainter:
            # P2: Sigmoid 대신 zero-init linear. 초기 x_inp = 0 = "ImageNet 평균색".
            nn.init.zeros_(self.inp_head.weight)
            nn.init.zeros_(self.inp_head.bias)

    # -- gate ---------------------------------------------------------------
    def glint_gate(self, dolp: Tensor) -> Tensor:
        """``g = sigmoid(softplus(a_hat)·DoLP + b)`` (헌법 R3).

        ``a > 0`` 제약이 "DoLP 가 높을수록 glint 가능성 증가" 라는 Fresnel 단조성을
        **구조적으로** 보장한다. init 에서 결정 경계는 DoLP=0.25 (water-leaving 상한
        0.15 와 glint 하한 0.30 의 중간)다.
        """
        if self.gate_mode != "pol":
            raise RuntimeError("glint_gate is only available when use_pol=True")
        a = F.softplus(self.a_hat)
        return torch.sigmoid(a * dolp + self.b)

    def rgb_gate(self, x_rgb: Tensor) -> Tensor:
        """RGB-only specular gate (01 §5.2.1). ★ 물리적 glint 가 아니다."""
        if self.gate_mode != "rgb_proxy":
            raise RuntimeError("rgb_gate is only available when rgb_proxy_gate=True")
        from bloomnet.data.indices import denormalize_imagenet  # 지연 import (순환 없음)

        x01 = denormalize_imagenet(x_rgb)
        spec = specular_proxy(x01)
        tot = x01.sum(dim=1, keepdim=True) + 1e-8
        exg = (2.0 * x01[:, 1:2] - x01[:, 0:1] - x01[:, 2:3]) / tot
        a = F.softplus(self.a_hat)
        c = F.softplus(self.c_hat)
        return torch.sigmoid(a * (spec - self.b) - c * F.relu(exg))

    # -- preconditioning ----------------------------------------------------
    def precondition(
        self, x_rgb: Tensor, x_pol: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Step 0~4. ``(x_clean, g_pol)`` 을 낸다.

        gate 가 없으면 ``x_clean is x_rgb`` (동일 객체 — 02 §9 테스트 계약).
        """
        if self.gate_mode == "none":
            return x_rgb, None  # Step 0 → goto Step 5 (연산 0, 파라미터 0)

        if self.gate_mode == "pol":
            if x_pol is None:
                return x_rgb, None  # 런타임 모달 dropout
            if x_pol.dim() != 4 or x_pol.shape[1] < 1:
                raise ValueError(f"x_pol must be (B,3,H,W), got {tuple(x_pol.shape)}")
            g = self.glint_gate(x_pol[:, 0:1])  # ch0 = DoLP. ch1/ch2 는 TPS 소관
        else:
            g = self.rgb_gate(x_rgb)

        # Step 2. Masking
        x_masked = x_rgb * (1.0 - g)
        u = torch.cat([x_masked, g], dim=1)  # (B, in_chans+1, H, W)

        # Step 3. Inpainter @ H/2 (1,371.5 → 342.9 MMAC)
        h = self.act(self.inp1(u))
        h = self.act(self.inp2(h))
        h = self.act(self.inp3(h))
        d = self.inp_head(h)
        x_inp = _up_to(d, x_rgb.shape[-2], x_rgb.shape[-1])

        # Step 4. Soft composite
        x_clean = (1.0 - g) * x_rgb + g * x_inp
        return x_clean, g

    def forward(
        self, x_rgb: Tensor, x_pol: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if x_rgb.dim() != 4 or x_rgb.shape[1] != self.in_chans:
            raise ValueError(
                f"x_rgb must be (B,{self.in_chans},H,W), got {tuple(x_rgb.shape)}"
            )
        x_clean, g_pol = self.precondition(x_rgb, x_pol)
        return self.patch_embed(x_clean), g_pol  # patch embed 뒤 activation 없음


def _up_to(x: Tensor, h: int, w: int) -> Tensor:
    """bilinear 업샘플. 배율이 정확히 정수면 정적 ``scale_factor`` 를 쓴다.

    ``size=`` 는 ONNX 에서 ``Shape→Gather`` 를 만든다(04 §9.3-B). H,W 가 32의 배수라는
    계약(헌법 C-4) 하에서 inpainter 출력은 항상 정확히 절반이므로 정적 경로만 탄다.
    """
    xh, xw = int(x.shape[-2]), int(x.shape[-1])
    if xh == h and xw == w:
        return x
    if h % xh == 0 and w % xw == 0 and (h // xh) == (w // xw):
        return F.interpolate(
            x, scale_factor=float(h // xh), mode="bilinear", align_corners=False
        )
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# SPS / TPS 공통 몸통
# ─────────────────────────────────────────────────────────────────────────────
class _SlotStem(nn.Module):
    """canonical slot projection → DW 잔차 → patch embed (02 §2.3 Step 3~5 / §2.4).

    SPS 는 ``in_ch = num_slots + bio_ch``, TPS 는 ``in_ch = num_slots`` 로 쓴다.
    ``C_abs`` 의 행 수는 항상 ``num_slots`` 다 (bio 는 slot 이 아니다).
    """

    def __init__(
        self,
        *,
        num_slots: int,
        in_ch: int,
        mid: int = 16,
        embed_dim: int = 32,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.mid = int(mid)
        self.scatter = CanonicalScatter(self.num_slots)

        self.proj = nn.Conv2d(int(in_ch), self.mid, 1, bias=True)
        self.c_abs = nn.Parameter(torch.zeros(self.num_slots, self.mid))
        self.norm1 = build_norm(norm, self.mid)
        self.dw = nn.Conv2d(self.mid, self.mid, 3, padding=1, groups=self.mid, bias=False)
        self.norm2 = build_norm(norm, self.mid)
        self.patch_embed = ConvBN(5, self.mid, int(embed_dim), stride=4, norm=norm)
        self.act = build_act("gelu")

    def project(self, u: Tensor, m: Tensor) -> Tensor:
        """Step 3~5. ``u (B,in_ch,H,W)``, ``m (B,num_slots)`` → ``(B,embed,H/4,W/4)``."""
        z = self.proj(u)
        # 결측 slot 의 기대 기여를 학습으로 보충. init 0 이라 초기엔 순수 zero-fill 과 동일.
        z = z + ((1.0 - m) @ self.c_abs).view(m.shape[0], self.mid, 1, 1)
        z = self.act(self.norm1(z))
        z = z + self.act(self.norm2(self.dw(z)))  # Step 4 residual
        return self.patch_embed(z)  # Step 5 (activation 없음)


class SPS(nn.Module):
    """Spectral Physics Stem (02 §2.3, ★X-01/X-02/X-03, 헌법 R1·R2).

    ``forward(x_msi, band_ids, x_bio=None, avail_msi=None) -> SPSOut``

    params 13,312 | 281.0 MMAC @512². 4/5/6밴드를 **단일 weight** 로 처리한다.

    폴백 사다리 (정정 B-3) — 판정은 ``m`` (slot presence) 으로 하며 per-sample 이다:

    ==== =========================== =========== ================== ===========
    #    present slot                ch0         ch1                bio_valid
    ==== =========================== =========== ================== ===========
    F0   2,3,5 전부                  NDCI        MCI_norm(c)        [1,1]
    F1   2,3 · 5 없음 · 4 있음       NDCI        MCI_norm(c_re2)    [1,1]
    F2   2,3 · 4·5 없음              NDCI        0                  [1,0]
    F3   2 또는 3 결측               0           0                  [0,0]
    ==== =========================== =========== ================== ===========

    F1 은 ``mci_c_re2`` (= ``mci_coefficient(λ_R, λ_RE1, λ_RE2)``) 가 주어졌을 때만
    발동한다. ``λ₄−λ₂ < λ₅−λ₂`` 라 baseline 이 짧아져 ``c`` 가 커지고 잡음이 증폭되므로
    (RedEdge-P 0.338983 → 0.533333, 1.57배) **경고 로그 + manifest 기록이 필수**이고
    ``allow_mci_re2=False`` 로 끌 수 있다. 실측 검증은 없다(정정 B §7-4).

    F3 에서도 spec path 를 끄지 않는다 — raw 6-slot projection 은 유효하고 BioGate 의
    ``p`` weight 가 0-init 이라 초기 동작이 정확히 항등이다. 다만 물리 prior(N2)가
    소멸하므로 리포트에 명시해야 한다.
    """

    def __init__(
        self,
        *,
        num_slots: int = len(MSI_SLOTS),
        bio_ch: int = 2,
        mid: int = 16,
        embed_dim: int = 32,
        bio_init_gain: float = 4.0,
        bio_kind: str = "mci",
        mci_c: float = 0.380952,
        mci_c_re2: Optional[float] = None,
        allow_mci_re2: bool = True,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        if int(num_slots) != len(MSI_SLOTS):
            raise ValueError(f"SPS num_slots must be {len(MSI_SLOTS)}, got {num_slots}")
        if bio_kind not in ("mci", "gndvi"):
            raise ValueError(f"bio_kind must be 'mci' or 'gndvi', got {bio_kind!r}")
        self.num_slots = int(num_slots)
        self.bio_ch = int(bio_ch)
        self.bio_kind = bio_kind
        self.mci_c = float(mci_c)
        self.mci_c_re2 = None if mci_c_re2 is None else float(mci_c_re2)
        self.allow_mci_re2 = bool(allow_mci_re2)
        self.bio_init_gain = float(bio_init_gain)

        self.body = _SlotStem(
            num_slots=self.num_slots,
            in_ch=self.num_slots + self.bio_ch,
            mid=mid,
            embed_dim=embed_dim,
            norm=norm,
        )
        self.apply(init_encoder)
        self._reset_specials()

    @torch.no_grad()
    def _reset_specials(self) -> None:
        """02 §2.3 Step3 초기화 특칙 + ``C_abs`` = 0."""
        self.body.c_abs.zero_()
        # bio 두 채널(열 6,7)의 초기 gain 을 ×4 — raw 밴드 6개 항의 합에 묻히지 않게.
        # ⚠ 근거는 "정규화 후 스케일"(01 §2.6 R4′ k_sensor) 전제에서만 유효하다 [U-U5].
        if self.bio_ch > 0:
            self.body.proj.weight[:, self.num_slots :] *= self.bio_init_gain

    # -- bio ----------------------------------------------------------------
    def compute_bio(self, s: Tensor, m: Tensor) -> Tuple[Tensor, Tensor]:
        """``(S, m)`` 에서 ``x_bio (B,2,H,W)`` 와 ``bio_valid (B,2)`` 를 만든다.

        F0/F2/F3 는 ``data.indices.compute_bio_canonical`` **단일 구현**(X-02)이
        slot gate 로 이미 처리한다. F1 은 NIR slot 자리에 REDEDGE2 를 넣은
        ``(S', m')`` 로 같은 함수를 재호출해 얻는다 — 공식 복제 없음.
        """
        base = compute_bio_canonical(s, m, mci_c=self.mci_c, kind=self.bio_kind)

        have_r, have_re1 = m[:, _SLOT_RED], m[:, _SLOT_RE1]
        have_re2, have_nir = m[:, _SLOT_RE2], m[:, _SLOT_NIR]
        ndci_ok = have_r * have_re1
        if self.bio_kind == "mci":
            ch1_ok = ndci_ok * have_nir
            use_f1 = ndci_ok * have_re2 * (1.0 - have_nir)
            if self.mci_c_re2 is not None and self.allow_mci_re2:
                s_f1 = s.clone()
                s_f1[:, _SLOT_NIR] = s[:, _SLOT_RE2]
                m_f1 = m.clone()
                m_f1[:, _SLOT_NIR] = m[:, _SLOT_RE2]
                alt = compute_bio_canonical(s_f1, m_f1, mci_c=self.mci_c_re2, kind="mci")
                g = use_f1.view(-1, 1, 1, 1)
                base = torch.cat(
                    [base[:, 0:1], base[:, 1:2] * (1.0 - g) + alt[:, 1:2] * g], dim=1
                )
                ch1_ok = ch1_ok + use_f1
        else:  # gndvi (ablation 대조군)
            ch1_ok = m[:, _SLOT_GREEN] * have_nir

        bio_valid = torch.stack([ndci_ok, ch1_ok], dim=1).clamp(0.0, 1.0)
        return base, bio_valid

    def bio_kind_of(self, bio_valid: Tensor) -> str:
        """배치 전체를 대표하는 ``meta["bio_kind"]`` 문자열 (F0~F3, 리포트/이식 판정용).

        01 §6.7 T6: 학습 시점과 다른 ``bio_kind`` 면 SPS 1×1 conv 의 bio 열을 이식하지 않는다.
        """
        bv = bio_valid.detach()
        if float(bv[:, 0].min()) < 0.5:
            return "none"
        if float(bv[:, 1].min()) < 0.5:
            return "ndci_only"
        if self.bio_kind == "gndvi":
            return "gndvi"
        return "mci"

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        x_msi: Tensor,
        band_ids: Sequence[int],
        x_bio: Optional[Tensor] = None,
        avail_msi: Optional[Tensor] = None,
        *,
        bio_valid: Optional[Tensor] = None,
    ) -> SPSOut:
        if x_msi.dim() != 4:
            raise ValueError(f"x_msi must be (B,K_in,H,W), got {tuple(x_msi.shape)}")
        b = x_msi.shape[0]
        s, m = self.body.scatter(x_msi, band_ids, avail_msi)

        if x_bio is None:
            x_bio, bio_valid = self.compute_bio(s, m)
        else:
            if x_bio.dim() != 4 or x_bio.shape[1] != self.bio_ch:
                raise ValueError(
                    f"x_bio must be (B,{self.bio_ch},H,W), got {tuple(x_bio.shape)}"
                )
            av = _as_avail(avail_msi, b, x_msi)
            if av is not None:  # A1: 결측 샘플의 bio 도 0 이어야 한다
                x_bio = x_bio * av.view(b, 1, 1, 1)
            if bio_valid is None:
                # 외부(dataset)가 계산했으면 bio_valid·bio_kind 도 함께 와야 한다(02 §2.3 규칙 4).
                # 없으면 "전부 유효" 로 가정하되 avail 은 반영한다.
                bio_valid = x_bio.new_ones((b, self.bio_ch))
                if av is not None:
                    bio_valid = bio_valid * av.view(b, 1)
            else:
                bio_valid = bio_valid.to(device=x_bio.device, dtype=x_bio.dtype)

        u = torch.cat([s, x_bio], dim=1)  # (B, 8, H, W)  ★X-03
        spec_stem = self.body.project(u, m)

        av = _as_avail(avail_msi, b, x_msi)
        if av is not None:  # ★ A7(ii) — 결측 샘플은 stem 출력도 0
            spec_stem = spec_stem * av.view(b, 1, 1, 1)
        return SPSOut(spec_stem, x_bio, bio_valid)


class TPS(nn.Module):
    """Thermal-Polarimetric Stem (02 §2.4, ★X-04).

    ``forward(x_phys, slot_ids, avail_phys=None) -> phys_stem (B,32,H/4,W/4)``

    SPS 와 동일 골격이되 bio 단계가 없다. params 13,216 | 264.2 MMAC @512².
    physics layer 를 넣지 않는 이유: IR·DoLP 에 대해 **본 프로젝트 데이터로 검증된
    파생 지수가 하나도 없다**(열화상·편광 파일 0개). 검증되지 않은 물리 지수 주입은
    "근거 없는 주장 금지" 위반이다.
    """

    def __init__(
        self,
        *,
        num_slots: int = len(PHYS_SLOTS),
        mid: int = 16,
        embed_dim: int = 32,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        if int(num_slots) != len(PHYS_SLOTS):
            raise ValueError(f"TPS num_slots must be {len(PHYS_SLOTS)} (X-04), got {num_slots}")
        self.num_slots = int(num_slots)
        self.body = _SlotStem(
            num_slots=self.num_slots,
            in_ch=self.num_slots,
            mid=mid,
            embed_dim=embed_dim,
            norm=norm,
        )
        self.apply(init_encoder)
        with torch.no_grad():
            self.body.c_abs.zero_()

    def forward(
        self,
        x_phys: Tensor,
        slot_ids: Sequence[int],
        avail_phys: Optional[Tensor] = None,
    ) -> Tensor:
        if x_phys.dim() != 4:
            raise ValueError(f"x_phys must be (B,K_in,H,W), got {tuple(x_phys.shape)}")
        b = x_phys.shape[0]
        s, m = self.body.scatter(x_phys, slot_ids, avail_phys)
        phys_stem = self.body.project(s, m)
        av = _as_avail(avail_phys, b, x_phys)
        if av is not None:  # ★ A7(ii)
            phys_stem = phys_stem * av.view(b, 1, 1, 1)
        return phys_stem
