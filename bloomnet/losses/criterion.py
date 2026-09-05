"""총 손실 조립 — 05 §0.2 / §1 / §3, API 동결 06 §3.5 (레벨 L3).

단일 방정식 (05 §1.1, 변형 금지)::

    L_total = λ_seg·(L_ohem + w_dice·L_dice)
            + λ_reg·L_reg
            + λ_bd ·L_edge
            + λ_bas·L_bas
            + Σ_tap λ_aux[tap]·L_aux,tap
            + λ_siam·L_siam

**λ 는 모드에 따라 바뀌지 않는다** (05 §1.4 / §3.2). 라벨이 없으면 λ 를 0 으로 바꾸는
대신 **항이 스스로 0.0 을 반환**한다. 그래야 (a) S0→S1 전환 시 손실 config diff 가 0,
(b) 로그 CSV 열 스키마가 모드 간 동일, (c) 헌법 C-5.5 를 실행시 마스킹으로 만족한다.

전 항 fp32 강제 (규칙 N7): forward 전체를 ``autocast(enabled=False)`` 로 감싸고
로짓을 ``.float()`` 로 올린다.

메모리 (정정 A-32(a)): ``ohem_ce`` / ``batch_soft_dice`` / ``bas_loss`` 는 각각 로짓을
라벨 해상도로 업샘플하므로, **criterion 이 1회만 업샘플해 셋에 공유**한다
(B=32·K=12·512² 에서 fp32 로짓 1장 = 402.7 MB, 3~4중 할당 제거).
softmax 까지 공유하려면 L1 동결 시그니처를 바꿔야 하므로 하지 않았다 —
남는 중복은 softmax 3회이며 업샘플 텐서 재할당보다 작다.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping as _ABCMapping
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import (
    IGNORE_INDEX,
    OUT_AUX,
    OUT_CHL,
    OUT_EDGE,
    OUT_LOGVAR,
    OUT_SEG,
    OUT_SIAM,
)
from bloomnet.data.boundary import make_boundary_target
from bloomnet.losses.boundary_loss import bas_loss, boundary_bce
from bloomnet.losses.distill import cwd_loss
from bloomnet.losses.regression import chl_reg_loss, u_ramp
from bloomnet.losses.seg import batch_soft_dice, ohem_ce, plain_ce

__all__ = [
    "StepCtx",
    "BloomNetCriterion",
    "build_criterion",
    "SIAM_PAIR_WEIGHTS",
    "TEACHER_KEY",
]

# 05 §2.6.3 표: SCTNet 의 feature 정렬 쌍 내부 가중치. λ_siam 은 이 위에 곱해진다.
SIAM_PAIR_WEIGHTS: Dict[str, float] = {"s3": 15.0, "s4": 15.0, "dec": 15.0}

# teacher feature 는 06 동결 시그니처(outputs/targets/step_ctx)에 통로가 없다.
# targets 의 이 키(값 = {"s3","s4","dec"} → Tensor)로 넘긴다. 없으면 L_siam = 0.
TEACHER_KEY: str = "siam_teacher"

_AUTOCAST_DEVICES = ("cpu", "cuda", "xpu", "hpu")


def _fp32_ctx(device: torch.device):
    """규칙 N7 — criterion 전체를 fp32 로 고정한다."""
    if device.type in _AUTOCAST_DEVICES:
        return torch.autocast(device_type=device.type, enabled=False)
    return contextlib.nullcontext()


@dataclass
class StepCtx:
    """trainer → (model, criterion) 로 전달되는 스텝 컨텍스트 (06 §3.5).

    Attributes:
        epoch: 0-base 현재 epoch.
        total_epochs: 총 epoch 수.
        global_step: 0-base 전역 스텝.
        total_steps: 총 스텝 수.
        prec_ramp: BMEF 정밀도 램프. **모델**이 소비한다 (정정 A-34).
            ``min(1, epoch / cfg.model.bmef.warmup_epochs)``.
        u: aleatoric 램프 ∈ [0,1]. **criterion 이 유일하게 소비한다** (정정 A-34).
            trainer 는 :meth:`BloomNetCriterion.u_at` 로 계산해 채운다 —
            그래야 ``u_warm_frac``/``u_ramp_frac`` 의 단일 출처가 criterion 이 된다.
            ``BloomNet.forward(unc_enabled=…)`` 는 기본 True 고정 디버그 스위치이며
            학습 스위치로 쓰지 않는다(이중 스위치 금지).
    """

    epoch: int
    total_epochs: int
    global_step: int
    total_steps: int
    prec_ramp: float = 1.0
    u: float = 0.0


class BloomNetCriterion(nn.Module):
    """BloomNet 총 손실 (05 §1.1).

    ``forward(outputs, targets, step_ctx) -> (L_total, breakdown)``.
    ``breakdown`` 의 손실 값은 전부 **λ 를 곱한 뒤**의 값이다 (06 §3.5).
    """

    #: 06 §3.5 가 요구하는 breakdown 손실 키(λ 적용 후). aux 는 tap 별로 추가된다.
    BREAKDOWN_LOSS_KEYS: Tuple[str, ...] = (
        "loss_total",
        "loss_ohem",
        "loss_dice",
        "loss_edge",
        "loss_bas",
        "loss_reg",
        "loss_siam",
    )

    def __init__(
        self,
        *,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        lambda_seg: float = 1.0,  # 고정 — 사업문서 확정값 (헌법 C-1)
        lambda_reg: float = 0.5,  # 고정 — 사업문서 확정값 (헌법 C-1)
        lambda_bd: float = 20.0,
        lambda_bas: float = 1.0,
        lambda_aux: Optional[Dict[str, float]] = None,  # ★X-06 tap 이름 키
        lambda_siam: float = 0.0,
        w_dice: float = 0.4,
        ohem_thresh: float = 0.7,
        ohem_keep_frac: float = 0.0625,
        bas_tau: float = 0.8,
        huber_beta: float = 1.0,  # ★X-11 (dense head)
        unc_clamp: Tuple[float, float] = (-7.0, 7.0),  # ★X-26 (정정 A-20)
        u_warm_frac: float = 0.30,
        u_ramp_frac: float = 0.20,
        boundary_source: str = "dataset",  # ★X-07
        boundary_radius: int = 1,
        boundary_stride: int = 4,
        aux_loss_stride: int = 1,  # {1,4} OOM 레버 (정정 A-32)
        class_weight: Optional[Tensor] = None,
        aux_decay: bool = False,
        debug_assert: bool = False,
        # ── 이하 동결표에 없는 keyword-only 추가분. 기본값 = 동결 동작 ────────
        boundary_radius_map: Optional[Mapping[int, int]] = None,
        siam_pair_weights: Optional[Mapping[str, float]] = None,
        siam_temperature: float = 4.0,
        u_source: str = "step_ctx",  # {"step_ctx", "auto"}
        allow_contract_break: bool = False,
    ) -> None:
        """
        Args:
            num_classes: K. S0-RGB = 12, S1 = 2 (05 §0.3).
            lambda_seg, lambda_reg: **변경 금지**. 다른 값이면 ``ValueError``
                (헌법 C-1 / config V2). 검증된 ablation 에서만
                ``allow_contract_break=True`` 로 우회한다.
            lambda_aux: tap 이름 → 가중치. 기본
                ``{"enc_s8":0.2, "enc_s16":0.4, "enc_s32":0.4, "p8":0.4}``.
                ``outputs`` 에 없는 tap 은 계산하지 않지만 breakdown 열은 0.0 으로 낸다
                (모드 간 CSV 스키마 동일 — 05 §3.2).
            unc_clamp: ``log_var`` clamp. ★X-26 으로 UncHead ``S_MIN/S_MAX`` 와 동일한
                ``(-7,7)`` 이므로 이중 clamp 가 무해한 항등이 된다.
            boundary_source: ``"criterion"`` 이면 ``targets`` 의 ``y_edge``/
                ``y_edge_valid`` 를 **무시하고** ``make_boundary_target`` 으로
                재생성한다 (정정 A-28). ``"dataset"`` 인데 타깃이 없어도 같은
                함수로 fallback 생성한다 (05 §0.2).
            boundary_radius: 재생성 시 반경. 512²→1 / 1024²→2 (01 C-10).
            boundary_radius_map: (추가) 해상도별 반경 ``{512:1, 1024:2}``.
                입력 H 가 키에 있으면 ``boundary_radius`` 대신 쓴다 — 동일 criterion
                객체로 여러 해상도를 평가할 때 반경이 조용히 틀리는 것을 막는다.
            aux_loss_stride: 1(기본) 또는 4. 4 면 aux 항의 로짓/타깃을 1/4 해상도에서
                계산한다 (정정 A-32(c)). 얇은 구조가 소실되므로 OOM 레버 전용.
            class_weight: ``(K,)`` 또는 None. ohem/bas/aux 모든 CE 에 동일 적용
                (PIDNet ``FullModel`` 관례). 기본 None (05 §4.2).
            u_source: (추가) ``"step_ctx"``(기본, 동결 계약) 이면 ``step_ctx.u`` 를
                그대로 소비한다. ``"auto"`` 면 ``step_ctx.epoch`` 에서 직접 유도한다
                (trainer 가 ``u`` 채우기를 잊어 aleatoric 이 영구 OFF 가 되는 사고 방지).
            siam_pair_weights: (추가) 쌍별 내부 가중치. 기본 SCTNet 15/15/15.
            siam_temperature: (추가) CWD 온도. SCTNet 공식값 4.0.
        """
        super().__init__()

        if not allow_contract_break:
            if float(lambda_seg) != 1.0 or float(lambda_reg) != 0.5:
                raise ValueError(
                    "헌법 C-1 / config V2: lambda_seg=1.0, lambda_reg=0.5 는 사업문서 "
                    f"확정값이라 변경 금지 (got {lambda_seg}, {lambda_reg}). "
                    "ablation 이면 allow_contract_break=True 를 명시하라."
                )
        if int(num_classes) < 2:
            raise ValueError(f"num_classes 는 2 이상 (got {num_classes})")
        if boundary_source not in ("dataset", "criterion"):
            raise ValueError(
                f"boundary_source ∈ {{dataset, criterion}} (got {boundary_source!r}) [X-07]"
            )
        if int(aux_loss_stride) not in (1, 4):
            raise ValueError(f"aux_loss_stride ∈ {{1, 4}} (got {aux_loss_stride}) [V22]")
        if int(boundary_stride) < 1:
            raise ValueError(f"boundary_stride 는 1 이상 (got {boundary_stride})")
        if u_source not in ("step_ctx", "auto"):
            raise ValueError(f"u_source ∈ {{step_ctx, auto}} (got {u_source!r})")
        if len(tuple(unc_clamp)) != 2 or float(unc_clamp[0]) >= float(unc_clamp[1]):
            raise ValueError(f"unc_clamp 은 (min, max) 이고 min < max (got {unc_clamp})")
        clamp = (float(unc_clamp[0]), float(unc_clamp[1]))

        lam_aux = dict(lambda_aux) if lambda_aux is not None else {
            "enc_s8": 0.2, "enc_s16": 0.4, "enc_s32": 0.4, "p8": 0.4
        }
        unknown = sorted(set(lam_aux) - set(OUT_AUX))
        if unknown:
            raise ValueError(f"lambda_aux 에 알 수 없는 tap {unknown} (허용 {sorted(OUT_AUX)}) [X-06]")

        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.lambda_seg = float(lambda_seg)
        self.lambda_reg = float(lambda_reg)
        self.lambda_bd = float(lambda_bd)
        self.lambda_bas = float(lambda_bas)
        self.lambda_aux: Dict[str, float] = {k: float(v) for k, v in lam_aux.items()}
        self.lambda_siam = float(lambda_siam)
        self.w_dice = float(w_dice)
        self.ohem_thresh = float(ohem_thresh)
        self.ohem_keep_frac = float(ohem_keep_frac)
        self.bas_tau = float(bas_tau)
        self.huber_beta = float(huber_beta)
        self.unc_clamp = clamp
        self.u_warm_frac = float(u_warm_frac)
        self.u_ramp_frac = float(u_ramp_frac)
        self.boundary_source = str(boundary_source)
        self.boundary_radius = int(boundary_radius)
        self.boundary_stride = int(boundary_stride)
        self.boundary_radius_map: Dict[int, int] = (
            {int(k): int(v) for k, v in boundary_radius_map.items()}
            if boundary_radius_map
            else {}
        )
        self.aux_loss_stride = int(aux_loss_stride)
        self.aux_decay = bool(aux_decay)
        self.debug_assert = bool(debug_assert)
        self.u_source = str(u_source)
        self.siam_pair_weights: Dict[str, float] = dict(
            siam_pair_weights if siam_pair_weights is not None else SIAM_PAIR_WEIGHTS
        )
        self.siam_temperature = float(siam_temperature)

        # class_weight 는 buffer 로 둬야 criterion.to(device) 가 따라온다.
        cw: Optional[Tensor] = None
        if class_weight is not None:
            cw = torch.as_tensor(class_weight, dtype=torch.float32).reshape(-1).clone()
            if cw.numel() != self.num_classes:
                raise ValueError(
                    f"class_weight 길이 {cw.numel()} != num_classes {self.num_classes}"
                )
        self.register_buffer("class_weight", cw, persistent=False)

    # ── 공개 헬퍼 ────────────────────────────────────────────────────────
    def u_at(self, epoch: int, total_epochs: int) -> float:
        """설정된 fracs 로 aleatoric 램프를 계산한다 (05 §2.5.4).

        trainer 는 ``step_ctx.u = criterion.u_at(epoch, total_epochs)`` 로 채운다.
        이렇게 해야 ``u_warm_frac``/``u_ramp_frac`` 의 출처가 criterion 하나로 남는다.
        """
        return u_ramp(
            int(epoch),
            int(total_epochs),
            warm_frac=self.u_warm_frac,
            ramp_frac=self.u_ramp_frac,
        )

    def aux_scale(self, epoch: int, total_epochs: int) -> float:
        """``aux_decay=True`` 일 때 학습 마지막 20% 구간의 선형 감쇠 계수 (05 §2.3.3)."""
        if not self.aux_decay or int(total_epochs) <= 0:
            return 1.0
        frac = float(epoch) / float(total_epochs)
        if frac <= 0.8:
            return 1.0
        return float(max(0.0, 1.0 - (frac - 0.8) / 0.2))

    # ── 내부 ────────────────────────────────────────────────────────────
    def _radius_for(self, size: int) -> int:
        return int(self.boundary_radius_map.get(int(size), self.boundary_radius))

    def _boundary_target(
        self, y_seg: Tensor, targets: Mapping[str, object]
    ) -> Tuple[Tensor, Tensor]:
        """(bd_gt, bd_valid) 를 만든다 — 정정 A-28 소유권 규칙.

        ``source == "criterion"`` 이면 targets 를 **무시하고** 재생성한다.
        그렇게 하지 않으면 dataset 이 넣은 (전부 0, valid=False) 자리표시자를 그대로 써서
        ``L_edge`` 가 학습 내내 조용히 0 이 되고 ``λ_bd = 20`` 이 무력화된다.
        """
        y_edge = targets.get("y_edge") if self.boundary_source == "dataset" else None
        if isinstance(y_edge, Tensor):
            valid = targets.get("y_edge_valid")
            if isinstance(valid, Tensor):
                v = valid.to(torch.bool)
            else:
                v = torch.ones_like(y_edge, dtype=torch.bool)
            gt = y_edge.float()
            if gt.dim() == 3:  # (B,h,w) -> (B,1,h,w) 관용
                gt = gt.unsqueeze(1)
                v = v.reshape(gt.shape)
            return gt, v

        return make_boundary_target(
            y_seg,
            ignore_index=self.ignore_index,
            radius=self._radius_for(int(y_seg.shape[-2])),
            out_stride=self.boundary_stride,
        )

    def _ohem_kw(self) -> Dict[str, object]:
        return {
            "thresh": self.ohem_thresh,
            "keep_frac": self.ohem_keep_frac,
            "class_weight": self.class_weight,
        }

    # ── forward ─────────────────────────────────────────────────────────
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        step_ctx: StepCtx,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """총 손실과 breakdown 을 낸다.

        Args:
            outputs: ``BloomNet`` train 반환 (키 = ``constants.OUT_*``).
                필수는 ``seg_logits_s4`` 뿐이고 나머지는 있으면 쓴다.
            targets: ``{"y_seg", "y_edge", "y_edge_valid", "y_chl", "y_chl_valid"}``.
                ``y_seg`` 만 필수. ``y_chl`` 은 **log1p 공간**이다 (★X-14).
                (추가) ``"siam_teacher"``: ``{"s3","s4","dec"} -> Tensor`` teacher feature.
            step_ctx: :class:`StepCtx`.

        Returns:
            ``(L_total, breakdown)``. breakdown 값은 **λ 적용 후**의 float 다.
        """
        if "y_seg" not in targets:
            raise KeyError("targets['y_seg'] 는 필수다 (★X-13)")
        if OUT_SEG not in outputs:
            raise KeyError(f"outputs['{OUT_SEG}'] 는 필수다 (05 §0.2)")

        device = outputs[OUT_SEG].device
        with _fp32_ctx(device):
            return self._forward_fp32(outputs, targets, step_ctx)

    def _forward_fp32(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        step_ctx: StepCtx,
    ) -> Tuple[Tensor, Dict[str, float]]:
        y_seg = targets["y_seg"]
        if y_seg.dim() == 4 and y_seg.shape[1] == 1:  # (B,1,H,W) 관용
            y_seg = y_seg[:, 0]
        if y_seg.dim() != 3:
            raise ValueError(f"y_seg 는 (B,H,W) 여야 한다 (got {tuple(y_seg.shape)})")
        y_seg = y_seg.long()
        hw = (int(y_seg.shape[-2]), int(y_seg.shape[-1]))

        seg_logits = outputs[OUT_SEG]
        if int(seg_logits.shape[1]) != self.num_classes:
            raise ValueError(
                f"seg 로짓 채널 {seg_logits.shape[1]} != num_classes {self.num_classes}"
            )
        if self.debug_assert:  # 규칙 N8 — 덮지 않고 드러낸다
            assert torch.isfinite(seg_logits).all(), "criterion: seg_logits 에 비유한 값"

        # ★ 정정 A-32(a): 업샘플을 여기서 **1회만** 하고 ohem/dice/bas 가 공유한다.
        seg_up = seg_logits.float()
        if tuple(seg_up.shape[-2:]) != hw:
            seg_up = F.interpolate(seg_up, size=hw, mode="bilinear", align_corners=False)

        u = float(step_ctx.u) if self.u_source == "step_ctx" else self.u_at(
            step_ctx.epoch, step_ctx.total_epochs
        )
        u = float(min(1.0, max(0.0, u)))

        zero = seg_up.new_zeros(())

        # ── seg (항상 활성) ──────────────────────────────────────────────
        l_ohem = ohem_ce(
            seg_up,
            y_seg,
            ignore_index=self.ignore_index,
            **self._ohem_kw(),  # type: ignore[arg-type]
        )
        l_dice = batch_soft_dice(
            seg_up, y_seg, num_classes=self.num_classes, ignore_index=self.ignore_index
        )

        # ── deep supervision ────────────────────────────────────────────
        y_aux = y_seg
        if self.aux_loss_stride > 1:
            s = self.aux_loss_stride
            if hw[0] % s or hw[1] % s:
                raise ValueError(
                    f"aux_loss_stride={s} 인데 (H,W)={hw} 가 나눠떨어지지 않는다"
                )
            # nearest 축소 = 격자 슬라이싱(좌상단 표본). ignore=255 도 그대로 보존된다.
            y_aux = y_seg[:, ::s, ::s].contiguous()

        # 규칙 N1 의 목적("DDP find_unused_parameters 오류 차단")을 손실에 들어가지 않는
        # 출력에도 적용한다. 아래 텐서들은 **정확히 0** 을 기여하되 graph 를 연결해
        # 해당 헤드의 파라미터가 grad=0 을 받게 한다. 값에는 영향이 없다.
        keepalive: List[Tensor] = []

        aux_scale = self.aux_scale(step_ctx.epoch, step_ctx.total_epochs)
        aux_terms: Dict[str, Tensor] = {}
        for tap, key in OUT_AUX.items():
            if tap in self.lambda_aux:
                continue
            unused = outputs.get(key)
            if isinstance(unused, Tensor):  # 모델이 냈지만 lambda_aux 에 없는 tap
                keepalive.append(unused.float().sum() * 0.0)
        for tap in sorted(self.lambda_aux):
            key = OUT_AUX[tap]
            logits = outputs.get(key)
            if not isinstance(logits, Tensor):
                continue
            if self.debug_assert:
                assert torch.isfinite(logits).all(), f"criterion: {key} 에 비유한 값"
            aux_terms[tap] = plain_ce(
                logits.float(),
                y_aux,
                ignore_index=self.ignore_index,
                class_weight=self.class_weight,
            )

        # ── 경계 (L_edge + L_bas) ───────────────────────────────────────
        edge_logits = outputs.get(OUT_EDGE)
        if isinstance(edge_logits, Tensor):
            if self.debug_assert:
                assert torch.isfinite(edge_logits).all(), "criterion: edge_logits 에 비유한 값"
            e = edge_logits.float()
            bd_gt, bd_valid = self._boundary_target(y_seg, targets)
            l_edge = boundary_bce(e, bd_gt, bd_valid)
            l_bas = bas_loss(
                seg_up,
                y_seg,
                e,
                tau=self.bas_tau,
                ignore_index=self.ignore_index,
                **self._ohem_kw(),  # type: ignore[arg-type]
            )
            vb = bd_valid.to(torch.bool)
            n_valid_bd = vb.sum().to(seg_up.dtype)
            edge_pos_ratio = ((bd_gt > 0.5) & vb).sum().to(seg_up.dtype) / n_valid_bd.clamp_min(1.0)
        else:
            # 헤드 자체가 없으면 연결할 파라미터도 없다 → 상수 0 으로 충분(규칙 N1 무관).
            l_edge = zero
            l_bas = zero
            edge_pos_ratio = zero

        # ── Chl-a 회귀 + aleatoric ──────────────────────────────────────
        chl = outputs.get(OUT_CHL)
        if isinstance(chl, Tensor):
            log_var = outputs.get(OUT_LOGVAR)
            # u == 0 이면 log_var 는 손실에 전혀 들어가지 않는다 — 불필요한
            # 업샘플 할당을 피하려고 None 으로 넘긴다 (수치적으로 동일).
            lv = log_var if (isinstance(log_var, Tensor) and u > 0.0) else None
            l_reg, reg_info = chl_reg_loss(
                chl.float(),
                lv.float() if lv is not None else None,
                targets.get("y_chl"),
                targets.get("y_chl_valid"),
                beta=self.huber_beta,
                u=u,
                clamp=self.unc_clamp,
                debug_assert=self.debug_assert,
            )
            n_chl_valid = float(reg_info.get("n_valid", 0.0))
            # UncHead 는 warmup(u=0) 또는 라벨 부재에서 손실에 전혀 들어가지 않는다.
            # 그대로 두면 DDP(find_unused_parameters=False)가 그 65개 파라미터에서 죽는다.
            if isinstance(log_var, Tensor) and (lv is None or n_chl_valid <= 0.0):
                keepalive.append(log_var.float().sum() * 0.0)
        else:
            l_reg = zero
            n_chl_valid = 0.0

        # ── SIAM distillation (기본 λ=0, 코드 경로만 유지 — 05 §2.6.3) ──
        l_siam = zero
        teacher = targets.get(TEACHER_KEY)
        if self.lambda_siam > 0.0 and isinstance(teacher, _ABCMapping):
            pairs: List[Tensor] = []
            for name, out_key in OUT_SIAM.items():
                s_feat = outputs.get(out_key)
                t_feat = teacher.get(name)
                if isinstance(s_feat, Tensor) and isinstance(t_feat, Tensor):
                    w = float(self.siam_pair_weights.get(name, 1.0))
                    pairs.append(
                        w * cwd_loss(s_feat.float(), t_feat.float(), T=self.siam_temperature)
                    )
            if pairs:
                l_siam = torch.stack(pairs).sum()

        # ── 총합 (05 §1.1 단일 방정식) ──────────────────────────────────
        t_ohem = self.lambda_seg * l_ohem
        t_dice = self.lambda_seg * self.w_dice * l_dice
        t_edge = self.lambda_bd * l_edge
        t_bas = self.lambda_bas * l_bas
        t_reg = self.lambda_reg * l_reg
        t_siam = self.lambda_siam * l_siam
        t_aux = {
            tap: (self.lambda_aux[tap] * aux_scale) * loss for tap, loss in aux_terms.items()
        }

        total = t_ohem + t_dice + t_edge + t_bas + t_reg + t_siam
        for term in t_aux.values():
            total = total + term
        for term in keepalive:  # 값 0, graph 연결 전용
            total = total + term

        # ── breakdown (λ 적용 후). 스칼라를 한 번에 옮겨 동기화 1회로 줄인다 ──
        names: List[str] = [
            "loss_total", "loss_ohem", "loss_dice", "loss_edge", "loss_bas",
            "loss_reg", "loss_siam", "n_edge_pos_ratio",
        ]
        vals: List[Tensor] = [
            total, t_ohem, t_dice, t_edge, t_bas, t_reg, t_siam, edge_pos_ratio,
        ]
        # aux 는 **설정된 전 tap** 을 낸다 — 출력에 없으면 0.0. 모드 간 CSV 열 고정(05 §3.2).
        for tap in sorted(self.lambda_aux):
            names.append(f"loss_aux_{tap}")
            vals.append(t_aux.get(tap, zero))

        flat = torch.stack([v.detach().reshape(()) for v in vals]).to("cpu", torch.float64)
        breakdown: Dict[str, float] = {n: float(x) for n, x in zip(names, flat.tolist())}
        breakdown["n_chl_valid"] = n_chl_valid
        breakdown["u_ramp"] = u
        return total, breakdown

    def extra_repr(self) -> str:  # pragma: no cover - 디버그 편의
        return (
            f"K={self.num_classes}, λ_seg={self.lambda_seg}, λ_reg={self.lambda_reg}, "
            f"λ_bd={self.lambda_bd}, λ_bas={self.lambda_bas}, λ_siam={self.lambda_siam}, "
            f"w_dice={self.w_dice}, boundary_source={self.boundary_source!r}"
        )


def build_criterion(cfg: object) -> BloomNetCriterion:
    """``BloomNetConfig`` → criterion. (동결표 밖의 편의 함수)

    ``cfg.loss`` / ``cfg.data.boundary`` / ``cfg.model`` 의 키를 그대로 옮긴다.
    ``boundary_radius`` 는 ``cfg.boundary_radius(crop_size)`` 로 해상도별 매핑을
    해소하고, 매핑 전체도 ``boundary_radius_map`` 으로 넘겨 평가 해상도가 달라져도
    반경이 맞게 한다 (01 C-10).
    """
    lo = cfg.loss  # type: ignore[attr-defined]
    data = cfg.data  # type: ignore[attr-defined]
    cw = torch.as_tensor(lo.class_weight, dtype=torch.float32) if lo.class_weight else None
    train_h = int(tuple(data.train_size)[0])
    return BloomNetCriterion(
        num_classes=int(data.num_classes),
        ignore_index=int(data.ignore_index),
        lambda_seg=lo.lambda_seg,
        lambda_reg=lo.lambda_reg,
        lambda_bd=lo.lambda_bd,
        lambda_bas=lo.lambda_bas,
        lambda_aux={t: lo.lambda_aux[t] for t in cfg.model.aux_taps},  # type: ignore[attr-defined]
        lambda_siam=lo.lambda_siam,
        w_dice=lo.w_dice,
        ohem_thresh=lo.ohem_thresh,
        ohem_keep_frac=lo.ohem_keep_frac,
        bas_tau=lo.bas_tau,
        huber_beta=lo.huber_beta,
        unc_clamp=(float(lo.unc_clamp[0]), float(lo.unc_clamp[1])),
        u_warm_frac=lo.u_warm_frac,
        u_ramp_frac=lo.u_ramp_frac,
        boundary_source=data.boundary.source,
        boundary_radius=int(cfg.boundary_radius(train_h)),  # type: ignore[attr-defined]
        boundary_stride=int(data.boundary.stride),
        aux_loss_stride=int(lo.aux_loss_stride),
        class_weight=cw,
        aux_decay=bool(lo.aux_decay),
        debug_assert=bool(lo.debug_assert),
        boundary_radius_map=dict(data.boundary.radius),
    )
