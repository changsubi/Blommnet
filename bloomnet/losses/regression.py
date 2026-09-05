"""Chl-a 회귀 + aleatoric 손실 프리미티브 — 05 §2.5 / API 동결 06 §3.5 (레벨 L1).

★ **X-14 (동결)**: ``y_chl`` 은 **이미 log1p(mg/m³) 공간**이며 dataset 이 변환한다.
   여기서 ``log1p`` 를 다시 적용하지 않는다 (05 초판 의사코드의 ``- log1p(y)`` 는 폐기).
★ **X-26 / 정정 A-20**: ``clamp`` 기본값은 ``(-7.0, +7.0)`` 으로 UncHead 의
   ``S_MIN/S_MAX`` 와 일치한다. 값이 같으므로 이중 clamp 는 무해한 항등이 된다.
★ **정정 A-33**: 이중-log 하드가드는 ``debug_assert=True`` 일 때만 수행한다.
   상시 실행하면 스텝마다 GPU→CPU 동기화 2회 + 가변 크기 할당 + graph break 가 생긴다.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["chl_reg_loss", "margin_rank_loss", "u_ramp"]

# X-14 이중-log 방어 임계: log1p(2980) = 8.0. 원단위 mg/m³ 가 들어오면 즉시 걸린다.
_LOG1P_MAX = 8.0


def chl_reg_loss(
    chl_u: Tensor,
    log_var: Optional[Tensor],
    y_chl: Optional[Tensor],
    m_chl: Optional[Tensor],
    *,
    beta: float = 1.0,
    u: float = 0.0,
    clamp: Tuple[float, float] = (-7.0, 7.0),
    debug_assert: bool = False,
) -> Tuple[Tensor, Dict[str, float]]:
    """Heteroscedastic Huber-NLL ("attenuated SmoothL1") — 05 §2.5.2/§2.5.3.

    ``per_px = exp(-u*s) * SmoothL1_beta(chl_u - y_chl) + 0.5*u*s``,  ``s = clamp(log_var)``

    ``u == 0`` (또는 ``log_var is None``) 에서 ``SmoothL1(δ=beta)`` 로 **정확히** 환원되므로
    사업문서 계약(δ=1.0 Huber)의 상위 호환 일반화다.

    Args:
        chl_u: ``(B,1,h,w)`` softplus 출력, **log1p 공간**. 임의 해상도.
        log_var: ``(B,1,h,w)`` raw ``s = log σ²`` 또는 None.
        y_chl: ``(B,1,H,W)`` **log1p 공간** 타깃 (★X-14) 또는 None.
        m_chl: ``(B,1,H,W)`` bool 유효 픽셀 마스크 또는 None.
        beta: SmoothL1 의 δ. dense head 는 1.0 (X-11).
        u: aleatoric ramp ∈ [0,1] (:func:`u_ramp`). criterion 이 유일 소유자(정정 A-34).
        clamp: ``log_var`` clamp 구간. 기본 ``(-7,7)`` = UncHead 와 동일.
        debug_assert: True 면 유한성 + 이중-log 하드가드를 검사한다(동기화 발생).

    Returns:
        ``(loss, {"n_valid": float})``. 분모는 ``numel`` 이 아니라 **유효 픽셀 수**다
        (D-LOSS-05). 라벨 부재/유효 픽셀 0 이면 ``chl_u.sum() * 0.0`` (규칙 N1).
    """
    if (y_chl is None) or (m_chl is None):
        return chl_u.sum() * 0.0, {"n_valid": 0.0}  # 규칙 N1

    m_bool = m_chl.to(torch.bool)

    if debug_assert:
        # N8: 업스트림 inf/NaN 은 덮지 않고 드러낸다.
        assert torch.isfinite(chl_u).all(), "chl_reg_loss: chl_u 에 비유한 값이 있다"
        if log_var is not None:
            assert torch.isfinite(log_var).all(), "chl_reg_loss: log_var 에 비유한 값이 있다"
        # X-14 이중-log 하드가드: 원단위 mg/m³ 가 들어오면 여기서 잡힌다.
        if bool(m_bool.any()):
            y_max = float(y_chl[m_bool].max())
            assert y_max < _LOG1P_MAX, (
                f"chl_reg_loss: y_chl 최대값 {y_max:.4f} >= {_LOG1P_MAX} — "
                "y_chl 은 log1p 공간이어야 한다 (★X-14). 원단위 mg/m³ 를 넣은 것 같다."
            )

    # 해상도 정렬 — 예측을 라벨 해상도로 올린다 (라벨을 내리지 않는다).
    if tuple(chl_u.shape[-2:]) != tuple(y_chl.shape[-2:]):
        size = (int(y_chl.shape[-2]), int(y_chl.shape[-1]))
        chl_u = F.interpolate(chl_u, size=size, mode="bilinear", align_corners=False)
        if log_var is not None:
            log_var = F.interpolate(log_var, size=size, mode="bilinear", align_corners=False)

    m = m_bool.to(chl_u.dtype)
    n_valid = m.sum()
    if float(n_valid) < 1.0:  # ← 오늘(라벨 부재)의 정상 경로
        return chl_u.sum() * 0.0, {"n_valid": 0.0}

    zero = chl_u.new_zeros(())
    # torch.where 로 무효 픽셀의 잔차를 0 으로 만든다. 단순 곱셈(`r * m`)은 라벨이
    # NaN/inf 일 때 0*NaN=NaN 으로 전파되지만, where 는 forward/backward 모두 안전하다.
    r = torch.where(m_bool, chl_u.float() - y_chl.float(), zero)
    h = F.smooth_l1_loss(r, torch.zeros_like(r), beta=beta, reduction="none")

    if (log_var is None) or (u <= 0.0):
        per_px = h
    else:
        s = torch.where(m_bool, log_var.float().clamp(clamp[0], clamp[1]), zero)  # 규칙 N5
        per_px = torch.exp(-u * s) * h + 0.5 * u * s

    loss = (per_px * m).sum() / n_valid.clamp_min(1.0)
    return loss, {"n_valid": float(n_valid)}


def margin_rank_loss(
    pred: Tensor,
    target: Tensor,
    *,
    margin: float = 0.1,
    n_pairs: Optional[int] = None,
) -> Tensor:
    """배치 내 무작위 쌍에 대한 margin ranking loss — 01 §6.6 (S0-Spec 보조).

    ``max(0, margin - sign(y_i - y_j) * (ŷ_i - ŷ_j))`` 의 쌍 평균.
    근거: [M8] 군간 분산 93.4% → 절대값 회귀는 site 편향에 지배되지만 서열은
    site 내부에서도 의미가 있다.

    Args:
        pred: ``(B,)`` 또는 ``(B,1)`` 예측.
        target: ``pred`` 와 같은 shape 의 타깃.
        margin: 기본 0.1 (01 §6.6).
        n_pairs: 샘플할 쌍 수. None 이면 ``B``. ``i != j`` 가 보장된다.

    Returns:
        스칼라. ``B < 2`` 면 ``pred.sum() * 0.0`` (규칙 N1 — 쌍을 만들 수 없다).
    """
    p = pred.reshape(-1)
    y = target.reshape(-1).to(p.dtype)
    b = p.numel()
    if b < 2:
        return pred.sum() * 0.0  # 규칙 N1

    k = int(n_pairs) if n_pairs is not None else b
    if k < 1:
        return pred.sum() * 0.0

    i = torch.randint(0, b, (k,), device=p.device)
    # offset ∈ [1, b-1] 로 j 를 만들어 i == j (sign=0 → 상수 margin) 자명 쌍을 배제한다.
    off = torch.randint(1, b, (k,), device=p.device)
    j = (i + off) % b

    sgn = torch.sign(y[i] - y[j])
    return F.relu(margin - sgn * (p[i] - p[j])).mean()


def u_ramp(
    epoch: int,
    total_epochs: int,
    *,
    warm_frac: float = 0.30,
    ramp_frac: float = 0.20,
) -> float:
    """Aleatoric degenerate-solution guard 램프 0.0 → 1.0 (05 §2.5.4).

    ``E_warm = round(warm_frac * total_epochs)``, ``E_ramp = round(ramp_frac * total_epochs)``.
    ``u`` 는 스위치가 아니라 곱(``exp(-u*s)``, ``0.5*u*s``)으로 들어가므로 손실 점프가 없다.
    total_epochs=80 → E_warm=24, E_ramp=16 (05 §5.2.3 과 일치).
    """
    e_warm = int(round(warm_frac * total_epochs))
    e_ramp = int(round(ramp_frac * total_epochs))
    if epoch < e_warm:
        return 0.0
    if e_ramp <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - e_warm) / e_ramp)))
