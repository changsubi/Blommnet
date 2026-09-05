"""분할(segmentation) 손실 프리미티브 — 05 §2.1~2.3 / API 동결 06 §3.5 (레벨 L1).

세 함수 모두 **임의 해상도 로짓**을 받아 내부에서 라벨 해상도로 bilinear 업샘플한다
(라벨을 내리지 않는다 — 04 §8.1). criterion(L3)이 업샘플 로짓을 1회만 계산해
공유하는 최적화(정정 A-32(a))를 쓰더라도, 이미 해상도가 맞으면 여기서는 no-op 이다.

NaN 차단 규칙 **N1** (05 §3.3): 유효 픽셀이 0개이면 ``logits.sum() * 0.0`` 을 반환한다.
``torch.tensor(0.0)`` 이 아니라 **예측 텐서를 곱해** 반환해야 DDP
``find_unused_parameters`` 오류를 피하고 autograd graph 가 끊기지 않는다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["ohem_ce", "batch_soft_dice", "plain_ce"]


def _upsample_logits_to(logits: Tensor, hw: Tuple[int, int]) -> Tensor:
    """로짓을 라벨 해상도로 올린다. 이미 같으면 원본을 그대로 돌려준다."""
    if tuple(logits.shape[-2:]) != tuple(hw):
        logits = F.interpolate(
            logits, size=(int(hw[0]), int(hw[1])), mode="bilinear", align_corners=False
        )
    return logits


def ohem_ce(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = 255,
    thresh: float = 0.7,
    keep_frac: float = 0.0625,
    class_weight: Optional[Tensor] = None,
) -> Tensor:
    """OHEM Cross-Entropy (PIDNet ``OhemCrossEntropy`` 의미론, 05 §2.1.1).

    Args:
        logits: ``(B, K, h, w)`` 임의 해상도.
        target: ``(B, H, W)`` int64, class id, ignore = ``ignore_index``.
        thresh: ``keep = p_gt < max(kth, thresh)``. **클수록 마이닝이 약하다**(05 §2.1.3).
        keep_frac: ``min_kept = max(1, int(keep_frac * n_valid))``. PIDNet 의 절대값
            131,072 를 해상도·배치 불변으로 상대화한 값.
        class_weight: ``(K,)`` 또는 None.

    Returns:
        스칼라 텐서. 유효 픽셀 0개면 ``logits.sum() * 0.0`` (규칙 N1).
    """
    logits = _upsample_logits_to(logits, target.shape[-2:])

    # 1) 픽셀별 CE. ignore 픽셀은 여기서 0 이 되지만 **평균 분모에서는 빠지지 않으므로**
    #    아래 valid 마스크로 반드시 별도 제외한다 (05 §2.1.2).
    ce = F.cross_entropy(
        logits,
        target,
        weight=class_weight,
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(-1)

    # 2) 유효 픽셀 마스크 — **정렬 이전에** 제거한다.
    valid = (target.reshape(-1) != ignore_index)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return logits.sum() * 0.0  # 규칙 N1

    # 3) GT 클래스의 예측 확률
    prob = F.softmax(logits.float(), dim=1)
    tmp = target.masked_fill(target == ignore_index, 0)  # gather 안전화
    p_gt = prob.gather(1, tmp.unsqueeze(1)).reshape(-1)[valid]  # (n_valid,)
    ce_v = ce[valid]

    # 4) 적응적 임계값: "적어도 min_kept 개는 유지" + "p_gt < thresh 는 모두 유지"
    min_kept = max(1, int(keep_frac * n_valid))
    p_sorted, _ = torch.sort(p_gt, stable=True)  # 오름차순 = 어려운 순 (재현성, 05 §7.1)
    kth = p_sorted[min(min_kept, n_valid) - 1]
    thr = torch.maximum(kth, p_gt.new_tensor(thresh))
    keep = p_gt < thr

    # 5) 동점 등으로 keep 이 비면 하위 min_kept 개를 강제 채택 (규칙 N3)
    if int(keep.sum()) == 0:
        idx = torch.argsort(p_gt, stable=True)[:min_kept]
        keep = torch.zeros_like(p_gt, dtype=torch.bool)
        keep[idx] = True

    return ce_v[keep].mean()


def batch_soft_dice(
    logits: Tensor,
    target: Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1.0,
) -> Tensor:
    """배치 전체를 하나로 집계하는 soft-Dice (05 §2.2.2).

    설계 3원칙:
      1. ``dims=(0,2,3)`` — 이미지 단위 집계는 양성 0 인 이미지에서 발산적으로 불안정.
      2. GT 부재 클래스는 평균에서 제외 (``present``) — 배치 구성에 따른 요동 제거.
      3. ``eps`` 는 **raw 픽셀 카운트**에 더한다 — 비율에 더하면 희소 클래스에서
         Dice≈1 이 되어 기울기가 사라진다.

    one-hot 은 ``F.one_hot``(int64) 대신 ``scatter_`` float 로 만든다
    (정정 A-32(b)/B-22(c): B=32,K=12,512² 에서 805 MB → 402 MB).
    """
    logits = _upsample_logits_to(logits, target.shape[-2:])

    prob = F.softmax(logits.float(), dim=1)  # (B,K,H,W)
    valid = (target != ignore_index)  # (B,H,W)
    if not bool(valid.any()):
        return logits.sum() * 0.0  # 규칙 N1

    t = target.masked_fill(~valid, 0)
    onehot = torch.zeros_like(prob).scatter_(1, t.unsqueeze(1), 1.0)  # float one-hot
    m = valid.unsqueeze(1).to(prob.dtype)
    prob = prob * m
    onehot = onehot * m

    dims = (0, 2, 3)  # 배치 전체 집계
    inter = (prob * onehot).sum(dims)  # (K,)
    denom = prob.sum(dims) + onehot.sum(dims)  # (K,)

    present = onehot.sum(dims) > 0  # GT 에 등장한 클래스만
    if not bool(present.any()):
        return logits.sum() * 0.0  # 규칙 N6

    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice[present].mean()


def plain_ce(
    logits: Tensor,
    target: Tensor,
    *,
    ignore_index: int = 255,
    class_weight: Optional[Tensor] = None,
) -> Tensor:
    """마이닝 없는 평균 CE — deep supervision(aux) 전용 (05 §2.3.3).

    ``class_weight`` 가 주어지면 torch 관례대로 **가중 평균**(분모 = 가중치 합)이다.
    유효 픽셀 0개면 규칙 N1 로 0.0 을 반환한다 — torch 의 ``reduction="mean"`` 은
    이 경우 0/0 = NaN 을 낸다.
    """
    logits = _upsample_logits_to(logits, target.shape[-2:])

    if not bool((target != ignore_index).any()):
        return logits.sum() * 0.0  # 규칙 N1

    return F.cross_entropy(
        logits.float(),
        target,
        weight=class_weight,
        ignore_index=ignore_index,
        reduction="mean",
    )
