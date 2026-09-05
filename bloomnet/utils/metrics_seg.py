"""Segmentation 지표 — 혼동행렬 누적 · mIoU(absent 제외) · per-class 행.

**mIoU 규약 (★X-20 / 01 C-12 / 05 §6.2)**
``union == 0`` 인 클래스는 평균에서 **제외**한다. 이 파일의 :func:`compute_mean_iou` 가
전 코드베이스의 **유일 구현**이며, ``eval_tta`` 계열의 두 번째 구현은 작성 금지다.

레벨 L0. ``constants.py`` 는 레벨 예외(정정 A-23)이므로 import 할 수 있다.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from bloomnet.constants import IGNORE_INDEX  # L−1 레벨 예외 (정정 A-23)

__all__ = [
    "IGNORE_INDEX",
    "new_confusion_matrix",
    "update_confusion_matrix",
    "confusion_from_logits",
    "ConfusionMatrix",
    "compute_mean_iou",
    "per_class_iou",
    "present_classes",
    "mean_iou_on_classes",
    "pixel_accuracy",
    "per_class_metric_rows",
    "bootstrap_ci",
]


def new_confusion_matrix(num_classes: int, *, device: Optional[torch.device] = None) -> Tensor:
    """``(K, K)`` int64 영행렬. 행 = GT, 열 = 예측."""
    return torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)


def update_confusion_matrix(
    confusion_matrix: Tensor,
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """혼동행렬을 **제자리** 누적한다 ([분석] §E.1.1 copy-as-is).

    ``targets`` 가 ``ignore_index`` 이거나 ``[0, num_classes)`` 밖이면 그 픽셀은 버린다.
    ``predictions`` 는 ``[0, num_classes-1]`` 로 clamp 한다(참조 구현과 동일 동작).
    """
    predictions = predictions.detach().reshape(-1).to(torch.int64)
    targets = targets.detach().reshape(-1).to(torch.int64)
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if not torch.any(valid):
        return
    indices = targets[valid] * num_classes + predictions[valid].clamp(0, num_classes - 1)
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion_matrix += counts.reshape(num_classes, num_classes).to(confusion_matrix.device)


def confusion_from_logits(
    logits: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """``(B,K,h,w)`` 로짓과 ``(B,H,W)`` 라벨에서 혼동행렬을 만든다.

    로짓 해상도가 라벨과 다르면 **로짓을 라벨 해상도로 올린다**(04 §8.1: 라벨을 nearest 로
    내리면 얇은 경계가 파괴된다).
    """
    if logits.shape[-2:] != target.shape[-2:]:
        logits = torch.nn.functional.interpolate(
            logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    pred = logits.argmax(dim=1)
    cm = new_confusion_matrix(num_classes, device=torch.device("cpu"))
    update_confusion_matrix(cm, pred.cpu(), target.cpu(), num_classes, ignore_index)
    return cm


class ConfusionMatrix:
    """혼동행렬 누적기 (지표 계산은 모듈 함수에 위임한다)."""

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.matrix = new_confusion_matrix(self.num_classes)

    def reset(self) -> None:
        self.matrix = new_confusion_matrix(self.num_classes)

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        update_confusion_matrix(
            self.matrix, predictions, targets, self.num_classes, self.ignore_index
        )

    def update_from_logits(self, logits: Tensor, targets: Tensor) -> None:
        self.matrix += confusion_from_logits(
            logits, targets, self.num_classes, self.ignore_index
        )

    def compute(self, *, class_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        return {
            "miou": compute_mean_iou(self.matrix),
            "per_class_iou": per_class_iou(self.matrix).tolist(),
            "present_classes": present_classes(self.matrix),
            "pixel_acc": pixel_accuracy(self.matrix),
            "rows": per_class_metric_rows(self.matrix, 0, class_names=class_names),
        }


def _iou_parts(confusion_matrix: Tensor) -> Tuple[Tensor, Tensor]:
    matrix = confusion_matrix.detach().cpu().to(torch.float64)
    intersection = torch.diag(matrix)
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - intersection
    return intersection, union


def compute_mean_iou(confusion_matrix: Tensor) -> float:
    """mIoU. ``union == 0`` 인 absent 클래스는 **평균에서 제외**한다 (★X-20 단일 구현).

    present 클래스가 하나도 없으면 ``0.0`` 을 반환한다(참조 구현과 동일).
    """
    intersection, union = _iou_parts(confusion_matrix)
    valid = union > 0
    if not torch.any(valid):
        return 0.0
    return float((intersection[valid] / union[valid].clamp_min(1)).mean().item())


def per_class_iou(confusion_matrix: Tensor) -> Tensor:
    """클래스별 IoU ``(K,)`` float64. absent 클래스는 ``nan``.

    ``nan`` 으로 두는 이유: 0.0 으로 채우면 호출자가 실수로 ``mean()`` 을 불러
    X-20 규약(제외)을 우회하게 된다.
    """
    intersection, union = _iou_parts(confusion_matrix)
    out = torch.full_like(intersection, float("nan"))
    valid = union > 0
    out[valid] = intersection[valid] / union[valid]
    return out


def present_classes(confusion_matrix: Tensor) -> List[int]:
    """``union > 0`` 인 클래스 id 목록 (compute_mean_iou 의 판정과 **동일** 기준)."""
    _, union = _iou_parts(confusion_matrix)
    return [int(i) for i in torch.nonzero(union > 0, as_tuple=False).reshape(-1).tolist()]


def mean_iou_on_classes(confusion_matrix: Tensor, class_ids: Sequence[int]) -> float:
    """지정한 클래스 집합 위에서만 평균한 mIoU (01 §8.1 ``mIoU_C`` / K2, 정정 A-4).

    ``class_ids`` 중 absent(union==0)인 클래스는 여전히 제외한다. 남는 클래스가 없으면
    ``nan`` 을 반환한다 — 0.0 은 "성능이 0" 과 구별되지 않아 K2 판정을 오염시킨다.
    """
    ious = per_class_iou(confusion_matrix)
    vals = [
        float(ious[c])
        for c in class_ids
        if 0 <= int(c) < ious.numel() and not math.isnan(float(ious[c]))
    ]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def pixel_accuracy(confusion_matrix: Tensor) -> float:
    """유효 픽셀에 대한 전체 정확도. 유효 픽셀 0 이면 ``nan``."""
    matrix = confusion_matrix.detach().cpu().to(torch.float64)
    total = float(matrix.sum().item())
    if total <= 0.0:
        return float("nan")
    return float(torch.diag(matrix).sum().item()) / total


def _safe_metric(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def per_class_metric_rows(
    confusion_matrix: Tensor,
    epoch: int,
    *,
    class_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """``val_per_class_metrics.csv`` 의 행 ([분석] §E.1.1 재사용).

    ``class_names`` 를 주면 generic ``class_{id}`` 대신 실제 이름을 넣는다 (05 §6.1).
    ``epoch`` 은 참조 구현과 동일하게 **+1 하여** 기록한다(1-based).
    """
    matrix = confusion_matrix.detach().cpu().to(torch.float64)
    true_positive = torch.diag(matrix)
    gt_pixels = matrix.sum(dim=1)
    pred_pixels = matrix.sum(dim=0)
    false_positive = pred_pixels - true_positive
    false_negative = gt_pixels - true_positive
    union = gt_pixels + pred_pixels - true_positive
    total_valid_pixels = float(matrix.sum().item())

    rows: List[Dict[str, Any]] = []
    for class_id in range(matrix.shape[0]):
        tp = float(true_positive[class_id].item())
        fp = float(false_positive[class_id].item())
        fn = float(false_negative[class_id].item())
        gt_count = float(gt_pixels[class_id].item())
        pred_count = float(pred_pixels[class_id].item())
        union_count = float(union[class_id].item())
        if class_names is not None and class_id < len(class_names):
            name = str(class_names[class_id])
        else:
            name = f"class_{class_id}"
        rows.append(
            {
                "epoch": epoch + 1,
                "class_id": class_id,
                "class_name": name,
                "gt_pixels": int(gt_count),
                "pred_pixels": int(pred_count),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": _safe_metric(tp, pred_count),
                "recall": _safe_metric(tp, gt_count),
                "iou": _safe_metric(tp, union_count),
                "gt_pixel_ratio": _safe_metric(gt_count, total_valid_pixels),
                "pred_pixel_ratio": _safe_metric(pred_count, total_valid_pixels),
                "gt_present": gt_count > 0,
                "pred_present": pred_count > 0,
                "gt_present_pred_missing": gt_count > 0 and pred_count == 0,
                "pred_present_gt_missing": pred_count > 0 and gt_count == 0,
            }
        )
    return rows


def bootstrap_ci(
    group_confusions: Sequence[Tensor],
    statistic: Callable[[Tensor], float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """군(flight-line) 단위 부트스트랩 백분위수 CI (정정 A-4 / 01 §8.1).

    Args:
        group_confusions: 군별 혼동행렬 리스트. **이미지 단위가 아니라 군 단위**로
            리샘플해야 한다 — 같은 flight-line 의 프레임은 근사 중복이라
            이미지 단위 리샘플은 CI 폭을 체계적으로 과소평가한다.
        statistic: 합산 혼동행렬 → 스칼라 (예: :func:`compute_mean_iou`).
        n_boot: 리샘플 횟수 (01 §8.1 규약 = 1,000).
        alpha: 0.05 → 95 % CI.

    Returns:
        ``(point, lo, hi)``. 군이 하나도 없으면 전부 ``nan``.
    """
    n = len(group_confusions)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    stacked = torch.stack([c.detach().cpu().to(torch.int64) for c in group_confusions], dim=0)
    point = float(statistic(stacked.sum(dim=0)))

    g = torch.Generator().manual_seed(int(seed))
    samples: List[float] = []
    for _ in range(int(n_boot)):
        idx = torch.randint(0, n, (n,), generator=g)
        val = float(statistic(stacked[idx].sum(dim=0)))
        if not math.isnan(val):
            samples.append(val)
    if not samples:
        return (point, float("nan"), float("nan"))
    vals = torch.tensor(sorted(samples), dtype=torch.float64)
    lo = float(torch.quantile(vals, alpha / 2.0).item())
    hi = float(torch.quantile(vals, 1.0 - alpha / 2.0).item())
    return (point, lo, hi)
