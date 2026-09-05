"""경계 지표 — BF-score(τ ∈ {1,3,5}) · Boundary IoU · edge head PRF.

정의 (01 §8.1):

* **BF-score**: pred/gt 경계를 01 §7.5 연산자의 ``radius=0`` 으로 뽑고
  ``P = |{p ∈ B_pred : dist(p, B_gt) <= τ}| / |B_pred|``,
  ``R = |{g ∈ B_gt : dist(g, B_pred) <= τ}| / |B_gt|``, ``BF = 2PR/(P+R)``.
  거리는 ``scipy.ndimage.distance_transform_edt`` (유클리드).
* **Boundary IoU** (Cheng et al. 2021): 마스크 내부의 폭 ``d`` 경계 띠끼리의 IoU.
  띠 추출은 침식(``-max_pool2d(-x)``)으로 하며 01 §7.5 경계 연산자와 **무관한 별개 정의**다.

경계 **추출**은 ``bloomnet.data.boundary.make_boundary_target`` 단일 구현에 위임한다 (★X-07).
이 파일은 두 번째 경계 연산자를 만들지 않는다.

레벨 L1.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import IGNORE_INDEX  # L−1 레벨 예외 (정정 A-23)
from bloomnet.data.boundary import make_boundary_target  # L0 (L1 -> L0 은 적법)

__all__ = [
    "DEFAULT_TOLERANCES",
    "boundary_from_mask",
    "bf_score_counts",
    "bf_score",
    "boundary_iou_counts",
    "boundary_iou",
    "edge_prf",
    "BoundaryAccumulator",
]

DEFAULT_TOLERANCES: Tuple[int, ...] = (1, 3, 5)


def boundary_from_mask(
    mask: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    radius: int = 0,
) -> Tuple[Tensor, Tensor]:
    """클래스 마스크 → 원해상도 경계/유효 마스크 (01 §8.1 = §7.5 연산자, radius=0, stride=1).

    Args:
        mask: ``(H,W)`` 또는 ``(B,H,W)`` int64.

    Returns:
        ``(edge, valid)`` — ``(B,1,H,W)`` float32 {0,1} / bool. ``(H,W)`` 입력이면 ``(1,H,W)``.
    """
    return make_boundary_target(mask, ignore_index=ignore_index, radius=radius, out_stride=1)


def _as_b1hw(x: Tensor) -> Tensor:
    """``(H,W)`` / ``(B,H,W)`` / ``(B,1,H,W)`` 를 ``(B,1,H,W)`` 로 통일한다."""
    if x.dim() == 2:
        return x[None, None]
    if x.dim() == 3:
        return x[:, None]
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError(f"channel dim must be 1, got {tuple(x.shape)}")
        return x
    raise ValueError(f"expected (H,W)/(B,H,W)/(B,1,H,W), got {tuple(x.shape)}")


def _edt_to_true(binary: np.ndarray) -> np.ndarray:
    """각 화소에서 ``binary == True`` 인 가장 가까운 화소까지의 유클리드 거리."""
    from scipy.ndimage import distance_transform_edt  # 런타임 의존성 (requirements.txt)

    if not binary.any():
        return np.full(binary.shape, np.inf, dtype=np.float64)
    return distance_transform_edt(~binary)


def bf_score_counts(
    pred_edge: Tensor,
    gt_edge: Tensor,
    *,
    tolerances: Sequence[int] = DEFAULT_TOLERANCES,
    valid: Optional[Tensor] = None,
) -> Dict[int, Tuple[int, int, int, int]]:
    """BF-score 의 **누적 가능한 카운트**를 낸다.

    Args:
        pred_edge, gt_edge: ``(B,1,H,W)`` (또는 ``(H,W)``/``(B,H,W)``) {0,1} 경계 맵.
        valid: 감독 유효 마스크. False 인 화소는 pred·gt 양쪽에서 제외한다.

    Returns:
        ``{τ: (n_pred, n_pred_matched, n_gt, n_gt_matched)}``.
    """
    p = _as_b1hw(pred_edge).detach().cpu() > 0.5
    g = _as_b1hw(gt_edge).detach().cpu() > 0.5
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {tuple(p.shape)} vs gt {tuple(g.shape)}")
    if valid is not None:
        v = _as_b1hw(valid).detach().cpu().to(torch.bool)
        p = p & v
        g = g & v

    out: Dict[int, List[int]] = {int(t): [0, 0, 0, 0] for t in tolerances}
    pn = p.numpy()[:, 0]
    gn = g.numpy()[:, 0]
    for b in range(pn.shape[0]):
        pb, gb = pn[b], gn[b]
        d_to_g = _edt_to_true(gb)
        d_to_p = _edt_to_true(pb)
        for t in out:
            acc = out[t]
            acc[0] += int(pb.sum())
            acc[1] += int((pb & (d_to_g <= t)).sum())
            acc[2] += int(gb.sum())
            acc[3] += int((gb & (d_to_p <= t)).sum())
    return {t: (v[0], v[1], v[2], v[3]) for t, v in out.items()}


def _prf_from_counts(
    n_pred: int, n_pred_hit: int, n_gt: int, n_gt_hit: int
) -> Tuple[float, float, float]:
    if n_pred == 0 and n_gt == 0:
        return (1.0, 1.0, 1.0)          # 경계가 없는 영상끼리 완전 일치
    if n_pred == 0 or n_gt == 0:
        return (0.0, 0.0, 0.0)
    prec = n_pred_hit / n_pred
    rec = n_gt_hit / n_gt
    f1 = 0.0 if (prec + rec) <= 0.0 else 2.0 * prec * rec / (prec + rec)
    return (prec, rec, f1)


def bf_score(
    pred_edge: Tensor,
    gt_edge: Tensor,
    *,
    tolerances: Sequence[int] = DEFAULT_TOLERANCES,
    valid: Optional[Tensor] = None,
) -> Dict[str, float]:
    """단발 BF-score. 반환 키: ``bf_{τ}px`` / ``bf_precision_{τ}px`` / ``bf_recall_{τ}px``."""
    counts = bf_score_counts(pred_edge, gt_edge, tolerances=tolerances, valid=valid)
    out: Dict[str, float] = {}
    for t, (np_, nph, ng, ngh) in counts.items():
        prec, rec, f1 = _prf_from_counts(np_, nph, ng, ngh)
        out[f"bf_{t}px"] = f1
        out[f"bf_precision_{t}px"] = prec
        out[f"bf_recall_{t}px"] = rec
    return out


def _erode(binary: Tensor, d: int) -> Tensor:
    """``(B,1,H,W)`` bool 을 반경 ``d`` 로 침식한다 (Chebyshev, ``-max_pool2d(-x)``)."""
    if d <= 0:
        return binary
    k = 2 * d + 1
    x = binary.to(torch.float32)
    # 영상 밖을 배경으로 보아 테두리도 침식되게 한다 (Boundary IoU 원 정의와 동일).
    x = F.pad(x, (d, d, d, d), mode="constant", value=0.0)
    return (-F.max_pool2d(-x, k, stride=1, padding=0)) > 0.5


def boundary_iou_counts(
    pred: Tensor,
    gt: Tensor,
    *,
    num_classes: int,
    tolerances: Sequence[int] = DEFAULT_TOLERANCES,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[int, Tensor]:
    """Boundary IoU 의 클래스별 (교집합, 합집합) 카운트.

    Returns:
        ``{d: (2, num_classes) int64}`` — 행 0 = 교집합, 행 1 = 합집합.
    """
    p = _as_b1hw(pred).detach().cpu().to(torch.int64)
    g = _as_b1hw(gt).detach().cpu().to(torch.int64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {tuple(p.shape)} vs gt {tuple(g.shape)}")
    keep = g != ignore_index

    out: Dict[int, Tensor] = {}
    for d in tolerances:
        acc = torch.zeros((2, num_classes), dtype=torch.int64)
        for c in range(num_classes):
            gm = (g == c) & keep
            pm = (p == c) & keep
            gb = gm & ~_erode(gm, int(d))
            pb = pm & ~_erode(pm, int(d))
            acc[0, c] = int((gb & pb).sum())
            acc[1, c] = int((gb | pb).sum())
        out[int(d)] = acc
    return out


def boundary_iou(
    pred: Tensor,
    gt: Tensor,
    *,
    num_classes: int,
    tolerances: Sequence[int] = DEFAULT_TOLERANCES,
    ignore_index: int = IGNORE_INDEX,
) -> Dict[str, float]:
    """단발 Boundary IoU. ``union == 0`` 인 클래스는 평균에서 제외한다 (C-12 규약과 정합).

    반환 키: ``biou_{d}px``.
    """
    counts = boundary_iou_counts(
        pred, gt, num_classes=num_classes, tolerances=tolerances, ignore_index=ignore_index
    )
    return {f"biou_{d}px": _mean_iou_from_counts(acc) for d, acc in counts.items()}


def _mean_iou_from_counts(acc: Tensor) -> float:
    inter = acc[0].to(torch.float64)
    union = acc[1].to(torch.float64)
    valid = union > 0
    if not bool(valid.any()):
        return float("nan")
    return float((inter[valid] / union[valid]).mean())


def edge_prf(
    edge_logits: Tensor,
    y_edge: Tensor,
    *,
    valid: Optional[Tensor] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """EdgeHead 로짓의 픽셀 단위 precision/recall/F1 + 예측 양성비.

    ``val_boundary_metrics.csv`` 의 ``edge_*`` 열(05 §6.1)에 대응한다.
    """
    logit = _as_b1hw(edge_logits).detach().float()
    tgt = _as_b1hw(y_edge).detach().float()
    if logit.shape[-2:] != tgt.shape[-2:]:
        logit = F.interpolate(logit, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
    pred = torch.sigmoid(logit) > threshold
    gt = tgt > 0.5
    if valid is not None:
        v = _as_b1hw(valid).detach().to(torch.bool)
        pred = pred & v
        gt = gt & v
        denom = int(v.sum())
    else:
        denom = int(pred.numel())

    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    prec = float("nan") if (tp + fp) <= 0 else tp / (tp + fp)
    rec = float("nan") if (tp + fn) <= 0 else tp / (tp + fn)
    if tp + fp <= 0 or tp + fn <= 0 or (prec + rec) <= 0:
        f1 = 0.0 if (tp + fn) > 0 or (tp + fp) > 0 else float("nan")
    else:
        f1 = 2.0 * prec * rec / (prec + rec)
    return {
        "edge_precision": prec,
        "edge_recall": rec,
        "edge_f1": f1,
        "edge_pos_ratio_pred": (float(pred.sum()) / denom) if denom > 0 else float("nan"),
    }


class BoundaryAccumulator:
    """BF-score + Boundary IoU 스트리밍 누적기.

    Args:
        num_classes: Boundary IoU 용 클래스 수.
        tolerances: τ (BF-score) 겸 d (Boundary IoU).
        boundary_fn: 경계 추출기. ``None`` 이면 :func:`boundary_from_mask` (= X-07 단일 구현).
            테스트에서 ``bloomnet.data.boundary`` 없이 돌리기 위한 주입 지점이다.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        tolerances: Sequence[int] = DEFAULT_TOLERANCES,
        ignore_index: int = IGNORE_INDEX,
        boundary_fn: Optional[Callable[..., Tuple[Tensor, Tensor]]] = None,
    ) -> None:
        self.num_classes = int(num_classes)
        self.tolerances = tuple(int(t) for t in tolerances)
        self.ignore_index = int(ignore_index)
        self._boundary_fn = boundary_fn
        self.reset()

    def reset(self) -> None:
        self._bf = {t: [0, 0, 0, 0] for t in self.tolerances}
        self._biou = {
            t: torch.zeros((2, self.num_classes), dtype=torch.int64) for t in self.tolerances
        }
        self._edge_tp = 0.0
        self._edge_fp = 0.0
        self._edge_fn = 0.0
        self._edge_pred_pos = 0.0
        self._edge_total = 0.0

    def update_from_edges(
        self, pred_edge: Tensor, gt_edge: Tensor, *, valid: Optional[Tensor] = None
    ) -> None:
        """이미 뽑아 둔 경계 맵으로 BF-score 카운트만 누적한다."""
        counts = bf_score_counts(pred_edge, gt_edge, tolerances=self.tolerances, valid=valid)
        for t, c in counts.items():
            acc = self._bf[t]
            for i in range(4):
                acc[i] += c[i]

    def update(self, pred: Tensor, gt: Tensor) -> None:
        """예측/GT 클래스 마스크로 BF-score 와 Boundary IoU 를 모두 누적한다."""
        fn = self._boundary_fn or boundary_from_mask
        p_edge, _ = fn(pred, ignore_index=self.ignore_index, radius=0)
        g_edge, g_valid = fn(gt, ignore_index=self.ignore_index, radius=0)
        self.update_from_edges(p_edge, g_edge, valid=g_valid)
        counts = boundary_iou_counts(
            pred,
            gt,
            num_classes=self.num_classes,
            tolerances=self.tolerances,
            ignore_index=self.ignore_index,
        )
        for d, acc in counts.items():
            self._biou[d] += acc

    def update_edge_head(
        self,
        edge_logits: Tensor,
        y_edge: Tensor,
        *,
        valid: Optional[Tensor] = None,
        threshold: float = 0.5,
    ) -> None:
        """EdgeHead 로짓 통계 누적."""
        logit = _as_b1hw(edge_logits).detach().float()
        tgt = _as_b1hw(y_edge).detach().float()
        if logit.shape[-2:] != tgt.shape[-2:]:
            logit = F.interpolate(logit, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
        pred = torch.sigmoid(logit) > threshold
        gt = tgt > 0.5
        if valid is not None:
            v = _as_b1hw(valid).detach().to(torch.bool)
            pred, gt = pred & v, gt & v
            self._edge_total += float(v.sum())
        else:
            self._edge_total += float(pred.numel())
        self._edge_tp += float((pred & gt).sum())
        self._edge_fp += float((pred & ~gt).sum())
        self._edge_fn += float((~pred & gt).sum())
        self._edge_pred_pos += float(pred.sum())

    def compute(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for t, c in self._bf.items():
            prec, rec, f1 = _prf_from_counts(c[0], c[1], c[2], c[3])
            out[f"bf_{t}px"] = f1
            out[f"bf_precision_{t}px"] = prec
            out[f"bf_recall_{t}px"] = rec
        for d, acc in self._biou.items():
            out[f"biou_{d}px"] = _mean_iou_from_counts(acc)
        tp, fp, fn = self._edge_tp, self._edge_fp, self._edge_fn
        out["edge_precision"] = float("nan") if (tp + fp) <= 0 else tp / (tp + fp)
        out["edge_recall"] = float("nan") if (tp + fn) <= 0 else tp / (tp + fn)
        p, r = out["edge_precision"], out["edge_recall"]
        if p != p or r != r or (p + r) <= 0:  # nan 체크
            out["edge_f1"] = float("nan") if (p != p and r != r) else 0.0
        else:
            out["edge_f1"] = 2.0 * p * r / (p + r)
        out["edge_pos_ratio_pred"] = (
            self._edge_pred_pos / self._edge_total if self._edge_total > 0 else float("nan")
        )
        return out
