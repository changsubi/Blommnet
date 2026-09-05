"""append-only CSV 로거 6종 (05 §6.1 — [분석] §E.1.4 패턴 계승).

첫 호출 시 헤더를 쓰고, 이후에는 행만 덧붙인다. ``encoding="utf-8"``, ``newline=""``.

이전 구현의 ``LOSS_FIELDS`` (Mask2Former 전용)는 05 §6.1 의 목록으로 **교체**했다.
손실 breakdown 은 **λ 를 곱한 뒤의 값**을 기록한다 — CSV 만 보고 어느 항이 지배하는지
즉시 판단할 수 있어야 한다.

레벨 L1.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from bloomnet.utils.metrics_seg import per_class_metric_rows

__all__ = [
    "EPOCH_FIELDS",
    "PER_CLASS_FIELDS",
    "PRESENCE_FIELDS",
    "BOUNDARY_FIELDS",
    "REGRESSION_FIELDS",
    "STABILITY_FIELDS",
    "CsvLogger",
    "append_epoch_metrics",
    "append_per_class_metrics",
    "append_presence_summary",
    "append_boundary_metrics",
    "regression_row",
    "append_regression_metrics",
    "append_stability",
]

_LOSS_TERMS: Tuple[str, ...] = (
    "ohem", "dice", "edge", "bas", "aux2", "aux3", "aux4", "reg", "siam",
)

EPOCH_FIELDS: Tuple[str, ...] = (
    "epoch",
    "lr_main", "lr_encoder", "lr_physics",
    "epoch_time_s", "gpu_peak_mb",
    "train_loss", "val_loss",
    *(f"train_loss_{t}" for t in _LOSS_TERMS),
    *(f"val_loss_{t}" for t in _LOSS_TERMS),
    "train_miou", "val_miou", "val_miou_ema", "best_val_miou",
    "val_algae_iou",                      # S1 전용. S0 에서는 NaN
    "n_nonfinite_skips", "grad_norm_mean", "grad_norm_p99", "clip_ratio",
    "n_chl_valid_px", "u_ramp",
    "ignore_ratio",                       # (정정 B-27) 배치별 ignore 픽셀 비율
)

PER_CLASS_FIELDS: Tuple[str, ...] = (
    "epoch", "class_id", "class_name",
    "gt_pixels", "pred_pixels", "tp", "fp", "fn",
    "precision", "recall", "iou",
    "gt_pixel_ratio", "pred_pixel_ratio",
    "gt_present", "pred_present",
    "gt_present_pred_missing", "pred_present_gt_missing",
)

PRESENCE_FIELDS: Tuple[str, ...] = (
    "epoch",
    "gt_present_pred_missing_count", "gt_present_pred_missing_class_ids",
    "pred_present_gt_missing_count", "pred_present_gt_missing_class_ids",
    "low_recall_lt_0_1_count", "low_recall_lt_0_1_class_ids",
    "low_precision_lt_0_1_count", "low_precision_lt_0_1_class_ids",
)

BOUNDARY_FIELDS: Tuple[str, ...] = (
    "epoch",
    "edge_f1", "edge_precision", "edge_recall", "edge_pos_ratio_pred",
    "biou_1px", "biou_3px", "biou_5px",
    # 01 §8.1 이 요구하는 BF-score(τ=1/3/5). 05 §6.1 초판 목록에는 없었으나
    # 합격 기준(경계 지표)이 BF-score 이므로 열로 고정한다.
    "bf_1px", "bf_3px", "bf_5px",
)

REGRESSION_FIELDS: Tuple[str, ...] = (
    "epoch", "n_valid_px",
    "mae_log", "rmse_log", "r2_log",
    "mae_mgm3", "rmse_mgm3", "mape_mgm3", "spearman",
    "nll", "sigma_mean", "sigma_err_spearman",
    "alarm_acc", "alarm_macro_f1",
    "f1_at_15", "f1_at_25", "f1_at_100",
    # (정정 A-30 / B-30) f1_at_100 은 235 에 표본 0개이므로 null + 사유를 남긴다.
    "alarm_excluded_levels", "alarm_exclusion_reason",
    "ignore_ratio",
)

STABILITY_FIELDS: Tuple[str, ...] = (
    "epoch",
    *(f"layerscale_gamma_mean_s{i}" for i in range(1, 5)),
    *(f"bio_beta_s{i}" for i in range(1, 5)),
    *(f"bio_mask_entropy_s{i}" for i in range(1, 5)),
    "bmef_omega_rgb", "bmef_omega_spec", "bmef_omega_phys",
    "glint_a", "glint_b", "log_var_mean", "log_var_std",
)


def _fmt(value: Any) -> Any:
    """CSV 셀 값 정규화. ``None`` → 빈 칸, ``nan`` → ``"nan"``, list → ``|`` join."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, dict):
        return "|".join(f"{k}={v}" for k, v in value.items())
    return value


class CsvLogger:
    """고정 컬럼 append-only CSV 라이터.

    Args:
        path: 출력 경로. 상위 디렉터리는 자동 생성한다.
        fieldnames: 컬럼 목록(동결). 행에 없는 키는 빈 칸으로 채운다.
        preamble: 첫 생성 시 헤더 **앞에** ``# k=v`` 주석 1줄로 기록할 메타데이터.
            (정정 A-18) ``iters_per_epoch``/``warmup_iters``/``total_iters`` 처럼
            런타임에 유도되는 값을 CSV 자체에 남겨 사후 추적을 가능하게 한다.
        strict: True 면 ``fieldnames`` 에 없는 키가 들어오면 ``ValueError``.
    """

    def __init__(
        self,
        path: os.PathLike[str] | str,
        fieldnames: Sequence[str],
        *,
        preamble: Optional[Mapping[str, Any]] = None,
        strict: bool = True,
    ) -> None:
        self.path = Path(path)
        self.fieldnames: List[str] = list(fieldnames)
        self.preamble = dict(preamble) if preamble else None
        self.strict = bool(strict)

    def _ensure_header(self) -> bool:
        exists = self.path.exists() and self.path.stat().st_size > 0
        if not exists:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        return exists

    def append(self, row: Mapping[str, Any]) -> None:
        self.append_many([row])

    def append_many(self, rows: Iterable[Mapping[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        if self.strict:
            allowed = set(self.fieldnames)
            for r in rows:
                unknown = sorted(set(r) - allowed)
                if unknown:
                    raise ValueError(
                        f"{self.path.name}: 미지 컬럼 {unknown} "
                        f"(허용 = {sorted(allowed)})"
                    )
        exists = self._ensure_header()
        with self.path.open("a", newline="", encoding="utf-8") as fp:
            if not exists and self.preamble:
                fp.write("# " + " ".join(f"{k}={v}" for k, v in self.preamble.items()) + "\n")
            writer = csv.DictWriter(fp, fieldnames=self.fieldnames, restval="")
            if not exists:
                writer.writeheader()
            for r in rows:
                writer.writerow({k: _fmt(r.get(k)) for k in self.fieldnames})


def append_epoch_metrics(
    path: os.PathLike[str] | str,
    row: Mapping[str, Any],
    *,
    preamble: Optional[Mapping[str, Any]] = None,
) -> None:
    """``epoch_metrics.csv`` 1행."""
    CsvLogger(path, EPOCH_FIELDS, preamble=preamble).append(row)


def append_per_class_metrics(
    path: os.PathLike[str] | str,
    epoch: int,
    confusion_matrix: Any,
    *,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """``val_per_class_metrics.csv`` — 클래스 수만큼의 행."""
    rows = per_class_metric_rows(confusion_matrix, epoch, class_names=class_names)
    CsvLogger(path, PER_CLASS_FIELDS).append_many(rows)


def append_presence_summary(
    path: os.PathLike[str] | str, epoch: int, confusion_matrix: Any
) -> None:
    """``val_presence_summary.csv`` — recall<0.1 / precision<0.1 / GT 있으나 미예측 클래스."""
    rows = per_class_metric_rows(confusion_matrix, epoch)

    def _ids(pred) -> List[str]:
        return [str(r["class_id"]) for r in rows if pred(r)]

    def _lt(r: Mapping[str, Any], key: str, gate: str) -> bool:
        v = r[key]
        return bool(r[gate]) and isinstance(v, float) and not math.isnan(v) and v < 0.1

    gpm = _ids(lambda r: bool(r["gt_present_pred_missing"]))
    pgm = _ids(lambda r: bool(r["pred_present_gt_missing"]))
    low_r = _ids(lambda r: _lt(r, "recall", "gt_present"))
    low_p = _ids(lambda r: _lt(r, "precision", "pred_present"))
    CsvLogger(path, PRESENCE_FIELDS).append(
        {
            "epoch": epoch + 1,
            "gt_present_pred_missing_count": len(gpm),
            "gt_present_pred_missing_class_ids": "|".join(gpm),
            "pred_present_gt_missing_count": len(pgm),
            "pred_present_gt_missing_class_ids": "|".join(pgm),
            "low_recall_lt_0_1_count": len(low_r),
            "low_recall_lt_0_1_class_ids": "|".join(low_r),
            "low_precision_lt_0_1_count": len(low_p),
            "low_precision_lt_0_1_class_ids": "|".join(low_p),
        }
    )


def append_boundary_metrics(path: os.PathLike[str] | str, row: Mapping[str, Any]) -> None:
    """``val_boundary_metrics.csv`` 1행."""
    CsvLogger(path, BOUNDARY_FIELDS).append(row)


def regression_row(
    epoch: int,
    metrics: Mapping[str, Any],
    *,
    ignore_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """``RegressionAccumulator.compute()`` 결과 → ``val_regression_metrics.csv`` 행.

    누적기의 키 이름과 CSV 컬럼을 **명시적으로** 잇는다. 자동 필터로 넘기면
    누적기 쪽 키 이름이 바뀌었을 때 조용히 빈 칸이 되어 리포트가 거짓말을 한다.
    ``f1_at_100`` 이 ``None`` 이면(정정 A-30) 사유 문자열이
    ``alarm_exclusion_reason`` 에 남는다.
    """
    reasons: List[str] = [
        str(v) for k, v in metrics.items() if k.endswith("_reason") and v
    ]
    extra = metrics.get("alarm_exclusion_reasons")
    if isinstance(extra, Mapping):
        reasons.extend(f"level{k}:{v}" for k, v in extra.items())

    row: Dict[str, Any] = {"epoch": epoch + 1, "n_valid_px": metrics.get("n_valid")}
    for key in (
        "mae_log", "rmse_log", "r2_log", "mae_mgm3", "rmse_mgm3", "mape_mgm3",
        "spearman", "nll", "sigma_mean", "sigma_err_spearman",
        "alarm_acc", "alarm_macro_f1", "f1_at_15", "f1_at_25", "f1_at_100",
    ):
        row[key] = metrics.get(key)
    row["alarm_excluded_levels"] = metrics.get("alarm_excluded_levels")
    row["alarm_exclusion_reason"] = "|".join(dict.fromkeys(reasons)) if reasons else None
    row["ignore_ratio"] = ignore_ratio
    return row


def append_regression_metrics(path: os.PathLike[str] | str, row: Mapping[str, Any]) -> None:
    """``val_regression_metrics.csv`` 1행. 행은 :func:`regression_row` 로 만든다."""
    CsvLogger(path, REGRESSION_FIELDS).append(row)


def append_stability(path: os.PathLike[str] | str, row: Mapping[str, Any]) -> None:
    """``stability.csv`` 1행 (05 §6.1 — bio_beta·layerscale 감시)."""
    CsvLogger(path, STABILITY_FIELDS).append(row)
