"""``bloomnet.utils`` — 재export 전용 (06 §2.1 원칙 3: ``__init__.py`` 에 로직 금지)."""

from __future__ import annotations

from bloomnet.utils.checkpoint import (
    atomic_save,
    collect_rng_state,
    git_commit,
    load_checkpoint,
    load_state_dict_shape_tolerant,
    prune_train_only_keys,
    restore_rng_state,
    save_checkpoint,
)
from bloomnet.utils.distributed import (
    all_reduce_mean,
    all_reduce_sum,
    barrier,
    convert_syncbn,
    get_rank,
    get_world_size,
    is_main_process,
    maybe_convert_syncbn,
)
from bloomnet.utils.flops import (
    MacReport,
    count_macs,
    count_macs_flop_counter,
    count_macs_hooks,
    count_parameters,
    scale_macs,
)
from bloomnet.utils.logging_csv import (
    BOUNDARY_FIELDS,
    EPOCH_FIELDS,
    PER_CLASS_FIELDS,
    PRESENCE_FIELDS,
    REGRESSION_FIELDS,
    STABILITY_FIELDS,
    CsvLogger,
    append_boundary_metrics,
    append_epoch_metrics,
    append_per_class_metrics,
    append_presence_summary,
    append_regression_metrics,
    append_stability,
    regression_row,
)
from bloomnet.utils.metrics_boundary import (
    BoundaryAccumulator,
    bf_score,
    boundary_from_mask,
    boundary_iou,
    edge_prf,
)
from bloomnet.utils.metrics_reg import (
    RegressionAccumulator,
    alarm_level,
    log1p_to_mgm3,
    mgm3_to_log1p,
    spearman_corr,
)
from bloomnet.utils.metrics_seg import (
    ConfusionMatrix,
    bootstrap_ci,
    compute_mean_iou,
    confusion_from_logits,
    mean_iou_on_classes,
    new_confusion_matrix,
    per_class_iou,
    per_class_metric_rows,
    pixel_accuracy,
    present_classes,
    update_confusion_matrix,
)
from bloomnet.utils.seed import make_generator, seed_everything_strict, worker_init_fn

__all__ = [
    # seed
    "seed_everything_strict", "worker_init_fn", "make_generator",
    # metrics_seg
    "new_confusion_matrix", "update_confusion_matrix", "confusion_from_logits",
    "ConfusionMatrix", "compute_mean_iou", "per_class_iou", "present_classes",
    "mean_iou_on_classes", "pixel_accuracy", "per_class_metric_rows", "bootstrap_ci",
    # metrics_reg
    "RegressionAccumulator", "log1p_to_mgm3", "mgm3_to_log1p", "alarm_level", "spearman_corr",
    # metrics_boundary
    "BoundaryAccumulator", "bf_score", "boundary_iou", "boundary_from_mask", "edge_prf",
    # logging_csv
    "CsvLogger", "EPOCH_FIELDS", "PER_CLASS_FIELDS", "PRESENCE_FIELDS",
    "BOUNDARY_FIELDS", "REGRESSION_FIELDS", "STABILITY_FIELDS",
    "append_epoch_metrics", "append_per_class_metrics", "append_presence_summary",
    "append_boundary_metrics", "append_regression_metrics", "append_stability",
    "regression_row",
    # checkpoint
    "atomic_save", "save_checkpoint", "load_checkpoint",
    "load_state_dict_shape_tolerant", "prune_train_only_keys",
    "collect_rng_state", "restore_rng_state", "git_commit",
    # flops
    "MacReport", "count_macs", "count_macs_hooks", "count_macs_flop_counter",
    "count_parameters", "scale_macs",
    # distributed
    "get_rank", "get_world_size", "is_main_process", "barrier",
    "all_reduce_mean", "all_reduce_sum", "convert_syncbn", "maybe_convert_syncbn",
]
