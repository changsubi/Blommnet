"""평가기 + 배치/autocast 공용 헬퍼 — 05 §6 / 01 §8.1 (06 §3.6 동결 시그니처).

세 평가기 모두 `utils/` 의 **스트리밍 누적기 위에 얹은 얇은 어댑터**다. 지표 계산을
여기서 다시 구현하지 않는다 (X-20: mIoU 두 번째 구현 금지).

* :class:`SegEvaluator`      → `utils.metrics_seg.ConfusionMatrix` + `bootstrap_ci`
* :class:`RegEvaluator`      → `utils.metrics_reg.RegressionAccumulator`
* :class:`BoundaryEvaluator` → `utils.metrics_boundary.BoundaryAccumulator`

레벨 L3. 따라서 **`losses/criterion.py`(같은 L3)와 `engine/trainer.py`(L6)를 import 하지
않는다.** 배치 언팩·autocast 헬퍼가 여기에 있는 이유가 그것이다 — trainer(L6)가
이 파일에서 가져다 쓴다(L6→L3, 합법). criterion 이 요구하는 `StepCtx` 는
:class:`EvalStepCtx` 로 **duck-typing** 한다(필드 이름만 같으면 되고 import 는 불필요).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from bloomnet.constants import (
    ALARM_THRESHOLDS_MGM3,
    IGNORE_INDEX,
    OUT_CHL,
    OUT_EDGE,
    OUT_LOGVAR,
    OUT_SEG,
    TRANSFER_CLASS_IDS,
)
from bloomnet.data.bundle import SAMPLE_META_KEY, SAMPLE_TARGET_KEYS
from bloomnet.utils.metrics_boundary import BoundaryAccumulator
from bloomnet.utils.metrics_reg import RegressionAccumulator
from bloomnet.utils.metrics_seg import (
    ConfusionMatrix,
    bootstrap_ci,
    compute_mean_iou,
    confusion_from_logits,
    mean_iou_on_classes,
    per_class_iou,
    per_class_metric_rows,
    pixel_accuracy,
    present_classes,
)

__all__ = [
    "SegEvaluator",
    "RegEvaluator",
    "BoundaryEvaluator",
    "evaluate",
    "EvalStepCtx",
    "DEFAULT_CI_WIDTH_MAX",
    "MODEL_INPUT_KEYS",
    "move_batch",
    "model_inputs_from_batch",
    "targets_from_batch",
    "autocast_dtype",
    "autocast_ctx",
    "dispatch_evaluators",
]

#: (정정 A-4) 군 단위 부트스트랩 95 % CI 폭이 이 값을 넘으면 '판정 불가(indeterminate)'.
#: 스펙이 수치를 동결하지 않아 본 구현이 정한 기본값이다. 판정 리포트에 함께 기록할 것.
DEFAULT_CI_WIDTH_MAX: float = 0.10

#: `BloomNet.forward` 가 받는 모달 텐서 키 (06 §3.4.3).
MODEL_INPUT_KEYS: Tuple[str, ...] = ("rgb", "msi", "bio", "ir", "pol")


# ═══════════════════════════════════════════════════════════════════════
#  배치 / autocast 헬퍼 (trainer 와 공유)
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class EvalStepCtx:
    """`losses.criterion.StepCtx` 와 **필드 이름이 같은** 값 컨테이너.

    evaluator(L3)가 criterion(L3)을 import 할 수 없어서 존재한다. criterion 은
    이 객체의 속성만 읽으므로 duck-typing 으로 충분하다. 학습 경로는 trainer(L6)가
    진짜 `StepCtx` 를 만들어 쓴다.
    """

    epoch: int = 0
    total_epochs: int = 1
    global_step: int = 0
    total_steps: int = 1
    prec_ramp: float = 1.0
    u: float = 0.0


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """텐서만 device 로 옮긴다. ``meta``(list[dict])·``all_missing``(dict)은 그대로."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, Tensor) else v
    return out


def model_inputs_from_batch(batch: Dict[str, Any], cfg: Any = None) -> Dict[str, Any]:
    """콜레이트 배치 → ``BloomNet.forward`` kwargs (06 §3.4.3).

    비활성 모달 키는 배치에 아예 없다(A4 계약)이므로 `None` 으로 남는다.
    ``band_ids``/``phys_slot_ids`` 는 cfg 에서 온다 — 배치 meta 의 값은 샘플별이라
    모델 생성 시점 계약과 어긋날 수 있어 쓰지 않는다.
    """
    kwargs: Dict[str, Any] = {k: batch.get(k) for k in MODEL_INPUT_KEYS}
    if kwargs["rgb"] is None:
        raise KeyError("배치에 'rgb' 가 없다 — 헌법 C-3 상 rgb 는 항상 존재한다(V4)")
    kwargs["avail"] = batch.get("avail")
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is not None:
        if getattr(data_cfg, "band_ids", None) is not None:
            kwargs["band_ids"] = list(data_cfg.band_ids)
        if getattr(data_cfg, "phys_slot_ids", None) is not None:
            kwargs["phys_slot_ids"] = list(data_cfg.phys_slot_ids)
    return kwargs


def targets_from_batch(batch: Dict[str, Any]) -> Dict[str, Tensor]:
    """콜레이트 배치 → criterion `targets` (06 §3.5 / X-13·X-14·X-16).

    ``y_chl`` 은 **log1p 공간**이며 dataset 이 이미 변환해 둔다(X-14).
    """
    return {k: batch[k] for k in SAMPLE_TARGET_KEYS if k in batch}


def autocast_dtype(cfg: Any) -> Optional[torch.dtype]:
    """``cfg.train.amp`` ∈ {off, bf16, fp16} → dtype (off 이면 None)."""
    mode = str(getattr(getattr(cfg, "train", None), "amp", "off")).lower()
    if mode in ("off", "none", "fp32", "false"):
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    raise ValueError(f"train.amp 는 off|bf16|fp16 이어야 한다 (받은 값 {mode!r})")


def autocast_ctx(cfg: Any, device: torch.device, *, enabled: bool = True) -> ContextManager:
    """모델 forward 용 autocast. 손실은 **이 컨텍스트 밖**에서 계산한다(05 §5.1.4 loss_dtype=fp32)."""
    dtype = autocast_dtype(cfg) if enabled else None
    if dtype is None or device.type not in ("cuda", "cpu", "xpu"):
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


# ═══════════════════════════════════════════════════════════════════════
#  Segmentation
# ═══════════════════════════════════════════════════════════════════════
class SegEvaluator:
    """mIoU / per-class IoU / pixel acc / transfer_score / 군 단위 부트스트랩 CI.

    mIoU 규약은 **union==0 클래스 제외** 하나뿐이다 (05 §6.2, X-20).

    군(flight-line) 단위 부트스트랩 (정정 A-4):
        :meth:`set_groups` 로 **다음 `update()` 배치의 샘플별 군 id** 를 주면 군별
        혼동행렬을 따로 쌓고 :meth:`compute` 가 95 % CI 를 낸다. 이미지 단위가 아니라
        군 단위인 이유는 같은 flight-line 프레임이 근사 중복이라 이미지 단위 리샘플이
        CI 폭을 체계적으로 과소평가하기 때문이다. 군 정보를 한 번도 주지 않으면
        CI 값은 nan 이고 `*_indeterminate` 가 True 다.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = IGNORE_INDEX,
        class_names: Optional[Sequence[str]] = None,
        *,
        transfer_class_ids: Sequence[int] = TRANSFER_CLASS_IDS,
        common_classes: Optional[Sequence[int]] = None,
        n_boot: int = 1000,
        ci_width_max: float = DEFAULT_CI_WIDTH_MAX,
        seed: int = 0,
    ) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.class_names = list(class_names) if class_names is not None else None
        self.transfer_class_ids = tuple(int(c) for c in transfer_class_ids)
        self.common_classes: Optional[Tuple[int, ...]] = (
            tuple(int(c) for c in common_classes) if common_classes is not None else None
        )
        self.n_boot = int(n_boot)
        self.ci_width_max = float(ci_width_max)
        self.seed = int(seed)
        self.cm = ConfusionMatrix(self.num_classes, self.ignore_index)
        self._groups: Dict[str, Tensor] = {}
        self._pending: Optional[List[str]] = None

    # ------------------------------------------------------------------ config
    def set_groups(self, group_ids: Sequence[str]) -> None:
        """**다음** :meth:`update` 배치의 샘플별 군 id (길이 = 배치 크기).

        `evaluate()` 는 `batch["meta"][i]["group_key"]` 에서 자동으로 채운다.
        """
        self._pending = [str(g) for g in group_ids]

    def set_common_classes(self, class_ids: Optional[Sequence[int]]) -> None:
        """`miou_common` 의 대상 `C_common` (정정 A-4 K2).

        None 이면 `miou_common == miou`. 합격 판정에서는 `group_split_manifest.json`
        의 면제 클래스(실측 `{8}`)를 뺀 공통 present 집합을 넣어야 한다 — 면제 목록은
        **데이터 산출물**이므로 평가기에 하드코딩하지 않는다.
        """
        self.common_classes = None if class_ids is None else tuple(int(c) for c in class_ids)

    def reset(self) -> None:
        self.cm.reset()
        self._groups.clear()
        self._pending = None

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(self, logits: Tensor, target: Tensor) -> None:
        """``logits`` (B,K,h,w) 임의 해상도 / ``target`` (B,H,W) int64.

        해상도가 다르면 **로짓을 타깃 해상도로** 올린다 (라벨을 내리지 않는다, 04 §8.1).
        """
        pending, self._pending = self._pending, None
        if pending is not None and len(pending) != int(target.shape[0]):
            raise ValueError(f"set_groups 길이 {len(pending)} != 배치 크기 {int(target.shape[0])}")
        if pending is None:
            self.cm.matrix += confusion_from_logits(
                logits, target, self.num_classes, self.ignore_index
            )
            return
        for i, gid in enumerate(pending):
            part = confusion_from_logits(
                logits[i : i + 1], target[i : i + 1], self.num_classes, self.ignore_index
            )
            self.cm.matrix += part
            acc = self._groups.get(gid)
            self._groups[gid] = part if acc is None else acc + part

    # ----------------------------------------------------------------- compute
    def compute(self) -> Dict[str, Any]:
        cm = self.cm.matrix
        present = present_classes(cm)
        common = self.common_classes if self.common_classes is not None else present
        out: Dict[str, Any] = {
            "miou": compute_mean_iou(cm),
            "per_class_iou": per_class_iou(cm).tolist(),
            "pixel_acc": pixel_accuracy(cm),
            "present_classes": present,
            "common_classes": list(common),
            "miou_common": mean_iou_on_classes(cm, common),
            "transfer_score": mean_iou_on_classes(cm, self.transfer_class_ids),
            "rows": per_class_metric_rows(cm, 0, class_names=self.class_names),
            "n_groups": len(self._groups),
        }
        out["bootstrap_ci"] = self._bootstrap()
        return out

    def _bootstrap(self) -> Dict[str, Any]:
        def _common(m: Tensor) -> float:
            ids = self.common_classes if self.common_classes is not None else present_classes(m)
            return mean_iou_on_classes(m, ids)

        stats = {
            "miou": compute_mean_iou,
            "miou_common": _common,
            "transfer_score": (lambda m: mean_iou_on_classes(m, self.transfer_class_ids)),
        }
        ci: Dict[str, Any] = {"ci_width_max": self.ci_width_max}
        mats = [self._groups[k] for k in sorted(self._groups)]
        for key, fn in stats.items():
            if not mats:
                ci[key] = (float("nan"), float("nan"))
                ci[key + "_indeterminate"] = True
                continue
            _, lo, hi = bootstrap_ci(mats, fn, n_boot=self.n_boot, seed=self.seed)
            ci[key] = (lo, hi)
            width = hi - lo
            ci[key + "_indeterminate"] = bool(math.isnan(width) or width > self.ci_width_max)
        return ci


# ═══════════════════════════════════════════════════════════════════════
#  Chl-a 회귀
# ═══════════════════════════════════════════════════════════════════════
class RegEvaluator:
    """Chl-a 회귀 지표 (05 §6.1 `val_regression_metrics.csv`).

    `RegressionAccumulator` 를 그대로 위임한다. 반환은 06 시그니처의
    ``Dict[str, float]`` 보다 넓은 ``Dict[str, Any]`` 다 — 정정 A-30 이 `f1_at_100` 을
    **값 대신 None + 사유 문자열**로 요구하므로 float 만으로는 계약을 표현할 수 없다.
    """

    def __init__(
        self,
        *,
        thresholds_mgm3: Sequence[float] = ALARM_THRESHOLDS_MGM3,
        max_rank_samples: int = 200_000,
        seed: int = 0,
        debug_assert: bool = False,
    ) -> None:
        self.acc = RegressionAccumulator(
            thresholds_mgm3=thresholds_mgm3,
            max_rank_samples=max_rank_samples,
            seed=seed,
            debug_assert=debug_assert,
        )

    def reset(self) -> None:
        self.acc.reset()

    @torch.no_grad()
    def update(
        self,
        chl_u: Tensor,
        y_chl: Tensor,
        m: Tensor,
        log_var: Optional[Tensor] = None,
    ) -> None:
        """``chl_u`` = softplus 출력(log1p 공간), ``y_chl`` = **log1p 공간** 타깃 (X-14)."""
        if chl_u.shape[-2:] != y_chl.shape[-2:]:
            size = tuple(int(s) for s in y_chl.shape[-2:])
            chl_u = F.interpolate(chl_u.float(), size=size, mode="bilinear", align_corners=False)
            if log_var is not None:
                log_var = F.interpolate(
                    log_var.float(), size=size, mode="bilinear", align_corners=False
                )
        self.acc.update(chl_u, y_chl, m, log_var)

    def compute(self) -> Dict[str, Any]:
        return self.acc.compute()


# ═══════════════════════════════════════════════════════════════════════
#  경계
# ═══════════════════════════════════════════════════════════════════════
class BoundaryEvaluator:
    """BF-score(τ=1/3/5) · Boundary IoU · EdgeHead PRF.

    ``num_classes`` 가 06 동결 시그니처에 없으므로 **첫 `update` 의 `seg_logits`
    채널 수로 지연 결정**한다. 명시하려면 keyword 로 넘겨라.
    """

    def __init__(
        self,
        tolerances: Tuple[int, ...] = (1, 3, 5),
        *,
        num_classes: Optional[int] = None,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.tolerances = tuple(int(t) for t in tolerances)
        self.ignore_index = int(ignore_index)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.acc: Optional[BoundaryAccumulator] = (
            self._make(self.num_classes) if self.num_classes is not None else None
        )

    def _make(self, num_classes: int) -> BoundaryAccumulator:
        return BoundaryAccumulator(
            num_classes=int(num_classes),
            tolerances=self.tolerances,
            ignore_index=self.ignore_index,
        )

    def reset(self) -> None:
        if self.acc is not None:
            self.acc.reset()

    @torch.no_grad()
    def update(
        self,
        seg_logits: Tensor,
        y_seg: Tensor,
        edge_logits: Optional[Tensor] = None,
        *,
        y_edge: Optional[Tensor] = None,
        y_edge_valid: Optional[Tensor] = None,
    ) -> None:
        if self.acc is None:
            self.num_classes = int(seg_logits.shape[1])
            self.acc = self._make(self.num_classes)
        logits = seg_logits.float()
        if logits.shape[-2:] != y_seg.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=tuple(int(s) for s in y_seg.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
        pred = logits.argmax(dim=1)
        # ignore 픽셀은 예측을 GT 로 덮어 경계 추출에서 배제한다 (X-07 의 valid 규약과 정합).
        pred = torch.where(y_seg == self.ignore_index, y_seg, pred)
        self.acc.update(pred, y_seg)
        if edge_logits is not None and y_edge is not None:
            self.acc.update_edge_head(edge_logits, y_edge, valid=y_edge_valid)

    def compute(self) -> Dict[str, float]:
        return {} if self.acc is None else self.acc.compute()


# ═══════════════════════════════════════════════════════════════════════
#  평가 루프
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: Optional[nn.Module],
    *,
    cfg: Any,
    evaluators: Sequence[Any],
    device: torch.device,
    amp: bool,
    step_ctx: Optional[Any] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """검증 1회 순회. 평가기별 `compute()` 를 소문자 이름(`seg`/`reg`/`boundary`)으로 묶는다.

    Args:
        criterion: None 이면 손실을 계산하지 않는다(지표 전용 평가).
        step_ctx: criterion 에 넘길 컨텍스트. None 이면 :class:`EvalStepCtx` 기본값
            (`u=0`, `prec_ramp=1`)을 쓴다 — 검증은 램프의 영향을 받지 않아야 한다.
        max_batches: dry-run/스모크용 조기 종료.
    """
    was_training = model.training
    model.eval()
    if criterion is not None:
        criterion.eval()
    for ev in evaluators:
        if hasattr(ev, "reset"):
            ev.reset()
    ctx = step_ctx if step_ctx is not None else EvalStepCtx(prec_ramp=1.0, u=0.0)

    total_loss, n_batches = 0.0, 0
    breakdown: Dict[str, float] = {}
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= int(max_batches):
            break
        batch = move_batch(batch, device)
        inputs = model_inputs_from_batch(batch, cfg)
        targets = targets_from_batch(batch)
        with autocast_ctx(cfg, device, enabled=amp):
            outputs = model(**inputs)
        outputs = {k: (v.float() if isinstance(v, Tensor) else v) for k, v in outputs.items()}

        if criterion is not None:
            loss, parts = criterion(outputs, targets, ctx)
            total_loss += float(loss.detach())
            for k, v in parts.items():
                breakdown[k] = breakdown.get(k, 0.0) + float(v)
            n_batches += 1
        dispatch_evaluators(evaluators, outputs, targets, batch)

    result: Dict[str, Any] = {}
    if n_batches:
        result["loss"] = total_loss / n_batches
        result["breakdown"] = {k: v / n_batches for k, v in breakdown.items()}
    for ev in evaluators:
        result[_evaluator_key(ev)] = ev.compute()
    model.train(was_training)
    return result


def _evaluator_key(ev: Any) -> str:
    name = type(ev).__name__
    return (name[: -len("Evaluator")] if name.endswith("Evaluator") else name).lower()


def dispatch_evaluators(
    evaluators: Sequence[Any],
    outputs: Dict[str, Any],
    targets: Dict[str, Tensor],
    batch: Dict[str, Any],
) -> None:
    """평가기 종류별로 필요한 텐서만 골라 `update` 한다 (trainer 와 공유)."""
    seg = outputs.get(OUT_SEG)
    y_seg = targets.get("y_seg")
    for ev in evaluators:
        if isinstance(ev, SegEvaluator):
            if seg is None or y_seg is None:
                continue
            metas = batch.get(SAMPLE_META_KEY)
            if isinstance(metas, list) and metas and isinstance(metas[0], dict):
                ev.set_groups(
                    [str(m.get("group_key", m.get("stem", i))) for i, m in enumerate(metas)]
                )
            ev.update(seg, y_seg)
        elif isinstance(ev, RegEvaluator):
            chl, y_chl = outputs.get(OUT_CHL), targets.get("y_chl")
            if chl is None or y_chl is None:
                continue
            m = targets.get("y_chl_valid")
            if m is None:
                m = torch.ones_like(y_chl, dtype=torch.bool)
            ev.update(chl, y_chl, m, outputs.get(OUT_LOGVAR))
        elif isinstance(ev, BoundaryEvaluator):
            if seg is None or y_seg is None:
                continue
            ev.update(
                seg,
                y_seg,
                outputs.get(OUT_EDGE),
                y_edge=targets.get("y_edge"),
                y_edge_valid=targets.get("y_edge_valid"),
            )
