"""학습 루프 — 05 §5 · §6 ([분석] §E.1.5 이전 구현 골격을 계승·적응).

이전 구현 대비 달라진 점 (전부 05/06 이 지시한 것):

1. **dict 배치 언팩** — `(images, labels)` 튜플이 아니라 `bloom_collate` 산출 dict.
2. **`prec_ramp = min(1, epoch / cfg.model.bmef.warmup_epochs)`** 를 모델로 전달
   (정정 A-34. 초판 주석의 하드코딩 `epoch/10` 은 config 를 무시했다).
   `u`(aleatoric 램프)는 `StepCtx.u` 로 **criterion 에만** 전달한다 —
   `unc_enabled` 는 항상 True 이며 학습 스위치가 아니다(이중 스위치 금지).
3. **modality dropout** 배치 단위 샘플링 (03 §8.4, X-21).
4. **비유한 손실 스텝 스킵** — 에폭당 `train.nonfinite_skip_max_per_epoch` 초과 시 중단.
5. **체크포인트 2개 고정** (`best.pt`/`last.pt`, 원자적 저장). 이전 구현 처럼
   `best_epoch_XXX_miou_YYYY.pt` 를 누적하지 않는다 (05 §6.3, 디스크 62 GB 여유).
6. **조기종료 카운터를 체크포인트에 저장** — 이전 구현 는 저장하지 않아 재개 시
   리셋되는 알려진 결함이 있다 ([분석] §E.1.9).

레벨 L6.

Note:
    `models/bloomnet.py`(L5)·`losses/criterion.py`(L3)·`data/build.py`(L3)는 :func:`fit`
    **안에서 지연 import** 한다. 레벨상 합법이지만 아직 존재하지 않는 파일이 있어
    모듈 import 시점에 터지면 `run_epoch` 단위 테스트조차 못 돌기 때문이다.
    `StepCtx` 만은 모듈 상단에서 시도하고, 없으면 필드가 동일한 `EvalStepCtx` 로
    폴백한다 — :data:`STEPCTX_FALLBACK` 이 그 사실을 노출하므로 criterion 이 도착하면
    테스트가 자동으로 정본 경로를 검증한다.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from bloomnet.constants import AUX_TAP_STRIDE, FINE_CLASS_NAMES, OUT_SEG, S1_CLASS_NAMES
from bloomnet.engine.ema import ModelEMA
from bloomnet.engine.evaluator import (
    BoundaryEvaluator,
    EvalStepCtx,
    RegEvaluator,
    SegEvaluator,
    autocast_ctx,
    dispatch_evaluators,
    evaluate,
    model_inputs_from_batch,
    move_batch,
    targets_from_batch,
)
from bloomnet.engine.optim import build_optimizer
from bloomnet.engine.sched import build_lr_scheduler
from bloomnet.losses.regression import u_ramp
from bloomnet.utils.checkpoint import (
    collect_rng_state,
    git_commit,
    load_checkpoint,
    load_state_dict_shape_tolerant,
    prune_train_only_keys,
    restore_rng_state,
    save_checkpoint,
)
from bloomnet.utils.logging_csv import (
    BOUNDARY_FIELDS,
    EPOCH_FIELDS,
    PER_CLASS_FIELDS,
    REGRESSION_FIELDS,
    STABILITY_FIELDS,
    CsvLogger,
    append_presence_summary,
    regression_row,
)
from bloomnet.utils.metrics_seg import ConfusionMatrix, compute_mean_iou
from bloomnet.utils.seed import seed_everything_strict

try:  # 정본 (L3). 없으면 필드가 동일한 폴백을 쓴다.
    from bloomnet.losses.criterion import StepCtx  # type: ignore[attr-defined]

    STEPCTX_FALLBACK = False
except ModuleNotFoundError:  # pragma: no cover - criterion 도착 시 자동 해소
    StepCtx = EvalStepCtx  # type: ignore[assignment,misc]
    STEPCTX_FALLBACK = True

__all__ = [
    "STEPCTX_FALLBACK",
    "EarlyStopper",
    "build_grad_scaler",
    "make_step_ctx",
    "prec_ramp_for_epoch",
    "sample_modality_dropout",
    "run_epoch",
    "fit",
    "checkpoint_payload",
    "class_names_for",
    "collect_stability_row",
    "breakdown_to_csv_columns",
    "AUX_CSV_TAPS",
]


# ═══════════════════════════════════════════════════════════════════════
#  작은 부품 (단위 테스트 대상)
# ═══════════════════════════════════════════════════════════════════════
class EarlyStopper:
    """조기종료 (05 §5.1.4). 카운터는 **체크포인트에 저장**된다 ([분석] §E.1.9 결함 수정).

    ``improved = (mode == "max") ? value > best + min_delta : value < best - min_delta``.
    NaN 은 절대 개선으로 보지 않는다 — val 배치에 클래스가 없어 지표가 nan 이 될 수 있다.
    """

    def __init__(self, *, patience: int = 10, min_delta: float = 1e-4, mode: str = "max") -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be max|min, got {mode!r}")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = -1

    def step(self, value: float, *, epoch: int = -1) -> bool:
        """개선이면 True. 개선이 없으면 카운터를 1 올린다."""
        v = float(value)
        if math.isnan(v):
            self.counter += 1
            return False
        improved = (
            v > self.best + self.min_delta
            if self.mode == "max"
            else v < self.best - self.min_delta
        )
        if improved:
            self.best = v
            self.counter = 0
            self.best_epoch = int(epoch)
        else:
            self.counter += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.counter >= self.patience

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best": self.best,
            "counter": self.counter,
            "best_epoch": self.best_epoch,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.best = float(state.get("best", self.best))
        self.counter = int(state.get("counter", self.counter))
        self.best_epoch = int(state.get("best_epoch", self.best_epoch))


def build_grad_scaler(cfg: Any, device: torch.device) -> torch.amp.GradScaler:
    """05 §5.1.4 — scaler 는 항상 만들되 **fp16 + cuda 에서만 활성**.

    bf16 은 지수 범위가 fp32 와 같아 스케일링이 불필요하고 CPU 는 애초에 대상이 아니다.
    비활성 scaler 는 `scale/unscale_/step/update` 가 전부 항등이라 학습 루프를 분기 없이
    유지할 수 있다 (05 이 "bf16 에서는 무해" 라고 적은 것과 같은 취지).
    """
    mode = str(getattr(getattr(cfg, "train", None), "amp", "off")).lower()
    enabled = mode == "fp16" and device.type == "cuda"
    return torch.amp.GradScaler(device.type, enabled=enabled)


def prec_ramp_for_epoch(cfg: Any, epoch: int) -> float:
    """(정정 A-34) ``prec_ramp = min(1, epoch / cfg.model.bmef.warmup_epochs)``.

    ``warmup_epochs <= 0`` 이면 1.0 (램프 없음). **하드코딩 `epoch/10` 금지.**
    """
    bmef = getattr(getattr(cfg, "model", None), "bmef", None)
    warm = int(getattr(bmef, "warmup_epochs", 0) or 0)
    if warm <= 0:
        return 1.0
    return float(min(1.0, max(0.0, int(epoch) / warm)))


def make_step_ctx(cfg: Any, *, epoch: int, global_step: int, iters_per_epoch: int) -> Any:
    """`StepCtx` 생성 (정정 A-34: `prec_ramp`·`u` 는 오직 여기서 유도된다)."""
    total_epochs = int(getattr(getattr(cfg, "schedule", None), "epochs", 1) or 1)
    loss_cfg = getattr(cfg, "loss", None)
    u = u_ramp(
        int(epoch),
        total_epochs,
        warm_frac=float(getattr(loss_cfg, "u_warm_frac", 0.30)),
        ramp_frac=float(getattr(loss_cfg, "u_ramp_frac", 0.20)),
    )
    return StepCtx(
        epoch=int(epoch),
        total_epochs=total_epochs,
        global_step=int(global_step),
        total_steps=max(1, total_epochs * int(iters_per_epoch)),
        prec_ramp=prec_ramp_for_epoch(cfg, epoch),
        u=u,
    )


def sample_modality_dropout(
    cfg: Any, rng: random.Random, *, active_paths: Optional[Sequence[str]] = None
) -> Dict[str, bool]:
    """배치 단위 modality dropout (03 §8.4, X-21). 반환은 **드롭된 path 만** True.

    * ``mode="subset"``: 확률 ``p_full`` 로 전체 유지, 아니면 보조 path 의 부분집합을
      **균등** 샘플. `rgb` 는 항상 남는다 — `BloomNet.forward` 의 필수 입력이고,
      그래야 S 가 구조적으로 공집합이 될 수 없다(03 §8.4 제약 "S 는 절대 공집합이 아니다").
    * ``mode="bernoulli"``: 보조 path 각각을 ``p_bernoulli`` 로 독립 드롭
      (05 §5.2.3 의 "0.15 per aux modality").

    S0-RGB 처럼 보조 path 가 없으면 항상 빈 dict 다. 결정론: 호출자가 넘긴 ``rng`` 가
    `(seed, epoch)` 로 시드되므로 같은 에폭은 같은 드롭 열을 본다 (05 §7.1).
    """
    md = getattr(getattr(cfg, "train", None), "modality_dropout", None)
    if md is None or not bool(getattr(md, "enabled", False)):
        return {}
    paths = (
        tuple(active_paths)
        if active_paths is not None
        else tuple(getattr(cfg, "active_paths", ("rgb",)))
    )
    aux = [p for p in paths if p != "rgb"]
    if not aux:
        return {}
    mode = str(getattr(md, "mode", "subset"))
    if mode == "bernoulli":
        p = float(getattr(md, "p_bernoulli", 0.15))
        return {m: True for m in aux if rng.random() < p}
    if mode != "subset":
        raise ValueError(f"modality_dropout.mode 는 subset|bernoulli 여야 한다 (받은 값 {mode!r})")
    if rng.random() < float(getattr(md, "p_full", 0.5)):
        return {}
    keep_mask = rng.randrange(2 ** len(aux))  # 보조 path 의 2^n 부분집합을 균등 샘플
    return {m: True for i, m in enumerate(aux) if not (keep_mask >> i) & 1}


def class_names_for(cfg: Any) -> Optional[List[str]]:
    """`val_per_class_metrics.csv` 의 `class_name` 열 (05 §6.1 — generic 이름 금지)."""
    k = int(getattr(getattr(cfg, "data", None), "num_classes", 0) or 0)
    if k == len(FINE_CLASS_NAMES):
        return list(FINE_CLASS_NAMES)
    if k == len(S1_CLASS_NAMES):
        return list(S1_CLASS_NAMES)
    return None


#: `epoch_metrics.csv` 의 `*_loss_aux{2,3,4}` 열에 대응하는 tap 순서 (05 §6.1 + X-06).
#: **stride 로 유도할 수 없다** — `AUX_TAP_STRIDE` 에서 `p8` 도 stride 8 이라
#: `enc_s8` 과 충돌한다. CSV 열은 인코더 tap 3개만 가리킨다.
AUX_CSV_TAPS: Tuple[str, ...] = ("enc_s8", "enc_s16", "enc_s32")


def _aux_column(tap: str) -> Optional[str]:
    """aux tap 이름 → CSV 접미사(`aux2`/`aux3`/`aux4`). 열이 없는 tap 은 None."""
    if tap not in AUX_CSV_TAPS:
        return None
    if AUX_TAP_STRIDE.get(tap) is None:  # constants 와의 정합성 방어
        return None
    return f"aux{AUX_CSV_TAPS.index(tap) + 2}"


def breakdown_to_csv_columns(prefix: str, breakdown: Mapping[str, Any]) -> Dict[str, float]:
    """criterion breakdown → `epoch_metrics.csv` 컬럼 (없는 컬럼은 버린다).

    ``p8`` tap 은 CSV 스키마에 열이 없다 — 05 §6.1 의 열 목록이 `aux2/aux3/aux4` 세 개로
    동결되어 있고, 그 목록을 임의로 넓히면 이전 구현 호환 파서가 깨진다.
    """
    out: Dict[str, float] = {}
    for key, value in breakdown.items():
        if not key.startswith("loss_"):
            continue
        term = key[len("loss_") :]
        if term.startswith("aux_"):
            col = _aux_column(term[len("aux_") :])
            if col is None:
                continue
            term = col
        name = f"{prefix}_loss_{term}"
        if name in EPOCH_FIELDS:
            out[name] = _as_float(value)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  진행 표시 (BLOOMNET_PROGRESS: auto|bar|plain|off, 기본 auto)
#
#  auto = TTY 면 tqdm 막대, 리다이렉트(nohup 로그)면 주기적 한 줄 출력.
#  로그 파일에 tqdm 의 \r 이 쌓이면 읽을 수 없게 되므로 모드를 나눈다.
#  학습 결과에는 영향이 없다 — 순수 표시용.
# ═══════════════════════════════════════════════════════════════════════
def resolve_progress_mode() -> str:
    mode = str(os.environ.get("BLOOMNET_PROGRESS", "auto")).strip().lower()
    if mode not in {"auto", "bar", "plain", "off"}:
        mode = "auto"
    if mode == "auto":
        try:
            mode = "bar" if sys.stderr.isatty() else "plain"
        except Exception:
            mode = "plain"
    if mode == "bar":
        try:
            import tqdm  # noqa: F401
        except Exception:
            mode = "plain"
    return mode


class EpochProgress:
    """loader 를 감싸 진행 상황을 표시한다. `postfix()` 로 손실 등을 갱신."""

    def __init__(
        self,
        iterable: Iterable[Any],
        *,
        total: Optional[int],
        desc: str,
        mode: str,
        plain_interval: float = 60.0,
    ) -> None:
        self._it = iterable
        self._total = total
        self._desc = desc
        self._mode = mode
        self._interval = float(plain_interval)
        self._bar = None
        self._post: Dict[str, float] = {}
        self._t0 = time.perf_counter()
        self._last = self._t0
        self._n = 0

    def __len__(self) -> int:  # DataLoader 호환
        return int(self._total or 0)

    def __iter__(self) -> Iterator[Any]:
        if self._mode == "off":
            yield from self._it
            return
        if self._mode == "bar":
            from tqdm.auto import tqdm

            self._bar = tqdm(self._it, total=self._total, desc=self._desc, dynamic_ncols=True, leave=False)
            yield from self._bar
            return
        # plain: 로그 친화적인 주기적 한 줄
        for item in self._it:
            self._n += 1
            yield item
            now = time.perf_counter()
            if now - self._last >= self._interval:
                self._last = now
                self._emit(now)

    def _emit(self, now: float) -> None:
        el = max(now - self._t0, 1e-9)
        rate = self._n / el
        msg = f"  {self._desc} {self._n}"
        if self._total:
            pct = 100.0 * self._n / self._total
            eta = (self._total - self._n) / rate if rate > 0 else 0.0
            msg += f"/{self._total} ({pct:4.1f}%) ETA {eta / 60:5.1f}m"
        msg += f" | {rate:5.2f} it/s"
        if self._post:
            msg += " | " + " ".join(f"{k}={v:.4f}" for k, v in self._post.items())
        print(msg, flush=True)

    def postfix(self, **kw: float) -> None:
        self._post = {k: float(v) for k, v in kw.items()}
        if self._bar is not None:
            self._bar.set_postfix(self._post, refresh=False)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


# ═══════════════════════════════════════════════════════════════════════
#  에폭 1회
# ═══════════════════════════════════════════════════════════════════════
def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device,
    cfg: Any,
    epoch: int,
    ema: Optional[ModelEMA] = None,
    iter_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    evaluators: Sequence[Any] = (),
) -> Dict[str, Any]:
    """1 에폭. ``optimizer=None`` 이면 **평가 모드**(grad 없음, 스케줄러/EMA 미갱신).

    Returns:
        ``loss``, ``breakdown``(λ 곱한 뒤 값의 에폭 평균), ``miou``, ``confusion``,
        ``n_batches``, ``n_nonfinite_skips``, ``grad_norm_mean``/``grad_norm_p99``,
        ``clip_ratio``, ``ignore_ratio``, ``n_chl_valid_px``, ``n_edge_pos_ratio``,
        ``u_ramp``, ``prec_ramp``, ``epoch_time_s``, ``global_step``, ``lr``(그룹별).

    Raises:
        RuntimeError: 비유한 손실 스킵이 ``train.nonfinite_skip_max_per_epoch`` 초과 시.
    """
    is_train = optimizer is not None
    model.train(is_train)
    if isinstance(criterion, nn.Module):
        criterion.train(is_train)

    ipe = len(loader)
    amp_on = str(getattr(getattr(cfg, "train", None), "amp", "off")).lower() != "off"
    grad_accum = max(1, int(getattr(getattr(cfg, "optim", None), "grad_accum_steps", 1)))
    clip_norm = float(getattr(getattr(cfg, "optim", None), "grad_clip_norm", 0.0) or 0.0)
    skip_max = int(getattr(getattr(cfg, "train", None), "nonfinite_skip_max_per_epoch", 10))
    num_classes = int(getattr(getattr(cfg, "data", None), "num_classes", 0) or 0)
    ignore_index = int(getattr(getattr(cfg, "data", None), "ignore_index", 255))
    dry_run = bool(getattr(cfg, "dry_run", False))
    prec_ramp = prec_ramp_for_epoch(cfg, epoch)
    # 05 §7.1 "RFS 샘플러의 에폭별 시드" 와 같은 규약: 같은 (seed, epoch) → 같은 드롭 열.
    rng = random.Random(int(getattr(cfg, "seed", 0)) * 1_000_003 + int(epoch))

    cm = ConfusionMatrix(num_classes, ignore_index) if num_classes > 0 else None
    total_loss, n_batches, n_skips, n_seen = 0.0, 0, 0, 0
    breakdown: Dict[str, float] = {}
    grad_norms: List[float] = []
    n_clipped = 0
    ignore_px, total_px = 0.0, 0.0
    n_chl_valid = 0.0
    edge_pos, edge_total = 0.0, 0.0
    u_value = 0.0
    scaler_ = scaler if scaler is not None else torch.amp.GradScaler(device.type, enabled=False)
    t0 = time.perf_counter()

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    try:
        n_total = len(loader)
    except TypeError:
        n_total = None
    progress = EpochProgress(
        loader,
        total=(3 if dry_run and n_total else n_total),
        desc=f"epoch {epoch + 1}/{int(cfg.schedule.epochs)} {'train' if is_train else 'val'}",
        mode=resolve_progress_mode(),
    )

    for step, batch in enumerate(progress, start=1):
        if dry_run and step > 3:
            break
        n_seen = step
        batch = move_batch(batch, device)
        inputs = model_inputs_from_batch(batch, cfg)
        inputs["prec_ramp"] = prec_ramp
        if is_train:
            drop = sample_modality_dropout(cfg, rng)
            if drop:
                inputs["drop_modal"] = drop
        targets = targets_from_batch(batch)
        ctx = make_step_ctx(
            cfg, epoch=epoch, global_step=int(epoch) * ipe + step - 1, iters_per_epoch=ipe
        )
        u_value = float(getattr(ctx, "u", 0.0))

        with torch.set_grad_enabled(is_train):
            with autocast_ctx(cfg, device, enabled=amp_on):
                outputs = model(**inputs)
            if not isinstance(outputs, dict):
                raise TypeError(
                    f"model.forward 는 dict 를 반환해야 한다 (06 §3.4.3), 받은 타입 {type(outputs)}"
                )
            # 손실은 autocast 밖 fp32 (05 §5.1.4 loss_dtype: fp32).
            outputs = {
                k: (v.float() if isinstance(v, Tensor) and v.is_floating_point() else v)
                for k, v in outputs.items()
            }
            loss, parts = criterion(outputs, targets, ctx)

        finite = bool(torch.isfinite(loss.detach()))
        if is_train:
            if not finite:
                n_skips += 1
                optimizer.zero_grad(set_to_none=True)  # 누적분까지 버린다
                if n_skips > skip_max:
                    raise RuntimeError(
                        f"epoch {epoch}: 비유한 손실 스킵 {n_skips}회 > "
                        f"train.nonfinite_skip_max_per_epoch({skip_max})"
                    )
                continue
            scaler_.scale(loss / grad_accum).backward()
            if (step % grad_accum == 0) or (step == ipe):
                scaler_.unscale_(optimizer)
                params = [
                    p for g in optimizer.param_groups for p in g["params"] if p.grad is not None
                ]
                if params:
                    # clip_norm <= 0 이면 클리핑 없이 노름만 측정한다(로깅 계약 유지).
                    max_norm = clip_norm if clip_norm > 0 else float("inf")
                    total_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
                    grad_norms.append(total_norm)
                    if clip_norm > 0 and total_norm > clip_norm:
                        n_clipped += 1
                scaler_.step(optimizer)
                scaler_.update()
                optimizer.zero_grad(set_to_none=True)
                if iter_scheduler is not None:
                    iter_scheduler.step()
                if ema is not None:
                    ema.update(model, int(epoch) * ipe + step)
        elif not finite:
            n_skips += 1

        if finite:
            total_loss += float(loss.detach())
            for k, v in parts.items():
                breakdown[k] = breakdown.get(k, 0.0) + _as_float(v)
            n_batches += 1
            progress.postfix(loss=total_loss / max(n_batches, 1))

        # ── 진단 통계 ─────────────────────────────────────────────────
        y_seg = targets.get("y_seg")
        if y_seg is not None:
            ignore_px += float((y_seg == ignore_index).sum())
            total_px += float(y_seg.numel())
            if cm is not None and OUT_SEG in outputs:
                cm.update_from_logits(outputs[OUT_SEG].detach(), y_seg)
        vc = targets.get("y_chl_valid")
        if vc is not None:
            n_chl_valid += float(vc.sum())
        ye, yv = targets.get("y_edge"), targets.get("y_edge_valid")
        if ye is not None:
            if yv is not None:
                edge_pos += float(((ye > 0.5) & yv.bool()).sum())
                edge_total += float(yv.sum())
            else:
                edge_pos += float((ye > 0.5).sum())
                edge_total += float(ye.numel())
        if evaluators:
            with torch.no_grad():
                dispatch_evaluators(
                    evaluators, {k: _detach(v) for k, v in outputs.items()}, targets, batch
                )

    progress.close()

    n = max(n_batches, 1)
    n_steps = max(len(grad_norms), 1)
    out: Dict[str, Any] = {
        "loss": total_loss / n,
        "breakdown": {k: v / n for k, v in breakdown.items()},
        "n_batches": n_batches,
        "n_nonfinite_skips": n_skips,
        "grad_norm_mean": (sum(grad_norms) / n_steps) if grad_norms else float("nan"),
        "grad_norm_p99": _percentile(grad_norms, 0.99),
        "clip_ratio": (n_clipped / n_steps) if grad_norms else float("nan"),
        "ignore_ratio": (ignore_px / total_px) if total_px > 0 else float("nan"),
        "n_chl_valid_px": n_chl_valid,
        "n_edge_pos_ratio": (edge_pos / edge_total) if edge_total > 0 else float("nan"),
        "u_ramp": u_value,
        "prec_ramp": prec_ramp,
        "epoch_time_s": time.perf_counter() - t0,
        "global_step": int(epoch) * ipe + n_seen,
        "miou": compute_mean_iou(cm.matrix) if cm is not None else float("nan"),
        "confusion": cm.matrix.clone() if cm is not None else None,
    }
    if optimizer is not None:
        out["lr"] = {
            str(g.get("name", i)): float(g["lr"]) for i, g in enumerate(optimizer.param_groups)
        }
    return out


def _detach(v: Any) -> Any:
    return v.detach() if isinstance(v, Tensor) else v


def _as_float(v: Any) -> float:
    return float(v.detach()) if isinstance(v, Tensor) else float(v)


def _percentile(values: Sequence[float], q: float) -> float:
    """선형 보간 백분위수 (numpy 없이 — utils 와 동일 규약)."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


# ═══════════════════════════════════════════════════════════════════════
#  체크포인트 / 안정성 로그
# ═══════════════════════════════════════════════════════════════════════
def checkpoint_payload(
    *,
    model: nn.Module,
    epoch: int,
    global_step: int,
    best_metric: float,
    metric_name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    ema: Optional[ModelEMA] = None,
    args: Optional[Mapping[str, Any]] = None,
    early_stopper: Optional[EarlyStopper] = None,
    class_stats_path: Optional[str] = None,
    include_train_only: bool = True,
) -> Dict[str, Any]:
    """05 §6.3 payload.

    Args:
        include_train_only: ``best.pt`` 는 **False** — 05 §6.3 이 teacher/AuxSegHead/
            SIAM projection 을 담지 말라고 못박았다. ``last.pt`` 는 재개용이므로 True.
            EMA 그림자에도 같은 필터를 적용한다.
    """
    sd = model.state_dict()
    ema_sd = ema.state_dict() if ema is not None else None
    if not include_train_only:
        sd = prune_train_only_keys(sd)
        if ema_sd is not None:
            ema_sd = dict(ema_sd)
            ema_sd["shadow"] = prune_train_only_keys(ema_sd["shadow"])
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "metric_name": str(metric_name),
        "model_state_dict": sd,
        "ema_state_dict": ema_sd,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": dict(args or {}),
        "rng_state": collect_rng_state(),
        "early_stop_counter": early_stopper.state_dict() if early_stopper is not None else None,
        "class_stats_path": class_stats_path,
        "git_commit": git_commit(),
    }


def collect_stability_row(model: nn.Module, epoch: int) -> Dict[str, Any]:
    """`stability.csv` 1행 (05 §6.1). 없는 항목은 nan 으로 남긴다.

    이름 규약에 의존하는 best-effort 수집기다 — 모듈 이름이 바뀌면 조용히 nan 이 되므로
    "학습 내내 nan" 은 값이 0 이라는 뜻이 아니라 **감시 장치가 죽었다**는 신호로 읽어야 한다.
    """
    row: Dict[str, Any] = {k: float("nan") for k in STABILITY_FIELDS}
    row["epoch"] = int(epoch) + 1
    gammas: Dict[int, List[float]] = {}
    betas: Dict[int, List[float]] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            stage = _stage_of(name)
            if stage and (name.endswith("gamma") or ".ls_" in name):
                gammas.setdefault(stage, []).append(float(p.detach().abs().mean()))
            if stage and "beta_hat" in name:
                betas.setdefault(stage, []).append(
                    float(nn.functional.softplus(p.detach()).mean())
                )
            if name.endswith("a_hat"):
                row["glint_a"] = float(nn.functional.softplus(p.detach()).mean())
            elif name.endswith("ppn.b"):
                row["glint_b"] = float(p.detach().mean())
    for s, vals in gammas.items():
        if 1 <= s <= 4:
            row[f"layerscale_gamma_mean_s{s}"] = sum(vals) / len(vals)
    for s, vals in betas.items():
        if 1 <= s <= 4:
            row[f"bio_beta_s{s}"] = sum(vals) / len(vals)
    return row


def _stage_of(name: str) -> Optional[int]:
    """dotted 이름에서 ``stageN`` 세그먼트를 찾는다. 없으면 None."""
    for seg in name.split("."):
        if seg.startswith("stage") and seg[5:].isdigit():
            return int(seg[5:])
    return None


# ═══════════════════════════════════════════════════════════════════════
#  전체 학습
# ═══════════════════════════════════════════════════════════════════════
def fit(cfg: Any) -> Dict[str, Any]:
    """전체 학습 루프 + 체크포인트 + 조기종료 (05 §5·§6).

    Returns:
        ``{"run_dir", "best_metric", "metric_name", "best_epoch", "epochs_run", "history"}``
    """
    from bloomnet.config import make_run_name, resolve_schedule  # L1
    from bloomnet.data.build import build_dataloaders, build_datasets  # L3
    from bloomnet.losses.criterion import BloomNetCriterion  # L3
    from bloomnet.models.bloomnet import build_bloomnet  # L5

    seed_everything_strict(int(cfg.seed), deterministic=bool(cfg.train.deterministic))
    device = torch.device(cfg.device)
    run_name = cfg.run_name or make_run_name(cfg)
    run_dir = Path(cfg.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(cfg)
    loaders = build_dataloaders(cfg, datasets)
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    resolve_schedule(cfg, len(train_loader))
    ipe = int(cfg.schedule.iters_per_epoch)

    (run_dir / "train_args.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    model = build_bloomnet(cfg).to(device)
    criterion = BloomNetCriterion(
        num_classes=int(cfg.data.num_classes),
        ignore_index=int(cfg.data.ignore_index),
        lambda_seg=cfg.loss.lambda_seg,
        lambda_reg=cfg.loss.lambda_reg,
        lambda_bd=cfg.loss.lambda_bd,
        lambda_bas=cfg.loss.lambda_bas,
        lambda_aux=dict(cfg.loss.lambda_aux),
        lambda_siam=cfg.loss.lambda_siam,
        w_dice=cfg.loss.w_dice,
        ohem_thresh=cfg.loss.ohem_thresh,
        ohem_keep_frac=cfg.loss.ohem_keep_frac,
        bas_tau=cfg.loss.bas_tau,
        huber_beta=cfg.loss.huber_beta,
        unc_clamp=tuple(cfg.loss.unc_clamp),
        u_warm_frac=cfg.loss.u_warm_frac,
        u_ramp_frac=cfg.loss.u_ramp_frac,
        boundary_source=cfg.data.boundary.source,
        boundary_radius=cfg.boundary_radius(int(cfg.data.train_size[0])),
        boundary_stride=int(cfg.data.boundary.stride),
        aux_loss_stride=int(cfg.loss.aux_loss_stride),
        aux_decay=bool(cfg.loss.aux_decay),
        debug_assert=bool(cfg.loss.debug_assert),
    ).to(device)

    pretrained_keys: Optional[List[str]] = None
    if cfg.train.init_from:
        ck = load_checkpoint(cfg.train.init_from)
        report = load_state_dict_shape_tolerant(model, ck["model_state_dict"])
        pretrained_keys = report["loaded"]  # 정정 B-28: encoder 그룹 멤버십의 정의

    optimizer = build_optimizer(model, cfg.optim, pretrained_keys=pretrained_keys)
    scheduler = build_lr_scheduler(
        optimizer,
        scheduler=cfg.schedule.scheduler,
        scheduler_kwargs=cfg.schedule.scheduler_kwargs,
        warmup_iters=int(cfg.schedule.warmup_iters or 0),
        warmup_start_factor=float(cfg.schedule.warmup_start_factor),
    )
    scaler = build_grad_scaler(cfg, device)
    ema = (
        ModelEMA(
            model,
            decay=float(cfg.train.ema.decay),
            start_step=int(cfg.train.ema.start_epoch) * ipe,
        )
        if cfg.train.ema.enabled
        else None
    )

    es_cfg = cfg.train.early_stopping
    stopper = EarlyStopper(
        patience=int(es_cfg.patience), min_delta=float(es_cfg.min_delta), mode=str(es_cfg.mode)
    )
    metric_name = str(es_cfg.metric)
    start_epoch = 0
    if cfg.train.resume_from:
        start_epoch = _resume(
            cfg.train.resume_from, model, optimizer, scheduler, scaler, ema, stopper, device
        )

    names = class_names_for(cfg)
    epoch_log = CsvLogger(
        run_dir / "epoch_metrics.csv",
        EPOCH_FIELDS,
        preamble={
            "iters_per_epoch": ipe,
            "warmup_iters": cfg.schedule.warmup_iters,
            "total_iters": cfg.schedule.scheduler_kwargs.get("total_iters"),
        },
    )
    percls_log = CsvLogger(run_dir / "val_per_class_metrics.csv", PER_CLASS_FIELDS)
    bnd_log = CsvLogger(run_dir / "val_boundary_metrics.csv", BOUNDARY_FIELDS)
    reg_log = CsvLogger(run_dir / "val_regression_metrics.csv", REGRESSION_FIELDS)
    stab_log = CsvLogger(run_dir / "stability.csv", STABILITY_FIELDS)

    history: List[Dict[str, Any]] = []
    for epoch in range(start_epoch, int(cfg.schedule.epochs)):
        tr = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch,
            ema=ema,
            iter_scheduler=scheduler,
        )
        va: Dict[str, Any] = {}
        seg_metrics: Dict[str, Any] = {}
        seg_ev: Optional[SegEvaluator] = None
        if val_loader is not None:
            seg_ev = SegEvaluator(
                int(cfg.data.num_classes), int(cfg.data.ignore_index), names, seed=int(cfg.seed)
            )
            bnd_ev = BoundaryEvaluator(
                tuple(int(t) for t in cfg.eval.boundary_tolerances),
                num_classes=int(cfg.data.num_classes),
                ignore_index=int(cfg.data.ignore_index),
            )
            reg_ev = RegEvaluator(
                thresholds_mgm3=tuple(float(t) for t in cfg.eval.alarm_thresholds)
            )
            va = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                scaler=None,
                device=device,
                cfg=cfg,
                epoch=epoch,
                evaluators=[seg_ev, bnd_ev, reg_ev],
            )
            seg_metrics = seg_ev.compute()
            # BoundaryAccumulator 는 `bf_precision_*` 등 CSV 스키마에 없는 키도 낸다.
            # CsvLogger 는 strict 이므로 여기서 열 목록으로 걸러야 한다.
            bnd = bnd_ev.compute()
            bnd_log.append(
                {"epoch": epoch + 1, **{k: v for k, v in bnd.items() if k in BOUNDARY_FIELDS}}
            )
            reg_log.append(
                regression_row(epoch, reg_ev.compute(), ignore_ratio=va.get("ignore_ratio"))
            )
            if ema is not None:
                ema_seg = SegEvaluator(
                    int(cfg.data.num_classes), int(cfg.data.ignore_index), names
                )
                with ema.swap_into(model):
                    ema_res = evaluate(
                        model,
                        val_loader,
                        None,
                        cfg=cfg,
                        evaluators=[ema_seg],
                        device=device,
                        amp=str(cfg.train.amp).lower() != "off",
                        # dry_run 은 run_epoch 만 3 step 으로 자른다. 이 경로도 자르지 않으면
                        # --dry_run 이 전체 val 순회가 되어 CPU 에서 수 시간 걸린다.
                        max_batches=3 if bool(getattr(cfg, "dry_run", False)) else None,
                    )
                va["miou_ema"] = ema_res["seg"]["miou"]

        selected = _select_metric(metric_name, tr, va, seg_metrics)
        improved = stopper.step(selected, epoch=epoch)
        payload_common = dict(
            model=model,
            epoch=epoch,
            global_step=tr["global_step"],
            best_metric=stopper.best,
            metric_name=metric_name,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            args=cfg.to_dict(),
            early_stopper=stopper,
            class_stats_path=cfg.data.sampler.class_stats,
        )
        save_checkpoint(
            run_dir / "last.pt", checkpoint_payload(include_train_only=True, **payload_common)
        )
        if improved:
            save_checkpoint(
                run_dir / "best.pt",
                checkpoint_payload(include_train_only=False, **payload_common),
            )

        row = _epoch_row(epoch, tr, va, seg_metrics, stopper, optimizer)
        epoch_log.append(row)
        if seg_ev is not None and seg_metrics:
            percls_log.append_many([{**r, "epoch": epoch + 1} for r in seg_metrics["rows"]])
            append_presence_summary(run_dir / "val_presence_summary.csv", epoch, seg_ev.cm.matrix)
        stab_log.append(collect_stability_row(model, epoch))
        history.append(row)
        print(
            f"epoch={epoch + 1} train_loss={tr['loss']:.4f} train_miou={tr['miou']:.4f} "
            f"{metric_name}={selected:.4f} best={stopper.best:.4f} "
            f"skips={tr['n_nonfinite_skips']} clip_ratio={tr['clip_ratio']:.3f}"
        )
        if tr["n_edge_pos_ratio"] == 0.0:
            # (정정 A-28) 경계 타깃이 전부 0 이면 λ_bd=20 이 조용히 무력화된 것이다.
            print("  [warn] n_edge_pos_ratio == 0 — boundary_source / y_edge 생성 경로를 확인하라")
        if stopper.should_stop:
            print(f"early stop: {stopper.counter} epochs without improvement")
            break

    return {
        "run_dir": str(run_dir),
        "best_metric": stopper.best,
        "metric_name": metric_name,
        "best_epoch": stopper.best_epoch,
        "epochs_run": len(history),
        "history": history,
    }


def _resume(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    ema: Optional[ModelEMA],
    stopper: EarlyStopper,
    device: torch.device,
) -> int:
    """05 §6.3 재개. rng_state·early_stop_counter 까지 되돌린다."""
    ck = load_checkpoint(path, map_location=device)
    load_state_dict_shape_tolerant(model, ck["model_state_dict"])
    if ck.get("optimizer_state_dict"):
        optimizer.load_state_dict(ck["optimizer_state_dict"])
    if scheduler is not None and ck.get("scheduler_state_dict"):
        scheduler.load_state_dict(ck["scheduler_state_dict"])
    if scaler is not None and ck.get("scaler_state_dict"):
        scaler.load_state_dict(ck["scaler_state_dict"])
    if ema is not None and ck.get("ema_state_dict"):
        ema.load_state_dict(ck["ema_state_dict"], strict=False)
    if ck.get("early_stop_counter"):
        stopper.load_state_dict(ck["early_stop_counter"])
    if ck.get("rng_state"):
        try:
            restore_rng_state(ck["rng_state"])
        except (TypeError, RuntimeError) as e:  # torch 2.13: map_location 로드 후 "RNG state must be a torch.ByteTensor"
            print(f"[resume] rng_state 복원 실패 ({e}); 새 난수 상태로 이어서 학습한다", flush=True)
    return int(ck["epoch"]) + 1


def _select_metric(
    name: str, tr: Mapping[str, Any], va: Mapping[str, Any], seg_metrics: Mapping[str, Any]
) -> float:
    """`train.early_stopping.metric` 문자열 → 값 (05 §6.2 모델 선택 지표).

    ``val_algae_iou`` 는 S1(K=2)의 class 1 IoU 다 (`constants.S1_CLASS_NAMES`).
    """
    if name == "val_algae_iou":
        ious = seg_metrics.get("per_class_iou") or []
        return float(ious[1]) if len(ious) > 1 else float("nan")
    table: Dict[str, Any] = {
        "train_loss": tr.get("loss"),
        "train_miou": tr.get("miou"),
        "val_loss": va.get("loss"),
        "val_miou": va.get("miou"),
        "val_miou_ema": va.get("miou_ema", va.get("miou")),
        "val_miou_common": seg_metrics.get("miou_common"),
        "val_transfer_score": seg_metrics.get("transfer_score"),
    }
    if name not in table:
        raise ValueError(
            f"early_stopping.metric {name!r} 을(를) 해석할 수 없다 — "
            f"{sorted(table)} 또는 'val_algae_iou'"
        )
    v = table[name]
    return float("nan") if v is None else float(v)


def _epoch_row(
    epoch: int,
    tr: Mapping[str, Any],
    va: Mapping[str, Any],
    seg_metrics: Mapping[str, Any],
    stopper: EarlyStopper,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    """`epoch_metrics.csv` 1행 (05 §6.1). breakdown 은 **λ 곱한 뒤** 값이다."""
    lrs = {str(g.get("name", i)): float(g["lr"]) for i, g in enumerate(optimizer.param_groups)}
    ious = seg_metrics.get("per_class_iou") or []
    row: Dict[str, Any] = {
        "epoch": epoch + 1,
        "lr_main": lrs.get("main", float("nan")),
        "lr_encoder": lrs.get("encoder", float("nan")),
        "lr_physics": lrs.get("physics", float("nan")),
        "epoch_time_s": tr.get("epoch_time_s"),
        "gpu_peak_mb": _gpu_peak_mb(),
        "train_loss": tr.get("loss"),
        "val_loss": va.get("loss", float("nan")),
        "train_miou": tr.get("miou"),
        "val_miou": va.get("miou", float("nan")),
        "val_miou_ema": va.get("miou_ema", float("nan")),
        "best_val_miou": stopper.best,
        # S1 전용. S0(K=12)에서는 nan 이어야 한다 (05 §6.1).
        "val_algae_iou": float(ious[1]) if len(ious) == 2 else float("nan"),
        "n_nonfinite_skips": tr.get("n_nonfinite_skips"),
        "grad_norm_mean": tr.get("grad_norm_mean"),
        "grad_norm_p99": tr.get("grad_norm_p99"),
        "clip_ratio": tr.get("clip_ratio"),
        "n_chl_valid_px": tr.get("n_chl_valid_px"),
        "u_ramp": tr.get("u_ramp"),
        "ignore_ratio": tr.get("ignore_ratio"),
    }
    row.update(breakdown_to_csv_columns("train", tr.get("breakdown") or {}))
    row.update(breakdown_to_csv_columns("val", va.get("breakdown") or {}))
    return row


def _gpu_peak_mb() -> float:
    if not torch.cuda.is_available():  # 헌법 C-5.2 — 개발/테스트에서는 항상 이 분기
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024**2)  # pragma: no cover
