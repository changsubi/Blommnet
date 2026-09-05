"""LR 스케줄러 — 이전 구현 `optim/schedulers.py` 계승 + 05 §5.1.3 warmup/Poly 조립.

정본 (05 §5.1.3, 정정 B-26)::

    warmup : LinearLR(start_factor=1e-3, end_factor=1.0, total_iters=warmup_iters)
    main   : PolynomialLR(power=0.9,   total_iters=epochs·ipe − warmup_iters)
    wrapper: SequentialLR(milestones=[warmup_iters])
    step   : **optimizer step 마다** (epoch 마다가 아니다)

**절대 iteration 을 이 파일에 하드코딩하지 않는다.** `warmup_iters` / `total_iters` 는
`config.resolve_schedule(cfg, len(train_loader))` 가 런타임에 유도하고 V15 로 검증한다.

레벨 L1 — `bloomnet.*` 를 하나도 import 하지 않는다. 특히 `bloomnet.config`(같은 L1)를
import 하면 레벨 규칙(§2.1 원칙 2)을 위반하므로 **cfg 객체가 아니라 평문 인자**를 받는다.
호출자(`engine/trainer.py`, L6)가 cfg 값을 풀어서 넘긴다.
"""

from __future__ import annotations

import math
from inspect import signature
from typing import Any, Dict, Final, Literal, Mapping, Optional, Sequence

from torch.optim import lr_scheduler as torch_schedulers
from torch.optim.optimizer import Optimizer

__all__ = [
    "SCHEDULERS_DICT",
    "WarmupOneCycleLR",
    "annealing_cos",
    "annealing_linear",
    "build_scheduler",
    "build_lr_scheduler",
    "lr_factor_at_step",
    "lr_curve",
]


# ═══════════════════════════════════════════════════════════════════════
#  이전 구현 계승분 ([분석] §E.1.7 copy-as-is)
# ═══════════════════════════════════════════════════════════════════════
def annealing_cos(start: float, end: float, pct: float) -> float:
    """``pct`` 0→1 에 따라 ``start``→``end`` 코사인 어닐링."""
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


def annealing_linear(start: float, end: float, pct: float) -> float:
    """``pct`` 0→1 에 따라 ``start``→``end`` 선형 어닐링."""
    return (end - start) * pct + start


class WarmupOneCycleLR(torch_schedulers.LRScheduler):
    """warmup 을 앞에 붙인 OneCycleLR 변형 (이전 구현 원본 이식).

    BloomNet 기본 레시피는 :func:`build_lr_scheduler` (LinearLR+PolynomialLR) 다.
    이 클래스는 ablation 용으로만 남긴다 — 05 §5.1.3 이 PolyLR power=0.9 를 동결했다.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Optional[float] = None,
        total_steps: int = 0,
        warmup_iters: int = 0,
        warmup_ratio: float = 0.0,
        pct_start: float = 0.295,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1000.0,
        use_beta1: bool = True,
        update_momentum: bool = True,
        last_epoch: int = -1,
    ) -> None:
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(f"anneal_strategy must be cos|linear, got {anneal_strategy!r}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.max_lr = max_lr
        self.min_point = float(pct_start * total_steps)
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.total_steps = total_steps
        self.use_beta1 = use_beta1
        self.anneal_strategy = anneal_strategy
        self.final_div_factor = final_div_factor
        self.update_momentum = update_momentum

        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                if last_epoch != -1:
                    raise ValueError("last_epoch != -1 이면 group['initial_lr'] 가 있어야 한다")
                ml = float(group["lr"])
                group["initial_lr"] = ml / div_factor
                group["max_lr"] = ml
                group["min_lr"] = group["initial_lr"] / final_div_factor
                group["lr"] = (
                    ml / final_div_factor if self.warmup_iters > 0 else group["initial_lr"]
                )
            if self.use_beta1 and "betas" in group:
                group["betas"] = (self.max_momentum, *group["betas"][1:])
            elif self.update_momentum and "momentum" in group:
                group["momentum"] = self.max_momentum

        super().__init__(optimizer, last_epoch)

    def _anneal_func(self, *args: Any, **kwargs: Any) -> float:
        fn = annealing_cos if self.anneal_strategy == "cos" else annealing_linear
        return fn(*args, **kwargs)

    def _compute_lr_momentum(self, group: Dict[str, Any]) -> tuple:
        step_num = (self._step_count - 1) if self.last_epoch != -1 else 0
        momentum = 0.0
        if step_num < self.warmup_iters:
            if self.warmup_ratio:
                k = (1 - step_num / self.warmup_iters) * (1 - self.warmup_ratio)
                warmup_lr = group["max_lr"] * (1 - k)
                thelr = warmup_lr * (1 - step_num / self.total_steps)
            else:
                gmax = (
                    group["max_lr"]
                    * (1 + math.cos(math.pi * step_num / float(self.total_steps)))
                    / 2
                )
                thelr = group["max_lr"] / self.final_div_factor + gmax * step_num / float(
                    self.warmup_iters
                )
        else:
            pct = (step_num - self.warmup_iters) / float(self.total_steps - self.warmup_iters)
            step_num_to_use = step_num
            momentum = self._anneal_func(self.base_momentum, self.max_momentum, pct)
            if self.anneal_strategy == "cos":
                step_num_to_use += 1
            thelr = self._anneal_func(
                group["max_lr"], group["min_lr"], step_num_to_use / float(self.total_steps)
            )
        return thelr, momentum

    def get_lr(self):  # noqa: D102 - torch API
        warn = getattr(torch_schedulers, "_warn_get_lr_called_within_step", None)
        if warn is not None:
            warn(self)
        if self.last_epoch > self.total_steps:
            raise ValueError(
                f"Tried to step {self.last_epoch} times; total_steps={self.total_steps}"
            )
        lrs = []
        for group in self.optimizer.param_groups:
            computed_lr, computed_momentum = self._compute_lr_momentum(group)
            lrs.append(computed_lr)
            if self.use_beta1 and "betas" in group:
                group["betas"] = (computed_momentum, *group["betas"][1:])
            elif self.update_momentum and "momentum" in group:
                group["momentum"] = computed_momentum
        return lrs


SCHEDULERS_DICT: Final[Dict[str, Any]] = {
    "ConstantLR": torch_schedulers.ConstantLR,
    "LinearLR": torch_schedulers.LinearLR,
    "MultiStepLR": torch_schedulers.MultiStepLR,
    "PolynomialLR": torch_schedulers.PolynomialLR,
    "StepLR": torch_schedulers.StepLR,
    "CosineAnnealingLR": torch_schedulers.CosineAnnealingLR,
    "OneCycleLR": torch_schedulers.OneCycleLR,
    "WarmupOneCycleLR": WarmupOneCycleLR,
}


def build_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    lr: float,
    total_iter: int,
    constructor_kwargs: Mapping[str, Any],
) -> torch_schedulers.LRScheduler:
    """이전 구현 `build_scheduler` 이식 (시그니처 동일).

    ``constructor_kwargs`` 중 생성자가 받지 않는 키는 조용히 버린다(원본 동작).
    BloomNet 기본 경로는 :func:`build_lr_scheduler` 이며, 이 함수는 ablation·호환용이다.
    """
    if scheduler_type not in SCHEDULERS_DICT:
        raise ValueError(
            f"unknown scheduler {scheduler_type!r} — {sorted(SCHEDULERS_DICT)} 중 하나여야 한다"
        )
    constructor_fn = SCHEDULERS_DICT[scheduler_type]
    accepted = signature(constructor_fn).parameters.keys()
    _kwargs = {k: v for k, v in dict(constructor_kwargs).items() if k in accepted}
    if scheduler_type in ("OneCycleLR", "WarmupOneCycleLR"):
        _kwargs.update(max_lr=lr, total_steps=total_iter)
    elif scheduler_type in ("ConstantLR", "LinearLR", "PolynomialLR"):
        _kwargs.update(total_iters=total_iter)
    return constructor_fn(optimizer, **_kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  BloomNet 정본 스케줄 (05 §5.1.3)
# ═══════════════════════════════════════════════════════════════════════
def build_lr_scheduler(
    optimizer: Optimizer,
    *,
    scheduler: str = "PolynomialLR",
    scheduler_kwargs: Optional[Mapping[str, Any]] = None,
    warmup_iters: int = 0,
    warmup_start_factor: float = 1.0e-3,
) -> torch_schedulers.LRScheduler:
    """LinearLR warmup + main 스케줄러를 SequentialLR 로 잇는다 (05 §5.1.3).

    Args:
        scheduler_kwargs: main 스케줄러 생성자 kwargs. ``PolynomialLR`` 이면
            ``{"power": 0.9, "total_iters": epochs·ipe − warmup_iters}`` 가 들어와야 한다.
            **``total_iters`` 는 여기서 계산하지 않는다** — `config.resolve_schedule` 이
            유도하고 V15 가 검증한 값을 그대로 받는다(정정 A-18/B-26).
        warmup_iters: 0 이하면 warmup 없이 main 만 반환한다.

    Note:
        스케줄러는 **optimizer step 마다** `.step()` 해야 한다(epoch 마다가 아니다).
        LR 곡선의 폐형식은 :func:`lr_factor_at_step` 에 있고, 두 값이 일치함을
        `test_engine.py` 가 회귀로 고정한다.
    """
    kwargs = dict(scheduler_kwargs or {})
    if scheduler not in SCHEDULERS_DICT:
        raise ValueError(
            f"unknown scheduler {scheduler!r} — {sorted(SCHEDULERS_DICT)} 중 하나여야 한다"
        )
    if scheduler == "PolynomialLR" and "total_iters" not in kwargs:
        raise ValueError(
            "PolynomialLR 에는 total_iters 가 필요하다. "
            "config.resolve_schedule(cfg, iters_per_epoch) 를 먼저 호출하라 (정정 A-18)."
        )
    main = SCHEDULERS_DICT[scheduler](optimizer, **kwargs)
    if int(warmup_iters) <= 0:
        return main
    warm = torch_schedulers.LinearLR(
        optimizer,
        start_factor=float(warmup_start_factor),
        end_factor=1.0,
        total_iters=int(warmup_iters),
    )
    return torch_schedulers.SequentialLR(
        optimizer, schedulers=[warm, main], milestones=[int(warmup_iters)]
    )


def lr_factor_at_step(
    step: int,
    *,
    warmup_iters: int,
    total_iters: int,
    warmup_start_factor: float = 1.0e-3,
    power: float = 0.9,
) -> float:
    """`build_lr_scheduler` 곡선의 **폐형식** (base_lr 대비 배율).

    Args:
        step: 지금까지 호출한 ``scheduler.step()`` 횟수 (0 = 생성 직후).
        total_iters: main PolynomialLR 의 ``total_iters``
            (= ``epochs × iters_per_epoch − warmup_iters``).

    학습 전체 스텝 수는 ``warmup_iters + total_iters`` 이고, 그 지점에서 배율이
    **정확히 0** 이 된다(정정 B-26 이 요구한 "LR 이 0 으로 수렴").
    """
    w = int(warmup_iters)
    t = int(total_iters)
    if t <= 0:
        raise ValueError(f"total_iters must be > 0, got {t}")
    k = int(step)
    if w > 0 and k < w:
        s0 = float(warmup_start_factor)
        return s0 + (1.0 - s0) * (k / w)
    kk = k - max(w, 0)
    if kk >= t:
        return 0.0
    return (1.0 - kk / t) ** float(power)


def lr_curve(
    base_lr: float,
    n_steps: int,
    *,
    warmup_iters: int,
    total_iters: int,
    warmup_start_factor: float = 1.0e-3,
    power: float = 0.9,
) -> Sequence[float]:
    """``lr_factor_at_step`` 를 ``n_steps`` 만큼 펼친 LR 목록 (검증·플롯용)."""
    return [
        base_lr
        * lr_factor_at_step(
            k,
            warmup_iters=warmup_iters,
            total_iters=total_iters,
            warmup_start_factor=warmup_start_factor,
            power=power,
        )
        for k in range(int(n_steps))
    ]
