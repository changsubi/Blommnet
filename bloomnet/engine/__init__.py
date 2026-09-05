"""``bloomnet.engine`` — 재export 전용 (06 §2.1 원칙 3: ``__init__.py`` 에 로직 금지).

L6 인 :mod:`bloomnet.engine.trainer` 는 여기서 재export 하지 **않는다**. 재export 하면
`engine.sched`(L1) 하나를 쓰려는 소비자가 trainer 를 통해 L5 모델까지 끌고 오게 되어
§2.1 원칙 2(상위 레벨 의존 금지)가 깨진다. trainer 는 항상
``from bloomnet.engine.trainer import run_epoch, fit`` 로 직접 import 한다.
"""

from __future__ import annotations

from bloomnet.engine.ema import ModelEMA
from bloomnet.engine.optim import (
    PARAM_GROUPS,
    assign_param_groups,
    build_optimizer,
    collect_no_decay,
    collect_physics,
)
from bloomnet.engine.sched import (
    SCHEDULERS_DICT,
    WarmupOneCycleLR,
    build_lr_scheduler,
    build_scheduler,
    lr_curve,
    lr_factor_at_step,
)

__all__ = [
    "ModelEMA",
    "PARAM_GROUPS",
    "SCHEDULERS_DICT",
    "WarmupOneCycleLR",
    "assign_param_groups",
    "build_lr_scheduler",
    "build_optimizer",
    "build_scheduler",
    "collect_no_decay",
    "collect_physics",
    "lr_curve",
    "lr_factor_at_step",
]
