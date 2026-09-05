"""``bloomnet.losses`` — 손실 함수 패키지.

06 §2.2 규칙 3에 따라 **re-export 만** 한다(로직 금지 — 순환 import 의 최대 원인).

여기서 재노출하는 것은 **레벨 L1 프리미티브뿐**이다. ``boundary_loss``(L2)·
``criterion``(L3)은 의도적으로 제외한다 — 패키지를 import 하는 것만으로 상위 레벨이
끌려오면 L1 소비자가 L3 의존성을 떠안게 되어 §2.2 레벨 규율이 깨진다.
상위 레벨은 ``from bloomnet.losses.criterion import BloomNetCriterion`` 처럼
모듈 경로로 직접 import 한다.
"""

from __future__ import annotations

# §2.1 원칙 2 는 "같은 레벨끼리도 import 금지"이고 원칙 3 은 "__init__.py 는 re-export"다.
# 둘을 동시에 만족시키기 위해 **상대 import** 만 쓴다 — T01 이 수집하는 `bloomnet.*`
# 절대 import 가 여기서는 0건이 되고, 패키지 내부 재노출이라는 의도도 분명해진다.
from .distill import cwd_loss
from .regression import chl_reg_loss, margin_rank_loss, u_ramp
from .seg import batch_soft_dice, ohem_ce, plain_ce

__all__ = [
    # seg.py (L1)
    "ohem_ce",
    "batch_soft_dice",
    "plain_ce",
    # regression.py (L1)
    "chl_reg_loss",
    "margin_rank_loss",
    "u_ramp",
    # distill.py (L1)
    "cwd_loss",
]
