"""``bloomnet.pretrain`` — S0-Spec 분광 사전학습 (01 §6).

06 §2.2 규칙 3 에 따라 **re-export 만** 한다 (로직 금지).

여기서 재노출하는 것은 **레벨 L2 의 ``spec_mlp`` 뿐**이다. ``loso``(L3)·``transplant``(L6)
는 의도적으로 제외한다 — 패키지 import 만으로 상위 레벨이 끌려오면 L2 소비자가 L6 의존성
(``modules.heads``, 장차 ``models.bloomnet``)을 떠안게 되어 §2.1 원칙 2 가 깨진다.
상위 레벨은 ``from bloomnet.pretrain.loso import run_loso`` 처럼 모듈 경로로 직접 import 한다.
"""

from __future__ import annotations

# 상대 import 만 쓴다 (losses/__init__.py 와 동일 규약) — T01 이 수집하는
# `bloomnet.*` 절대 import 가 이 파일에서 0건이 된다.
from .spec_mlp import SpecMLP, init_spec_mlp, spec_loss

__all__ = ["SpecMLP", "init_spec_mlp", "spec_loss"]
