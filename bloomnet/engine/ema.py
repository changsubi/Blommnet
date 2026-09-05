"""Exponential Moving Average of model weights (05 §5.1.4, 06 §3.6 동결 시그니처).

정본 설정: ``decay = 0.9998``(S0-RGB) / ``0.999``(S1), ``start_epoch = 5``.
`start_step` 은 trainer 가 ``start_epoch × iters_per_epoch`` 로 유도해 넘긴다 —
이 파일에 절대 iteration 을 하드코딩하지 않는다(정정 A-18 과 같은 취지).

모델 선택 지표가 ``val_miou_ema``(05 §6.2)이므로 EMA 는 **선택 사항이 아니라
평가 경로의 일부**다. 05 §6.3 payload 의 ``ema_state_dict`` 로 저장된다.

레벨 L1 — `bloomnet.*` 를 하나도 import 하지 않는다.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["ModelEMA"]


class ModelEMA:
    """가중치 EMA 그림자 사본.

    갱신식 (부동소수 항목만)::

        v <- d·v + (1 - d)·p,      d = 0.0 (step < start_step) 또는 decay

    ``step < start_step`` 에서 ``d = 0`` 이므로 그림자는 모델을 **그대로 추종**한다.
    즉 ``start_step`` 이전에는 "EMA 를 시작하지 않은" 것이 아니라 "지연 없이 따라가는"
    것이고, 시작 시점에서 초기 랜덤 가중치의 잔향이 남지 않는다.

    정수 항목(``num_batches_tracked`` 등)은 EMA 가 정의되지 않으므로 **그대로 복사**한다.
    BN 의 ``running_mean``/``running_var`` 는 부동소수라 EMA 대상이다(timm 관례).

    Args:
        model: 그림자를 뜰 원본. 생성 시점의 ``state_dict`` 를 clone 한다.
        decay: EMA 계수.
        start_step: 이 스텝 이전에는 그림자 = 모델.

    Note:
        그림자는 원본과 **같은 device** 에 만든다. CPU 오프로딩이 필요하면
        생성 전에 ``model.cpu()`` 사본을 넘겨라.
    """

    def __init__(self, model: nn.Module, *, decay: float = 0.9998, start_step: int = 0) -> None:
        if not (0.0 <= float(decay) < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if int(start_step) < 0:
            raise ValueError(f"start_step must be >= 0, got {start_step}")
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.num_updates = 0
        self.last_step = -1
        self.shadow: Dict[str, Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """한 optimizer step 뒤에 호출한다. ``step`` 은 **global step**(0-기준)."""
        d = 0.0 if int(step) < self.start_step else self.decay
        src = model.state_dict()
        missing = [k for k in self.shadow if k not in src]
        if missing:
            raise KeyError(
                f"ModelEMA: 모델에 없는 그림자 키 {missing[:5]}"
                f"{'…' if len(missing) > 5 else ''} — 아키텍처가 바뀌었다"
            )
        for k, v in self.shadow.items():
            new = src[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(new.detach().to(v.device, v.dtype), alpha=1.0 - d)
            else:
                v.copy_(new.detach().to(v.device))
        self.num_updates += 1
        self.last_step = int(step)

    # ------------------------------------------------------------- (de)serialize
    def state_dict(self) -> Dict[str, Any]:
        """05 §6.3 payload 의 ``ema_state_dict``.

        가중치뿐 아니라 ``decay``/``start_step``/``num_updates`` 도 담는다 —
        재개 시 EMA 하이퍼가 config 와 어긋나면 조용히 다른 궤적을 그리기 때문이다.
        """
        return {
            "decay": self.decay,
            "start_step": self.start_step,
            "num_updates": self.num_updates,
            "last_step": self.last_step,
            "shadow": {k: v.detach().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any], *, strict: bool = True) -> None:
        """:meth:`state_dict` 산출물(또는 순수 가중치 dict)을 되돌린다."""
        shadow = state.get("shadow", state)
        if not isinstance(shadow, Mapping):
            raise TypeError("ModelEMA.load_state_dict: 'shadow' 가 매핑이 아니다")
        if strict:
            unexpected = sorted(set(shadow) - set(self.shadow))
            missing = sorted(set(self.shadow) - set(shadow))
            if unexpected or missing:
                raise KeyError(f"ModelEMA 키 불일치: missing={missing[:5]} unexpected={unexpected[:5]}")
        for k, v in shadow.items():
            if k in self.shadow:
                self.shadow[k].copy_(torch.as_tensor(v).to(self.shadow[k].device))
        if "decay" in state:
            self.decay = float(state["decay"])
        if "start_step" in state:
            self.start_step = int(state["start_step"])
        self.num_updates = int(state.get("num_updates", self.num_updates))
        self.last_step = int(state.get("last_step", self.last_step))

    # ------------------------------------------------------------------ apply
    def copy_to(self, model: nn.Module) -> None:
        """그림자를 모델에 **덮어쓴다**(되돌릴 수 없다). 평가 후 복구가 필요하면
        :meth:`swap_into` 를 쓸 것."""
        model.load_state_dict(self.shadow, strict=True)

    @contextmanager
    def swap_into(self, model: nn.Module) -> Iterator[nn.Module]:
        """with 블록 안에서만 EMA 가중치로 평가하고 원래 가중치를 복구한다.

        06 동결 시그니처에는 없지만, ``val_miou_ema``(05 §6.2)를 매 에폭 계산하려면
        "덮어쓰고 되돌리기" 가 필요하다. `copy_to` 만 쓰면 학습 가중치가 파괴된다.
        """
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            self.copy_to(model)
            yield model
        finally:
            model.load_state_dict(backup, strict=True)
