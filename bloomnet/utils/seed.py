"""재현성 시드 고정 (05 §7.1 — 이전 구현 `seed.py` 의 강화판).

이전 구현의 ``seed_everything`` 은 ``cudnn.deterministic`` / ``use_deterministic_algorithms``
를 설정하지 않는다. 본 모듈은 그 둘을 포함한 강화판을 제공한다.

레벨 L0 — ``bloomnet.*`` 를 import 하지 않는다(``constants.py`` 예외도 쓰지 않는다).
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = [
    "DEFAULT_SEED",
    "seed_everything_strict",
    "worker_init_fn",
    "make_generator",
    "get_base_seed",
]

DEFAULT_SEED: int = 1234

# ``seed_everything_strict`` 가 갱신하는 프로세스 전역 기준 시드.
# DataLoader 가 ``worker_init_fn`` 을 위치인자 1개로만 호출하므로 전역이 필요하다.
_BASE_SEED: int = DEFAULT_SEED


def seed_everything_strict(
    seed: int = DEFAULT_SEED,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
) -> None:
    """python / numpy / torch RNG 와 cudnn 결정론 플래그를 한 번에 고정한다.

    Args:
        seed: 기준 시드. 이후 ``worker_init_fn`` 의 기준값으로도 쓰인다.
        deterministic: False 면 RNG 만 고정하고 결정론 플래그를 **끈다**(cudnn autotuning 허용, 속도 우선).
        warn_only: ``torch.use_deterministic_algorithms`` 의 warn_only.
            **True 가 필수**다 — 05 §7.2 R1 이 기록하듯 bilinear ``F.interpolate`` 의
            backward 가 CUDA atomicAdd 를 쓰므로 False 면 학습이 즉시 예외로 죽는다.

    Note:
        ``CUBLAS_WORKSPACE_CONFIG`` 는 **torch import 이전에** 설정되어야 실제로 반영된다.
        여기서 설정하는 것은 이후 spawn 되는 자식 프로세스(DataLoader worker)를 위한 것이다.
    """
    global _BASE_SEED
    _BASE_SEED = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - 헌법 C-5.2 로 테스트에서는 항상 False
        torch.cuda.manual_seed_all(seed)

    if not deterministic:
        # 속도 우선: cudnn autotuning 을 켠다. RNG seed 는 위에서 이미 고정됐다.
        # A100 실측(S0-RGB, B=32, 512²): 1128 ms/step → 429 ms/step.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def get_base_seed() -> int:
    """``seed_everything_strict`` 가 마지막으로 설정한 기준 시드."""
    return _BASE_SEED


def worker_init_fn(worker_id: int, *, base_seed: Optional[int] = None) -> None:
    """DataLoader worker 별 RNG 고정 (05 §7.1 ``_seed_worker``).

    ``DataLoader(worker_init_fn=worker_init_fn)`` 으로 그대로 넘길 수 있다
    (위치인자 1개 호출 형태를 유지한다).
    """
    base = _BASE_SEED if base_seed is None else int(base_seed)
    s = (base + int(worker_id)) % (2**32)
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)


def make_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """DataLoader ``generator=`` 용 CPU 제너레이터."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
