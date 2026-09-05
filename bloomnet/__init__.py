"""BloomNet — 녹조 영역 추정 모델 (T1).

패키지 `__init__` 은 **re-export 만** 한다 (06 §2.1 규칙 3 — 순환 import 의 최대 원인).
로직·부작용을 여기 두지 않는다.
"""

from bloomnet.config import (
    BloomNetConfig,
    apply_overrides,
    default_config,
    dump_config,
    load_config,
)
from bloomnet.version import __version__, git_revision

__all__ = [
    "BloomNetConfig",
    "apply_overrides",
    "default_config",
    "dump_config",
    "load_config",
    "git_revision",
    "__version__",
]
