"""BloomNet 신경망 부품 (`bloomnet.modules`).

이 `__init__` 은 **re-export 만** 한다 (06 §2.1 규칙 3). 로직·부작용을 두지 않는다.
레벨이 다른 모듈을 한 이름공간에서 노출하므로, 소비자는 필요한 레벨만 골라 쓴다.
여기서 노출하는 것은 L1(`common`) 과 L2(`stems`) 뿐이다 — 상위 레벨(bmef/heads 등)을
추가하면 L1 소비자가 L3 를 끌고 오게 되어 §2.1 원칙 2 가 깨진다.
"""

from bloomnet.modules.common import (
    BN_KWARGS,
    ConvBN,
    Downsample,
    DropPath,
    LayerScale,
    SEGate,
    act_layer,
    build_act,
    build_bio_pyramid,
    build_norm,
    init_encoder,
)
from bloomnet.modules.stems import PPN, SPS, TPS, CanonicalScatter, SPSOut

__all__ = [
    # common (L1)
    "BN_KWARGS",
    "ConvBN",
    "Downsample",
    "DropPath",
    "LayerScale",
    "SEGate",
    "act_layer",
    "build_act",
    "build_bio_pyramid",
    "build_norm",
    "init_encoder",
    # stems (L2)
    "CanonicalScatter",
    "PPN",
    "SPS",
    "SPSOut",
    "TPS",
]
