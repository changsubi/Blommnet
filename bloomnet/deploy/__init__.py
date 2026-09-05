"""``bloomnet.deploy`` — 배포(ONNX/TensorRT) 경로.

06 §2.1 원칙 3 에 따라 **re-export 만** 한다.
``export_onnx``(L6)는 여기서 re-export 하지 않는다 — L1 소비자(`trt_policy`,
`postprocess`)가 L6 를 끌고 오면 레벨 규약이 깨지고, onnx 미설치 환경에서
``import bloomnet.deploy`` 자체가 실패한다. 소비자는
``from bloomnet.deploy.export_onnx import export`` 로 직접 import 한다.
"""

from .postprocess import chl_to_alert_level, confidence_map, sigma_chl
from .trt_policy import (
    FP16_LOCKED_MODULES,
    FP32_FORCED_MODULES,
    RISK_TABLE,
    assert_precision_policy,
    matches_module_pattern,
)

__all__ = [
    "chl_to_alert_level",
    "confidence_map",
    "sigma_chl",
    "FP32_FORCED_MODULES",
    "FP16_LOCKED_MODULES",
    "RISK_TABLE",
    "assert_precision_policy",
    "matches_module_pattern",
]
