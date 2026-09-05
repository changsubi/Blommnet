"""ONNX export + 계약 검증 (L6) — 04 §9.4, 06 §3.6.

의존성 격리 (06 §5.0)
---------------------
onnx / onnxruntime / onnxsim 은 ``requirements-deploy.txt`` 이며 현재 venv 에
**설치되어 있지 않다**. 이 모듈은 **import 시점에는 onnx 를 요구하지 않는다** —
:func:`export` 등 실제로 필요한 함수 안에서 지연 import 하고, 미설치 시
무엇을 설치해야 하는지 적힌 :class:`ModuleNotFoundError` 를 낸다.
조용한 skip 은 금지다.

torch 2.13 주의 (정정 A-29)
---------------------------
``torch.onnx.export`` 의 기본값이 **dynamo=True** 다. 04 §9.4 가 지시한
``dynamic_axes=`` 는 legacy TorchScript exporter 전용 인자라 dynamo 경로에서는
조용히 무시되고, 그 경로는 onnxscript 미설치로 ``ModuleNotFoundError`` 를 낸다.
따라서 ``dynamo=False`` 를 **명시**한다.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from bloomnet.models.bloomnet import BloomNet, ExportWrapper

__all__ = [
    "export",
    "assert_no_shape_nodes",
    "verify_onnx",
    "SHAPE_DEPENDENT_OPS",
    "DEPLOY_PACKAGES",
]

#: 04 §9.3-B/O — export 그래프에 있으면 안 되는 shape 의존 연산자.
SHAPE_DEPENDENT_OPS: Tuple[str, ...] = ("Shape", "Gather")

#: 이 모듈이 요구하는 패키지 (requirements-deploy.txt).
DEPLOY_PACKAGES: Tuple[str, ...] = ("onnx", "onnxruntime", "onnxsim")


def _require(pkg: str) -> Any:
    """지연 import. 미설치면 설치 방법이 적힌 예외를 낸다 (조용한 skip 금지)."""
    if importlib.util.find_spec(pkg) is None:
        missing = [p for p in DEPLOY_PACKAGES if importlib.util.find_spec(p) is None]
        raise ModuleNotFoundError(
            f"'{pkg}' 미설치 — ONNX export 게이트(04 §9.4)를 실행할 수 없다.\n"
            f"  미설치 목록: {missing}\n"
            f"  설치:        pip install -r bloomnet/requirements-deploy.txt\n"
            f"  (헌법 C-5.3: 인터넷 다운로드가 필요하므로 오프라인 wheel 을 미리 확보한다 — §7.3 B5)"
        )
    return importlib.import_module(pkg)


def export(
    model: BloomNet,
    out_path: Path,
    *,
    input_hw: Tuple[int, int] = (1024, 1024),
    opset: int = 17,
    dynamic_batch: bool = True,
    use_dynamo: bool = False,
) -> Path:
    """``model`` → ONNX 파일. 04 §9.4 step 1~2.

    Args:
        model: :meth:`BloomNet.deploy` 를 **먼저 호출한** 모델.
        out_path: 산출 ``.onnx`` 경로.
        input_hw: ``deploy.input_hw``. ``ExportWrapper`` 가 이 값으로 ``PagFM``/``PAPPM``
            의 ``out_hw`` 를 int 리터럴로 굽는다 → step 3(Shape/Gather 0개)의 전제.
        opset: ONNX opset. GELU tanh 근사를 위해 17 이상 권장 (04 §9.3-G).
        dynamic_batch: batch 축만 동적으로. H/W 는 정적 고정 (04 §9.3-C).
        use_dynamo: **기본 False** (정정 A-29). True 면 ``dynamic_axes`` 대신
            ``dynamic_shapes`` 를 써야 하고 onnxscript 가 필요하다.

    Returns:
        ``out_path``.

    Raises:
        AssertionError: ``deploy()`` 를 부르지 않았거나 train 모드일 때 (ExportWrapper 계약).
        ModuleNotFoundError: onnx 미설치.
    """
    _require("onnx")  # 산출물 검증까지 못 할 export 는 애초에 시작하지 않는다.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = ExportWrapper(model, input_hw=input_hw)
    dummies = wrapper.dummy_inputs(1)
    input_names = list(wrapper.input_names)
    output_names = ["seg", "chl", "conf"]

    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    if dynamic_batch:
        dynamic_axes = {n: {0: "B"} for n in input_names + output_names}

    torch.onnx.export(
        wrapper,
        tuple(dummies),
        str(out_path),
        opset_version=int(opset),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=bool(use_dynamo),  # ★ 정정 A-29 — torch 2.13 기본은 True 다
    )
    return out_path


def assert_no_shape_nodes(onnx_path: Path) -> None:
    """그래프에 ``Shape``/``Gather`` 노드가 0개인지 검사한다 (04 §9.4 step 3).

    ★ 정정 A-21/B-14: **ExportWrapper 가 산출한 ONNX 에만** 적용한다.
    학습 그래프의 ``F.interpolate(size=...)`` 는 규칙 위반이 아니다 — H=64/128 에서
    고정 ``scale_factor`` 가 실제로 터지기 때문에 ``size=`` 가 학습 정본이다.

    Raises:
        AssertionError: shape 의존 노드가 남아 있을 때 (op_type → 개수를 메시지에 담는다).
    """
    onnx = _require("onnx")
    model = onnx.load(str(onnx_path))
    found: Dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type in SHAPE_DEPENDENT_OPS:
            found[node.op_type] = found.get(node.op_type, 0) + 1
    if found:
        raise AssertionError(
            f"04 §9.3-B/O 위반: {onnx_path.name} 에 shape 의존 노드가 남아 있다 {found}. "
            "ExportWrapper(input_hw=...) 로 out_hw 를 주입했는지 확인하라 (정정 B-14)."
        )


def verify_onnx(onnx_path: Path, model: BloomNet, *, atol: float = 1e-2) -> Dict[str, float]:
    """onnxruntime 출력과 PyTorch 출력을 대조한다 (04 §9.4 step 4~5의 CPU 부분).

    TRT 대조(polygraphy)는 실기에서 수행한다. 여기서는 ORT(CPU) vs PyTorch(fp32) 만
    본다 — 이 단계가 깨지면 TRT 이전에 그래프 자체가 틀린 것이다.

    Args:
        onnx_path: :func:`export` 산출물.
        model: 같은 ``deploy()`` 상태의 모델.
        atol: 절대 허용 오차.

    Returns:
        ``{"max_abs_diff_seg": …, "max_abs_diff_chl": …, "max_abs_diff_conf": …,
        "conf_min": …, "conf_max": …, "chl_min": …, "chl_max": …}``

    Raises:
        AssertionError: 오차 초과 또는 conf/chl 범위 계약 위반.
    """
    ort = _require("onnxruntime")
    onnx_path = Path(onnx_path)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()
    hw = [d for d in in_meta[0].shape[-2:]]
    h = int(hw[0]) if isinstance(hw[0], int) else 1024
    w = int(hw[1]) if isinstance(hw[1], int) else 1024

    wrapper = ExportWrapper(model, input_hw=(h, w))
    torch.manual_seed(0)
    dummies: List[Tensor] = [torch.randn_like(t) for t in wrapper.dummy_inputs(1)]

    with torch.no_grad():
        ref = wrapper(*dummies)

    feed = {m.name: d.numpy() for m, d in zip(in_meta, dummies)}
    got = sess.run(None, feed)

    stats: Dict[str, float] = {}
    for name, r, g in zip(("seg", "chl", "conf"), ref, got):
        diff = float((r.numpy() - g).__abs__().max())
        stats[f"max_abs_diff_{name}"] = diff
        if diff > atol:
            raise AssertionError(f"ORT vs PyTorch {name} maxdiff {diff:.3e} > atol {atol}")

    conf = got[2]
    chl = got[1]
    stats["conf_min"] = float(conf.min())
    stats["conf_max"] = float(conf.max())
    stats["chl_min"] = float(chl.min())
    stats["chl_max"] = float(chl.max())
    # 정정 A-20 / X-25·X-26 — s ∈ [-7,7] clamp 가 유도하는 도달 가능 구간.
    assert stats["conf_min"] >= 0.0293 - 1e-3, f"conf 하한 위반 {stats['conf_min']}"
    assert stats["conf_max"] <= 0.9705 + 1e-3, f"conf 상한 위반 {stats['conf_max']}"
    assert stats["chl_min"] >= -1e-3, f"chl 음수 {stats['chl_min']}"
    return stats
