"""MAC/FLOP 카운터 — ``FlopCounterMode`` 래퍼 + conv/linear forward hook 카운터.

**MAC 규약 (06 §6)**: conv/matmul 의 multiply-accumulate 만 센다. BN·activation·
``interpolate``·elementwise 는 제외한다. ``GFLOP = 2 × GMAC``. 기본 B=1.

두 경로를 **모두** 제공한다.

* :func:`count_macs_hooks` — ``nn.Module`` forward hook. 모듈 단위 내역이 나오므로
  T24 의 "모듈 단위 assert"(정정 A-14)에 쓴다. functional 호출(``F.conv2d``,
  ``torch.matmul``)은 못 센다.
* :func:`count_macs_flop_counter` — ``torch.utils.flop_counter.FlopCounterMode``.
  functional/attention 까지 세지만 모듈 귀속이 거칠다.

**측정 해상도 (정정 A-24)**: 1024² S2-Full 1회 = 62.678 GMAC = 125.4 GFLOP 로 4-스레드
CPU 규약(전 테스트 합 < 30 s)과 양립하지 않는다. 256² 에서 1회 재고 :func:`scale_macs`
로 ×4(512²)·×16(1024²) 한다.

레벨 L1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "MacReport",
    "count_parameters",
    "count_macs_hooks",
    "count_macs_flop_counter",
    "count_macs",
    "scale_macs",
    "conv_macs",
    "linear_macs",
]


@dataclass
class MacReport:
    """MAC 측정 결과.

    Attributes:
        total: 총 MAC (multiply-accumulate 수).
        by_module: 모듈 경로 → MAC. hook 경로에서만 채워진다.
        by_op: 연산 종류 → MAC.
        method: ``"hooks"`` | ``"flop_counter"``.
        input_hw: 측정에 쓴 ``(H, W)``. 알 수 없으면 None.
    """

    total: int = 0
    by_module: Dict[str, int] = field(default_factory=dict)
    by_op: Dict[str, int] = field(default_factory=dict)
    method: str = "hooks"
    input_hw: Optional[Tuple[int, int]] = None

    @property
    def gmac(self) -> float:
        return self.total / 1e9

    @property
    def gflop(self) -> float:
        """``GFLOP = 2 × GMAC`` (06 §6 규약)."""
        return 2.0 * self.total / 1e9

    def top(self, n: int = 10) -> List[Tuple[str, int]]:
        return sorted(self.by_module.items(), key=lambda kv: -kv[1])[:n]


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    """파라미터 수. ``trainable_only=True`` 면 ``requires_grad`` 인 것만."""
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def conv_macs(module: nn.modules.conv._ConvNd, out_numel_per_sample: int) -> int:
    """conv MAC = ``k_prod · c_in/groups · c_out · (출력 공간 크기)``. bias 는 세지 않는다.

    Args:
        out_numel_per_sample: 배치 1장의 출력 공간 원소 수 (= ``C_out · H_out · W_out``).
    """
    k = 1
    for s in module.kernel_size:
        k *= int(s)
    return int(k * (module.in_channels // module.groups) * out_numel_per_sample)


def linear_macs(module: nn.Linear, out_numel_per_sample: int) -> int:
    """linear MAC = ``in_features · (출력 원소 수)``."""
    return int(module.in_features * out_numel_per_sample)


_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_DECONV_TYPES = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def count_macs_hooks(
    model: nn.Module,
    inputs: Sequence[Any] | Mapping[str, Any] | Tensor,
    *,
    input_hw: Optional[Tuple[int, int]] = None,
    include: Tuple[type, ...] = (),
) -> MacReport:
    """forward hook 으로 conv/linear MAC 을 센다. 모듈 단위 내역을 낸다.

    Args:
        inputs: 텐서 1개, 위치인자 tuple/list, 또는 키워드인자 dict.
        include: 추가로 세고 싶은 모듈 타입(현재는 미사용, 확장 지점).

    Note:
        ``F.conv2d`` / ``torch.matmul`` 같은 functional 호출은 잡지 못한다.
        LiteMLA 의 ``q@kv`` 처럼 functional matmul 이 지배적인 모듈은
        :func:`count_macs_flop_counter` 로 교차검증하라.
    """
    report = MacReport(method="hooks", input_hw=input_hw)
    handles: List[Any] = []
    name_of: Dict[nn.Module, str] = {m: n for n, m in model.named_modules()}

    def _acc(module: nn.Module, macs: int, op: str) -> None:
        name = name_of.get(module, module.__class__.__name__)
        report.total += macs
        report.by_module[name] = report.by_module.get(name, 0) + macs
        report.by_op[op] = report.by_op.get(op, 0) + macs

    def conv_hook(module: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
        if not isinstance(out, Tensor):
            return
        b = out.shape[0] if out.dim() > 1 else 1
        per_sample = out.numel() // max(b, 1)
        _acc(module, b * conv_macs(module, per_sample), module.__class__.__name__)

    def deconv_hook(module: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
        # ConvTranspose 는 입력 원소마다 커널을 뿌린다: k_prod · c_out/groups · (입력 크기).
        if not isinstance(out, Tensor) or not inp or not isinstance(inp[0], Tensor):
            return
        x = inp[0]
        k = 1
        for s in module.kernel_size:
            k *= int(s)
        macs = int(k * (module.out_channels // module.groups) * x.numel())
        _acc(module, macs, module.__class__.__name__)

    def linear_hook(module: nn.Module, inp: Tuple[Any, ...], out: Any) -> None:
        if not isinstance(out, Tensor):
            return
        _acc(module, linear_macs(module, out.numel()), "Linear")

    for m in model.modules():
        if isinstance(m, _CONV_TYPES):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, _DECONV_TYPES):
            handles.append(m.register_forward_hook(deconv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            _call(model, inputs)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    if input_hw is None:
        report.input_hw = _infer_hw(inputs)
    return report


def count_macs_flop_counter(
    model: nn.Module,
    inputs: Sequence[Any] | Mapping[str, Any] | Tensor,
    *,
    input_hw: Optional[Tuple[int, int]] = None,
    depth: int = 3,
) -> MacReport:
    """``torch.utils.flop_counter.FlopCounterMode`` 래퍼.

    FlopCounterMode 는 conv/matmul 을 **FLOP = 2 × MAC** 으로 센다. 06 §6 규약에 맞추어
    2 로 나눈 MAC 을 반환한다.
    """
    from torch.utils.flop_counter import FlopCounterMode

    report = MacReport(method="flop_counter", input_hw=input_hw)
    was_training = model.training
    model.eval()
    counter = FlopCounterMode(display=False, depth=depth)
    try:
        with torch.no_grad(), counter:
            _call(model, inputs)
    finally:
        model.train(was_training)

    report.total = int(counter.get_total_flops()) // 2
    counts = counter.get_flop_counts()
    for mod_name, ops in counts.items():
        sub = 0
        for op, flops in ops.items():
            name = getattr(op, "__name__", str(op))
            if mod_name == "Global":
                report.by_op[name] = report.by_op.get(name, 0) + int(flops) // 2
            sub += int(flops) // 2
        if mod_name != "Global":
            report.by_module[mod_name] = sub
    if input_hw is None:
        report.input_hw = _infer_hw(inputs)
    return report


def count_macs(
    model: nn.Module,
    inputs: Sequence[Any] | Mapping[str, Any] | Tensor,
    *,
    method: str = "hooks",
    input_hw: Optional[Tuple[int, int]] = None,
) -> MacReport:
    """``method`` ∈ {``"hooks"``, ``"flop_counter"``} 로 위임한다."""
    if method == "hooks":
        return count_macs_hooks(model, inputs, input_hw=input_hw)
    if method == "flop_counter":
        return count_macs_flop_counter(model, inputs, input_hw=input_hw)
    raise ValueError(f"unknown method: {method!r} (hooks | flop_counter)")


def scale_macs(macs: int | float, *, from_hw: Tuple[int, int], to_hw: Tuple[int, int]) -> float:
    """해상도 스케일링 (정정 A-24).

    conv MAC 은 출력 화소 수에 선형이므로 배율은 면적비다.
    256² 측정 → 512² 는 ×4, 1024² 는 ×16.

    Note:
        전역 pooling branch(PAPPM 의 global branch = 전체의 5e-5 %)처럼 해상도에
        비선형인 항이 소량 섞이므로, 이 함수의 결과는 근사다. T24 는 ±0.5 % 예산으로 본다.
    """
    fh, fw = from_hw
    th, tw = to_hw
    if fh <= 0 or fw <= 0:
        raise ValueError(f"from_hw must be positive, got {from_hw}")
    return float(macs) * (float(th) * float(tw)) / (float(fh) * float(fw))


def _call(model: nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, Tensor):
        return model(inputs)
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _infer_hw(inputs: Any) -> Optional[Tuple[int, int]]:
    cands: List[Tensor] = []
    if isinstance(inputs, Tensor):
        cands = [inputs]
    elif isinstance(inputs, Mapping):
        cands = [v for v in inputs.values() if isinstance(v, Tensor)]
    elif isinstance(inputs, (tuple, list)):
        cands = [v for v in inputs if isinstance(v, Tensor)]
    for t in cands:
        if t.dim() == 4:
            return (int(t.shape[-2]), int(t.shape[-1]))
    return None
