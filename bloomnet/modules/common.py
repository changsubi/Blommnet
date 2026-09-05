"""인코더/디코더 공통 부품 — 02 §1.2~1.5, §4.3, §6.2 / 06 §3.3.1 (레벨 L1).

동결 사항
    * **norm 은 :func:`build_norm` 을 통해서만 만든다.** 모듈이 ``nn.BatchNorm2d`` 를 직접
      호출하는 것을 금지한다 (정정 B-1 / A-37 / V20). ``gn8`` 은 문자 그대로의
      ``GroupNorm(8, C)`` 가 아니라 ``GroupNorm(gcd(8, C), C)`` 다 — BioGate 의
      ``C_r = max(8, C//8)`` 이 stage3 에서 **20** 이라 ``GroupNorm(8, 20)`` 은
      ``ValueError`` 로 모델 생성 자체를 실패시킨다. ``gcd`` 방식은 파라미터 수를
      보존한다(BN(20) = GN(4,20) = 40) — 06 §6.1/§6.2 회귀 상수가 ``norm`` 과 무관해진다.
    * **bio pyramid 는 ``adaptive_avg_pool2d`` 를 쓰지 않는다** (정정 B-7 / 04 §9.3 항목 O).
      출력 크기가 feature 텐서 shape 에서 오면 ONNX 에 ``Shape→Gather`` 서브그래프가 생겨
      04 §9.4 step 3("Shape/Gather 0개")을 통과할 수 없다. 축소비가 정확히 정수이므로
      고정 커널 ``avg_pool2d(k=s, s)`` 와 **수학적으로 동등**하다(CPU 실측 maxdiff 0.0).
    * **GAP 은 ``x.mean(dim=(2,3), keepdim=True)``** 로 구현한다 (04 §9.3-A,
      ``AdaptiveAvgPool2d`` 금지 — 같은 이유).

레벨 L1. `bloomnet.*` 를 import 하지 않는다 (constants(L−1) 조차 필요 없다 —
이 파일의 리터럴은 전부 torch 규약값이다).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "BN_KWARGS",
    "build_norm",
    "build_act",
    "act_layer",
    "ConvBN",
    "Downsample",
    "LayerScale",
    "DropPath",
    "SEGate",
    "build_bio_pyramid",
    "init_encoder",
]

# 02 §1.2: PIDNet 관례 `bn_mom=0.1` 과 동일 ([분석] §D.2).
BN_KWARGS = dict(eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)

NORM_KINDS: Tuple[str, ...] = ("bn", "syncbn", "gn8")
_ACTS = {
    "hswish": nn.Hardswish,
    "relu": nn.ReLU,
    "gelu": lambda: nn.GELU(approximate="tanh"),
    "sigmoid": nn.Sigmoid,
    "hsigmoid": nn.Hardsigmoid,
    "hardsigmoid": nn.Hardsigmoid,  # 06 §3.3.1 SEGate 기본값 표기와의 호환
    "identity": nn.Identity,
    "none": nn.Identity,
}


# ─────────────────────────────────────────────────────────────────────────────
# 빌더 (02 §1.2 / §1.3 동결)
# ─────────────────────────────────────────────────────────────────────────────
def build_norm(kind: str, num_channels: int) -> nn.Module:
    """norm 단일 진입점 (02 §1.2 동결, 정정 B-1).

    Args:
        kind: ``"bn"`` | ``"syncbn"`` | ``"gn8"``.
        num_channels: 정규화할 채널 수.

    Returns:
        ``nn.Module``. 학습 파라미터는 어느 kind 에서도 ``2·C`` 로 동일하다.

    Note:
        ``gn8`` 의 group 수는 ``gcd(8, C)`` 다. ``C=20 → 4`` 그룹(그룹당 5채널),
        ``C = 8k → 8`` 그룹(원 의도 그대로). ``bn ↔ gn8`` 전환은 GN 에
        ``running_mean/var/num_batches_tracked`` 버퍼가 없으므로
        ``load_state_dict(strict=False)`` + 누락/여분 키 로깅이 필수다.
    """
    c = int(num_channels)
    if c <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")
    if kind == "bn":
        return nn.BatchNorm2d(c, **BN_KWARGS)
    if kind == "syncbn":
        return nn.SyncBatchNorm(c, **BN_KWARGS)
    if kind == "gn8":
        return nn.GroupNorm(math.gcd(8, c), c, eps=1e-5, affine=True)
    raise ValueError(f"unknown norm kind: {kind!r} (expected {'|'.join(NORM_KINDS)})")


def build_act(kind: str) -> nn.Module:
    """activation 단일 진입점 (02 §1.3 동결).

    ``'hswish' | 'gelu' | 'relu' | 'hsigmoid' | 'sigmoid' | 'identity'``.
    GELU 는 ``approximate="tanh"`` 로 고정한다.
    """
    try:
        ctor = _ACTS[kind]
    except KeyError:  # pragma: no cover - 오타 방어
        raise ValueError(
            f"unknown act kind: {kind!r} (expected one of {sorted(_ACTS)})"
        ) from None
    return ctor()


# 06 §2.1.3 매니페스트가 common.py 의 공개 이름으로 `act_layer` 를 지정했고,
# 02 §1.2(정정 B-1)의 동결 코드 블록은 같은 함수를 `build_act` 로 부른다. 동일 객체다.
act_layer = build_act


def ConvBN(
    k: int,
    cin: int,
    cout: int,
    *,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    norm: str = "bn",
) -> nn.Sequential:
    """``Conv2d(bias=False) → norm``. activation 없음 (06 §3.3.1).

    padding 은 ``dilation * (k // 2)`` 로 자동 산출되어 stride=1 에서 shape 을 보존한다.
    ``params = k·k·cin·cout/groups + 2·cout``.

    ``norm`` 은 06 동결 시그니처에 없는 keyword-only 추가분이다 —
    02 §1.2(정정 B-1)가 "모듈이 `nn.BatchNorm2d` 를 직접 호출하는 것을 금지"하므로
    호출자가 kind 를 넘길 통로가 필요하다. 기본값 ``"bn"`` 이라 기존 호출은 불변.
    """
    return nn.Sequential(
        nn.Conv2d(
            cin,
            cout,
            kernel_size=k,
            stride=stride,
            padding=dilation * (k // 2),
            dilation=dilation,
            groups=groups,
            bias=False,
            padding_mode="zeros",
        ),
        build_norm(norm, cout),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 모듈
# ─────────────────────────────────────────────────────────────────────────────
class Downsample(nn.Module):
    """stage 간 축소. ``Conv3x3 s2 p1 (bias=False) + norm``, activation 없음 (02 §6.2).

    모든 path·경계 공용이며 path 별로 **사설 인스턴스**를 갖는다(02 §6.3).
    params: 32→64 = 18,560 / 64→160 = 92,480 / 160→320 = 461,440.
    """

    def __init__(self, in_ch: int, out_ch: int, *, norm: str = "bn") -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.norm = build_norm(norm, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.conv(x))


class LayerScale(nn.Module):
    """per-channel residual 스케일 ``γ ∈ ℝ^C`` (02 §1.4).

    스칼라가 아니라 채널별이다 (CMNeXt ``MSPABlock.layer_scale_init_value=1e-2`` 근거).
    학습 로그에 stage 별 ``mean(|γ|)`` 기록 의무가 있다(02 §1.4 ⚠, M-4).
    """

    def __init__(self, dim: int, init_value: float = 0.01) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma.view(1, -1, 1, 1)

    def extra_repr(self) -> str:
        return f"dim={self.gamma.numel()}"


class DropPath(nn.Module):
    """stochastic depth (per-sample). eval 및 ``p==0`` 에서 **정확히 항등**."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= float(p) < 1.0:
            raise ValueError(f"drop_path p must be in [0,1), got {p}")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # (B,1,1,...) 브로드캐스트 — 샘플 단위 drop.
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"p={self.p}"


class SEGate(nn.Module):
    """Squeeze-Excitation gate (02 §5.1, 06 §3.3.1).

    ``GAP → 1x1(C→C_r) → act → 1x1(C_r→C) → gate`` 후 곱한다.
    ``C_r = max(se_min, C // reduction)`` (floor 8 — 02 §5.2).
    2번째 conv 의 weight/bias 를 0 으로 초기화해 **초기 gate = 0.5** (중립)로 둔다.

    gate 는 무한계가 아니어야 한다([분석] D-PHY-01 "무한계 곱셈 gate = 발산 장치").
    GAP 은 ``mean(dim=(2,3))`` 이다 — ``AdaptiveAvgPool2d`` 는 ONNX 에서
    tensor-shape 의존 서브그래프를 만든다(04 §9.3-A).
    """

    def __init__(
        self,
        dim: int,
        *,
        reduction: int = 8,
        se_min: int = 8,
        act: str = "relu",
        gate: str = "hardsigmoid",
    ) -> None:
        super().__init__()
        c_r = max(int(se_min), int(dim) // int(reduction))
        self.fc1 = nn.Conv2d(dim, c_r, 1, bias=True)
        self.act = build_act(act)
        self.fc2 = nn.Conv2d(c_r, dim, 1, bias=True)
        self.gate = build_act(gate)
        self.reset_gate_()

    @torch.no_grad()
    def reset_gate_(self) -> None:
        """02 §1.5 초기화 특칙: 2nd conv weight/bias = 0 → 초기 gate 0.5."""
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)  # ★ AdaptiveAvgPool2d 금지 (04 §9.3-A)
        s = self.fc2(self.act(self.fc1(s)))
        return x * self.gate(s)


# ─────────────────────────────────────────────────────────────────────────────
# bio pyramid (파라미터 0, MAC 0)
# ─────────────────────────────────────────────────────────────────────────────
def build_bio_pyramid(
    x_bio: Optional[Tensor], sizes: Sequence[Tuple[int, int]]
) -> List[Optional[Tensor]]:
    """``x_bio`` 를 각 stage 해상도로 area-average 축소한다 (02 §4.3, 정정 B-7).

    Args:
        x_bio: ``(B, 2, H, W)`` 또는 ``None``.
        sizes: stage 별 목표 ``(H_i, W_i)`` (보통 stride 4/8/16/32).

    Returns:
        길이 ``len(sizes)`` 리스트. ``x_bio is None`` 이면 ``[None] * len(sizes)``.

    Raises:
        ValueError: ``H`` 가 ``H_i`` 로 나눠떨어지지 않을 때. 조용한 반올림 축소를
            허용하면 지수 맵과 feature 의 픽셀 대응이 어긋난다.

    Note:
        ``adaptive_avg_pool2d`` 가 아니라 고정 커널 ``avg_pool2d(k=s, s)`` 를 쓴다.
        축소비가 정수이므로 수치는 동일하고(CPU 실측 maxdiff 0.0) ONNX 에
        ``Shape→Gather`` 가 생기지 않는다. NDCI/MCI_norm 은 비율량이므로
        receptive field 내 **평균 지수**가 물리적으로 옳다(max/nearest 아님).
    """
    n = len(sizes)
    if x_bio is None:
        return [None] * n
    if x_bio.dim() != 4:
        raise ValueError(f"x_bio must be (B,C,H,W), got {tuple(x_bio.shape)}")
    h, w = int(x_bio.shape[-2]), int(x_bio.shape[-1])

    out: List[Optional[Tensor]] = []
    for hi, wi in sizes:
        hi, wi = int(hi), int(wi)
        if hi <= 0 or wi <= 0:
            raise ValueError(f"bio pyramid size must be positive, got {(hi, wi)}")
        if (hi, wi) == (h, w):
            out.append(x_bio)
            continue
        if h % hi != 0 or w % wi != 0:
            raise ValueError(
                f"bio pyramid: ({h},{w}) is not divisible by ({hi},{wi}); "
                "H,W must be a multiple of the stage stride (02 §1.1 / 헌법 C-4)"
            )
        sh, sw = h // hi, w // wi
        out.append(F.avg_pool2d(x_bio, kernel_size=(sh, sw), stride=(sh, sw)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 초기화
# ─────────────────────────────────────────────────────────────────────────────
def init_encoder(m: nn.Module) -> None:
    """모듈 1개를 제자리 초기화한다 (02 §1.5). ``model.apply(init_encoder)`` 로 쓴다.

    * ``Conv2d`` k==1 → ``trunc_normal_(std=0.02)`` (pointwise/projection)
    * ``Conv2d`` k>=3 (DW 포함) → ``kaiming_normal_(fan_out, relu)``
    * bias → 0, ``BatchNorm2d``/``GroupNorm``/``SyncBatchNorm`` → weight 1, bias 0

    ★ 예외는 **각 클래스의 ``__init__`` 말미가 덮어쓴다** (여기서 처리하지 않는다):
    LayerScale γ=0.01, PPN ``a_hat``/``b``, PPN inpainter 최종 1×1(=0),
    ``C_abs``(=0), BioGate ``β̂`` 및 2nd conv, MSPA gate conv, SE 2nd conv.
    따라서 ``apply(init_encoder)`` 는 반드시 그 특칙보다 **먼저** 호출해야 한다.
    """
    if isinstance(m, nn.Conv2d):
        if m.kernel_size == (1, 1):
            nn.init.trunc_normal_(m.weight, std=0.02)
        else:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
