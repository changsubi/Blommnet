"""RGB path 블록 — MBConv / LiteMLA / ELALocal / ELAGlobal.

정본: 02 §3 (ELA Block), 06 §3.3.3 (API 동결표), 정정 B-1 / B-5 / B-6.

핵심 계약
---------
* **fewer-norm** — MBConv 의 유일한 norm 은 마지막 pointwise 뒤, LiteMLA 의 유일한 norm 은
  ``proj`` 뒤. pre-norm 없음 (EfficientViT 원본, 02 §3.3).
* **residual 형태 고정** — ``x = x + DropPath(gamma * Branch(x))``, gamma 는 per-channel
  ``LayerScale`` (init 0.01, 02 §1.4).
* **(정정 B-5) qkv 채널 배치는 head 별 인터리브** ``[q_0,k_0,v_0, q_1,k_1,v_1, ...]``.
  ``[Q_all|K_all|V_all]`` 로 구현하면 shape/param/finite 테스트를 전부 통과하는 **조용한
  오모델**이 된다 (test_blocks_ela.py 의 naive 참조 비교가 이를 고정한다).
* **(정정 B-6) attention 은 fp32 강제** — ``autocast(enabled=False)`` + ``.float()``.
  fp16 에서 먼저 터지는 것은 ``kv`` 가 아니라 ``out = q @ kv`` 다(02 §3.3 실측표).
  ε=1e-5 없이는 ReLU 출력이 전부 0 인 토큰에서 0-division → NaN 이 확정적이다.
* **(정정 B-1) 모든 norm 은 ``build_norm`` 을 통해서만 만든다.** 모듈이 ``nn.BatchNorm2d`` 를
  직접 호출하는 것을 금지한다 (``norm="gn8"`` 스위치가 깨진다).

동결표(06 §3.3.3) 대비 추가한 인자는 ``norm: str = "bn"`` **하나뿐**이며 keyword-only 다.
기본값이 초판 동작과 동일하므로 위치 호출 계약은 변하지 않는다. 이 인자가 없으면
``model.norm ∈ {bn, syncbn, gn8}`` (06 §4.2 / V20 / 02 §9 "norm 파라메트릭") 를
블록까지 전달할 방법이 없다.

배포 정책 토큰 대응 (deploy/trt_policy.py 담당자 인계)
-----------------------------------------------------
``deploy.fp32_forced`` / ``fp16_locked`` 의 ``encoder.*.litemla.attention`` 은
본 파일에서 다음에 해당한다:

* ``litemla``   → ``ELAGlobal.context_module`` (``LiteMLA`` 인스턴스). stage3/4 × 7 block.
* ``attention`` → ``LiteMLA.forward`` 가 호출하는 모듈 전역 함수 ``relu_linear_attention``
  의 ``_fp32_ctx`` 블록. **nn.Module 이 아니라 함수**이므로 이름 기반 layer 매칭으로는
  잡히지 않는다. TRT 에서는 ``LiteMLA.proj`` 입력 텐서를 출력으로 마킹해
  (02 §3.3, 04 §9.4 polygraphy 대조) 정밀도 경계를 확인해야 한다.
"""

from __future__ import annotations

import contextlib
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bloomnet.modules.common import DropPath, LayerScale, build_act, build_norm, init_encoder

__all__ = ["relu_linear_attention", "MBConv", "LiteMLA", "ELALocal", "ELAGlobal"]

# torch.autocast 가 인정하는 device type. 그 외(meta 등)에서는 컨텍스트를 걸지 않는다.
_AUTOCAST_DEVICES = frozenset({"cpu", "cuda", "xpu", "hpu", "mps"})


def _fp32_ctx(device_type: str):
    """autocast 를 끄는 컨텍스트. 지원되지 않는 device 에서는 no-op."""
    if device_type in _AUTOCAST_DEVICES:
        return torch.autocast(device_type=device_type, enabled=False)
    return contextlib.nullcontext()


def relu_linear_attention(q: Tensor, k: Tensor, v: Tensor, eps: float = 1e-5) -> Tensor:
    """ReLU linear attention — 02 §3.3 수식 정확판 (ones-column 트릭, fp32 강제).

    ``O_i = ReLU(q_i)·(Σ_j ReLU(k_j)^T v_j) / (ReLU(q_i)·(Σ_j ReLU(k_j)^T) + eps)``

    Args:
        q, k, v: ``(B, G, N, d)``. G = head-group 수, N = 토큰 수, d = head_dim.
                 v 에는 activation 을 적용하지 않는다.
        eps: 분모 하한. ReLU 출력이 전부 0 이면 분모가 정확히 0 이 되므로 필수다.

    Returns:
        ``(B, G, N, d)``, dtype 은 ``q`` 와 동일 (내부 누산은 항상 float32).

    Note:
        1/N 스케일링은 없다 — 분모가 이미 ``Σ_j ReLU(k_j)`` 로 정규화 항이다.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"q/k/v shape mismatch: {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}"
        )
    orig_dtype = q.dtype
    q = torch.relu(q)
    k = torch.relu(k)
    v = F.pad(v, (0, 1), mode="constant", value=1.0)  # (B,G,N,d+1)
    with _fp32_ctx(q.device.type):
        qf, kf, vf = q.float(), k.float(), v.float()
        kv = kf.transpose(-1, -2) @ vf  # (B,G,d,d+1)
        out = qf @ kv  # (B,G,N,d+1)
        out = out[..., :-1] / (out[..., -1:] + eps)  # (B,G,N,d)
    return out.to(orig_dtype)


class MBConv(nn.Module):
    """EfficientViT fewer-norm MBConv (02 §3.2).

    ``use_bias=(True, True, False)``, ``norm=(None, None, BN)``, ``act=(hswish, hswish, None)``.
    params = ``8C² + 46C`` (e=4, k=3). 잔차·LayerScale 은 포함하지 않는다 — 상위 블록 소관.
    """

    def __init__(
        self,
        dim: int,
        *,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if expand_ratio <= 0:
            raise ValueError(f"expand_ratio must be positive, got {expand_ratio}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd (symmetric padding), got {kernel_size}")
        mid = dim * expand_ratio
        self.dim = dim
        self.expand_ratio = expand_ratio
        self.inverted_conv = nn.Conv2d(dim, mid, 1, bias=True)
        self.act1 = build_act("hswish")
        self.depth_conv = nn.Conv2d(
            mid, mid, kernel_size, padding=kernel_size // 2, groups=mid, bias=True
        )
        self.act2 = build_act("hswish")
        self.point_conv = nn.Conv2d(mid, dim, 1, bias=False)
        self.norm = build_norm(norm, dim)  # branch 유일 norm. 뒤에 activation 없음
        self.apply(init_encoder)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act1(self.inverted_conv(x))
        h = self.act2(self.depth_conv(h))
        return self.norm(self.point_conv(h))


class LiteMLA(nn.Module):
    """ReLU linear attention + multi-scale token aggregation (02 §3.3).

    params = ``6C² + 6·C·head_dim + 104C`` (scales=(3,5)).
    multi-scale DW 는 **concat 된 QKV 에 scale 당 1 개** (Q/K/V 개별 6 개가 아니다) —
    QKV 가 채널 축으로 이어붙어 있어 depthwise conv 는 Q/K/V 를 섞지 않으므로 수학적으로 동등.
    """

    def __init__(
        self,
        dim: int,
        *,
        head_dim: int = 32,
        scales: Tuple[int, ...] = (3, 5),
        eps: float = 1e-5,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        if dim <= 0 or head_dim <= 0:
            raise ValueError(f"dim/head_dim must be positive, got {dim}/{head_dim}")
        if dim % head_dim != 0:
            raise ValueError(f"dim({dim}) must be divisible by head_dim({head_dim})")
        scales = tuple(int(s) for s in scales)
        if any(s % 2 == 0 or s <= 0 for s in scales):
            raise ValueError(f"scales must be positive odd ints, got {scales}")
        self.dim = dim
        self.head_dim = head_dim
        self.heads = dim // head_dim
        self.scales = scales
        self.eps = eps

        total = 3 * dim
        self.qkv = nn.Conv2d(dim, total, 1, bias=False)  # norm/act 없음 (fewer-norm)
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(total, total, s, padding=s // 2, groups=total, bias=False),
                    # groups=3h -> 그룹 크기 = head_dim. 정정 B-5 의 인터리브 배치를 전제한다.
                    nn.Conv2d(total, total, 1, groups=3 * self.heads, bias=False),
                )
                for s in scales
            ]
        )
        self.proj = nn.Conv2d(dim * (1 + len(scales)), dim, 1, bias=False)
        self.proj_norm = build_norm(norm, dim)
        self.apply(init_encoder)

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        n = h * w
        base = self.qkv(x)  # (B,3C,H,W)
        if self.aggreg:
            ms = torch.cat([base, *(agg(base) for agg in self.aggreg)], dim=1)
        else:
            ms = base
        # (B, 3C(1+S), H, W) -> (B, G=h(1+S)*3, 3d, N) -> (B, G, N, 3d)
        qkv = ms.reshape(b, -1, 3 * self.head_dim, n).transpose(-1, -2)
        q, k, v = qkv.chunk(3, dim=-1)  # 각 (B, G, N, d)
        out = relu_linear_attention(q, k, v, self.eps)  # 모듈 전역 참조(테스트가 감쌀 수 있게)
        out = out.transpose(-1, -2).reshape(b, -1, h, w)  # (B, C(1+S), H, W)
        return self.proj_norm(self.proj(out))


class ELALocal(nn.Module):
    """stage 1–2 블록 — MBConv residual 단독 (02 §3.2). params = ``8C² + 47C``."""

    def __init__(
        self,
        dim: int,
        *,
        expand_ratio: int = 4,
        layer_scale_init: float = 0.01,
        drop_path: float = 0.0,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.local_module = MBConv(dim, expand_ratio=expand_ratio, norm=norm)
        self.ls_local = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path)
        self.apply(init_encoder)  # LayerScale γ 는 Parameter 라 덮어쓰이지 않는다

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.ls_local(self.local_module(x)))


class ELAGlobal(nn.Module):
    """stage 3–4 블록 — LiteMLA residual + MBConv residual (02 §3.3).

    params = ``14C² + 6·C·head_dim + 152C``.
    두 잔차 분기는 **각각 독립된 DropPath** 를 갖는다(같은 p, 독립 표본).
    """

    def __init__(
        self,
        dim: int,
        *,
        head_dim: int = 32,
        scales: Tuple[int, ...] = (3, 5),
        expand_ratio: int = 4,
        eps: float = 1e-5,
        layer_scale_init: float = 0.01,
        drop_path: float = 0.0,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.context_module = LiteMLA(dim, head_dim=head_dim, scales=scales, eps=eps, norm=norm)
        self.ls_ctx = LayerScale(dim, layer_scale_init)
        self.drop_path_ctx = DropPath(drop_path)
        self.local_module = MBConv(dim, expand_ratio=expand_ratio, norm=norm)
        self.ls_local = LayerScale(dim, layer_scale_init)
        self.drop_path_local = DropPath(drop_path)
        self.apply(init_encoder)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path_ctx(self.ls_ctx(self.context_module(x)))
        x = x + self.drop_path_local(self.ls_local(self.local_module(x)))
        return x
