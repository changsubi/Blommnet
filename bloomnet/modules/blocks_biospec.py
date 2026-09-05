"""Spec path 블록 — 02 §4 / 06 §3.3.4 (레벨 L2).

구성 (`BioSpecBlock` 1개 = 아래 3단)::

    x = x + DropPath(γ_a * MSPA(BN_a(x)))     # (1) 다중 스케일 공간 증거
    x = BioGate(x, b)                         # (2) 물리 prior gating (residual 아님)
    x = x + DropPath(γ_b * MLP(BN_b(x)))      # (3) 채널 혼합

동결 사항 / 함정
    * **MSPA 는 pooling 이 아니라 parallel atrous DW conv** 다 (02 §4.2 E10).
      [PPT-s34/35] 의 AvgPool 3/7/11 은 학습 자유도가 0 이고 stage4(16²)에서
      11×11 pool 이 사실상 GAP 으로 붕괴한다.
    * **gate 는 sigmoid 를 1회만** 쓴다. `sigmoid(공간)·sigmoid(채널)` 로 곱하면
      초기값이 0.25 로 눌린다. logit 을 더한 뒤 한 번만 씌워 **초기 att = 0.5 정확히**.
    * **MSPA 는 입력 `u` 가 아니라 `s`(atrous branch 의 출력)를 반환**한다.
      `u * att` 로 만들면 block 의 표현 capacity 가 gate 하나뿐이 된다 (02 §4.2 근거 3).
    * **BioGate 의 query 는 학습된 feature 가 아니라 물리 지수 맵 `[NDCI, MCI_norm]`**
      (X-01/X-27, 정정 B-3) 을 각 stage 해상도로 area-pool 한 것이다.
      `b is None` 이면 **정확한 항등**이다 — 부재를 zeros 로 채우면 BioGate 가 bias
      만으로 동작해 상수 gain 으로 수렴하고 Novelty N2 의 근거가 사라진다(정정 B-9).
    * `gain = 1 + softplus(β̂)·(2·sigmoid(p) − 1)` 이라 **초기 gain 은 비트 단위로 1.0**
      이고 크기가 stage 수·채널 수와 무관하다. 원설계의 `softmax(bio_mask)` 는
      ×1/44.8 로 크기를 파괴했다(02 §4.3 E12).
    * norm 은 :func:`bloomnet.modules.common.build_norm` 을 통해서만 만든다
      (정정 B-1). `gn8` 에서 BioGate 의 `C_r = max(8, C//8)` 이 stage3 에서 **20** 이라
      `GroupNorm(8, 20)` 은 ValueError 다 — 빌더가 `gcd(8, 20) = 4` 그룹을 쓴다.

초기화 계약 (호출자 주의)
    02 §1.5 의 특칙(gate 최종 conv weight/bias = 0)은 `__init__` 말미에서 적용된다.
    상위 모델이 `model.apply(init_encoder)` 를 **나중에** 부르면 이 0 초기화가 지워지므로,
    그 뒤에 `m.reset_gate_()` 를 다시 호출해야 한다(`SEGate.reset_gate_` 와 동일 규약)::

        model.apply(init_encoder)
        for m in model.modules():
            if hasattr(m, "reset_gate_"):
                m.reset_gate_()

레벨 L2 — L1(`modules/common.py`)만 import 한다. 다른 L2 모듈은 import 하지 않는다.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bloomnet.modules.common import DropPath, LayerScale, build_act, build_norm

__all__ = ["MSPA", "BioGate", "BioSpecMLP", "BioSpecBlock"]


def _inv_softplus(y: float) -> float:
    """`softplus(x) = y` 를 만족하는 x. `log(exp(y) − 1)`.

    `torch.expm1` 은 ONNX 에 없어 금지 목록(06 §10-13)에 있으나 여기는 그래프가 아니라
    파이썬 상수 계산이다. 그래도 이름 혼동을 피하려 exp−1 형태로 쓴다.
    """
    y = float(y)
    if y <= 0.0:
        raise ValueError(f"beta_init must be > 0 (softplus 의 치역), got {y}")
    return math.log(math.exp(y) - 1.0)


class MSPA(nn.Module):
    """Multi-Scale Parallel Atrous attention (02 §4.2).

    3개의 dilated depthwise 3×3 branch → 1×1 cross-scale mixing → norm →
    `sigmoid(공간 logit + 채널 logit)` gate.

    Args:
        dim: 채널 수 C.
        dilations: branch 3개의 dilation. stage 별 (1,3,5)/(1,3,5)/(1,2,4)/(1,2,3).
        se_ratio: 채널 gate 병목비. `C_r = max(se_min, C // se_ratio)`.
        se_min: `C_r` 하한.
        norm: `bn|syncbn|gn8` — 06 동결 시그니처에 없는 keyword-only 추가분
            (정정 B-1 이 norm 직접 생성을 금지하므로 kind 를 받을 통로가 필요하다).
            기본값 `"bn"` 이라 동결 호출은 불변.

    params: `27C + 3C² + 2C + (C+1) + 2·C·C_r + C_r + C`
    → C=32 4,585 / 64 16,337 / 160 94,601 / 320 368,401.
    """

    def __init__(
        self,
        dim: int,
        *,
        dilations: Tuple[int, int, int] = (1, 3, 5),
        se_ratio: int = 4,
        se_min: int = 8,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        dils: Tuple[int, ...] = tuple(int(d) for d in dilations)
        if len(dils) != 3:
            raise ValueError(f"MSPA needs exactly 3 dilations, got {dilations!r}")
        if min(dils) < 1:
            raise ValueError(f"dilation must be >= 1, got {dilations!r}")
        self.dim = int(dim)
        self.dilations = dils
        self.c_r = max(int(se_min), int(dim) // int(se_ratio))

        # 3×3 DW, bias 없음. padding = dilation 이라 stride 1 에서 shape 보존
        # (2×2 입력 + dilation 5 도 보존됨 — 02 §1.7 검증).
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim, bias=False)
                for d in dils
            ]
        )
        self.mix = nn.Conv2d(3 * dim, dim, 1, bias=False)
        self.norm = build_norm(norm, dim)

        self.spatial = nn.Conv2d(dim, 1, 1, bias=True)  # 공간 saliency logit
        self.fc1 = nn.Conv2d(dim, self.c_r, 1, bias=True)  # 채널 prior logit
        self.act = build_act("relu")
        self.fc2 = nn.Conv2d(self.c_r, dim, 1, bias=True)
        self.reset_gate_()

    @torch.no_grad()
    def reset_gate_(self) -> None:
        """02 §1.5 특칙: gate 최종 conv weight/bias = 0 → 초기 att = 0.5 정확히."""
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward_parts(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        """`(s, att)` 를 함께 돌려준다 (진단·테스트용). `forward` 는 `s * att`."""
        s = self.norm(self.mix(torch.cat([br(u) for br in self.branches], dim=1)))
        l_sp = self.spatial(s)  # (B,1,h,w)
        # ★ GAP 은 mean(dim=(2,3)) — AdaptiveAvgPool2d 금지 (04 §9.3-A)
        g = s.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        l_ch = self.fc2(self.act(self.fc1(g)))  # (B,C,1,1)
        att = torch.sigmoid(l_sp + l_ch)  # sigmoid 1회, broadcast (B,C,h,w)
        return s, att

    def forward(self, u: Tensor) -> Tensor:
        s, att = self.forward_parts(u)
        return s * att

    def extra_repr(self) -> str:
        return f"dim={self.dim}, dilations={self.dilations}, c_r={self.c_r}"


class BioGate(nn.Module):
    """물리 지수 맵을 query 로 spec feature 의 (채널, 픽셀)을 gating 한다 (02 §4.3).

    Args:
        dim: gating 대상 채널 수 C.
        bio_ch: 지수 채널 수 (=2, `[NDCI, MCI_norm]`).
        reduction: `C_r = max(se_min, C // reduction)`.
        se_min: `C_r` 하한. stage3 에서 `C_r = 20` 이 되는 유일한 비-8배수 폭이다.
        beta_init: `softplus(β̂)` 의 초기값. 0.1 → β̂ = −2.25217.
        norm: `bn|syncbn|gn8` (keyword-only 추가분, 정정 B-1).

    `forward(x, b)` 에서 `b is None` 이면 **연산 없이 `x` 를 그대로** 돌려준다.
    params: `5·C_r + C_r·C + C + 1` → 329 / 617 / 3,461 / 13,321.
    """

    def __init__(
        self,
        dim: int,
        *,
        bio_ch: int = 2,
        reduction: int = 8,
        se_min: int = 8,
        beta_init: float = 0.1,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.bio_ch = int(bio_ch)
        self.c_r = max(int(se_min), int(dim) // int(reduction))

        self.fc1 = nn.Conv2d(self.bio_ch, self.c_r, 1, bias=True)
        self.norm = build_norm(norm, self.c_r)
        self.act = build_act("gelu")
        self.fc2 = nn.Conv2d(self.c_r, self.dim, 1, bias=True)
        self.beta_hat = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
        self.reset_gate_()

    @torch.no_grad()
    def reset_gate_(self) -> None:
        """02 §1.5 특칙: 2nd conv = 0 → `p ≡ 0` → 초기 gain 이 정확히 1.0."""
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @property
    def beta(self) -> Tensor:
        """`β = softplus(β̂) > 0`. 학습 로그에 stage 별로 기록할 것 (02 §1.4/§4.3)."""
        return F.softplus(self.beta_hat)

    def gain(self, b: Tensor) -> Tensor:
        """`1 + β·(2·sigmoid(p) − 1) ∈ (1−β, 1+β)`. 초기값은 비트 단위로 1.0."""
        p = self.fc2(self.act(self.norm(self.fc1(b))))
        alpha = torch.sigmoid(p)
        return 1.0 + self.beta * (2.0 * alpha - 1.0)

    def forward(self, x: Tensor, b: Optional[Tensor] = None) -> Tensor:
        if b is None:
            # 부재 = 항등. zeros 로 대체하면 bias 만으로 동작하는 상수 gain 이 된다.
            return x
        if b.shape[0] != x.shape[0] or b.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"BioGate: bio {tuple(b.shape)} 와 feature {tuple(x.shape)} 의 "
                "(B,H,W) 가 다르다. bio pyramid 를 stage 해상도로 만들어 넘겨라"
            )
        if b.shape[1] != self.bio_ch:
            raise ValueError(f"BioGate expects bio_ch={self.bio_ch}, got {b.shape[1]}")
        return x * self.gain(b)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, bio_ch={self.bio_ch}, c_r={self.c_r}"


class BioSpecMLP(nn.Module):
    """`1x1(C→rC) → DW3x3 → GELU → 1x1(rC→C)` (02 §4.4, CMNeXt `Mlp` 와 동일 구조).

    norm 이 없다 — pre-norm(`BN_b`)은 `BioSpecBlock` 이 소유한다.
    params: `2rC² + 11rC + C` → 19,232 / 71,232 / 212,000 / 833,600.
    """

    def __init__(self, dim: int, *, mlp_ratio: int = 8) -> None:
        super().__init__()
        hidden = int(dim) * int(mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1, bias=True)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.act = build_act("gelu")
        self.fc2 = nn.Conv2d(hidden, dim, 1, bias=True)

    def forward(self, v: Tensor) -> Tensor:
        return self.fc2(self.act(self.dw(self.fc1(v))))


class BioSpecBlock(nn.Module):
    """Spec path 의 residual block (02 §4.1/§4.5).

    Args:
        dim: 채널 수 C.
        mlp_ratio: MLP 확장비 r. stage 스케줄 (8, 8, 4, 4).
        dilations: MSPA branch dilation 3개.
        bio_ch: 지수 채널 수.
        se_ratio: MSPA 채널 gate 병목비 (BioGate 의 reduction 과 다르다 — 4 vs 8).
        layer_scale_init: γ 초기값 (per-channel).
        drop_path: stochastic depth 확률.
        beta_init: BioGate β 초기값 (keyword-only 추가분 — `model.biospec.beta_init`).
        norm: `bn|syncbn|gn8` (keyword-only 추가분, 정정 B-1).

    params = MSPA + BioGate + MLP + 6C → 24,338 / 88,570 / 311,022 / 1,217,242.
    """

    def __init__(
        self,
        dim: int,
        *,
        mlp_ratio: int = 8,
        dilations: Tuple[int, int, int] = (1, 3, 5),
        bio_ch: int = 2,
        se_ratio: int = 4,
        layer_scale_init: float = 0.01,
        drop_path: float = 0.0,
        beta_init: float = 0.1,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm_a = build_norm(norm, dim)
        self.mspa = MSPA(dim, dilations=dilations, se_ratio=se_ratio, norm=norm)
        self.ls_a = LayerScale(dim, layer_scale_init)
        self.dp_a = DropPath(drop_path)

        self.biogate = BioGate(dim, bio_ch=bio_ch, beta_init=beta_init, norm=norm)

        self.norm_b = build_norm(norm, dim)
        self.mlp = BioSpecMLP(dim, mlp_ratio=mlp_ratio)
        self.ls_b = LayerScale(dim, layer_scale_init)
        self.dp_b = DropPath(drop_path)

    def forward(self, x: Tensor, b: Optional[Tensor] = None) -> Tensor:
        x = x + self.dp_a(self.ls_a(self.mspa(self.norm_a(x))))
        x = self.biogate(x, b)  # residual 아님 — 곱셈 gate
        x = x + self.dp_b(self.ls_b(self.mlp(self.norm_b(x))))
        return x
