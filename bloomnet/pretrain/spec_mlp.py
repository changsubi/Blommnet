"""S0-Spec 분광 회귀 모델 — 01 §6.5/§6.6, 06 §3.6 (★X-09), 레벨 L2.

``SpecMLP`` 의 **유일한 존재 이유는 이식(transplant)** 이다. SPS(``modules/stems.py``)의
Step 2~4 와 op·shape 이 bit-exact 로 대응하도록 동결되어 있다:

===============  ==========================================  ==========
SpecMLP          SPS (``sps.body`` = ``_SlotStem``)           이식 규칙
===============  ==========================================  ==========
``proj``         ``proj``   ``Conv2d(8,16,1,bias=True)``      T1
``c_abs``        ``c_abs``  ``Parameter(6,16)``               T1′
``n1``           ``norm1``  affine ``(16,)×2``                T2
``dw``           ``dw``     ``Conv2d(16,16,3,g=16)``          **T3 금지**
``mix``          ``patch_embed[0]`` ``Conv2d(16,32,5,s=4)``   T4 (25탭 균등)
``head.bias``    ``ChlHead.out.bias``                         T5
===============  ==========================================  ==========

파라미터 1,089 (``head_out=2`` → 1,122). 01 초판의 961(c_in=6, ``C_abs`` 없음)은 폐기됐다
(정정 A-9 / 06 X-03·X-09).

Note:
    ``n1`` 만 :func:`bloomnet.modules.common.build_norm` 을 쓰지 않는다 — 01 §6.5 가
    ``GroupNorm(num_groups=1)``(= NCHW-safe LayerNorm)로 동결했는데 ``build_norm`` 의
    ``gn8`` 은 ``GroupNorm(gcd(8,16)=8, 16)`` 이라 그룹 수가 다르다. BN 계열은 전부
    ``build_norm`` 을 경유한다(정정 B-1).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import K235_LOG1P_MEAN, MSI_SLOTS
from bloomnet.losses.regression import chl_reg_loss, margin_rank_loss
from bloomnet.modules.common import build_act, build_norm

__all__ = ["SpecMLP", "init_spec_mlp", "spec_loss", "HEAD_BIAS_DEFAULT", "softplus_inv"]

#: ``softplus(1.5662) = 1.75589 = K235_LOG1P_MEAN`` (01 §6.5 초기화 검산, 오차 1e-5).
HEAD_BIAS_DEFAULT: float = 1.5662


def softplus_inv(y: float) -> float:
    """``softplus`` 역함수. ``head.bias`` 를 원하는 초기 예측값에서 역산할 때 쓴다."""
    if y <= 0.0:
        raise ValueError(f"softplus_inv requires y > 0, got {y}")
    return float(torch.special.expm1(torch.tensor(float(y), dtype=torch.float64)).log())


class SpecMLP(nn.Module):
    """S0-Spec: 칩 분광 → ``log1p(Chl-a)`` 회귀 (01 §6.5, 06 §3.6 동결).

    Args:
        c_in: 입력 채널. canonical 6 slot + bio 2 = **8** (X-03).
        num_slots: canonical slot 수 = 6. ``C_abs`` 의 행 수다(bio 는 slot 이 아니다).
        c1: proj 출력 폭 16 (SPS ``mid``).
        c2: mix 출력 폭 32 (SPS ``embed_dim``).
        head_out: 1 → ``z``; 2 → ``[z, log_var]`` (05 Huber-NLL 재사용, X-09).

    Shape:
        ``forward(x (B,8,H,W), m (B,6)) -> (B, head_out)``.
        칩은 3×3 이지만 H,W 는 임의다 — 마지막에 공간 평균을 취한다.

    Note:
        출력은 **log1p 공간의 raw** ``z`` 다. ``softplus`` 는 손실(:func:`spec_loss`)에서
        적용한다 — ``ChlHead`` 와 동일한 규약(04 §8.2.1).
    """

    def __init__(
        self,
        *,
        c_in: int = 8,
        num_slots: int = len(MSI_SLOTS),
        c1: int = 16,
        c2: int = 32,
        head_out: int = 1,
    ) -> None:
        super().__init__()
        if int(head_out) not in (1, 2):
            raise ValueError(f"head_out must be 1 or 2, got {head_out}")
        if int(c_in) < int(num_slots):
            raise ValueError(f"c_in({c_in}) < num_slots({num_slots}) — bio 열이 음수가 된다")
        self.c_in = int(c_in)
        self.num_slots = int(num_slots)
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.head_out = int(head_out)
        #: bio 열의 시작 index. canonical 8ch 에서 6 (= [NDCI, MCI_norm] 위치, 정정 A-9).
        self.bio_start = self.num_slots

        # --- SPS Step 2 대응 ---
        self.proj = nn.Conv2d(self.c_in, self.c1, kernel_size=1, bias=True)
        self.c_abs = nn.Parameter(torch.zeros(self.num_slots, self.c1))
        self.n1 = nn.GroupNorm(num_groups=1, num_channels=self.c1)  # LayerNorm 등가
        # --- SPS Step 3 대응 (T3: 이식 금지 대상) ---
        self.dw = nn.Conv2d(self.c1, self.c1, 3, padding=1, groups=self.c1, bias=False)
        self.bn = build_norm("bn", self.c1)
        # --- SPS Step 5 (patch_embed) 대응부 ---
        self.mix = nn.Conv2d(self.c1, self.c2, kernel_size=1, bias=True)
        self.bn2 = build_norm("bn", self.c2)
        # --- ChlHead 최종 1×1 대응부 ---
        self.head = nn.Conv2d(self.c2, self.head_out, kernel_size=1, bias=True)
        self.act = build_act("gelu")

        init_spec_mlp(self)

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"x must be (B,{self.c_in},H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.c_in:
            raise ValueError(f"x channel {x.shape[1]} != c_in {self.c_in}")
        if m.dim() != 2 or m.shape[1] != self.num_slots:
            raise ValueError(f"m must be (B,{self.num_slots}), got {tuple(m.shape)}")
        if m.shape[0] != x.shape[0]:
            raise ValueError(f"batch mismatch: x {x.shape[0]} vs m {m.shape[0]}")

        b = x.shape[0]
        m = m.to(x.dtype)
        z = self.proj(x)
        # 결측 slot 의 기대 기여를 학습으로 보충. init 0 이라 초기엔 순수 zero-fill 과 동일.
        z = z + ((1.0 - m) @ self.c_abs).view(b, self.c1, 1, 1)
        z = self.act(self.n1(z))
        z = z + self.act(self.bn(self.dw(z)))  # residual (SPS Step 4 와 동일)
        z = self.act(self.bn2(self.mix(z)))
        out = self.head(z)  # (B, head_out, H, W)
        return out.mean(dim=(2, 3))  # (B, head_out) — 칩 전체 평균


@torch.no_grad()
def init_spec_mlp(
    m: SpecMLP,
    *,
    bio_gain: float = 4.0,
    head_bias: float = HEAD_BIAS_DEFAULT,
) -> None:
    """01 §6.5 초기화 특칙 (06 §3.6 동결).

    * ``proj.weight`` Kaiming-uniform(fan_in = ``c_in``), **bio 열 ``[:, 6:8]`` 만 ×bio_gain**
    * ``proj.bias = 0``, ``C_abs = 0``, ``head.weight = 0``, ``head.bias = head_bias``
    * norm 은 weight=1 / bias=0

    ``head.weight = 0`` + ``head.bias = 1.5662`` 이므로 **첫 스텝의 예측은 입력과 무관하게
    정확히 전역 평균** ``softplus(1.5662) = 1.75589 = log1p(Chla) 평균`` 이다.

    Warning:
        ``bio_gain`` 의 근거는 raw 밴드가 O(1e-2) 스케일일 때만 성립한다
        (01 §2.6 R4′ ``k_sensor`` 확정 후 재산정 대상, [U-11] / 06 M-7).
    """
    nn.init.kaiming_uniform_(m.proj.weight, a=5 ** 0.5)  # nn.Conv2d 기본과 동일
    if m.proj.bias is not None:
        nn.init.zeros_(m.proj.bias)
    if m.bio_start < m.c_in:
        m.proj.weight[:, m.bio_start :] *= float(bio_gain)
    m.c_abs.zero_()

    nn.init.kaiming_uniform_(m.dw.weight, a=5 ** 0.5)  # T3 — 이식 대상이 아니다
    nn.init.kaiming_uniform_(m.mix.weight, a=5 ** 0.5)
    if m.mix.bias is not None:
        nn.init.zeros_(m.mix.bias)

    for norm in (m.n1, m.bn, m.bn2):
        if getattr(norm, "weight", None) is not None:
            nn.init.ones_(norm.weight)
        if getattr(norm, "bias", None) is not None:
            nn.init.zeros_(norm.bias)

    nn.init.zeros_(m.head.weight)
    nn.init.zeros_(m.head.bias)
    m.head.bias[0] = float(head_bias)  # head_out=2 면 log_var 채널은 0 (σ²=1) 에서 출발


def spec_loss(
    out: Tensor,
    y: Tensor,
    *,
    beta: float = 0.3,
    rank_weight: float = 0.2,
    rank_margin: float = 0.1,
    use_aleatoric: bool = False,
    u: float = 1.0,
) -> Dict[str, Tensor]:
    """S0-Spec 손실 — 01 §6.6 / 06 §3.6 (★X-11).

    ``y_hat = softplus(out[:, 0:1])`` (log1p 공간 비음수 보장) 위에서

    * ``use_aleatoric=False`` (기본): ``SmoothL1(y_hat, y; β) + rank_weight · L_rank``
    * ``use_aleatoric=True``: 같은 항의 SmoothL1 을 :func:`chl_reg_loss` 의
      Huber-NLL 로 대체한다 (``head_out=2`` 필요 — ``out[:,1]`` 이 ``log σ²``).

    Args:
        out: ``(B, head_out)`` SpecMLP 출력 (log1p 공간 raw).
        y: ``(B,)`` 또는 ``(B,1)`` **log1p 공간** 타깃 (★X-14 — 여기서 log1p 를 다시 걸지 않는다).
        beta: SmoothL1 의 δ. 기본 0.3 ≈ 0.36·σ(log1p) (X-11).
        rank_weight: 서열 보조손실 가중. 0 이면 끈다.
        rank_margin: ranking margin (01 §6.6 m=0.1).
        use_aleatoric: Huber-NLL 사용 여부.
        u: aleatoric 램프 ∈ [0,1] (:func:`bloomnet.losses.u_ramp`). ``use_aleatoric`` 일 때만 쓴다.

    Returns:
        ``{"loss", "reg", "rank", "y_hat"}``. ``y_hat`` 은 detach 되지 않은 예측이며
        평가기가 그대로 소비할 수 있다.

    Raises:
        ValueError: ``use_aleatoric=True`` 인데 ``out`` 이 2열이 아닐 때.
    """
    if out.dim() != 2:
        raise ValueError(f"out must be (B,head_out), got {tuple(out.shape)}")
    z = out[:, 0:1]
    y_t = y.reshape(-1, 1).to(z.dtype)
    if y_t.shape[0] != z.shape[0]:
        raise ValueError(f"batch mismatch: out {z.shape[0]} vs y {y_t.shape[0]}")
    y_hat = F.softplus(z)

    if use_aleatoric:
        if out.shape[1] < 2:
            raise ValueError("use_aleatoric=True 는 head_out=2 를 요구한다 (out[:,1] = log σ²)")
        log_var: Optional[Tensor] = out[:, 1:2]
        mask = torch.ones_like(y_t, dtype=torch.bool)
        reg, _ = chl_reg_loss(y_hat, log_var, y_t, mask, beta=beta, u=float(u))
    else:
        reg = F.smooth_l1_loss(y_hat, y_t, beta=beta)

    if rank_weight != 0.0:
        rank = margin_rank_loss(y_hat, y_t, margin=rank_margin)
    else:
        rank = y_hat.sum() * 0.0  # 규칙 N1 — 텐서 0.0 이 아니라 그래프에 연결된 0

    return {
        "loss": reg + float(rank_weight) * rank,
        "reg": reg,
        "rank": rank,
        "y_hat": y_hat,
    }
