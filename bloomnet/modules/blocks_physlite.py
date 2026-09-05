"""Phys path 블록 — 02 §5 / 06 §3.3.5 (레벨 L2).

`PhysLiteBlock` 1개 = inverted residual(expand 2) + SE::

    h = HSwish(BN(1x1(C→2C)(x)))
    h = HSwish(BN(DWk×k(2C)(h)))
    h = BN(1x1(2C→C)(h))                      # activation 없음
    h = h * Hardsigmoid(SE(h))                # SENet 배치: branch 끝, add 직전
    x = x + DropPath(γ * h)

동결 사항 / 함정
    * **SE gate 는 `Hardsigmoid` 가 필수**다 ([분석] D-PHY-01). [PPT-s36] 의
      `1x1(4→32) → dot product` 는 활성화가 없어 부호 반전 가능한 무한계 채널
      스케일링, 즉 발산 장치다. 2nd conv zero-init 으로 **초기 gate = 0.5**.
    * **SE 병목에 floor 8** — `C_se = max(se_min, C // se_ratio)` = 8/8/20/40.
      floor 없이 C=32 를 4차원으로 병목시키면 rank 제약이 심하고, floor 비용은
      stage1 에서 256 params 뿐이다 (02 §5.2).
    * **SE 위치는 branch 끝**(SENet 원본). DW 직후(2C)에 두면 C=320 에서
      `2·640·80 = 102,400` 으로 block 의 22% 를 먹는다.
    * **activation 은 GELU 가 아니라 Hardswish**. "경량 경로"에 가장 비싼
      activation 을 쓰는 것은 자기모순이고 TensorRT 에서 느리다 ([분석] D-PHY-07).
    * **DoLP 를 재주입하지 않는다** ([분석] D-PHY-06). DoLP 의 역할은 *증거*가 아니라
      *신뢰도*이고 그 자리는 BMEF 의 `r_rgb = 1 − g_pol` 이다. BioSpec 의 BioGate 와
      대칭이 아닌 것은 의도적이다 — 측정된 상관이 없는 양을 gate query 로 쓰지 않는다.
    * **`drop_path` 는 항상 0.0** 이다 (06 §3.3.5 ★). phys path 는 전체 5 block 이라
      stochastic depth 의 통계적 의미가 없다. 인자는 계약 유지를 위해 남긴다.
    * norm 은 :func:`bloomnet.modules.common.build_norm` 을 통해서만 만든다(정정 B-1).
      PhysLite 의 norm 폭은 2C·C 로 전부 8의 배수라 `gn8` 에서도 8 그룹이다
      (SE 의 `C_se=20` 은 norm 이 아니라 conv 이므로 무관).

초기화 계약: `model.apply(init_encoder)` 를 나중에 부르면 SE 의 zero-init 이 지워진다.
그 뒤 `m.reset_gate_()` 를 다시 호출할 것 (`SEGate.reset_gate_` 와 동일 규약).

레벨 L2 — L1(`modules/common.py`)만 import 한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from bloomnet.modules.common import DropPath, LayerScale, SEGate, build_act, build_norm

__all__ = ["PhysLiteBlock"]


class PhysLiteBlock(nn.Module):
    """열/편광 path 의 경량 residual block (02 §5.1).

    Args:
        dim: 채널 수 C.
        expand_ratio: 중간 확장비 e (=2). 02 §5.3 의 "왜 얕은가" 4근거 참조 —
            이 path 는 데이터가 0개라 capacity 를 늘릴 근거가 없다.
        dw_kernel: DW 커널 k. stage1–2 = 3, stage3–4 = 5.
        se_ratio: `C_se = max(se_min, C // se_ratio)`.
        se_min: `C_se` 하한(=8).
        layer_scale_init: γ 초기값 (per-channel).
        drop_path: 계약상 존재하지만 **항상 0.0** 으로 쓴다.
        norm: `bn|syncbn|gn8` — 06 동결 시그니처에 없는 keyword-only 추가분
            (정정 B-1 이 norm 직접 생성을 금지하므로 kind 를 받을 통로가 필요하다).

    params = `2eC² + 4eC + k²·eC + 2C + (2·C·C_se + C_se + C) + C`
    → 5,576 (C=32,k=3) / 19,336 (64,3) / 118,740 (160,5) / 455,080 (320,5).
    """

    def __init__(
        self,
        dim: int,
        *,
        expand_ratio: int = 2,
        dw_kernel: int = 3,
        se_ratio: int = 8,
        se_min: int = 8,
        layer_scale_init: float = 0.01,
        drop_path: float = 0.0,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        k = int(dw_kernel)
        if k % 2 == 0:
            raise ValueError(f"dw_kernel must be odd (shape 보존), got {dw_kernel}")
        self.dim = int(dim)
        self.hidden = int(dim) * int(expand_ratio)

        self.pw1 = nn.Conv2d(dim, self.hidden, 1, bias=False)
        self.norm1 = build_norm(norm, self.hidden)
        self.act1 = build_act("hswish")

        self.dw = nn.Conv2d(
            self.hidden, self.hidden, k, padding=k // 2, groups=self.hidden, bias=False
        )
        self.norm2 = build_norm(norm, self.hidden)
        self.act2 = build_act("hswish")

        self.pw2 = nn.Conv2d(self.hidden, dim, 1, bias=False)
        self.norm3 = build_norm(norm, dim)  # activation 없음

        self.se = SEGate(dim, reduction=se_ratio, se_min=se_min, act="relu", gate="hardsigmoid")
        self.ls = LayerScale(dim, layer_scale_init)
        self.dp = DropPath(drop_path)

    @torch.no_grad()
    def reset_gate_(self) -> None:
        """02 §1.5 특칙 재적용 — SE 2nd conv weight/bias = 0 (초기 gate 0.5)."""
        self.se.reset_gate_()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act1(self.norm1(self.pw1(x)))
        h = self.act2(self.norm2(self.dw(h)))
        h = self.norm3(self.pw2(h))
        h = self.se(h)
        return x + self.dp(self.ls(h))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, hidden={self.hidden}, k={self.dw.kernel_size[0]}"
