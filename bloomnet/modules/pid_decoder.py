"""PID Trident Decoder — 04 §2.3 op-by-op (06 §3.3.7, 레벨 L3).

세 분기를 갖는다:

* **I** (context)  : ``PAPPM(F4)`` → ``lat3(F3)`` 가산 → ``IBlock3`` → ``ISep`` → bilinear ↑
* **P** (detail)   : ``lat2(F2)`` → ``PagFM_8`` → ``PBlock8`` → ``lat1(F1)`` 가산
  → ``PagFM_4`` → ``PSep``
* **D** (boundary) : ``dlat(F1)`` + ``diff3(c3)↑`` → ``DBlock`` + ``diff4(c4)↑``
  → ``d32`` / ``d128``

``LightBag(p4, c1, d128)`` 이 셋을 합쳐 ``F_dec`` 을 낸다.

params **1,382,272** | MAC **3.309 GMAC @512² / 13.235 GMAC @1024²** (정정 A-14).

구현상 주의 (앞 단계 인계 사항):

* ``PagFM_4`` 는 ``pag4(x=p4, y=c2, y_up=c1)`` 로 호출한다 — ``y=c1`` 을 넘기면 ``f_y`` 가
  H/4 에서 돌아 MAC 이 0.671 → 1.074 GMAC 로 늘어난다 (04 §4.1, 정정 B-16).
* ``d128 = dexp(relu(d))`` 의 ReLU 는 **비-inplace** 여야 한다. inplace 면 ``d32`` tap 이
  오염되어 EdgeHead 가 활성화 이후 값을 보게 된다 (04 §7.3).
* ``out_hw`` (정정 A-21/B-14) 는 **export 경로 전용**이다. 학습·CPU 테스트는 ``None`` 으로
  두어 ``size=`` 정본을 쓴다 — H=64(h=2)/H=128(h=4) 에서 PAPPM 의 pooled 가 전부 1×1 이
  되므로 고정 ``scale_factor`` 로는 반드시 터진다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch.nn.functional as F
from torch import Tensor, nn

from bloomnet.constants import BD_CH, CHANNELS, DEC_CH
from bloomnet.modules.common import ConvBN, init_encoder
from bloomnet.modules.decoder_blocks import PAPPM, BasicBlock, LightBag, PagFM, SepConvBNAct

__all__ = ["PIDDecoder"]

_HW = Tuple[int, int]
_DEC_NORM = "bn"  # 04 §1.2 — 디코더는 BatchNorm 고정 (TRT conv+BN fusion)


def _up(t: Tensor, f: int) -> Tensor:
    """``F.interpolate(scale_factor=f)``. 배율이 상수라 ONNX 에 ``Shape`` 이 생기지 않는다."""
    return F.interpolate(t, scale_factor=float(f), mode="bilinear", align_corners=False)


class PIDDecoder(nn.Module):
    """04 §2.3 의 배선을 그대로 구현한다.

    Args:
        in_channels: ``(C1,C2,C3,C4)`` = BMEF 출력 채널 (헌법 C-4 = 32/64/160/320).
        dec_ch: ``Cd`` = 128 (고정, 헌법 C-4).
        bd_ch: ``Cb`` = 32 (D 분기 채널).
        ppm_branch: PAPPM branch 채널 = 96.
        zero_init_residual: ``BasicBlock`` 의 ``BN2.weight = 0`` (저데이터 항등 출발).
        dot_scale: ``PagFM`` 내적 스케일. ``None`` → ``mid**-0.5`` = 1/8.

    Note:
        ``self.apply(init_encoder)`` 를 **모듈 전체에 다시 돌리면 안 된다** —
        ``BasicBlock.BN2.weight=0`` 과 ``PAPPM``/``PagFM`` 의 특칙이 전부 날아간다.
        여기서는 자체 초기화가 없는 bare ``ConvBN`` 6개에만 개별 적용한다.
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int, int] = CHANNELS,
        *,
        dec_ch: int = DEC_CH,
        bd_ch: int = BD_CH,
        ppm_branch: int = 96,
        zero_init_residual: bool = True,
        dot_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"in_channels 는 4원소: {tuple(in_channels)}")
        c1, c2, c3, c4 = (int(c) for c in in_channels)
        self.in_channels = (c1, c2, c3, c4)
        self.dec_ch = int(dec_ch)
        self.bd_ch = int(bd_ch)

        # ---- I 분기 -------------------------------------------------------
        self.pappm = PAPPM(c4, int(ppm_branch), self.dec_ch)
        self.lat3 = ConvBN(1, c3, self.dec_ch, norm=_DEC_NORM)
        self.iblock3 = BasicBlock(self.dec_ch, zero_init_residual=zero_init_residual)
        self.isep = SepConvBNAct(self.dec_ch, self.dec_ch)

        # ---- P 분기 -------------------------------------------------------
        self.lat2 = ConvBN(1, c2, self.dec_ch, norm=_DEC_NORM)
        self.pag8 = PagFM(self.dec_ch, dot_scale=dot_scale)
        self.pblock8 = BasicBlock(self.dec_ch, zero_init_residual=zero_init_residual)
        self.lat1 = ConvBN(1, c1, self.dec_ch, norm=_DEC_NORM)
        self.pag4 = PagFM(self.dec_ch, dot_scale=dot_scale)
        self.psep = SepConvBNAct(self.dec_ch, self.dec_ch)

        # ---- D 분기 -------------------------------------------------------
        self.dlat = ConvBN(1, c1, self.bd_ch, norm=_DEC_NORM)
        self.diff3 = ConvBN(3, self.dec_ch, self.bd_ch, norm=_DEC_NORM)
        self.dblock = BasicBlock(self.bd_ch, zero_init_residual=zero_init_residual)
        self.diff4 = ConvBN(1, self.dec_ch, self.bd_ch, norm=_DEC_NORM)
        self.dexp = ConvBN(1, self.bd_ch, self.dec_ch, norm=_DEC_NORM)

        # ---- fusion -------------------------------------------------------
        self.bag = LightBag(self.dec_ch, self.dec_ch)

        # bare ConvBN 만 초기화 (자체 특칙이 있는 블록은 건드리지 않는다)
        for mod in (self.lat3, self.lat2, self.lat1, self.dlat, self.diff3, self.diff4, self.dexp):
            mod.apply(init_encoder)

        # export 전용 정적 크기 (정정 A-21). None = 학습/테스트 정본 (size=)
        self._export_hw: Optional[_HW] = None

    # ------------------------------------------------------------------ export
    def set_export_hw(self, input_hw: Optional[Tuple[int, int]]) -> None:
        """``ExportWrapper`` 전용 — 입력 ``(H,W)`` 로부터 하위 모듈의 정적 ``out_hw`` 를 굽는다.

        ``None`` 을 주면 학습 정본(``size=x.shape[-2:]``)으로 되돌린다.
        ``PAPPM.up_mode`` 도 함께 전환한다 — 04 §9.3-B/§9.4 step 3 (Shape/Gather 0개).
        """
        if input_hw is None:
            self._export_hw = None
            self.pappm.up_mode = "size"
            return
        h, w = int(input_hw[0]), int(input_hw[1])
        if h % 32 or w % 32:
            raise ValueError(f"input_hw 는 32 의 배수여야 한다 (헌법 C-4): {(h, w)}")
        self._export_hw = (h, w)
        self.pappm.up_mode = "scale"

    def _hw(self, div: int) -> Optional[_HW]:
        if self._export_hw is None:
            return None
        return (self._export_hw[0] // div, self._export_hw[1] // div)

    # ----------------------------------------------------------------- forward
    def forward(
        self, F1: Tensor, F2: Tensor, F3: Tensor, F4: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            F1: (B,32,H/4,W/4) — ``F_fused^(1)``.
            F2: (B,64,H/8,W/8).
            F3: (B,160,H/16,W/16).
            F4: (B,320,H/32,W/32).

        Returns:
            ``(F_dec (B,128,H/4,W/4), d32 (B,32,H/4,W/4), p8 (B,128,H/8,W/8))``.
            ``d32`` 는 EdgeHead tap(활성화 **이전**), ``p8`` 은 AuxSegHead tap 이다.
        """
        # ---------------- I branch (context) ----------------
        c4 = self.pappm(F4, self._hw(32))  # (B,128,H/32)
        c3 = self.iblock3(self.lat3(F3) + _up(c4, 2))  # (B,128,H/16)
        c2 = self.isep(_up(c3, 2))  # (B,128,H/8)
        c1 = _up(c2, 2)  # (B,128,H/4)  ← LightBag i · PagFM_4 y_up 공유

        # ---------------- P branch (detail) -----------------
        p8 = self.lat2(F2)  # (B,128,H/8)
        p8 = self.pag8(p8, c2, self._hw(8))  # x,y 동일 해상도 → 보간 skip
        p8 = self.pblock8(p8)  # ← AuxSegHead tap
        p4 = _up(p8, 2) + self.lat1(F1)  # (B,128,H/4)
        p4 = self.pag4(p4, c2, self._hw(4), c1)  # ★ y 는 H/8 원본, y_up 은 c1 재사용
        p4 = self.psep(p4)

        # ---------------- D branch (boundary) ---------------
        d = self.dlat(F1)  # (B,32,H/4)
        d = d + _up(self.diff3(c3), 4)
        d = self.dblock(d)
        d = d + _up(self.diff4(c4), 8)
        d32 = d  # ← EdgeHead tap (활성화 이전)
        d128 = self.dexp(F.relu(d))  # ★ 비-inplace: d32 오염 금지

        # ---------------- Fusion ----------------------------
        f_dec = self.bag(p4, c1, d128)
        return f_dec, d32, p8
