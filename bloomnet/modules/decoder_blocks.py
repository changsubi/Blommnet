"""PID Trident 디코더 기본 블록 — 04 §3~§6 / 06 §3.3.7 (레벨 L2).

정의 클래스: :class:`BasicBlock`, :class:`SepConvBNAct`, :class:`PagFM`,
:class:`LightBag`, :class:`PAPPM`.  조립(`PIDDecoder`)은 `modules/pid_decoder.py`(L3) 소관이다.

동결 사항
    * **norm 은 BatchNorm2d 뿐이다** (04 §1.2). GN/LN 은 TensorRT 에서 conv+BN fusion 을
      불가능하게 하므로 디코더 본체에서 금지된다. 그래도 ``nn.BatchNorm2d`` 를 직접 부르지
      않고 :func:`bloomnet.modules.common.build_norm` 을 ``kind="bn"`` 으로 경유한다
      (02 §1.2 / 정정 B-1 의 단일 진입점 규약).
    * **디코더 본체 activation = ReLU(inplace=True)** (04 §1.3). GELU 는 헤드 전용.
    * **업샘플의 정본은 ``size=`` 다** (정정 A-21 / B-14). :class:`PAPPM` 의 pooled branch
      업샘플 배율은 입력 해상도에 의존해서 고정 ``scale_factor`` 로 표현할 수 없다 —
      H=64 (h=2) 에서 pooled 가 전부 1×1 이 되어 필요 배율이 전부 2 가 되는데, 헌법 C-5.1 이
      바로 그 크기를 필수 테스트로 요구한다. 고정 (2,4,8) 을 쓰면
      ``RuntimeError: The size of tensor a (4) must match the size of tensor b (2)`` 로 터진다.
      ``out_hw`` 에 **파이썬 int 리터럴**을 주입하는 경로(ExportWrapper)만 정적 상수가 된다.
    * :class:`PagFM` 의 ``f_y`` 는 반드시 **y 의 원해상도**에서 계산한 뒤 보간한다
      (먼저 올리면 MAC 4배, 04 §4.1). ``y_up`` 인자로 호출부가 이미 만든 업샘플을 재사용한다
      (정정 B-16).

MAC 정본 (정정 A-14, @1024² / @512²)
    PAPPM 0.4864 / 0.1216 · PagFM_8 0.2684 / 0.0671 · PagFM_4 0.6711 / 0.1678 ·
    LightBag 2.1475 / 0.5369 · BasicBlock(128)@H/8 4.8318 / 1.2080.

배포 정밀도 정책 (04 §9.3, `deploy/trt_policy.py` 소관)
    * :class:`PagFM` 는 **INT8 엔진에서 fp16 고정**(``FP16_LOCKED``) 대상이다 — 채널 내적의
      동적 범위가 커서 스케일 추정이 실패하기 쉽다. FP16 자체는 안전하다:
      ``dot_scale = 1/8`` 로 합의 표준편차가 ≈ 1 이라 실사용 범위가 ±40 이내다(§9.3-H).
    * ``fp32_forced`` 는 ``bmef.fusion`` 과 ``encoder.*.litemla.attention`` 두 곳이며
      (정정 A-22 / B-6, LiteMLA 는 ``out = q @ kv`` 가 먼저 overflow 한다),
      **디코더 블록에는 fp32 강제 대상이 없다** — 전부 conv/BN 이다.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import CHANNELS, DEC_CH
from bloomnet.modules.common import ConvBN, build_norm, init_encoder

__all__ = ["BasicBlock", "SepConvBNAct", "PagFM", "LightBag", "PAPPM"]

_HW = Tuple[int, int]

# 04 §1.2: 디코더 본체는 BatchNorm 고정. 이 상수를 바꾸는 것은 헌법 개정 사항이다.
_DEC_NORM = "bn"


def _as_hw(hw: Sequence[int]) -> _HW:
    """``torch.Size`` / 튜플 → 파이썬 int 2-튜플.

    ``out_hw`` 로 넘어온 값을 그대로 ``size=`` 에 실어야 ONNX 에서 상수로 접힌다.
    """
    return (int(hw[0]), int(hw[1]))


def _resize(t: Tensor, hw: _HW, *, up_mode: str = "size") -> Tensor:
    """``t`` 를 ``hw`` 로 bilinear 보간한다. 이미 같은 크기면 **보간 자체를 건너뛴다**.

    Args:
        t: (B,C,h,w).
        hw: 목표 (H,W). 파이썬 int 튜플.
        up_mode: ``"size"`` (정본) 또는 ``"scale"`` (export 전용, 배율이 정확한 정수일 때만).

    Note:
        ``size=(h,w)`` 와 ``scale_factor=k`` 는 배율이 정확한 정수일 때 bit-identical 하다
        (04 §4.1 CPU 실측). 따라서 export 시 정적 치환이 수치를 바꾸지 않는다.
    """
    src = (int(t.shape[-2]), int(t.shape[-1]))
    if src == hw:
        return t
    if up_mode == "scale":
        fh, fw = hw[0] // src[0], hw[1] // src[1]
        if fh * src[0] != hw[0] or fw * src[1] != hw[1]:
            raise RuntimeError(
                f"up_mode='scale' 은 정확한 정수 배율에서만 쓸 수 있다: {src} -> {hw}. "
                "학습·CPU 테스트 경로는 up_mode='size' 를 쓴다 (정정 A-21/B-14)."
            )
        return F.interpolate(t, scale_factor=(fh, fw), mode="bilinear", align_corners=False)
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=False)


class BasicBlock(nn.Module):
    """PIDNet BasicBlock (expansion 1, stride 1, downsample 없음) — 04 §3.2.

    ``out = ReLU(x + BN2(Conv3(ReLU(BN1(Conv3(x))))))``.
    params ``2·(9C² + 2C)``: C=128 → 295,424 / C=32 → 18,560.

    Args:
        c: 입출력 채널.
        zero_init_residual: ``BN2.weight = 0``. 저데이터(헌법 C-1)에서 블록이 항등 근처에서
            출발하게 한다. PIDNet 원본에는 없는 편차이므로 끌 수 있게 노출한다.
    """

    def __init__(self, c: int, *, zero_init_residual: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False)
        self.bn1 = build_norm(_DEC_NORM, c)
        self.conv2 = nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False)
        self.bn2 = build_norm(_DEC_NORM, c)
        self.relu = nn.ReLU(inplace=True)
        self.apply(init_encoder)
        if zero_init_residual:  # ★ init_encoder(weight=1) 뒤에 덮어쓴다
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: Tensor) -> Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(x + y)


class SepConvBNAct(nn.Module):
    """depthwise-separable refine — 04 §3.3.

    ``ReLU(BN_pw(PW1x1(ReLU(BN_dw(DW3x3(x))))))``.
    params ``9·cin + 2·cin + cin·cout + 2·cout``: 128→128 = 18,048.
    full ``Conv3x3(128→128)`` 대비 MAC 8.4배 절감(147,456 → 17,536 MAC/px).
    """

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.dw = ConvBN(3, cin, cin, groups=cin, norm=_DEC_NORM)
        self.pw = ConvBN(1, cin, cout, norm=_DEC_NORM)
        self.act = nn.ReLU(inplace=True)
        self.apply(init_encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.pw(self.act(self.dw(x))))


class PagFM(nn.Module):
    """Pixel-attention-guided Fusion — 04 §4 (PIDNet 공식형).

    ``out = (1−σ)·x + σ·y↑``,  ``σ = sigmoid(⟨f_x(x), f_y(y)↑⟩ · dot_scale)``.
    x = P 분기(고해상도), y = I 분기(저해상도). params 16,640.

    내적이 크면 두 분기가 같은 대상을 보고 있다는 뜻이므로 context 를 주입하고(σ→1),
    어긋나면 detail 을 유지한다(σ→0).

    Args:
        C: 두 입력의 채널 수.
        mid: 사영 채널.
        dot_scale: ``None`` 이면 ``mid**-0.5`` (= 1/8). PIDNet 원본은 1.0 이지만 BN 직후
            64채널 내적의 표준편차가 ≈ 8 이라 σ 가 초기부터 포화해 gradient 가 죽는다
            (04 §4.2). ``dot_scale=1.0`` 은 PIDNet 정확 재현 — ablation 항목.
    """

    def __init__(self, C: int = DEC_CH, mid: int = 64, dot_scale: Optional[float] = None) -> None:
        super().__init__()
        self.f_x = ConvBN(1, C, mid, norm=_DEC_NORM)
        self.f_y = ConvBN(1, C, mid, norm=_DEC_NORM)
        self.dot_scale = float(mid**-0.5) if dot_scale is None else float(dot_scale)
        self.apply(init_encoder)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        out_hw: Optional[_HW] = None,
        y_up: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (B,C,hx,wx) P 분기.
            y: (B,C,hy,wy) I 분기 (``hy <= hx``).
            out_hw: ``None`` 이면 ``x.shape[-2:]`` (학습·CPU 테스트 정본). 파이썬 int 튜플을
                주면 그 정적 상수로 보간한다 → ONNX 에 ``Shape``/``Gather`` 가 생기지 않는다
                (정정 A-21 / B-14).
            y_up: 호출부가 이미 만들어 둔 ``y`` 의 ``out_hw`` 업샘플. 주면 재계산하지 않는다
                (정정 B-16 — @1024² B=1 fp16 기준 16.8 MB 임시 텐서 1회 절감).

        Returns:
            (B,C,hx,wx).
        """
        hw = _as_hw(out_hw) if out_hw is not None else _as_hw(x.shape[-2:])
        # 1) I 를 **저해상도에서** 사영한 뒤 올린다 (반대로 하면 MAC 4배, 04 §4.1)
        q = _resize(self.f_y(y), hw)
        k = self.f_x(x)
        sig = torch.sigmoid((k * q).sum(dim=1, keepdim=True) * self.dot_scale)
        if y_up is None:
            y_up = _resize(y, hw)
        return (1.0 - sig) * x + sig * y_up


class LightBag(nn.Module):
    """Boundary-attention-guided Fusion (PIDNet Light_Bag) — 04 §5.

    ``σ = sigmoid(d)`` 는 **128채널 elementwise 게이트**이며 1채널 edge logit 이 아니다.
    ``out = conv_p(p + (1−σ)·i) + conv_i(i + σ·p)``.

    전개: σ=1(경계) → ``conv_p(p) + conv_i(i+p)``, σ=0(내부) → ``conv_p(p+i) + conv_i(i)``.
    즉 게이트는 어느 분기를 버릴지가 아니라 **cross-term 을 어느 projection 에 실을지**를 정한다.
    params 33,280 | 2.1475 GMAC @1024².
    """

    def __init__(self, C: int = DEC_CH, C_out: int = DEC_CH) -> None:
        super().__init__()
        self.conv_p = ConvBN(1, C, C_out, norm=_DEC_NORM)
        self.conv_i = ConvBN(1, C, C_out, norm=_DEC_NORM)
        self.apply(init_encoder)

    def forward(self, p: Tensor, i: Tensor, d: Tensor) -> Tensor:
        """
        Args:
            p: (B,C,H/4,W/4) P 분기.
            i: (B,C,H/4,W/4) I 분기 (= ``c1``).
            d: (B,C,H/4,W/4) D 분기 **feature** (= ``d128``). 1채널 edge logit 이 아니다.
        """
        sig = torch.sigmoid(d)
        p_add = self.conv_p(p + (1.0 - sig) * i)
        i_add = self.conv_i(i + sig * p)
        return p_add + i_add


class PAPPM(nn.Module):
    """Parallel Aggregation PPM — 04 §6. BloomNet 편차 2건 포함.

    ``PAPPM(320, 96, 128)`` 을 ``F4`` (H/32) 에 둔다. params 590,144 | 0.4864 GMAC @1024².

    편차 1 — **BN+ReLU 를 5 branch 앞에서 1회만** 수행한다. PIDNet 원형은 branch 마다
        ``BN(C_in)→ReLU→Conv1x1`` 인데, 그러면 global branch 의 BN 이 ``(B,320,1,1)`` 에 걸려
        **B=1 학습 모드에서 ``ValueError``** 가 난다(04 §6.2 CPU 실측, 06 R-9).
        공유 pre-BN 은 ``scale0``/``shortcut`` 에 대해 원형과 수학적으로 동일하고,
        pooled branch 만 "pool→BN→ReLU" 가 "BN→ReLU→pool" 로 바뀐다(PSPNet 계열 표준 순서).
    편차 2 — ``count_include_pad=False``. ``k=17,p=8`` 을 h=16 맵에 걸면 윈도우 대부분이
        padding 이라 True 면 값이 크게 감쇠하는데, 편차 1 로 BN 이 앞으로 갔으므로 그 감쇠를
        보정해 줄 BN 이 뒤에 없다. TRT ``average_count_excludes_padding`` 기본값과도 일치한다.

    Args:
        up_mode: ``"size"`` 가 **정본**. ``"scale"`` 은 ExportWrapper 가 정적 shape 로
            트레이스할 때만 켠다 — 배율이 정확한 정수가 아니면 ``RuntimeError``.
    """

    def __init__(
        self,
        C_in: int = CHANNELS[3],
        br: int = 96,
        C_out: int = DEC_CH,
        up_mode: str = "size",
    ) -> None:
        super().__init__()
        if up_mode not in ("size", "scale"):
            raise ValueError(f"up_mode must be 'size' or 'scale', got {up_mode!r}")
        self.up_mode = up_mode
        self.pre = nn.Sequential(build_norm(_DEC_NORM, C_in), nn.ReLU(inplace=True))
        self.s0 = nn.Conv2d(C_in, br, 1, bias=False)
        self.s1 = nn.Conv2d(C_in, br, 1, bias=False)
        self.s2 = nn.Conv2d(C_in, br, 1, bias=False)
        self.s3 = nn.Conv2d(C_in, br, 1, bias=False)
        self.s4 = nn.Conv2d(C_in, br, 1, bias=False)  # global branch
        self.pool1 = nn.AvgPool2d(5, stride=2, padding=2, count_include_pad=False)
        self.pool2 = nn.AvgPool2d(9, stride=4, padding=4, count_include_pad=False)
        self.pool3 = nn.AvgPool2d(17, stride=8, padding=8, count_include_pad=False)
        self.proc = nn.Sequential(
            build_norm(_DEC_NORM, 4 * br),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * br, 4 * br, 3, stride=1, padding=1, groups=4, bias=False),
        )
        self.comp = nn.Sequential(
            build_norm(_DEC_NORM, 5 * br),
            nn.ReLU(inplace=True),
            nn.Conv2d(5 * br, C_out, 1, bias=False),
        )
        self.short = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.apply(init_encoder)

    def forward(self, x: Tensor, out_hw: Optional[_HW] = None) -> Tensor:
        """
        Args:
            x: (B,C_in,h,w), h = H/32.
            out_hw: ``None`` 이면 ``x.shape[-2:]``. export 경로에서만 파이썬 int 튜플 주입.

        Returns:
            (B,C_out,h,w).
        """
        hw = _as_hw(out_hw) if out_hw is not None else _as_hw(x.shape[-2:])
        xr = self.pre(x)
        x0 = self.s0(xr)
        # ★ AdaptiveAvgPool2d(1) 금지 (04 §9.3-A) — ReduceMean 으로 나가야 한다
        g = xr.mean(dim=(2, 3), keepdim=True)
        s1 = _resize(self.s1(self.pool1(xr)), hw, up_mode=self.up_mode) + x0
        s2 = _resize(self.s2(self.pool2(xr)), hw, up_mode=self.up_mode) + x0
        s3 = _resize(self.s3(self.pool3(xr)), hw, up_mode=self.up_mode) + x0
        s4 = _resize(self.s4(g), hw, up_mode=self.up_mode) + x0
        so = self.proc(torch.cat([s1, s2, s3, s4], dim=1))
        return self.comp(torch.cat([x0, so], dim=1)) + self.short(xr)
