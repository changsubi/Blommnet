"""경계(boundary) 타깃 생성 — 01 §7.5 / 06 §3.2.3 (★X-07 단일 구현).

pure-torch 형태학적 gradient 를 쓴다. **OpenCV Canny 와 동일하지 않다** (분석 §D.2.6).
`cv2` 의존을 만들지 않는 것이 X-07 동결의 핵심이며, 문서·보고서에 비동일성을 명시해야 한다.

레벨 L0. `bloomnet.constants` (L−1 예외, 정정 A-23) 외에는 `bloomnet.*` 를 import 하지 않는다.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import IGNORE_INDEX

__all__ = ["make_boundary_target", "boundary_pos_weight", "POS_RATIO_ANCHOR"]

# 01 [M4] radius=1 → H/4 max-pool 에서 N_neg/N_pos ≈ 31.1 (양성비 3.112 %).
# 고정 상수로 쓰지 않는다 — 온전성 검사 anchor 전용 (05 §1.3 / X-08 / M-11).
POS_RATIO_ANCHOR: float = 31.1
_POS_RATIO_SANE: Tuple[float, float] = (10.0, 100.0)


def make_boundary_target(
    mask: Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    radius: int = 1,
    out_stride: int = 4,
) -> Tuple[Tensor, Tensor]:
    """클래스 라벨에서 경계 타깃과 감독 유효 마스크를 만든다 (01 §7.5, 5단계 하드 순서).

    Args:
        mask: ``(H, W)`` 또는 ``(B, H, W)`` int64 클래스 id. ignore = ``ignore_index``.
        ignore_index: 무효 라벨 값.
        radius: 경계 두께 반경. 512² → 1(폭 3px), 1024² → 2(폭 5px) 로 물리 폭을 유지한다.
        out_stride: 다운샘플 배율. 헌법 C-4 의 디코더 작업 해상도 H/4 이므로 4.

    Returns:
        ``(edge, valid)``.
        ``(H,W)`` 입력 → ``edge (1, H/os, W/os) float32 {0,1}``, ``valid`` 동 shape bool.
        ``(B,H,W)`` 입력 → ``edge (B,1,H/os,W/os)``, ``valid`` 동 shape bool.
    """
    if mask.dim() == 2:
        batched = False
        m4 = mask[None, None]
    elif mask.dim() == 3:
        batched = True
        m4 = mask[:, None]
    else:
        raise ValueError(f"mask must be (H,W) or (B,H,W), got {tuple(mask.shape)}")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if out_stride < 1:
        raise ValueError(f"out_stride must be >= 1, got {out_stride}")

    h, w = m4.shape[-2:]
    if h % out_stride or w % out_stride:
        # 조용한 잘림 금지: max_pool2d 는 나머지를 버리므로 타깃이 라벨과 어긋난다.
        raise ValueError(f"(H,W)=({h},{w}) must be divisible by out_stride={out_stride}")

    m = m4.to(torch.float32)
    invalid = (m4 == ignore_index)

    # 1) 1픽셀 형태학적 gradient (클래스 무관 경계).
    #    ignore(255)는 값이 가장 크므로 경계를 유발한다 → 3) 에서 제거한다.
    mx = F.max_pool2d(m, 3, stride=1, padding=1)
    mn = -F.max_pool2d(-m, 3, stride=1, padding=1)
    edge = (mx != mn).to(torch.float32)

    # 2) 두께 확장 (dilate, 폭 = 2*radius+1)
    if radius > 0:
        k = 2 * radius + 1
        edge = (F.max_pool2d(edge, k, stride=1, padding=radius) > 0).to(torch.float32)

    # 3) ignore 인접 경계 제거 — ignore 로부터 (radius+1) 이내는 감독하지 않는다.
    ki = 2 * (radius + 1) + 1
    near_ignore = F.max_pool2d(invalid.to(torch.float32), ki, stride=1, padding=radius + 1) > 0
    valid = ~near_ignore
    edge = edge * valid.to(torch.float32)

    # 4) 영상 테두리 폭 (radius+1) 무효화 (PIDNet edge_pad 대응)
    b = radius + 1
    valid = valid.clone()
    valid[..., :b, :] = False
    valid[..., -b:, :] = False
    valid[..., :, :b] = False
    valid[..., :, -b:] = False
    edge = edge * valid.to(torch.float32)

    # 5) H/out_stride 다운샘플. edge 는 max-pool(얇은 구조 보존), valid 는 과반 규칙.
    edge_s = F.max_pool2d(edge, out_stride, stride=out_stride)
    valid_s = F.avg_pool2d(valid.to(torch.float32), out_stride, stride=out_stride) > 0.5

    if batched:
        return edge_s.contiguous(), valid_s.contiguous()
    return edge_s[0].contiguous(), valid_s[0].contiguous()


def boundary_pos_weight(edge: Tensor, valid: Tensor) -> Tuple[float, float]:
    """PIDNet 방식 배치별 동적 가중치 (분석 §D.2.5). ``w_pos + w_neg == 1``.

    Args:
        edge: ``{0,1}`` 경계 타깃. `valid` 와 broadcast 가능해야 한다.
        valid: 감독 유효 마스크 (bool).

    Returns:
        ``(w_pos, w_neg) = (N_neg/N, N_pos/N)``, ``N = N_pos + N_neg``.
        유효 화소가 0 이면 ``(1.0, 0.0)`` (합 계약 유지, 손실은 0 이 된다).
    """
    v = valid.to(torch.bool)
    pos = ((edge > 0.5) & v).sum()
    n_pos = int(pos.item())
    n_all = int(v.sum().item())
    n_neg = n_all - n_pos
    if n_all == 0:
        return 1.0, 0.0
    w_pos = n_neg / n_all
    w_neg = n_pos / n_all
    return w_pos, w_neg


def pos_ratio_is_sane(edge: Tensor, valid: Tensor) -> bool:
    """`N_neg/N_pos` 가 01 [M4] anchor(≈31.1) 주변의 상식 범위 안인지 (경고 로깅용).

    고정 상수를 손실에 쓰지 않는 이유는 S1 녹조 마스크에서 경계 밀도가 완전히 달라지기
    때문이다 (01 §7.5). 이 함수는 파이프라인 이상 탐지에만 쓴다.
    """
    v = valid.to(torch.bool)
    n_pos = int(((edge > 0.5) & v).sum().item())
    n_neg = int(v.sum().item()) - n_pos
    if n_pos == 0:
        return False
    lo, hi = _POS_RATIO_SANE
    return lo <= (n_neg / n_pos) <= hi
