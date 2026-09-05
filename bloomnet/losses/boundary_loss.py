"""경계 손실 — 05 §2.4.2 / §2.4.3, API 동결 06 §3.5 (레벨 L2).

두 함수 모두 **PIDNet 공식 구현의 의미론**을 따른다.

``boundary_bce`` 는 `pos_weight` 방식 BCE 가 **아니다**. PIDNet 의 ``weighted_bce`` 는
픽셀 가중치 ``w_pos = N_neg/N_all``, ``w_neg = N_pos/N_all`` 를 곱한 뒤 ``N_all`` 로
나누므로 가중치의 평균이 ``2·p·(1−p) ≈ 0.0603`` (p = 0.03112, 05 §1.3 / 정정 B-20)
까지 내려간다. 이 형태에 ``λ_bd = 20`` 을 곱해야 seg 항의 1/3 수준(0.836)이 된다.
`pos_weight` 방식(≈ 1/p 배)을 쓰면 **약 1,000 배 과대**해진다.

``bas_loss`` 는 "D 분기가 경계라고 믿는 픽셀에 대해서만 seg CE 를 다시 건다".
학습 초기에는 ``sigmoid(edge) ≈ 0.5 < τ=0.8`` 이라 전 픽셀이 ignore 가 되는데,
**PIDNet 원본은 여기서 빈 텐서의 ``.mean()`` → NaN 을 낸다.** 우리는
``ohem_ce`` 의 규칙 N1(``n_valid == 0 → logits.sum() * 0.0``)로 이 경로를 차단한다.
이것은 우연히 피하는 것이 아니라 **실제로 발생하는 버그**다 (05 §2.4.3).

레벨 L2 — L1 이하만 import 한다 (``losses/seg.py`` = L1, ``constants`` = L−1 예외).
경계 **타깃 생성**은 여기서 하지 않는다. 단일 구현은 ``data/boundary.py::
make_boundary_target`` (★X-07) 이며 dataset 또는 criterion(L3) 이 호출한다.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import IGNORE_INDEX
from bloomnet.losses.seg import ohem_ce

__all__ = ["boundary_bce", "bas_loss"]


def _to_bd_hw(x: Tensor, hw: Tuple[int, int]) -> Tensor:
    """경계 로짓을 타깃 격자로 올린다. 이미 같으면 no-op."""
    if tuple(x.shape[-2:]) != tuple(hw):
        x = F.interpolate(
            x, size=(int(hw[0]), int(hw[1])), mode="bilinear", align_corners=False
        )
    return x


def boundary_bce(edge_logits: Tensor, bd_gt: Tensor, bd_valid: Tensor) -> Tensor:
    """Class-balanced weighted BCE (PIDNet ``weighted_bce``) — 05 §2.4.2.

    ``L = (1/N_all) · Σ_{p ∈ valid} w_p · BCEwithLogits(logit_p, tgt_p)``
    with ``w_p = N_neg/N_all`` (양성) / ``N_pos/N_all`` (음성), ``N_all = |valid|``.

    구조적으로 **0-나눗셈이 불가능**하다. 배치에 경계가 하나도 없으면
    ``w_pos = 1, w_neg = 0`` 이 되어 손실이 정확히 0 이다 (규칙 N4).
    유효 픽셀이 0개면 분모를 1 로 clamp 해 ``0/1 = 0`` 을 낸다 (규칙 N1) —
    이때도 반환값은 ``edge_logits`` 에 연결된 텐서라 autograd graph 가 끊기지 않는다.

    구현 주: 05 의사코드는 ``reshape(-1)[v]`` boolean-mask 인덱싱과
    ``if int(v.sum()) == 0`` 을 쓰지만, 여기서는 **가변 크기 할당과 GPU→CPU 동기화가
    없는 등가식**으로 구현했다 (정정 A-33 이 같은 이유로 금지한 패턴이다).
    ``F.binary_cross_entropy_with_logits(..., weight=w, reduction="mean")`` 는
    ``sum(w·bce)/numel`` 이므로 valid 부분집합 위에서 두 식은 동일하다.
    ``test_criterion.py::test_boundary_bce_matches_pidnet_reference`` 가 05 의사코드를
    그대로 옮긴 참조 구현과의 일치를 고정한다.

    Args:
        edge_logits: ``(B,1,h,w)`` raw 로짓. 임의 해상도(타깃 격자로 업샘플).
        bd_gt: ``(B,1,H,W)`` {0,1} 경계 타깃.
        bd_valid: ``(B,1,H,W)`` 감독 유효 마스크. bool/float 모두 허용 (★X-16).

    Returns:
        스칼라 텐서.
    """
    e = _to_bd_hw(edge_logits, bd_gt.shape[-2:]).float()
    tgt = bd_gt.float()
    v = bd_valid.to(torch.bool).to(tgt.dtype)

    n_all = v.sum()
    n_pos = (tgt * v).sum()
    n_neg = n_all - n_pos
    denom = n_all.clamp_min(1.0)  # 규칙 N1/N4 — 유효 0개여도 0-나눗셈 없음

    w_pos = n_neg / denom
    w_neg = n_pos / denom
    w = torch.where(tgt > 0.5, w_pos, w_neg) * v

    bce = F.binary_cross_entropy_with_logits(e, tgt, reduction="none")
    return (w * bce).sum() / denom


def bas_loss(
    seg_logits: Tensor,
    y_seg: Tensor,
    edge_logits: Tensor,
    *,
    tau: float = 0.8,
    ignore_index: int = IGNORE_INDEX,
    **ohem_kw: object,
) -> Tensor:
    """Boundary-Aware Semantic loss (PIDNet ``loss_sb``) — 05 §2.4.3.

    ``bd_label = where(sigmoid(edge) > tau, y_seg, ignore)`` 에 ``ohem_ce`` 를 건다.

    Args:
        seg_logits: ``(B,K,h,w)`` 임의 해상도. criterion 이 이미 라벨 해상도로
            업샘플해 넘기면 ``ohem_ce`` 내부 업샘플이 no-op 이 된다 (정정 A-32(a)).
        y_seg: ``(B,H,W)`` int64 (★X-13).
        edge_logits: ``(B,1,h,w)`` raw 로짓.
        tau: 경계 확률 임계. PIDNet 0.8.
        ignore_index: 무효 라벨 값.
        **ohem_kw: ``ohem_ce`` 로 그대로 전달 (``thresh``/``keep_frac``/``class_weight``).

    Returns:
        스칼라. 임계를 넘는 픽셀이 없으면 ``ohem_ce`` 가 0.0 을 반환한다(규칙 N1/N2).
    """
    e = _to_bd_hw(edge_logits, y_seg.shape[-2:])
    bd_prob = torch.sigmoid(e[:, 0].float())  # (B,H,W)
    bd_label = torch.where(bd_prob > tau, y_seg, torch.full_like(y_seg, ignore_index))
    return ohem_ce(  # type: ignore[arg-type]
        seg_logits, bd_label, ignore_index=ignore_index, **ohem_kw
    )
