"""L1 손실 프리미티브 테스트 — 05 §2.1~2.3 / §2.5 / §2.6, 06 §3.5 (T19 선행분).

헌법 C-5.1/C-5.2: **CPU 전용**, B<=2, 64x64 이하 소형 텐서.
중점: (a) 손계산 일치, (b) ignore 처리, (c) 유효 픽셀 0 -> 0.0 반환 + backward NaN 없음,
      (d) OHEM 이 실제로 어려운 픽셀만 남기는지, (e) 완벽 예측에서 CE/Dice 최소.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # cwd 와 무관하게 bloomnet 패키지를 찾게 한다
    sys.path.insert(0, str(_ROOT))

from bloomnet.losses import (  # noqa: E402
    batch_soft_dice,
    chl_reg_loss,
    cwd_loss,
    margin_rank_loss,
    ohem_ce,
    plain_ce,
    u_ramp,
)

IGNORE = 255
DEV = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def _logits_from_probs(probs: list[list[float]]) -> torch.Tensor:
    """probs[b][pixel] = P(class 0). K=2, shape (B,2,1,W). softmax(log p)=p 를 이용."""
    b = len(probs)
    w = len(probs[0])
    out = torch.empty(b, 2, 1, w, dtype=torch.float64)
    for i, row in enumerate(probs):
        for j, p in enumerate(row):
            out[i, 0, 0, j] = math.log(p)
            out[i, 1, 0, j] = math.log(1.0 - p)
    return out.float()


def _tiny_seg_batch(b: int = 2, k: int = 4, h: int = 8, w: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, k, h, w, generator=g, requires_grad=True)
    target = torch.randint(0, k, (b, h, w), generator=g, dtype=torch.int64)
    return logits, target


def _assert_backward_finite(loss: torch.Tensor, *tensors: torch.Tensor) -> None:
    """backward 후 모든 입력 grad 가 존재하고 유한한지 확인 (05 §3.3 필수 목록 6)."""
    assert torch.isfinite(loss).all(), "loss 자체가 비유한하다"
    loss.backward()
    for t in tensors:
        assert t.grad is not None, "grad 가 None — graph 가 끊겼다 (규칙 N1 위반)"
        assert torch.isfinite(t.grad).all(), "grad 에 NaN/inf 가 있다"


# ═════════════════════════════════════════════════════════════════════════════
# ohem_ce
# ═════════════════════════════════════════════════════════════════════════════
def test_ohem_ce_all_ignore_returns_exact_zero_and_grad_is_finite():
    """유효 픽셀 0개 -> 0.0 (규칙 N1). tensor(0.0) 이 아니라 graph 가 살아 있어야 한다."""
    logits, _ = _tiny_seg_batch()
    target = torch.full((2, 8, 8), IGNORE, dtype=torch.int64)

    loss = ohem_ce(logits, target, ignore_index=IGNORE)
    assert loss.item() == 0.0
    assert loss.requires_grad, "graph 가 끊겼다 — DDP find_unused_parameters 오류 유발"
    _assert_backward_finite(loss, logits)
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_ohem_ce_keeps_only_hard_pixels_hand_computed():
    """thresh=0.7, min_kept=1 -> p_gt < 0.7 인 4개만 채택. 손계산 일치."""
    probs = [0.05, 0.15, 0.30, 0.60, 0.80, 0.90, 0.95, 0.99]
    logits = _logits_from_probs([probs])
    target = torch.zeros(1, 1, 8, dtype=torch.int64)

    # n_valid=8 -> min_kept = max(1, int(0.0625*8)) = 1 -> kth=0.05 -> thr=max(0.05,0.7)=0.7
    hard = [p for p in probs if p < 0.7]
    assert len(hard) == 4
    expected = sum(-math.log(p) for p in hard) / len(hard)

    got = ohem_ce(logits, target, thresh=0.7, keep_frac=0.0625)
    assert got.item() == pytest.approx(expected, abs=1e-5)

    # 전체 평균 CE 보다 반드시 커야 한다 (어려운 픽셀만 남겼으므로)
    mean_ce = sum(-math.log(p) for p in probs) / len(probs)
    assert got.item() > mean_ce


def test_ohem_ce_thresh_direction_higher_thresh_keeps_more():
    """05 §2.1.3: thresh 가 클수록 더 많은 픽셀을 유지 = 마이닝이 약해진다."""
    probs = [0.05, 0.15, 0.30, 0.60, 0.80, 0.90, 0.95, 0.99]
    logits = _logits_from_probs([probs])
    target = torch.zeros(1, 1, 8, dtype=torch.int64)

    loose = ohem_ce(logits, target, thresh=0.99).item()  # 거의 전부 채택
    tight = ohem_ce(logits, target, thresh=0.30).item()  # 어려운 것만
    assert tight > loose


def test_ohem_ce_thresh_one_reduces_to_mean_ce():
    """thresh=1.0 이면 (p_gt<1) 전 픽셀 채택 -> 유효 픽셀 평균 CE 와 동일."""
    logits, target = _tiny_seg_batch(seed=3)
    target[0, 0, :] = IGNORE  # ignore 도 섞는다

    got = ohem_ce(logits, target, ignore_index=IGNORE, thresh=1.0)
    ref = F.cross_entropy(logits, target, ignore_index=IGNORE, reduction="mean")
    assert got.item() == pytest.approx(ref.item(), abs=1e-6)


def test_ohem_ce_ignore_pixels_excluded_from_sort_and_mean():
    """ignore 픽셀이 정렬·분모에 들어가면 임계값이 오염된다 (05 §2.1.2)."""
    # 유효 4개 + ignore 4개. ignore 픽셀은 p_gt(class0)=4.5e-5 로 '가장 어려워' 보인다.
    probs = [0.05, 0.15, 0.30, 0.60, 4.5e-5, 4.5e-5, 4.5e-5, 4.5e-5]
    logits = _logits_from_probs([probs])
    target = torch.zeros(1, 1, 8, dtype=torch.int64)
    target[0, 0, 4:] = IGNORE

    # thresh=0.0 -> thr = kth 만으로 결정. n_valid=4, min_kept=int(0.75*4)=3
    #  -> p_sorted=[0.05,0.15,0.30,0.60], kth=0.30, keep = p<0.30 -> {0.05,0.15}
    expected = (-math.log(0.05) - math.log(0.15)) / 2.0
    got = ohem_ce(logits, target, ignore_index=IGNORE, thresh=0.0, keep_frac=0.75)
    assert got.item() == pytest.approx(expected, abs=1e-5)


def test_ohem_ce_tie_fallback_keeps_min_kept():
    """전부 동점이라 keep 이 비는 경계 사례 -> 하위 min_kept 강제 채택 (규칙 N3)."""
    probs = [0.5] * 8
    logits = _logits_from_probs([probs])
    target = torch.zeros(1, 1, 8, dtype=torch.int64)

    # kth=0.5, thr=max(0.5, 0.3)=0.5 -> keep = p<0.5 -> 공집합 -> fallback
    got = ohem_ce(logits, target, thresh=0.3, keep_frac=0.0625)
    assert got.item() == pytest.approx(-math.log(0.5), abs=1e-6)
    assert torch.isfinite(got)


def test_ohem_ce_upsamples_logits_not_downsamples_labels():
    """임의 해상도 로짓 허용. 라벨은 내리지 않는다 (04 §8.1)."""
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 64, 64), dtype=torch.int64)

    got = ohem_ce(logits, target)
    up = F.interpolate(logits, size=(64, 64), mode="bilinear", align_corners=False)
    ref = ohem_ce(up, target)
    assert got.item() == pytest.approx(ref.item(), abs=1e-6)
    _assert_backward_finite(got, logits)


def test_ohem_ce_class_weight_scales_per_pixel_ce():
    logits, target = _tiny_seg_batch(k=3, seed=7)
    w = torch.tensor([1.0, 2.0, 4.0])

    got = ohem_ce(logits, target, thresh=1.0, class_weight=w)
    ce = F.cross_entropy(logits, target, weight=w, reduction="none")
    assert got.item() == pytest.approx(ce.mean().item(), abs=1e-6)


def test_ohem_ce_minimal_at_perfect_prediction():
    _, target = _tiny_seg_batch(seed=11)
    perfect = F.one_hot(target, 4).permute(0, 3, 1, 2).float() * 30.0
    wrong = perfect.roll(1, dims=1)
    assert ohem_ce(perfect, target).item() < 1e-6
    assert ohem_ce(wrong, target).item() > 10.0


# ═════════════════════════════════════════════════════════════════════════════
# batch_soft_dice
# ═════════════════════════════════════════════════════════════════════════════
def test_dice_all_ignore_returns_exact_zero_and_grad_is_finite():
    logits, _ = _tiny_seg_batch()
    target = torch.full((2, 8, 8), IGNORE, dtype=torch.int64)

    loss = batch_soft_dice(logits, target, num_classes=4, ignore_index=IGNORE)
    assert loss.item() == 0.0
    assert loss.requires_grad
    _assert_backward_finite(loss, logits)


def test_dice_hand_computed_value():
    """K=2, 4픽셀. eps=1.0 을 raw 카운트에 더하는 규약을 손계산으로 고정."""
    probs = [0.9, 0.8, 0.3, 0.4]  # P(class 0)
    logits = _logits_from_probs([probs])
    target = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.int64)  # (1,1,4)

    inter0, denom0 = 0.9 + 0.8, (0.9 + 0.8 + 0.3 + 0.4) + 2.0
    inter1, denom1 = 0.7 + 0.6, (0.1 + 0.2 + 0.7 + 0.6) + 2.0
    d0 = (2 * inter0 + 1.0) / (denom0 + 1.0)
    d1 = (2 * inter1 + 1.0) / (denom1 + 1.0)
    expected = 1.0 - (d0 + d1) / 2.0
    assert expected == pytest.approx(0.2012882, abs=1e-6)

    got = batch_soft_dice(logits, target, num_classes=2)
    assert got.item() == pytest.approx(expected, abs=1e-6)


def test_dice_absent_gt_classes_are_excluded_from_mean():
    """GT 부재 클래스를 0 으로 세면 배치 구성에 따라 손실이 요동친다 (05 §2.2.2-2)."""
    probs = [0.9, 0.8, 0.3, 0.4]
    two = _logits_from_probs([probs])  # (1,2,1,4)
    four = torch.cat([two, torch.full((1, 2, 1, 4), -40.0)], dim=1)  # class 2,3 확률 ~0
    target = torch.tensor([[[0, 0, 1, 1]]], dtype=torch.int64)

    got2 = batch_soft_dice(two, target, num_classes=2).item()
    got4 = batch_soft_dice(four, target, num_classes=4).item()
    assert got4 == pytest.approx(got2, abs=1e-5)

    # 부재 클래스를 포함했다면 dice_2=dice_3=1 이 되어 손실이 절반으로 떨어졌을 것
    naive = 1.0 - ((1.0 - got2) * 2 + 1.0 + 1.0) / 4.0
    assert abs(naive - got4) > 0.05


def test_dice_scatter_onehot_matches_f_one_hot_reference():
    """정정 A-32(b): F.one_hot(int64) -> scatter_ float. 값이 달라지면 안 된다."""
    logits, target = _tiny_seg_batch(k=5, seed=13)
    target[0, 0, :] = IGNORE

    def _ref(lg, tg, k, ignore=IGNORE, eps=1.0):
        prob = F.softmax(lg.float(), dim=1)
        valid = tg != ignore
        t = tg.clone()
        t[~valid] = 0
        onehot = F.one_hot(t, k).permute(0, 3, 1, 2).float()
        m = valid.unsqueeze(1).float()
        prob, onehot = prob * m, onehot * m
        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dims)
        denom = prob.sum(dims) + onehot.sum(dims)
        present = onehot.sum(dims) > 0
        return 1.0 - ((2 * inter + eps) / (denom + eps))[present].mean()

    got = batch_soft_dice(logits, target, num_classes=5, ignore_index=IGNORE)
    assert got.item() == pytest.approx(_ref(logits, target, 5).item(), abs=1e-7)


def test_dice_minimal_at_perfect_prediction():
    _, target = _tiny_seg_batch(seed=17)
    perfect = F.one_hot(target, 4).permute(0, 3, 1, 2).float() * 30.0
    assert batch_soft_dice(perfect, target, num_classes=4).item() < 1e-5

    wrong = perfect.roll(1, dims=1)
    bad = batch_soft_dice(wrong, target, num_classes=4).item()
    assert 0.9 < bad <= 1.0


def test_dice_single_class_batch_is_finite():
    """05 §3.3 필수 목록 4: 배치에 단 1개 클래스만 존재해도 유한."""
    logits = torch.randn(2, 4, 8, 8, requires_grad=True)
    target = torch.zeros(2, 8, 8, dtype=torch.int64)

    loss = batch_soft_dice(logits, target, num_classes=4)
    assert torch.isfinite(loss)
    _assert_backward_finite(loss, logits)


def test_dice_eps_on_raw_counts_not_ratios():
    """eps 를 raw 카운트에 더한다. 카운트 대비 eps 가 커지면 희소 클래스의 Dice 가 1 에
    붙어 손실(=기울기)이 사라진다 — 05 §2.2.2-3 이 경고한 실패 모드."""
    # 클래스 1 이 64픽셀 중 1픽셀뿐인데 모델은 전부 class 0 이라고 예측한다
    logits = torch.full((1, 2, 1, 64), 0.0)
    logits[0, 0] = 10.0
    target = torch.zeros(1, 1, 64, dtype=torch.int64)
    target[0, 0, 0] = 1

    # 손계산: p0=sigmoid(10)=0.99995460, p1=4.5398e-5
    #   dice0=(2*63*p0+1)/(64*p0+63+1)=0.9921653,  dice1=(2*p1+1)/(64*p1+1+1)=0.4993200
    p0 = 1.0 / (1.0 + math.exp(-10.0))
    p1 = 1.0 - p0
    d0 = (2 * 63 * p0 + 1.0) / (64 * p0 + 63 + 1.0)
    d1 = (2 * p1 + 1.0) / (64 * p1 + 1 + 1.0)
    expected = 1.0 - (d0 + d1) / 2.0
    assert expected == pytest.approx(0.2542573, abs=1e-6)

    loss = batch_soft_dice(logits, target, num_classes=2).item()
    assert loss == pytest.approx(expected, abs=1e-6)

    # eps 가 카운트를 지배하면(비율에 더한 것과 같은 효과) 손실이 사실상 사라진다
    swamped = batch_soft_dice(logits, target, num_classes=2, eps=1000.0).item()
    assert swamped < 0.01
    assert loss > 100 * swamped


# ═════════════════════════════════════════════════════════════════════════════
# plain_ce
# ═════════════════════════════════════════════════════════════════════════════
def test_plain_ce_matches_torch_and_handles_ignore():
    logits, target = _tiny_seg_batch(seed=19)
    target[1, :, 0] = IGNORE

    got = plain_ce(logits, target, ignore_index=IGNORE)
    ref = F.cross_entropy(logits, target, ignore_index=IGNORE, reduction="mean")
    assert got.item() == pytest.approx(ref.item(), abs=1e-6)


def test_plain_ce_all_ignore_returns_zero_where_torch_returns_nan():
    logits, _ = _tiny_seg_batch()
    target = torch.full((2, 8, 8), IGNORE, dtype=torch.int64)

    torch_ref = F.cross_entropy(logits, target, ignore_index=IGNORE, reduction="mean")
    assert torch.isnan(torch_ref), "torch 기본 동작이 바뀌었다면 이 가드의 근거를 재확인할 것"

    loss = plain_ce(logits, target, ignore_index=IGNORE)
    assert loss.item() == 0.0
    _assert_backward_finite(loss, logits)


def test_plain_ce_upsamples_logits():
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 64, 64), dtype=torch.int64)
    up = F.interpolate(logits, size=(64, 64), mode="bilinear", align_corners=False)
    assert plain_ce(logits, target).item() == pytest.approx(
        F.cross_entropy(up, target).item(), abs=1e-6
    )


# ═════════════════════════════════════════════════════════════════════════════
# chl_reg_loss
# ═════════════════════════════════════════════════════════════════════════════
def test_chl_reg_u0_reduces_exactly_to_smooth_l1():
    """T19: u=0 에서 SmoothL1(δ=1) 로 정확히 환원 (atol 1e-6)."""
    g = torch.Generator().manual_seed(23)
    chl = torch.rand(2, 1, 8, 8, generator=g, requires_grad=True)
    y = torch.rand(2, 1, 8, 8, generator=g) * 4.0  # log1p 공간 (★X-14)
    m = torch.ones(2, 1, 8, 8, dtype=torch.bool)

    loss, stats = chl_reg_loss(chl, None, y, m, beta=1.0, u=0.0)
    ref = F.smooth_l1_loss(chl, y, beta=1.0, reduction="mean")
    assert loss.item() == pytest.approx(ref.item(), abs=1e-6)
    assert stats["n_valid"] == float(m.numel())

    # log_var 가 있어도 u=0 이면 동일해야 한다
    lv = torch.randn(2, 1, 8, 8, generator=g)
    loss2, _ = chl_reg_loss(chl, lv, y, m, beta=1.0, u=0.0)
    assert loss2.item() == pytest.approx(ref.item(), abs=1e-6)


def test_chl_reg_does_not_apply_log1p_again():
    """★X-14: y_chl 은 이미 log1p 공간. 내부에서 log1p 를 또 적용하면 안 된다."""
    chl = torch.full((1, 1, 2, 2), 1.5)
    y = torch.full((1, 1, 2, 2), 1.5)
    m = torch.ones(1, 1, 2, 2, dtype=torch.bool)

    loss, _ = chl_reg_loss(chl, None, y, m)
    assert loss.item() == pytest.approx(0.0, abs=1e-12), "잔차가 0 이어야 한다"

    # 만약 내부에서 log1p(y) 를 했다면 잔차 = 1.5 - log1p(1.5) = 0.5837 이 되었을 것
    wrong_r = 1.5 - math.log1p(1.5)
    assert abs(loss.item() - (abs(wrong_r) - 0.5 if wrong_r >= 1 else 0.5 * wrong_r**2)) > 0.1


def test_chl_reg_no_labels_returns_zero_and_grad_is_finite():
    """오늘(2026-07-31)의 정상 경로: Chl-a 픽셀 라벨이 존재하지 않는다."""
    chl = torch.rand(2, 1, 8, 8, requires_grad=True)

    loss, stats = chl_reg_loss(chl, None, None, None)
    assert loss.item() == 0.0
    assert stats["n_valid"] == 0.0
    assert loss.requires_grad
    _assert_backward_finite(loss, chl)
    assert torch.equal(chl.grad, torch.zeros_like(chl))


def test_chl_reg_all_masked_out_returns_zero_and_grad_is_finite():
    chl = torch.rand(2, 1, 8, 8, requires_grad=True)
    lv = torch.randn(2, 1, 8, 8, requires_grad=True)
    y = torch.rand(2, 1, 8, 8) * 4.0
    m = torch.zeros(2, 1, 8, 8, dtype=torch.bool)

    loss, stats = chl_reg_loss(chl, lv, y, m, u=1.0)
    assert loss.item() == 0.0
    assert stats["n_valid"] == 0.0
    loss.backward()
    assert torch.isfinite(chl.grad).all() and torch.equal(chl.grad, torch.zeros_like(chl))
    assert lv.grad is None or torch.isfinite(lv.grad).all()


def test_chl_reg_denominator_is_n_valid_not_numel():
    """D-LOSS-05: numel 로 나누면 유효 픽셀 비율에 따라 실효 λ_reg 가 배치마다 변한다."""
    chl = torch.zeros(1, 1, 1, 8)
    y = torch.full((1, 1, 1, 8), 0.5)  # 잔차 -0.5 -> SmoothL1 = 0.125
    m_full = torch.ones(1, 1, 1, 8, dtype=torch.bool)
    m_half = m_full.clone()
    m_half[..., 4:] = False

    l_full, s_full = chl_reg_loss(chl, None, y, m_full)
    l_half, s_half = chl_reg_loss(chl, None, y, m_half)
    assert l_full.item() == pytest.approx(0.125, abs=1e-7)
    assert l_half.item() == pytest.approx(0.125, abs=1e-7)
    assert (s_full["n_valid"], s_half["n_valid"]) == (8.0, 4.0)


def test_chl_reg_aleatoric_formula_hand_computed():
    """per_px = exp(-u*s)*SmoothL1 + 0.5*u*s (05 §2.5.2)."""
    chl = torch.full((1, 1, 1, 4), 1.0)
    y = torch.full((1, 1, 1, 4), 0.5)  # r=0.5 -> h = 0.5*0.25 = 0.125
    lv = torch.full((1, 1, 1, 4), 1.0)  # s=1
    m = torch.ones(1, 1, 1, 4, dtype=torch.bool)

    loss, _ = chl_reg_loss(chl, lv, y, m, beta=1.0, u=1.0)
    expected = math.exp(-1.0) * 0.125 + 0.5 * 1.0
    assert loss.item() == pytest.approx(expected, abs=1e-6)

    # u=0.5 (램프 중간)
    loss_h, _ = chl_reg_loss(chl, lv, y, m, beta=1.0, u=0.5)
    assert loss_h.item() == pytest.approx(math.exp(-0.5) * 0.125 + 0.25, abs=1e-6)


def test_chl_reg_sigma_star_equals_abs_residual():
    """05 §2.5.2-3: 2차 영역에서 ∂L/∂s=0 의 해는 σ*=|r| (즉 s* = 2·log|r|)."""
    r = 0.3
    chl = torch.full((1, 1, 1, 1), r)
    y = torch.zeros(1, 1, 1, 1)
    m = torch.ones(1, 1, 1, 1, dtype=torch.bool)

    grid = torch.arange(-4.0, 2.0, 0.001)
    vals = torch.stack(
        [chl_reg_loss(chl, torch.full((1, 1, 1, 1), float(s)), y, m, u=1.0)[0] for s in grid]
    )
    s_star = float(grid[int(vals.argmin())])
    assert s_star == pytest.approx(2.0 * math.log(r), abs=5e-3)


def test_chl_reg_clamps_extreme_log_var():
    """05 §3.3 필수 목록 5 / 규칙 N5: log_var=±1e4 -> clamp 후 유한."""
    chl = torch.full((1, 1, 1, 4), 1.0, requires_grad=True)
    y = torch.full((1, 1, 1, 4), 0.5)
    m = torch.ones(1, 1, 1, 4, dtype=torch.bool)

    for extreme, clipped in ((1e4, 7.0), (-1e4, -7.0)):
        lv = torch.full((1, 1, 1, 4), extreme)
        loss, _ = chl_reg_loss(chl, lv, y, m, u=1.0, clamp=(-7.0, 7.0))
        ref, _ = chl_reg_loss(chl, torch.full((1, 1, 1, 4), clipped), y, m, u=1.0)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(ref.item(), abs=1e-6)


def test_chl_reg_clamp_default_matches_unc_head_range():
    """★X-26 / 정정 A-20: 기본 clamp 는 UncHead 의 S_MIN/S_MAX = (-7,+7) 과 같아야 한다."""
    import inspect

    default = inspect.signature(chl_reg_loss).parameters["clamp"].default
    assert tuple(default) == (-7.0, 7.0)


def test_chl_reg_debug_assert_catches_raw_mgm3_labels():
    """T19 / ★X-14 하드가드: 원단위 mg/m³ 를 넣으면 AssertionError (정정 A-33: debug 전용)."""
    chl = torch.zeros(1, 1, 1, 4)
    y_raw = torch.full((1, 1, 1, 4), 50.0)  # mg/m³ 원단위 -> log1p 공간이면 불가능한 값
    m = torch.ones(1, 1, 1, 4, dtype=torch.bool)

    with pytest.raises(AssertionError, match="log1p"):
        chl_reg_loss(chl, None, y_raw, m, debug_assert=True)

    # 기본(debug_assert=False)에서는 동기화 비용 때문에 검사하지 않는다
    loss, _ = chl_reg_loss(chl, None, y_raw, m, debug_assert=False)
    assert torch.isfinite(loss)

    # 정상 범위(log1p(90.34)=4.51)는 통과
    y_ok = torch.full((1, 1, 1, 4), math.log1p(90.34))
    chl_reg_loss(chl, None, y_ok, m, debug_assert=True)


def test_chl_reg_survives_nonfinite_labels_outside_the_mask():
    """무효 픽셀의 라벨이 NaN/inf 여도 forward/backward 가 오염되면 안 된다."""
    chl = torch.rand(1, 1, 1, 8, requires_grad=True)
    y = torch.rand(1, 1, 1, 8) * 3.0
    y[..., 4:] = float("nan")
    m = torch.zeros(1, 1, 1, 8, dtype=torch.bool)
    m[..., :4] = True

    loss, stats = chl_reg_loss(chl, None, y, m)
    assert stats["n_valid"] == 4.0
    assert torch.isfinite(loss)

    ref, _ = chl_reg_loss(chl, None, torch.nan_to_num(y), m)
    assert loss.item() == pytest.approx(ref.item(), abs=1e-7)
    _assert_backward_finite(loss, chl)


def test_chl_reg_upsamples_prediction_to_label_resolution():
    chl = torch.rand(1, 1, 4, 4, requires_grad=True)
    lv = torch.randn(1, 1, 4, 4, requires_grad=True)
    y = torch.rand(1, 1, 16, 16) * 3.0
    m = torch.ones(1, 1, 16, 16, dtype=torch.bool)

    loss, stats = chl_reg_loss(chl, lv, y, m, u=1.0)
    assert stats["n_valid"] == 256.0
    _assert_backward_finite(loss, chl, lv)


# ═════════════════════════════════════════════════════════════════════════════
# margin_rank_loss
# ═════════════════════════════════════════════════════════════════════════════
def test_margin_rank_zero_for_correctly_ordered_predictions():
    """모든 쌍이 margin 이상 벌어져 있으면 0. i==j 쌍이 섞이면 margin>0 이 되므로,
    이 테스트는 쌍의 i != j 보장도 함께 검증한다."""
    torch.manual_seed(0)
    y = torch.arange(8, dtype=torch.float32)
    pred = y.clone().requires_grad_(True)

    loss = margin_rank_loss(pred, y, margin=0.1, n_pairs=256)
    assert loss.item() == 0.0


def test_margin_rank_penalizes_inverted_ranking():
    torch.manual_seed(0)
    y = torch.arange(8, dtype=torch.float32)
    pred = (-y).clone().requires_grad_(True)

    loss = margin_rank_loss(pred, y, margin=0.1, n_pairs=256)
    assert loss.item() > 0.1  # margin + |i-j| >= 1.1
    _assert_backward_finite(loss, pred)


def test_margin_rank_ties_cost_exactly_margin():
    """sign(y_i-y_j)=0 -> max(0, margin) = margin (torch MarginRankingLoss 와 동일 규약)."""
    torch.manual_seed(0)
    y = torch.zeros(8)
    pred = torch.randn(8, requires_grad=True)
    loss = margin_rank_loss(pred, y, margin=0.1, n_pairs=64)
    assert loss.item() == pytest.approx(0.1, abs=1e-6)


def test_margin_rank_batch_of_one_returns_zero():
    pred = torch.rand(1, 1, requires_grad=True)
    y = torch.rand(1, 1)
    loss = margin_rank_loss(pred, y)
    assert loss.item() == 0.0
    _assert_backward_finite(loss, pred)


def test_margin_rank_accepts_b1_shape():
    torch.manual_seed(0)
    y = torch.arange(6, dtype=torch.float32).view(6, 1)
    pred = y.clone().requires_grad_(True)
    assert margin_rank_loss(pred, y, n_pairs=64).item() == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# u_ramp
# ═════════════════════════════════════════════════════════════════════════════
def test_u_ramp_boundaries_match_spec_example():
    """05 §5.2.3: total=80 -> E_warm=24, E_ramp=16."""
    total = 80
    assert u_ramp(0, total) == 0.0
    assert u_ramp(23, total) == 0.0
    assert u_ramp(24, total) == 0.0
    assert u_ramp(25, total) == pytest.approx(1 / 16)
    assert u_ramp(39, total) == pytest.approx(15 / 16)
    assert u_ramp(40, total) == 1.0
    assert u_ramp(79, total) == 1.0


def test_u_ramp_is_monotone_and_bounded():
    prev = -1.0
    for e in range(0, 120):
        v = u_ramp(e, 100)
        assert 0.0 <= v <= 1.0
        assert v >= prev
        prev = v


def test_u_ramp_degenerate_total_epochs():
    for total in (0, 1, 2):
        v = u_ramp(0, total)
        assert 0.0 <= v <= 1.0 and math.isfinite(v)
    assert u_ramp(10, 0) == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# cwd_loss
# ═════════════════════════════════════════════════════════════════════════════
def test_cwd_zero_when_student_matches_teacher():
    g = torch.Generator().manual_seed(29)
    t = torch.randn(2, 6, 8, 8, generator=g)
    s = t.clone().requires_grad_(True)
    loss = cwd_loss(s, t, T=4.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert loss.item() >= -1e-6  # KL >= 0


def test_cwd_matches_naive_reference():
    g = torch.Generator().manual_seed(31)
    s = torch.randn(2, 3, 4, 4, generator=g, requires_grad=True)
    t = torch.randn(2, 3, 4, 4, generator=g)
    T = 4.0

    # 05 §2.6.1 의 정의를 그대로 옮긴 참조 구현 (정규화 = T²/(C·N), N = 배치)
    b, c = s.shape[0], s.shape[1]
    acc = 0.0
    for bi in range(b):
        for ci in range(c):
            pt = F.softmax(t[bi, ci].reshape(-1) / T, dim=0)
            ps = F.softmax(s[bi, ci].reshape(-1) / T, dim=0)
            acc = acc + (pt * (pt.log() - ps.log())).sum()
    ref = acc * (T * T) / (c * b)

    got = cwd_loss(s, t, T=T)
    assert got.item() == pytest.approx(ref.detach().item(), abs=1e-5)
    _assert_backward_finite(got, s)


def test_cwd_teacher_receives_no_gradient():
    """05 §2.6.3 teacher 위생 규칙: teacher 로는 gradient 가 흐르면 안 된다."""
    s = torch.randn(1, 4, 4, 4, requires_grad=True)
    t = torch.randn(1, 4, 4, 4, requires_grad=True)
    cwd_loss(s, t).backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    assert t.grad is None


def test_cwd_upsamples_student_to_teacher_grid():
    s = torch.randn(1, 4, 4, 4, requires_grad=True)
    t = torch.randn(1, 4, 16, 16)
    got = cwd_loss(s, t)
    up = F.interpolate(s, size=(16, 16), mode="bilinear", align_corners=False)
    assert got.item() == pytest.approx(cwd_loss(up, t).item(), abs=1e-6)
    _assert_backward_finite(got, s)


def test_cwd_extreme_magnitudes_stay_finite():
    s = torch.randn(1, 4, 8, 8, requires_grad=True) * 100.0
    s.retain_grad()
    t = torch.randn(1, 4, 8, 8) * 100.0
    loss = cwd_loss(s, t)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(s.grad).all()


# ═════════════════════════════════════════════════════════════════════════════
# 통합: 헌법 C-5.1 소형 텐서에서 전 손실 backward 시 grad NaN 없음
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("degenerate", [False, True])
def test_all_primitives_backward_through_a_module_without_nan(degenerate: bool):
    """B=2, 64x64, CPU. degenerate=True 는 '라벨 전무' 상태를 재현한다."""
    torch.manual_seed(37)
    net = torch.nn.Conv2d(3, 6, 3, padding=1)
    x = torch.randn(2, 3, 64, 64)
    out = net(x)

    seg_logits = out[:, :4]
    chl = F.softplus(out[:, 4:5])
    log_var = out[:, 5:6].clamp(-7.0, 7.0)

    if degenerate:
        y = torch.full((2, 64, 64), IGNORE, dtype=torch.int64)
        m_chl = torch.zeros(2, 1, 64, 64, dtype=torch.bool)
    else:
        y = torch.randint(0, 4, (2, 64, 64), dtype=torch.int64)
        m_chl = torch.ones(2, 1, 64, 64, dtype=torch.bool)
    y_chl = torch.rand(2, 1, 64, 64) * 4.0

    total = (
        ohem_ce(seg_logits, y, ignore_index=IGNORE)
        + 0.4 * batch_soft_dice(seg_logits, y, num_classes=4, ignore_index=IGNORE)
        + 0.2 * plain_ce(seg_logits, y, ignore_index=IGNORE)
        + 0.5 * chl_reg_loss(chl, log_var, y_chl, m_chl, u=1.0)[0]
        + 0.0 * cwd_loss(seg_logits, seg_logits.detach() + 0.1)
    )
    assert torch.isfinite(total)
    if degenerate:
        assert total.item() == 0.0

    total.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"{name}: grad 가 None"
        assert torch.isfinite(p.grad).all(), f"{name}: grad 에 NaN/inf"
