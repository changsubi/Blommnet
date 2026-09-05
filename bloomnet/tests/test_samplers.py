"""`data/samplers.py` — RFS / GroupBatchSampler (05 §4.3 / 06 §3.2.6).

CPU 전용, 소형 배열. GPU 미사용.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bloomnet.data.samplers import GroupBatchSampler, RepeatFactorSampler

N, K = 100, 3


def _presence(n_class2: int = 2) -> np.ndarray:
    p = np.zeros((N, K), dtype=bool)
    p[:, 0] = True                 # background: 전 이미지
    p[:50, 1] = True               # 흔한 클래스
    p[:n_class2, 2] = True         # 희소 클래스
    return p


# ─────────────────────────────────────────────────────────────────────────────
# RepeatFactorSampler
# ─────────────────────────────────────────────────────────────────────────────
def test_class_repeat_factor_formula():
    s = RepeatFactorSampler(_presence(), t=0.05, seed=1234)
    # r_c = max(1, sqrt(t / f_img_c)),  ignore 클래스는 1 로 고정
    assert s.class_repeat_factors[0] == pytest.approx(1.0)
    assert s.class_repeat_factors[1] == pytest.approx(1.0)          # f=0.5 -> sqrt(0.1)<1
    assert s.class_repeat_factors[2] == pytest.approx(math.sqrt(0.05 / 0.02))
    assert s.class_repeat_factors[2] == pytest.approx(1.5811388, abs=1e-6)


def test_image_repeat_factor_is_max_over_present_classes():
    s = RepeatFactorSampler(_presence(), t=0.05)
    assert s.repeat_factors[0] == pytest.approx(math.sqrt(2.5))
    assert s.repeat_factors[1] == pytest.approx(math.sqrt(2.5))
    assert s.repeat_factors[2] == pytest.approx(1.0)
    assert s.expected_epoch_size == pytest.approx(98 + 2 * math.sqrt(2.5))


def test_ignore_class_ids_forced_to_one():
    """ignore 목록의 클래스는 희소하더라도 반복계수를 만들지 않아야 한다."""
    p = np.zeros((10, 2), dtype=bool)
    p[:2, 0] = True                    # 합성: 클래스 0 을 일부러 희소하게 둔다
    p[:1, 1] = True
    with_ignore = RepeatFactorSampler(p, t=0.5, seed=0, ignore_class_ids=(0,))
    without = RepeatFactorSampler(p, t=0.5, seed=0, ignore_class_ids=())
    assert with_ignore.class_repeat_factors[0] == pytest.approx(1.0)
    assert without.class_repeat_factors[0] == pytest.approx(math.sqrt(0.5 / 0.2))
    assert without.class_repeat_factors[0] > 1.0
    # ignore 여부와 무관하게 다른 클래스의 계수는 같다
    assert with_ignore.class_repeat_factors[1] == pytest.approx(math.sqrt(0.5 / 0.1))


def test_epoch_length_and_index_range():
    s = RepeatFactorSampler(_presence(), t=0.05, seed=1234)
    idx = list(s)
    assert len(idx) == len(s)
    assert 100 <= len(idx) <= 102             # floor 98*1 + 2*(1 or 2)
    assert min(idx) >= 0 and max(idx) < N
    # 희소 클래스 이미지는 최소 1회 이상 등장한다
    assert idx.count(0) >= 1 and idx.count(1) >= 1


def test_deterministic_for_same_seed_and_epoch():
    a = RepeatFactorSampler(_presence(), seed=7)
    b = RepeatFactorSampler(_presence(), seed=7)
    assert list(a) == list(b)
    a.set_epoch(3)
    b.set_epoch(3)
    assert list(a) == list(b)
    assert a.epoch == 3


def test_set_epoch_reshuffles():
    s = RepeatFactorSampler(_presence(), seed=7)
    first = list(s)
    s.set_epoch(1)
    assert list(s) != first, "에폭이 바뀌어도 순서가 같다면 재현성 설계가 잘못됐다"
    s.set_epoch(0)
    assert list(s) == first


def test_stochastic_rounding_is_unbiased_over_epochs():
    s = RepeatFactorSampler(_presence(), t=0.05, seed=0)
    lens = []
    for e in range(40):
        s.set_epoch(e)
        lens.append(len(s))
    assert min(lens) >= 100 and max(lens) <= 102
    assert abs(float(np.mean(lens)) - s.expected_epoch_size) < 0.5


def test_rare_class_is_oversampled():
    """t=0.05 에서 희소 클래스 노출이 실제로 늘어야 한다 (05 §4.3 전략 1)."""
    s = RepeatFactorSampler(_presence(n_class2=2), t=0.05, seed=0)
    total_rare = 0
    for e in range(20):
        s.set_epoch(e)
        idx = list(s)
        total_rare += sum(1 for i in idx if i < 2)
    assert total_rare / 20 > 2.0, total_rare / 20


def test_rfs_validation():
    with pytest.raises(ValueError):
        RepeatFactorSampler(np.zeros((5,), dtype=bool))
    with pytest.raises(ValueError):
        RepeatFactorSampler(np.zeros((0, 3), dtype=bool))
    with pytest.raises(ValueError):
        RepeatFactorSampler(_presence(), t=-0.1)


def test_zero_frequency_class_does_not_explode():
    p = np.zeros((10, 3), dtype=bool)
    p[:, 0] = True                     # 클래스 1,2 는 한 번도 등장하지 않는다
    s = RepeatFactorSampler(p, t=0.05)
    assert np.isfinite(s.class_repeat_factors).all()
    assert s.class_repeat_factors.tolist() == [1.0, 1.0, 1.0]
    assert len(s) == 10


# ─────────────────────────────────────────────────────────────────────────────
# GroupBatchSampler
# ─────────────────────────────────────────────────────────────────────────────
def test_group_batches_never_mix_groups():
    groups = ["a"] * 5 + ["b"] * 4 + ["c"] * 2
    s = GroupBatchSampler(groups, batch_size=2, drop_last=True, seed=0)
    batches = list(s)
    assert len(batches) == len(s) == s.expected_num_batches == 2 + 2 + 1
    for b in batches:
        assert len({groups[i] for i in b}) == 1, b
        assert len(b) == 2
    flat = [i for b in batches for i in b]
    assert len(set(flat)) == len(flat), "인덱스 중복"


def test_group_drop_last_false_keeps_remainder():
    groups = ["a"] * 5 + ["b"] * 4
    s = GroupBatchSampler(groups, batch_size=2, drop_last=False, seed=0)
    batches = list(s)
    assert len(batches) == s.expected_num_batches == 3 + 2
    assert sorted(i for b in batches for i in b) == list(range(9))
    assert s.num_groups == 2


def test_group_deterministic_and_epoch_dependent():
    groups = [i // 3 for i in range(30)]
    a = GroupBatchSampler(groups, batch_size=3, seed=11)
    b = GroupBatchSampler(groups, batch_size=3, seed=11)
    assert list(a) == list(b)
    a.set_epoch(1)
    assert list(a) != list(b)
    a.set_epoch(0)
    assert list(a) == list(b)


def test_group_no_shuffle_is_first_seen_order():
    groups = ["z", "z", "y", "y"]
    s = GroupBatchSampler(groups, batch_size=2, shuffle=False)
    assert list(s) == [[0, 1], [2, 3]]


def test_group_validation():
    with pytest.raises(ValueError):
        GroupBatchSampler(["a"], batch_size=0)
    with pytest.raises(ValueError):
        GroupBatchSampler([], batch_size=2)
