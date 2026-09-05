"""T07 — `data/bundle.py` (01 §2 / 06 §3.2.1, 정정 A-11 / A-25).

CPU 전용, 소형 텐서. GPU 미사용.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict

import pytest
import torch

from bloomnet.constants import MODALITY_ORDER, PATHS, PHYS_SLOTS
from bloomnet.data.bundle import (
    MODALITY_KEYS,
    OPTIONAL_MODALITY_KEYS,
    PHYS_SLOT_TO_MODALITY_INDEX,
    assert_availability_contract,
    bloom_collate,
    derive_present,
    phys_modality_indices,
)

H = W = 8
CH = {"rgb": 3, "msi": 4, "bio": 2, "ir": 1, "pol": 3}


def make_sample(
    *,
    avail: Dict[str, float] | None = None,
    keys: tuple = ("rgb", "msi", "bio", "ir", "pol"),
    h: int = H,
    w: int = W,
    bio_kind: str = "mci",
    seed: int = 0,
) -> Dict[str, Any]:
    """A1~A6 를 만족하는 표준 샘플. `avail` 로 개별 modality 를 결측 처리한다."""
    g = torch.Generator().manual_seed(seed)
    av = {k: 1.0 for k in MODALITY_ORDER}
    av.update(avail or {})
    s: Dict[str, Any] = {}
    for k in keys:
        a = av[k]
        t = torch.rand(CH[k], h, w, generator=g) + 0.5 if a == 1.0 else torch.zeros(CH[k], h, w)
        s[k] = t
    s["avail"] = torch.tensor([av[k] for k in MODALITY_ORDER], dtype=torch.float32)
    s["y_seg"] = torch.randint(0, 3, (h, w), generator=g, dtype=torch.int64)
    s["y_edge"] = torch.zeros(1, h // 4, w // 4)
    s["y_edge_valid"] = torch.ones(1, h // 4, w // 4, dtype=torch.bool)
    s["y_chl"] = torch.zeros(1, h, w)
    s["y_chl_valid"] = torch.zeros(1, h, w, dtype=torch.bool)
    s["y_chl_scalar"] = torch.tensor(0.0)
    s["y_chl_scalar_valid"] = torch.tensor(False)
    s["meta"] = {
        "stem": "L01_11000_130_20230801_N01_00001", "scene": "01.한강",
        "group_key": ("01.한강", "11000", "20230801", "N01"), "split": "train",
        "site": None, "date": "20230801", "flight_line": "N01", "alt_m": 130.0,
        "gsd_m": 0.03, "sensor": "m3m", "band_ids": (1, 2, 3, 5),
        "phys_slot_ids": (0, 1, 2, 3), "band_centers_nm": {}, "bio_kind": bio_kind,
        "chl_space": "log1p", "aug": {},
    }
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 스키마 상수
# ─────────────────────────────────────────────────────────────────────────────
def test_schema_constants():
    assert MODALITY_KEYS == MODALITY_ORDER == ("rgb", "msi", "bio", "ir", "pol")
    # A5: 샘플 수준 결측이 허용되는 key. rgb 는 절대 포함되지 않는다 (A2)
    assert OPTIONAL_MODALITY_KEYS == ("msi", "bio", "ir", "pol")
    assert "rgb" not in OPTIONAL_MODALITY_KEYS
    assert PATHS == ("rgb", "spec", "phys")


def test_phys_slot_to_modality_index():
    ir_i, pol_i = MODALITY_ORDER.index("ir"), MODALITY_ORDER.index("pol")
    assert PHYS_SLOTS == ("ir", "dolp", "aolp_sin", "aolp_cos")
    assert PHYS_SLOT_TO_MODALITY_INDEX == (ir_i, pol_i, pol_i, pol_i)
    assert phys_modality_indices((0, 1, 2, 3)) == (ir_i, pol_i)
    assert phys_modality_indices((0,)) == (ir_i,)
    assert phys_modality_indices((1, 2, 3)) == (pol_i,)


# ─────────────────────────────────────────────────────────────────────────────
# bloom_collate — C1 ~ C7
# ─────────────────────────────────────────────────────────────────────────────
def test_collate_shapes_and_keys():
    batch = bloom_collate([make_sample(seed=0), make_sample(seed=1)])
    assert batch["rgb"].shape == (2, 3, H, W)
    assert batch["msi"].shape == (2, 4, H, W)
    assert batch["bio"].shape == (2, 2, H, W)
    assert batch["ir"].shape == (2, 1, H, W)
    assert batch["pol"].shape == (2, 3, H, W)
    assert batch["avail"].shape == (2, 5)                    # C5
    assert batch["y_seg"].shape == (2, H, W)
    assert batch["y_edge"].shape == (2, 1, H // 4, W // 4)
    assert batch["y_chl"].shape == (2, 1, H, W)
    assert batch["y_chl_scalar"].shape == (2,)
    assert batch["y_chl_scalar_valid"].shape == (2,)


def test_collate_c1_spatial_mismatch_raises():
    with pytest.raises(RuntimeError, match="C1"):
        bloom_collate([make_sample(h=8, w=8), make_sample(h=8, w=16)])


def test_collate_c2_key_mismatch_raises():
    a = make_sample()
    b = make_sample(keys=("rgb", "msi", "bio", "ir"))
    with pytest.raises(RuntimeError, match="C2"):
        bloom_collate([a, b])


def test_collate_c4_meta_stays_list():
    batch = bloom_collate([make_sample(seed=0), make_sample(seed=1)])
    assert isinstance(batch["meta"], list) and len(batch["meta"]) == 2
    assert isinstance(batch["meta"][0], dict)
    assert batch["meta"][0]["chl_space"] == "log1p"


def test_collate_c6_all_missing():
    s0 = make_sample(avail={"msi": 0.0, "bio": 0.0}, seed=0)
    s1 = make_sample(avail={"msi": 0.0, "bio": 0.0}, seed=1)
    batch = bloom_collate([s0, s1])
    assert batch["all_missing"]["msi"] is True
    assert batch["all_missing"]["bio"] is True
    assert batch["all_missing"]["rgb"] is False
    mixed = bloom_collate([make_sample(avail={"ir": 0.0}, seed=0), make_sample(seed=1)])
    assert mixed["all_missing"]["ir"] is False


def test_collate_c7_no_dtype_promotion():
    batch = bloom_collate([make_sample(seed=0), make_sample(seed=1)])
    assert batch["y_seg"].dtype == torch.int64
    assert batch["y_edge_valid"].dtype == torch.bool
    assert batch["y_chl_valid"].dtype == torch.bool
    assert batch["y_chl_scalar_valid"].dtype == torch.bool
    assert batch["avail"].dtype == torch.float32
    assert batch["rgb"].dtype == torch.float32


def test_collate_c7_dtype_mismatch_raises():
    a = make_sample()
    b = make_sample()
    b["y_seg"] = b["y_seg"].to(torch.int32)
    with pytest.raises(RuntimeError, match="C7"):
        bloom_collate([a, b])


def test_collate_empty_raises():
    with pytest.raises(RuntimeError):
        bloom_collate([])


# ─────────────────────────────────────────────────────────────────────────────
# assert_availability_contract — A1 ~ A6
# ─────────────────────────────────────────────────────────────────────────────
def test_contract_positive():
    assert_availability_contract(make_sample())
    assert_availability_contract(make_sample(avail={"msi": 0.0, "bio": 0.0}))
    assert_availability_contract(
        make_sample(keys=("rgb",), avail={"msi": 0.0, "bio": 0.0, "ir": 0.0, "pol": 0.0}),
        active_modalities=("rgb",),
    )


def test_a1_zero_avail_requires_all_zero_tensor():
    s = make_sample()
    s["avail"][1] = 0.0                       # msi 결측 선언인데 텐서는 비영
    with pytest.raises(AssertionError, match="A1"):
        assert_availability_contract(s)


def test_a1_one_avail_requires_nonzero_tensor():
    s = make_sample()
    s["msi"] = torch.zeros_like(s["msi"])     # 가용 선언인데 텐서는 전부 0
    with pytest.raises(AssertionError, match="A1"):
        assert_availability_contract(s)


def test_a2_rgb_always_available():
    s = make_sample()
    s["avail"][0] = 0.0
    s["rgb"] = torch.zeros_like(s["rgb"])
    with pytest.raises(AssertionError, match="A2"):
        assert_availability_contract(s)


def test_a3_bio_requires_msi():
    s = make_sample(avail={"msi": 0.0})       # bio=1, msi=0  → 위반
    with pytest.raises(AssertionError, match="A3"):
        assert_availability_contract(s)


def test_a3_relaxed_for_rgb_proxy():
    """정정 A-11: rgb_proxy 경로는 msi 없이 bio 가용이 허용되는 유일한 경우."""
    s = make_sample(avail={"msi": 0.0}, bio_kind="rgb_proxy")
    assert_availability_contract(s, bio_source="rgb_proxy")
    # V17: meta["bio_kind"] 기록이 없으면 실패해야 한다
    s2 = make_sample(avail={"msi": 0.0}, bio_kind="mci")
    with pytest.raises(AssertionError, match="V17"):
        assert_availability_contract(s2, bio_source="rgb_proxy")


def test_a4_inactive_modality_key_must_be_absent():
    s = make_sample()
    with pytest.raises(AssertionError, match="A4"):
        assert_availability_contract(s, active_modalities=("rgb",))


def test_a5_rgb_may_not_be_sample_level_missing():
    # 구조적 계약: rgb 는 결측 허용 목록에 없다 (실행 경로는 A2 가 먼저 잡는다)
    assert set(OPTIONAL_MODALITY_KEYS) == {"msi", "bio", "ir", "pol"}
    s = make_sample()
    s["avail"][0] = 0.0
    s["rgb"] = torch.zeros_like(s["rgb"])
    with pytest.raises(AssertionError):
        assert_availability_contract(s)


def test_a6_target_zero_fill_and_valid_flag():
    s = make_sample()
    s["y_chl"] = torch.ones_like(s["y_chl"])          # valid 전부 False 인데 값이 있다
    with pytest.raises(AssertionError, match="A6"):
        assert_availability_contract(s)
    s2 = make_sample()
    s2["y_seg"] = s2["y_seg"].to(torch.float32)
    with pytest.raises(AssertionError, match="A6"):
        assert_availability_contract(s2)


def test_contract_rejects_bad_avail_shape_and_dtype():
    s = make_sample()
    s["avail"] = torch.ones(4, dtype=torch.float32)
    with pytest.raises(AssertionError, match="A1"):
        assert_availability_contract(s)
    s2 = make_sample()
    s2["avail"] = s2["avail"].to(torch.float64)
    with pytest.raises(AssertionError, match="A1"):
        assert_availability_contract(s2)


# ─────────────────────────────────────────────────────────────────────────────
# derive_present — A7 전파 / A8 AND (정정 A-25)
# ─────────────────────────────────────────────────────────────────────────────
def test_derive_present_truth_table_full_slots():
    """avail 32조합 전수. phys 는 slot 집합 AND 이며 OR 가 **아니다**."""
    combos = list(itertools.product([0.0, 1.0], repeat=5))
    avail = torch.tensor(combos, dtype=torch.float32)
    present = derive_present(avail)
    assert present.shape == (32, 3) and present.dtype == torch.bool
    or_count = 0
    for i, (rgb, msi, _bio, ir, pol) in enumerate(combos):
        assert bool(present[i, 0]) == (rgb > 0.5)
        assert bool(present[i, 1]) == (msi > 0.5)          # bio 는 부속이라 미반영
        assert bool(present[i, 2]) == ((ir > 0.5) and (pol > 0.5))
        if ((ir > 0.5) or (pol > 0.5)) != ((ir > 0.5) and (pol > 0.5)):
            or_count += 1
            assert bool(present[i, 2]) is False, "OR 규약으로 퇴행했다"
    assert or_count == 16, "AND/OR 가 갈리는 조합이 존재해야 판별력이 있다"


@pytest.mark.parametrize(
    "slot_ids,expect",
    [((0, 1, 2, 3), ("ir", "pol")), ((0,), ("ir",)), ((1, 2, 3), ("pol",))],
)
def test_derive_present_respects_phys_slot_ids(slot_ids, expect):
    combos = list(itertools.product([0.0, 1.0], repeat=5))
    avail = torch.tensor(combos, dtype=torch.float32)
    present = derive_present(avail, phys_slot_ids=slot_ids)
    idxs = [MODALITY_ORDER.index(e) for e in expect]
    for i, c in enumerate(combos):
        want = all(c[j] > 0.5 for j in idxs)
        assert bool(present[i, 2]) is want


def test_derive_present_propagates_sample_level_missing():
    """A7: 샘플 수준 결측이 path presence 로 전파된다 (샘플별 독립)."""
    avail = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    present = derive_present(avail)
    assert present[0].tolist() == [True, True, True]
    assert present[1].tolist() == [True, False, False]


def test_derive_present_rejects_bad_shape():
    with pytest.raises(ValueError):
        derive_present(torch.ones(5, dtype=torch.float32))
    with pytest.raises(ValueError):
        derive_present(torch.ones(2, 4, dtype=torch.float32))


def test_derive_present_consistent_with_collated_batch():
    batch = bloom_collate(
        [make_sample(seed=0), make_sample(avail={"msi": 0.0, "bio": 0.0}, seed=1)]
    )
    present = derive_present(batch["avail"])
    assert present[:, 1].tolist() == [True, False]
