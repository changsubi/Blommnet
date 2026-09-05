"""T05 — `data/boundary.py` (01 §7.5, ★X-07 동결 연산자).

CPU 전용, 소형 텐서. GPU 미사용.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest
import torch

from bloomnet.constants import IGNORE_INDEX
from bloomnet.data.boundary import (
    POS_RATIO_ANCHOR,
    boundary_pos_weight,
    make_boundary_target,
    pos_ratio_is_sane,
)

AIHUB_LABELS = Path("<AIHUB092_ROOT>/labels/train")


def _blob_mask(h: int = 64, w: int = 64, size: int = 20) -> torch.Tensor:
    m = torch.zeros(h, w, dtype=torch.int64)
    c0, c1 = (h - size) // 2, (w - size) // 2
    m[c0 : c0 + size, c1 : c1 + size] = 3
    return m


# ── shape / dtype / 값 범위 ───────────────────────────────────────────────────
def test_shape_dtype_unbatched():
    edge, valid = make_boundary_target(_blob_mask(), radius=1, out_stride=4)
    assert edge.shape == (1, 16, 16)
    assert valid.shape == (1, 16, 16)
    assert edge.dtype == torch.float32
    assert valid.dtype == torch.bool
    assert torch.isin(edge, torch.tensor([0.0, 1.0])).all()


def test_shape_dtype_batched():
    mask = torch.stack([_blob_mask(), _blob_mask(size=10)], dim=0)
    edge, valid = make_boundary_target(mask, radius=1, out_stride=4)
    assert edge.shape == (2, 1, 16, 16)
    assert valid.shape == (2, 1, 16, 16)
    assert edge.dtype == torch.float32 and valid.dtype == torch.bool
    # 배치 판과 개별 판이 일치해야 한다 (dataset/criterion 공용 계약)
    e0, v0 = make_boundary_target(mask[0], radius=1, out_stride=4)
    assert torch.equal(edge[0], e0)
    assert torch.equal(valid[0], v0)


def test_uniform_mask_has_no_edge():
    for cls in (0, 1, 7):
        m = torch.full((64, 64), cls, dtype=torch.int64)
        edge, _ = make_boundary_target(m, radius=1, out_stride=4)
        assert float(edge.sum()) == 0.0, f"uniform class {cls} produced edges"


# ── 테두리 무효화 (4단계) ────────────────────────────────────────────────────
@pytest.mark.parametrize("radius", [1, 2])
def test_border_band_is_invalid(radius: int):
    edge, valid = make_boundary_target(_blob_mask(), radius=radius, out_stride=4)
    assert not valid[0, 0, :].any(), "첫 블록 행이 valid 로 남았다"
    assert not valid[0, -1, :].any()
    assert not valid[0, :, 0].any()
    assert not valid[0, :, -1].any()
    assert valid[0, 2:-2, 2:-2].all()
    # edge 는 valid 밖에서 반드시 0
    assert float((edge * (~valid).float()).sum()) == 0.0


# ── ignore 인접 제거 (3단계) ─────────────────────────────────────────────────
def test_ignore_neighbourhood_is_removed():
    m = torch.zeros(64, 64, dtype=torch.int64)
    m[30:34, 30:34] = IGNORE_INDEX          # ignore 패치만 존재
    edge, valid = make_boundary_target(m, radius=1, out_stride=4)
    # ignore 는 값이 가장 커서 형태학적 gradient 를 만들지만 3단계에서 전부 제거된다
    assert float(edge.sum()) == 0.0
    # 패치 주변 (radius+1=2) 은 감독 대상에서 빠진다
    assert not valid[0, 7, 7]


def test_ignore_removes_real_boundary_nearby():
    m = torch.zeros(64, 64, dtype=torch.int64)
    m[:, 32:] = 1                            # x=32 세로 경계
    edge_clean, _ = make_boundary_target(m, radius=1, out_stride=4)
    m2 = m.clone()
    m2[20:28, 30:35] = IGNORE_INDEX          # 경계 일부를 ignore 로 덮는다
    edge_ign, valid_ign = make_boundary_target(m2, radius=1, out_stride=4)
    assert float(edge_ign.sum()) < float(edge_clean.sum())
    assert not valid_ign[0, 5:7, 7:9].any()


# ── radius 단조성 ────────────────────────────────────────────────────────────
def test_radius_monotone_increase():
    m = _blob_mask()
    n = []
    for r in (0, 1, 2):
        edge, _ = make_boundary_target(m, radius=r, out_stride=1)
        n.append(float(edge.sum()))
    assert n[0] < n[1] < n[2], f"radius 증가가 양성 수를 늘리지 않았다: {n}"


# ── 입력 검증 ────────────────────────────────────────────────────────────────
def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        make_boundary_target(torch.zeros(2, 1, 8, 8, dtype=torch.int64))
    with pytest.raises(ValueError):
        make_boundary_target(torch.zeros(10, 10, dtype=torch.int64), out_stride=4)  # 10 % 4
    with pytest.raises(ValueError):
        make_boundary_target(torch.zeros(8, 8, dtype=torch.int64), radius=-1)


# ── pos_weight ───────────────────────────────────────────────────────────────
def test_pos_weight_sums_to_one():
    edge, valid = make_boundary_target(_blob_mask(), radius=1, out_stride=4)
    w_pos, w_neg = boundary_pos_weight(edge, valid)
    assert abs(w_pos + w_neg - 1.0) < 1e-12
    assert 0.0 < w_pos < 1.0 and 0.0 < w_neg < 1.0
    # 경계는 희소하므로 양성 가중치가 훨씬 크다
    assert w_pos > w_neg


def test_pos_weight_no_valid_pixel():
    edge = torch.zeros(1, 4, 4)
    valid = torch.zeros(1, 4, 4, dtype=torch.bool)
    w_pos, w_neg = boundary_pos_weight(edge, valid)
    assert (w_pos, w_neg) == (1.0, 0.0)
    assert abs(w_pos + w_neg - 1.0) < 1e-12


def test_pos_weight_no_positive():
    m = torch.zeros(64, 64, dtype=torch.int64)
    edge, valid = make_boundary_target(m, radius=1, out_stride=4)
    w_pos, w_neg = boundary_pos_weight(edge, valid)
    assert (w_pos, w_neg) == (1.0, 0.0)
    assert not pos_ratio_is_sane(edge, valid)


# ── X-08 / M-11 판정: 동결 연산자의 실측 양성비 ──────────────────────────────
@pytest.mark.data
@pytest.mark.skipif(
    not AIHUB_LABELS.is_dir(),
    reason=(
        "aihub092 라벨 경로 없음 -> X-08(경계 양성비 3.112 % vs 1.876 %) 미판정. "
        f"기대 경로: {AIHUB_LABELS}"
    ),
)
def test_pos_ratio_sane():
    """01 [M4] radius=1 → H/4 양성비 3.112 % 를 동결 연산자에서 재측정한다 (X-08 / M-11)."""
    from PIL import Image

    files = sorted(glob.glob(str(AIHUB_LABELS / "*" / "*_labelids.png")))[:100]
    assert len(files) >= 20, f"라벨 파일이 부족하다: {len(files)}"
    n_pos = 0.0
    n_all = 0.0
    for f in files:
        arr = np.array(Image.open(f))
        mask = torch.from_numpy(arr.astype(np.int64))
        edge, valid = make_boundary_target(mask, radius=1, out_stride=4)
        n_pos += float(((edge > 0.5) & valid).sum())
        n_all += float(valid.sum())
    ratio = n_pos / n_all
    neg_over_pos = (n_all - n_pos) / n_pos
    assert 0.005 <= ratio <= 0.06, f"H/4 경계 양성비 {ratio:.5f} 가 [0.005, 0.06] 밖"
    # 01 [M4] anchor 31.1 근방이어야 한다 (1.876 % 계열이면 ≈53 이 나온다)
    assert 0.5 * POS_RATIO_ANCHOR <= neg_over_pos <= 2.0 * POS_RATIO_ANCHOR
