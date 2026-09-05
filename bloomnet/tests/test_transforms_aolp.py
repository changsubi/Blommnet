"""T06 — `data/transforms.py` (01 §7.4 / 06 §3.2.4).

AoLP 부호 변환 누락은 **조용한 물리 오염 버그**다 (분석 I-23, 06 R-10).
편광 데이터가 0개인 지금도 구현 + 단위 테스트를 남긴다.

CPU 전용, 소형 텐서. GPU 미사용.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from bloomnet.constants import IGNORE_INDEX
from bloomnet.data.transforms import (
    JointGeometricTransform,
    PhotometricRGB,
    photometric_forbidden_keys,
    transform_aolp,
)

H = W = 16


def _wrap(theta: torch.Tensor) -> torch.Tensor:
    """θ 를 [-π/2, π/2) 로 감는다 (AoLP 는 주기 180°)."""
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def _make_pol(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    theta = (torch.rand(H, W, generator=g) - 0.5) * math.pi          # ∈ [-π/2, π/2)
    dolp = torch.rand(H, W, generator=g)
    pol = torch.stack([dolp, torch.sin(2 * theta), torch.cos(2 * theta)], dim=0)
    return pol, theta


def _theta_of(pol: torch.Tensor) -> torch.Tensor:
    return torch.atan2(pol[1], pol[2]) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# transform_aolp — 물리 정합
# ─────────────────────────────────────────────────────────────────────────────
def test_hflip_negates_theta():
    pol, theta = _make_pol()
    out = transform_aolp(pol, hflip=True, vflip=False, rot90_k=0)
    assert torch.allclose(_wrap(_theta_of(out)), _wrap(-theta), atol=1e-6)


def test_vflip_negates_theta():
    pol, theta = _make_pol(seed=1)
    out = transform_aolp(pol, hflip=False, vflip=True, rot90_k=0)
    assert torch.allclose(_wrap(_theta_of(out)), _wrap(-theta), atol=1e-6)


def test_hflip_and_vflip_is_180deg_rotation():
    """flip 2회 = 180° 회전이므로 θ 는 그대로다 (부호 반전이 두 번)."""
    pol, theta = _make_pol(seed=2)
    out = transform_aolp(pol, hflip=True, vflip=True, rot90_k=0)
    assert torch.allclose(_wrap(_theta_of(out)), _wrap(theta), atol=1e-6)
    assert torch.allclose(out, pol, atol=0.0)


def test_hflip_twice_is_identity():
    pol, _ = _make_pol(seed=3)
    once = transform_aolp(pol, hflip=True, vflip=False, rot90_k=0)
    twice = transform_aolp(once, hflip=True, vflip=False, rot90_k=0)
    assert torch.allclose(twice, pol, atol=0.0)
    assert not torch.allclose(once, pol, atol=1e-3)      # 실제로 바뀌긴 해야 한다


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_rot90_theta_shift(k: int):
    pol, theta = _make_pol(seed=4)
    out = transform_aolp(pol, hflip=False, vflip=False, rot90_k=k)
    want = _wrap(theta + k * math.pi / 2)
    assert torch.allclose(_wrap(_theta_of(out)), want, atol=1e-6)


def test_rot90_four_times_is_identity():
    pol, _ = _make_pol(seed=5)
    cur = pol
    for _ in range(4):
        cur = transform_aolp(cur, hflip=False, vflip=False, rot90_k=1)
    assert torch.allclose(cur, pol, atol=0.0)
    assert torch.allclose(transform_aolp(pol, hflip=False, vflip=False, rot90_k=4), pol, atol=0.0)


def test_dolp_channel_is_invariant():
    pol, _ = _make_pol(seed=6)
    for hf, vf, k in [(True, False, 0), (False, True, 1), (True, True, 3)]:
        out = transform_aolp(pol, hflip=hf, vflip=vf, rot90_k=k)
        assert torch.allclose(out[0], pol[0], atol=0.0)


def test_transform_aolp_batched_and_validation():
    pol, _ = _make_pol(seed=7)
    batched = pol[None].repeat(2, 1, 1, 1)
    out = transform_aolp(batched, hflip=True, vflip=False, rot90_k=0)
    assert out.shape == batched.shape
    assert torch.allclose(out[0], transform_aolp(pol, hflip=True, vflip=False, rot90_k=0))
    with pytest.raises(ValueError):
        transform_aolp(torch.zeros(2, H, W), hflip=True, vflip=False, rot90_k=0)


# ─────────────────────────────────────────────────────────────────────────────
# JointGeometricTransform — 전 모달 · 전 타깃 동일 파라미터
# ─────────────────────────────────────────────────────────────────────────────
def _aligned_inputs(h: int = 32, w: int = 32):
    pattern = (torch.arange(h * w, dtype=torch.float32).reshape(h, w) % 200)
    tensors = {
        "rgb": pattern[None].repeat(3, 1, 1).clone(),
        "ir": pattern[None].clone(),
    }
    targets = {
        "y_seg": pattern.to(torch.int64),
        "y_chl": pattern[None].clone(),
        "y_chl_valid": (pattern % 2 == 0)[None].clone(),
        "y_chl_scalar": torch.tensor(1.25),
        "y_chl_scalar_valid": torch.tensor(True),
    }
    return tensors, targets


def test_geometric_applies_identical_params_to_all_modalities_and_targets():
    tf = JointGeometricTransform(crop_size=(16, 16), scale_range=(1.0, 1.0))
    for seed in range(8):
        tensors, targets = _aligned_inputs()
        out_t, out_y, aug = tf(tensors, targets, random.Random(seed))
        seg = out_y["y_seg"].to(torch.float32)
        assert out_t["rgb"].shape == (3, 16, 16)
        assert out_y["y_seg"].shape == (16, 16)
        assert torch.equal(out_t["rgb"][0], seg), f"rgb/y_seg 정렬 깨짐 (seed={seed})"
        assert torch.equal(out_t["rgb"][2], seg)
        assert torch.equal(out_t["ir"][0], seg)
        assert torch.equal(out_y["y_chl"][0], seg)
        assert torch.equal(out_y["y_chl_valid"][0], (seg % 2 == 0))
        assert set(aug) == {"scale", "crop_box", "hflip", "vflip", "rot90_k"}
        # 공간축 없는 타깃은 그대로
        assert float(out_y["y_chl_scalar"]) == 1.25
        assert bool(out_y["y_chl_scalar_valid"])


def test_geometric_is_deterministic_given_rng():
    tf = JointGeometricTransform(crop_size=(16, 16))
    a_t, a_y, a_aug = tf(*_aligned_inputs(), random.Random(42))
    b_t, b_y, b_aug = tf(*_aligned_inputs(), random.Random(42))
    assert a_aug == b_aug
    assert torch.equal(a_t["rgb"], b_t["rgb"])
    assert torch.equal(a_y["y_seg"], b_y["y_seg"])


def test_geometric_rot90_disabled_when_pol_present():
    """01 §7.4.1: rot90 은 태양-관측 기하를 깨므로 pol 존재 시 금지된다."""
    tf = JointGeometricTransform(crop_size=(16, 16), scale_range=(1.0, 1.0), rot90=True)
    ks = set()
    for seed in range(20):
        tensors, targets = _aligned_inputs()
        tensors["pol"] = torch.rand(3, 32, 32)
        _, _, aug = tf(tensors, targets, random.Random(seed))
        ks.add(aug["rot90_k"])
    assert ks == {0}, f"pol 존재 시 rot90 이 발동했다: {ks}"

    tf_ok = JointGeometricTransform(
        crop_size=(16, 16), scale_range=(1.0, 1.0), rot90=True, allow_rot90_with_pol=True
    )
    ks2 = set()
    for seed in range(20):
        tensors, targets = _aligned_inputs()
        tensors["pol"] = torch.rand(3, 32, 32)
        _, _, aug = tf_ok(tensors, targets, random.Random(seed))
        ks2.add(aug["rot90_k"])
    assert ks2 - {0}, "opt-in 인데도 rot90 이 한 번도 발동하지 않았다"


def test_geometric_applies_aolp_correction():
    """기하 변환이 pol 채널의 부호 보정을 실제로 수행하는지 (누락 = 조용한 오염)."""
    tf = JointGeometricTransform(
        crop_size=(32, 32), scale_range=(1.0, 1.0), hflip_p=1.0, vflip_p=0.0, rot90=False
    )
    pol, theta = _make_pol(seed=9)
    pol32 = torch.zeros(3, 32, 32)
    pol32[:, :H, :W] = pol
    tensors = {"rgb": torch.rand(3, 32, 32) + 0.1, "pol": pol32}
    targets = {"y_seg": torch.zeros(32, 32, dtype=torch.int64)}
    out_t, _, aug = tf(tensors, targets, random.Random(0))
    assert aug["hflip"] is True and aug["vflip"] is False and aug["rot90_k"] == 0
    flipped = torch.flip(pol32, dims=(-1,))
    got = out_t["pol"]
    assert torch.allclose(got[0], flipped[0], atol=0.0)          # DoLP 는 공간만 뒤집힌다
    assert torch.allclose(got[1], -flipped[1], atol=0.0)         # sin2θ 부호 반전
    assert torch.allclose(got[2], flipped[2], atol=0.0)          # cos2θ 불변


def test_geometric_padding_uses_ignore_label():
    tf = JointGeometricTransform(
        crop_size=(32, 32), scale_range=(0.5, 0.5), pad_label=IGNORE_INDEX, rot90=False,
        hflip_p=0.0, vflip_p=0.0, cat_max_ratio=1.0,
    )
    tensors = {"rgb": torch.rand(3, 32, 32) + 0.1}
    targets = {"y_seg": torch.zeros(32, 32, dtype=torch.int64)}
    out_t, out_y, aug = tf(tensors, targets, random.Random(0))
    assert aug["scale"] == 0.5
    assert out_t["rgb"].shape == (3, 32, 32) and out_y["y_seg"].shape == (32, 32)
    assert int((out_y["y_seg"] == IGNORE_INDEX).sum()) == 32 * 32 - 16 * 16
    assert float(out_t["rgb"][:, 16:, :].abs().max()) == 0.0       # 이미지 패딩은 0


def test_geometric_rejects_post_geometric_targets():
    """01 §7.4.3: y_edge 는 step 7 에서 만든다. 여기 들어오면 H/4 계약이 깨진다."""
    tf = JointGeometricTransform(crop_size=(16, 16))
    tensors, targets = _aligned_inputs()
    targets["y_edge"] = torch.zeros(1, 8, 8)
    with pytest.raises(ValueError, match="y_edge"):
        tf(tensors, targets, random.Random(0))


def test_geometric_rejects_unknown_modality_key():
    tf = JointGeometricTransform(crop_size=(16, 16))
    tensors, targets = _aligned_inputs()
    tensors["thermal"] = torch.zeros(1, 32, 32)
    with pytest.raises(ValueError, match="thermal"):
        tf(tensors, targets, random.Random(0))


def test_geometric_scale_changes_resolution_before_crop():
    tf = JointGeometricTransform(
        crop_size=(16, 16), scale_range=(2.0, 2.0), hflip_p=0.0, vflip_p=0.0, rot90=False
    )
    tensors = {"rgb": torch.rand(3, 32, 32) + 0.1}
    targets = {"y_seg": torch.randint(0, 5, (32, 32), dtype=torch.int64)}
    out_t, out_y, aug = tf(tensors, targets, random.Random(1))
    assert aug["scale"] == 2.0
    top, left, ch, cw = aug["crop_box"]
    assert (ch, cw) == (16, 16)
    assert 0 <= top <= 64 - 16 and 0 <= left <= 64 - 16
    assert out_t["rgb"].shape == (3, 16, 16)
    assert out_y["y_seg"].dtype == torch.int64


def test_geometric_bad_args():
    with pytest.raises(ValueError):
        JointGeometricTransform(crop_size=(0, 16))
    with pytest.raises(ValueError):
        JointGeometricTransform(scale_range=(2.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# PhotometricRGB — RGB 에만
# ─────────────────────────────────────────────────────────────────────────────
def test_photometric_forbidden_keys():
    assert photometric_forbidden_keys() == ("msi", "bio", "ir", "pol")


def test_photometric_identity_when_disabled():
    pm = PhotometricRGB(p=0.0, blur_p=0.0, noise_std=0.0)
    x = torch.rand(3, 8, 8)
    assert torch.allclose(pm(x, random.Random(0)), x, atol=0.0)


def test_photometric_range_and_shape():
    pm = PhotometricRGB(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05, p=1.0)
    for seed in range(10):
        x = torch.rand(3, 8, 8)
        y = pm(x, random.Random(seed))
        assert y.shape == x.shape and y.dtype == torch.float32
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
        assert torch.isfinite(y).all()


def test_photometric_deterministic_given_rng():
    pm = PhotometricRGB(p=1.0, blur_p=1.0, noise_std=0.01)
    x = torch.rand(3, 8, 8)
    a = pm(x, random.Random(7))
    b = pm(x, random.Random(7))
    assert torch.allclose(a, b, atol=0.0)
    c = pm(x, random.Random(8))
    assert not torch.allclose(a, c, atol=1e-6)


def test_photometric_hue_preserves_gray():
    """무채색은 hue 이동에 불변이어야 한다 (HSV 왕복 정합)."""
    pm = PhotometricRGB(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.5, p=1.0)
    x = torch.full((3, 4, 4), 0.42)
    y = pm(x, random.Random(0))
    assert torch.allclose(y, x, atol=1e-6)


def test_photometric_blur_reduces_variance():
    pm = PhotometricRGB(p=0.0, blur_p=1.0, blur_sigma=(1.0, 1.0), noise_std=0.0)
    x = torch.rand(3, 16, 16)
    y = pm(x, random.Random(0))
    assert float(y.var()) < float(x.var())
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_photometric_bad_args():
    with pytest.raises(ValueError):
        PhotometricRGB(hue=0.9)
    with pytest.raises(ValueError):
        PhotometricRGB(brightness=-0.1)
    with pytest.raises(ValueError):
        PhotometricRGB()(torch.rand(4, 8, 8), random.Random(0))
