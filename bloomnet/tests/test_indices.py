"""T04 — `data/indices.py` (01 §2.6, §4 / ★X-01/X-02, 정정 A-2/A-27/A-40).

CPU 전용, 소형 텐서. GPU 미사용.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bloomnet.constants import BAND_CENTERS_NM, K_SENSOR, MCI_C, MSI_SLOT_ID, SENSOR_BAND_IDS
from bloomnet.data.indices import (
    apply_k_sensor,
    assert_msi_scale_contract,
    canonical_scatter_np,
    compute_bio_canonical,
    compute_rgb_proxy,
    coregister_m3m,
    denormalize_imagenet,
    m3m_dn_to_relative_reflectance,
    mci_coefficient,
    normalize_imagenet,
    specular_proxy,
)

B, H, W = 2, 8, 8
NSLOT = len(MSI_SLOT_ID)
G, R, RE1, NIR = (MSI_SLOT_ID[k] for k in ("green", "red", "rededge1", "nir"))


def _m3m_S(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """M3M 4밴드가 채운 canonical scatter 결과. 값은 O(1e-1) 반사율 자릿수."""
    g = torch.Generator().manual_seed(seed)
    s = torch.zeros(B, NSLOT, H, W)
    m = torch.zeros(B, NSLOT)
    for slot in SENSOR_BAND_IDS["m3m"]:
        s[:, slot] = torch.rand(B, H, W, generator=g) * 1.0 + 0.2
        m[:, slot] = 1.0
    return s, m


# ── mci_coefficient (3센서 회귀 고정) ────────────────────────────────────────
@pytest.mark.parametrize("sensor", ["m3m", "rededge_mx", "rededge_p"])
def test_mci_coefficient(sensor: str):
    b = BAND_CENTERS_NM[sensor]
    c = mci_coefficient(b["red"], b["rededge1"], b["nir"])
    assert abs(c - MCI_C[sensor]) < 1e-6, f"{sensor}: runtime {c} vs constants {MCI_C[sensor]}"


def test_mci_coefficient_hand_values():
    assert abs(mci_coefficient(650, 730, 860) - 80 / 210) < 1e-12
    assert abs(mci_coefficient(668, 717, 840) - 49 / 172) < 1e-12
    # ★ 정정 A-27: rededge_p 는 0.475410 이 아니라 40/118 = 0.338983 이다
    assert abs(mci_coefficient(665, 705, 783) - 40 / 118) < 1e-12
    assert abs(MCI_C["rededge_p"] - 0.338983) < 1e-9


def test_mci_coefficient_zero_denominator():
    with pytest.raises(ValueError):
        mci_coefficient(650.0, 730.0, 650.0)


# ── compute_bio_canonical ────────────────────────────────────────────────────
def test_bio_shape_and_range():
    s, m = _m3m_S()
    out = compute_bio_canonical(s, m, mci_c=MCI_C["m3m"])
    assert out.shape == (B, 2, H, W)
    assert out.dtype == torch.float32
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0
    assert torch.isfinite(out).all()


def test_bio_matches_hand_formula():
    s, m = _m3m_S(seed=3)
    eps = 1e-6
    c = MCI_C["m3m"]
    r, re1, nir = s[:, R : R + 1], s[:, RE1 : RE1 + 1], s[:, NIR : NIR + 1]
    ndci = (re1 - r) / (re1 + r + eps)
    mci = (re1 - r - c * (nir - r)) / (r + re1 + nir + eps)
    want = torch.clamp(torch.cat([ndci, mci], dim=1), -1.0, 1.0)
    got = compute_bio_canonical(s, m, mci_c=c, eps=eps)
    assert torch.allclose(got, want, atol=0.0), "손계산과 불일치"


@pytest.mark.parametrize("k", [0.5, 2.0])
def test_bio_scale_invariance(k: float):
    """NDCI 도 MCI_norm 도 절대 캘리브레이션(R5)에 불변이어야 한다 (01 §4.2.3 근거 3)."""
    s, m = _m3m_S(seed=7)
    a = compute_bio_canonical(s, m, mci_c=MCI_C["m3m"])
    b = compute_bio_canonical(s * k, m, mci_c=MCI_C["m3m"])
    assert torch.allclose(a, b, atol=1e-6), float((a - b).abs().max())


def test_bio_missing_slot_zeroes_channel():
    s, m = _m3m_S(seed=11)
    m_nonir = m.clone()
    m_nonir[:, NIR] = 0.0
    out = compute_bio_canonical(s, m_nonir, mci_c=MCI_C["m3m"])
    assert float(out[:, 1].abs().max()) == 0.0, "NIR 결측인데 MCI_norm 이 0 이 아니다"
    assert float(out[:, 0].abs().max()) > 0.0, "NDCI 는 NIR 과 무관해야 한다"

    m_nore = m.clone()
    m_nore[:, RE1] = 0.0
    out2 = compute_bio_canonical(s, m_nore, mci_c=MCI_C["m3m"])
    assert float(out2.abs().max()) == 0.0, "RE1 결측이면 두 채널 모두 0"


def test_bio_missing_slot_is_per_sample():
    """A7 계열: slot presence 는 (B,K) 이므로 샘플별로 독립 발동해야 한다."""
    s, m = _m3m_S(seed=13)
    m2 = m.clone()
    m2[1, NIR] = 0.0
    out = compute_bio_canonical(s, m2, mci_c=MCI_C["m3m"])
    assert float(out[1, 1].abs().max()) == 0.0
    assert float(out[0, 1].abs().max()) > 0.0


def test_bio_gndvi_ablation_kind():
    s, m = _m3m_S(seed=17)
    out = compute_bio_canonical(s, m, mci_c=MCI_C["m3m"], kind="gndvi")
    eps = 1e-6
    want = (s[:, NIR : NIR + 1] - s[:, G : G + 1]) / (s[:, NIR : NIR + 1] + s[:, G : G + 1] + eps)
    assert torch.allclose(out[:, 1:2], torch.clamp(want, -1, 1), atol=0.0)
    # ch0 는 kind 와 무관하게 NDCI
    assert torch.allclose(
        out[:, 0:1], compute_bio_canonical(s, m, mci_c=MCI_C["m3m"])[:, 0:1], atol=0.0
    )


def test_bio_rejects_bad_args():
    s, m = _m3m_S()
    with pytest.raises(ValueError):
        compute_bio_canonical(s[:, :4], m, mci_c=0.38)
    with pytest.raises(ValueError):
        compute_bio_canonical(s, m[:, :4], mci_c=0.38)
    with pytest.raises(ValueError):
        compute_bio_canonical(s, m, mci_c=0.38, kind="flh")


# ── canonical_scatter_np ─────────────────────────────────────────────────────
def test_canonical_scatter_np_m3m():
    x = np.random.RandomState(0).rand(4, 5, 5).astype(np.float32) + 0.1
    s, m = canonical_scatter_np(x, SENSOR_BAND_IDS["m3m"], NSLOT)
    assert s.shape == (NSLOT, 5, 5) and m.shape == (NSLOT,)
    assert m.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    assert float(np.abs(s[0]).max()) == 0.0 and float(np.abs(s[4]).max()) == 0.0
    assert np.array_equal(s[1], x[0]) and np.array_equal(s[5], x[3])


def test_canonical_scatter_np_validation():
    x = np.zeros((4, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        canonical_scatter_np(x, (1, 2, 3), NSLOT)          # 길이 불일치
    with pytest.raises(ValueError):
        canonical_scatter_np(x, (1, 1, 2, 3), NSLOT)       # 중복 slot
    with pytest.raises(ValueError):
        canonical_scatter_np(x, (1, 2, 3, 9), NSLOT)       # 범위 밖


# ── RGB 프록시 / specular ────────────────────────────────────────────────────
def test_rgb_proxy_range_and_formula():
    rgb = torch.rand(B, 3, H, W)
    out = compute_rgb_proxy(rgb)
    assert out.shape == (B, 2, H, W)
    assert float(out[:, 0].min()) >= -1.0 and float(out[:, 0].max()) <= 2.0   # ExG
    assert float(out[:, 1].min()) >= -1.0 and float(out[:, 1].max()) <= 1.0   # NGRDI
    tot = rgb.sum(dim=1, keepdim=True) + 1e-8
    exg = (2 * rgb[:, 1:2] - rgb[:, 0:1] - rgb[:, 2:3]) / tot
    assert torch.allclose(out[:, 0:1], exg, atol=0.0)


def test_rgb_proxy_pure_green_and_gray():
    green = torch.zeros(1, 3, 2, 2)
    green[:, 1] = 1.0
    out = compute_rgb_proxy(green)
    assert float(out[0, 0].mean()) == pytest.approx(2.0, abs=1e-6)
    assert float(out[0, 1].mean()) == pytest.approx(1.0, abs=1e-6)
    gray = torch.full((1, 3, 2, 2), 0.5)
    out_g = compute_rgb_proxy(gray)
    assert float(out_g.abs().max()) < 1e-6


def test_specular_proxy_range():
    rgb = torch.rand(B, 3, H, W)
    sp = specular_proxy(rgb)
    assert sp.shape == (B, 1, H, W)
    assert float(sp.min()) >= 0.0 and float(sp.max()) <= 1.0
    # 무채색은 Sat=0 이므로 spec == V
    gray = torch.full((1, 3, 2, 2), 0.6)
    assert torch.allclose(specular_proxy(gray), torch.full((1, 1, 2, 2), 0.6), atol=1e-5)
    # 순수 채도색은 Sat≈1 이므로 spec≈0
    pure = torch.zeros(1, 3, 2, 2)
    pure[:, 1] = 0.9
    assert float(specular_proxy(pure).max()) < 1e-5


# ── ImageNet 정규화 왕복 ─────────────────────────────────────────────────────
def test_normalize_denormalize_roundtrip():
    x = torch.rand(B, 3, H, W)
    assert torch.allclose(denormalize_imagenet(normalize_imagenet(x)), x, atol=1e-6)


def test_normalize_matches_literal_constants():
    x = torch.rand(1, 3, 2, 2)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    assert torch.allclose(normalize_imagenet(x), (x - mean) / std, atol=0.0)


# ── M3M 방사 보정 + R4' + [H1] 하드 검증 ─────────────────────────────────────
# 01 §2.6 [M13] 실측 XMP 값 (vignetting/노출·게인은 1 로 두어 표의 R1/R4 를 재현)
_M13 = {
    "green": (13831.5, 19744.299, 8.638e-06),
    "red": (10209.5, 16193.244, 6.944e-06),
    "rededge1": (15852.3, 13127.500, 1.546e-05),
    "nir": (15303.4, 11662.058, 1.665e-05),
}


def _rho_rel(band: str) -> np.ndarray:
    dn_mean, irr, _ = _M13[band]
    dn = np.full((4, 4), dn_mean, dtype=np.float64)
    return m3m_dn_to_relative_reflectance(
        dn,
        black_level=3200,
        vignetting_coeffs=(0.0,) * 6,
        optical_center=(2.0, 2.0),
        exposure_time=1.0,
        sensor_gain=1.0,
        sensor_gain_adj=1.0,
        irradiance=irr,
    )


@pytest.mark.parametrize("band", list(_M13))
def test_m3m_r1_r4_reproduces_measured_table(band: str):
    got = float(_rho_rel(band).mean())
    want = _M13[band][2]
    assert got == pytest.approx(want, rel=2e-3), f"{band}: {got:.4e} vs [M13] {want:.4e}"


def test_msi_scale_contract_fails_without_k_sensor():
    """★ 정정 A-2: R4' 없이 학습에 투입하는 것은 06 §10 금지 항목 14 다."""
    rho_rel = np.stack([_rho_rel(b) for b in _M13])
    with pytest.raises(RuntimeError, match="msi radiometric scale contract violated"):
        assert_msi_scale_contract(rho_rel)


def test_msi_scale_contract_passes_after_k_sensor():
    rho_rel = np.stack([_rho_rel(b) for b in _M13])
    rho = apply_k_sensor(rho_rel, "m3m")
    med = assert_msi_scale_contract(rho)
    assert 1e-3 < med < 1.0, med
    assert K_SENSOR["m3m"] == pytest.approx(3.5e3)


def test_msi_scale_contract_torch_and_empty():
    t = torch.full((2, 4, 4, 4), 0.05)
    assert assert_msi_scale_contract(t) == pytest.approx(0.05)
    with pytest.raises(RuntimeError):
        assert_msi_scale_contract(torch.zeros(2, 2))
    with pytest.raises(RuntimeError):
        assert_msi_scale_contract(torch.full((2, 2), 5.0))     # 상한 위반


def test_m3m_vignetting_and_gain_paths():
    dn = np.full((8, 8), 20000.0)
    base = m3m_dn_to_relative_reflectance(
        dn, black_level=3200, vignetting_coeffs=(0.0,) * 6, optical_center=(3.0, 3.0),
        exposure_time=1.0, sensor_gain=1.0, sensor_gain_adj=1.0, irradiance=1.0,
    )
    vig = m3m_dn_to_relative_reflectance(
        dn, black_level=3200, vignetting_coeffs=(0.0, 0.01, 0.0, 0.0, 0.0, 0.0),
        optical_center=(3.0, 3.0), exposure_time=1.0, sensor_gain=1.0,
        sensor_gain_adj=1.0, irradiance=1.0,
    )
    assert float(vig[0, 0]) < float(vig[3, 3])          # 주변부가 더 크게 나눠진다
    # 광학 중심(r=0)에서는 V=1 이므로 vignetting 유무가 값을 바꾸지 않는다
    assert float(base[3, 3]) == pytest.approx(float(vig[3, 3]), rel=1e-12)
    gained = m3m_dn_to_relative_reflectance(
        dn, black_level=3200, vignetting_coeffs=(0.0,) * 6, optical_center=(3.0, 3.0),
        exposure_time=2.0, sensor_gain=1.0, sensor_gain_adj=1.0, irradiance=1.0,
    )
    assert np.allclose(gained * 2.0, base)
    with pytest.raises(ValueError):
        m3m_dn_to_relative_reflectance(
            dn, black_level=3200, vignetting_coeffs=(0.0,) * 6, optical_center=(0.0, 0.0),
            exposure_time=1.0, sensor_gain=1.0, sensor_gain_adj=1.0, irradiance=0.0,
        )


# ── R6 밴드 정합 ─────────────────────────────────────────────────────────────
def test_coregister_identity_is_identity():
    rs = np.random.RandomState(0)
    bands = [rs.rand(6, 6) for _ in range(4)]
    eye = np.eye(3)
    out = coregister_m3m(bands, [eye] * 4)
    assert out.shape == (4, 6, 6)
    for k in range(4):
        assert np.allclose(out[k], bands[k].astype(np.float32), atol=1e-6)


def test_coregister_translation():
    img = np.arange(36, dtype=np.float64).reshape(6, 6)
    shift = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = coregister_m3m([img, img], [np.eye(3), shift])
    assert np.allclose(out[1][:, :5], img[:, 1:], atol=1e-6)
    assert np.allclose(out[1][:, 5], 0.0)          # 격자 밖은 0
    with pytest.raises(ValueError):
        coregister_m3m([img], [np.eye(3), np.eye(3)])
