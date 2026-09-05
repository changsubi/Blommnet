"""T02 — `bloomnet/constants.py` 계약 회귀 고정 (06 §5.1 T02).

실행:
    cd <repo_root> && CUDA_VISIBLE_DEVICES="" \
        python -m pytest bloomnet/tests/test_constants.py -q
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest
import torch

from bloomnet import constants as C
from bloomnet import version as V

# ── 모달 계약 ──────────────────────────────────────────────────────────


def test_modality_order_and_channels() -> None:
    assert C.MODALITY_ORDER == ("rgb", "msi", "bio", "ir", "pol")  # avail 인덱스 0..4
    assert isinstance(C.MODALITY_ORDER, tuple)
    assert set(C.MODALITY_CHANNELS) == set(C.MODALITY_ORDER)
    assert C.MODALITY_CHANNELS["rgb"] == 3
    assert C.MODALITY_CHANNELS["msi"] == -1  # 센서 의존 (4/5/6)
    assert C.MODALITY_CHANNELS["bio"] == 2
    assert C.MODALITY_CHANNELS["ir"] == 1
    assert C.MODALITY_CHANNELS["pol"] == 3  # ★X-04 (2 -> 3, [DoLP, sin2θ, cos2θ])


def test_paths_order() -> None:
    assert C.PATHS == ("rgb", "spec", "phys")  # present 인덱스 0..2 (03)


def test_imagenet_stats_bit_exact() -> None:
    """이전 구현 `aihub092_semseg/data/dataset.py` 와 bit-exact (헌법 C-4)."""
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
    assert torch.equal(C.IMAGENET_MEAN, mean)
    assert torch.equal(C.IMAGENET_STD, std)
    assert C.IMAGENET_MEAN.shape == (3, 1, 1)
    assert C.IMAGENET_MEAN.dtype is torch.float32 and C.IMAGENET_STD.dtype is torch.float32


def test_ignore_index_and_schedule() -> None:
    assert C.IGNORE_INDEX == 255
    assert C.CHANNELS == (32, 64, 160, 320)
    assert C.STRIDES == (4, 8, 16, 32)
    assert C.DEC_CH == 128
    assert C.BD_CH == 32
    assert C.SIZE_DIVISOR == 32
    assert all(isinstance(v, int) for v in C.CHANNELS + C.STRIDES)


# ── canonical slot ─────────────────────────────────────────────────────


def test_canonical_slots() -> None:
    assert C.MSI_SLOTS == ("blue", "green", "red", "rededge1", "rededge2", "nir")
    assert len(C.MSI_SLOTS) == 6
    assert C.PHYS_SLOTS == ("ir", "dolp", "aolp_sin", "aolp_cos")  # ★X-04
    assert len(C.PHYS_SLOTS) == 4
    assert C.MSI_SLOT_ID == {n: i for i, n in enumerate(C.MSI_SLOTS)}
    assert C.PHYS_SLOT_ID == {n: i for i, n in enumerate(C.PHYS_SLOTS)}


def test_sensor_band_ids_are_valid_slots() -> None:
    for sensor, ids in C.SENSOR_BAND_IDS.items():
        assert list(ids) == sorted(set(ids)), f"{sensor}: band_ids 는 중복 없는 오름차순"
        assert all(0 <= i < len(C.MSI_SLOTS) for i in ids), sensor
    assert C.SENSOR_BAND_IDS["m3m"] == (1, 2, 3, 5)  # G560 R650 RE730 NIR860
    assert C.SENSOR_BAND_IDS["rededge_mx"] == (1, 2, 3, 5)  # ★X-03 use_blue=False 기본
    assert C.SENSOR_BAND_IDS["rededge_p"] == (0, 1, 2, 3, 4, 5)


def test_band_centers_match_band_ids() -> None:
    assert set(C.BAND_CENTERS_NM) == set(C.SENSOR_BAND_IDS)
    for sensor, centers in C.BAND_CENTERS_NM.items():
        assert set(centers) <= set(C.MSI_SLOTS), sensor
        expected = {C.MSI_SLOTS[i] for i in C.SENSOR_BAND_IDS[sensor]}
        if sensor == "rededge_mx":
            expected |= {"blue"}  # blue 포함 변형을 위해 중심파장은 유지한다
        assert set(centers) == expected, sensor


@pytest.mark.parametrize("sensor", sorted(C.MCI_C))
def test_mci_c_matches_formula(sensor: str) -> None:
    """c = (λ_RE1 − λ_R) / (λ_NIR − λ_R)  (01 §4.2.2 / X-01)."""
    ctr = C.BAND_CENTERS_NM[sensor]
    expected = (ctr["rededge1"] - ctr["red"]) / (ctr["nir"] - ctr["red"])
    assert C.MCI_C[sensor] == pytest.approx(expected, abs=1e-6)


def test_mci_c_rededge_p_regression() -> None:
    """★ (정정 A-27) 0.475410 은 오기다. 40/118 = 0.3389831."""
    assert C.MCI_C["rededge_p"] == 0.338983
    assert C.MCI_C["m3m"] == 0.380952
    assert C.MCI_C["rededge_mx"] == 0.284884


# ── msi 방사 스케일 (정정 A-40 / X-28) ─────────────────────────────────


def test_k_sensor_contract() -> None:
    assert set(C.K_SENSOR) == set(C.SENSOR_BAND_IDS)
    assert all(v > 0 for v in C.K_SENSOR.values())
    assert C.K_SENSOR["m3m"] == 3.5e3  # 잠정값 [01 U-11]
    assert C.K_SENSOR["rededge_mx"] == 1.0  # 235 는 이미 절대 반사율
    assert C.K_SENSOR["rededge_p"] == 1.0
    assert C.MSI_MEDIAN_RANGE == (1e-3, 1.0)  # 로더 하드 검증 [H1]
    assert C.MSI_MEDIAN_RANGE[0] < C.MSI_MEDIAN_RANGE[1]


# ── 분할 (정정 A-1 / A-39) ─────────────────────────────────────────────


def test_split_constants() -> None:
    assert C.SPLIT_SEED == 20260731
    assert C.SPLIT_TARGET == (0.70, 0.15, 0.15)
    assert sum(C.SPLIT_TARGET) == pytest.approx(1.0, abs=1e-12)
    assert C.SPLIT_TOL == 0.02  # 완화 불가
    assert C.SPLIT_BAND == 0.50  # 출현율 허용 배율 [0.5, 1.5]
    assert C.SPLIT_MIN_GROUPS == 3
    assert C.SPLIT_RETRIES == 8


# ── AI Hub 092 ─────────────────────────────────────────────────────────


def test_class_names() -> None:
    assert len(C.FINE_CLASS_NAMES) == 12
    assert C.FINE_CLASS_NAMES[0] == "background"
    assert C.FINE_CLASS_NAMES[8] == "목장"  # 정정 A-1 exempt 대상 (군 8개뿐)
    assert len(set(C.FINE_CLASS_NAMES)) == 12
    assert C.S1_CLASS_NAMES == ("background", "algae")
    assert all(0 <= i < len(C.FINE_CLASS_NAMES) for i in C.TRANSFER_CLASS_IDS)
    assert C.TRANSFER_CLASS_IDS == (3, 4, 10, 11)
    assert len(C.SCENES) == 5


def test_stem_regex() -> None:
    pat = re.compile(C.STEM_RE)
    m = pat.match("L01_12345_130_20210506_N01_0001")
    assert m is not None
    assert m.group("river") == "01"
    assert m.group("alt") == "130"
    assert m.group("date") == "20210506"
    assert m.group("line") == "N01"
    assert pat.match("L1_12345_130_20210506_N01_0001") is None  # river 는 2자리
    assert pat.match("X01_12345_130_20210506_N01_0001") is None


# ── 235 ────────────────────────────────────────────────────────────────


def test_k235_sites_partition() -> None:
    assert set(C.K235_TRAIN_SITES) | set(C.K235_VAL_SITES) == set(C.K235_ALL_SITES)
    assert not (set(C.K235_TRAIN_SITES) & set(C.K235_VAL_SITES))  # X-22 배타
    assert len(C.K235_ALL_SITES) == 8
    assert C.K235_BAND_ORDER_DEFAULT == ("blue", "green", "red", "rededge1", "nir")  # 가설 H2
    assert set(C.K235_BAND_ORDER_DEFAULT) <= set(C.MSI_SLOTS)
    assert C.K235_LOG1P_MEAN == 1.7559 and C.K235_LOG1P_STD == 0.8265
    assert "WaterTemp" in C.K235_QC and "ODOMg" in C.K235_QC


def test_alarm_thresholds_log1p_consistency() -> None:
    assert C.ALARM_THRESHOLDS_MGM3 == (15.0, 25.0, 100.0)  # 사업문서 s5
    for mgm3, log1p_v in zip(C.ALARM_THRESHOLDS_MGM3, C.ALARM_THRESHOLDS_LOG1P):
        assert math.log1p(mgm3) == pytest.approx(log1p_v, abs=1e-6)
    assert len(C.ALARM_LEVEL_NAMES) == len(C.ALARM_THRESHOLDS_MGM3) + 1
    assert C.ALARM_LEVEL_NAMES == ("정상", "관심", "경계", "대발생")


# ── 출력 키 (X-15) ─────────────────────────────────────────────────────


def test_output_keys_unique() -> None:
    keys = [C.OUT_SEG, C.OUT_CHL, C.OUT_LOGVAR, C.OUT_EDGE, C.OUT_VACUITY]
    keys += list(C.OUT_AUX.values()) + list(C.OUT_SIAM.values())
    assert len(keys) == len(set(keys)), "OUT_* 키 중복"
    assert C.OUT_SEG == "seg_logits_s4"
    assert C.OUT_CHL == "chl_u_s4"  # log1p 공간 (X-14)
    assert all(isinstance(k, str) and k for k in keys)


def test_aux_tap_tables_are_aligned() -> None:
    taps = set(C.AUX_TAP_STRIDE)
    assert taps == set(C.AUX_TAP_CH) == set(C.AUX_TAP_MID) == set(C.OUT_AUX)
    assert taps == {"enc_s8", "enc_s16", "enc_s32", "p8"}
    assert C.AUX_TAP_STRIDE == {"enc_s8": 8, "enc_s16": 16, "enc_s32": 32, "p8": 8}
    assert C.AUX_TAP_CH == {"enc_s8": 64, "enc_s16": 160, "enc_s32": 320, "p8": 128}
    # ★ (I-6 / 정정 B-21) `C_mid = C_in // 2` 규칙 폐기 — 이 리터럴이 정본
    assert C.AUX_TAP_MID == {"enc_s8": 32, "enc_s16": 64, "enc_s32": 128, "p8": 64}
    assert C.AUX_TAP_MID["enc_s16"] != C.AUX_TAP_CH["enc_s16"] // 2
    assert C.AUX_TAP_MID["enc_s32"] != C.AUX_TAP_CH["enc_s32"] // 2
    # 인코더 stage 채널과 tap 채널의 정합 (32→64→160→320)
    assert (C.AUX_TAP_CH["enc_s8"], C.AUX_TAP_CH["enc_s16"], C.AUX_TAP_CH["enc_s32"]) == (
        C.CHANNELS[1],
        C.CHANNELS[2],
        C.CHANNELS[3],
    )
    assert C.AUX_TAP_CH["p8"] == C.DEC_CH


def test_train_only_modules() -> None:
    assert C.TRAIN_ONLY_MODULES == ("edge_head", "aux_heads", "siam_proj")


# ── 레벨 예외 (정정 A-23) ──────────────────────────────────────────────


@pytest.mark.parametrize("module", [C, V])
def test_level_minus_one_files_import_no_bloomnet(module) -> None:
    """constants.py / version.py 는 `bloomnet.*` 를 하나도 import 하지 않는다 (06 §2.1 규칙 2 예외)."""
    src = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            if node.level:  # 상대 import 도 금지
                imported.add("." * node.level + node.module)
    assert not [n for n in imported if n.split(".")[0] == "bloomnet"], imported


def test_version_contract() -> None:
    assert isinstance(V.__version__, str) and V.__version__
    rev = V.git_revision()
    assert isinstance(rev, str) and rev
    assert rev == V.GIT_REVISION_UNKNOWN or len(rev.split("-")[0]) == 40
    assert len(V.short_revision(7)) == 7
