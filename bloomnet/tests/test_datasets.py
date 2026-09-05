"""T08/T09 — L2 데이터셋 어댑터 + group 분할 도구 (01 §6~§7, 06 §3.2.5/§3.6).

실행:
    cd <repo_root> && CUDA_VISIBLE_DEVICES="" \
        python -m pytest bloomnet/tests/test_datasets.py -q

규약:
    * GPU 미사용 (autouse fixture 가 `torch.cuda.is_available() == False` 를 강제).
    * 임시 파일은 `/tmp` 가 아니라 `k_water/.pytest_tmp` 안에만 만든다.
    * 실데이터(aihub092 / 235 / M3M)는 **읽기 전용**으로 몇 개 샘플만 만진다.
      경로가 없으면 해당 테스트만 skip 한다 (조용한 pass 금지 — skip 사유를 남긴다).
"""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, Sequence, Tuple

import numpy as np
import pytest
import torch

import bloomnet
from bloomnet.config import default_config, from_dict
from bloomnet.constants import IGNORE_INDEX, MODALITY_ORDER
from bloomnet.data.aihub092 import (
    AIHub092Dataset,
    LABEL_SUFFIX,
    find_pairs,
    group_key,
    parse_stem,
    scan_class_presence,
)
from bloomnet.data.build import (
    assert_resolution_contract,
    build_dataloaders,
    build_datasets,
)
from bloomnet.data.bundle import SAMPLE_REQUIRED_META, bloom_collate
from bloomnet.data.drone_m3m import (
    M3MFrame,
    find_m300_frames,
    find_m3m_frames,
    frame_reflectance,
    parse_xmp,
    read_ms_tiff,
)
from bloomnet.data.indices import compute_rgb_proxy
from bloomnet.data.k235 import (
    K235ChipDataset,
    assert_band_order,
    band_order_evidence,
    build_k235_cache,
    read_chip_geojson,
    read_chip_tiff,
)
from bloomnet.data.transforms import JointGeometricTransform
from bloomnet.tools.make_group_split import (
    ALGORITHM_VERSION,
    SPLIT_NAMES,
    build_group_table,
    make_group_split,
    plan_group_split,
    scan_dataset,
)

REPO_ROOT = Path(bloomnet.__file__).resolve().parent.parent

AIHUB_ROOT = Path("<AIHUB092_ROOT>")
K235_ROOT = Path(
    "<DATA_ROOT>/aihub235"
    "/01-1.정식개방데이터"
)
M3M_CLEAR = Path("<DATA_ROOT>/flight_clear_m3m")
M300_CLEAR = Path("<DATA_ROOT>/flight_clear_p1")
VERIFY_DIR = REPO_ROOT / "methods" / "_verify"

def _needs_data(missing: bool, reason: str):
    """실데이터 의존 표시 + 경로 부재 시 skip.

    06 §5.0 은 실데이터 접근 테스트를 ``@pytest.mark.data`` 로 표시하고 CI 기본 게이트를
    ``-m "not data"`` 로 돌 것을 요구한다. skipif 만 걸면 데이터가 **있는** 머신에서
    기본 게이트가 실데이터 테스트를 그대로 돌아 30초 목표를 지킬 수 없다.
    """
    def deco(fn):
        return pytest.mark.data(pytest.mark.skipif(missing, reason=reason)(fn))
    return deco


needs_aihub092 = _needs_data(not AIHUB_ROOT.is_dir(), f"missing {AIHUB_ROOT}")
needs_k235 = _needs_data(not K235_ROOT.is_dir(), f"missing {K235_ROOT}")
needs_m3m = _needs_data(not M3M_CLEAR.is_dir(), f"missing {M3M_CLEAR}")


@pytest.fixture(autouse=True)
def _no_gpu() -> None:
    """헌법 C-5.2 — 구현·테스트 중 GPU 를 절대 쓰지 않는다."""
    assert not torch.cuda.is_available(), "GPU must be invisible (CUDA_VISIBLE_DEVICES='')"


@pytest.fixture()
def kw_tmp() -> Iterator[Path]:
    base = REPO_ROOT / ".pytest_tmp"
    base.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            base.rmdir()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  합성 트리 생성기
# ═══════════════════════════════════════════════════════════════════════════
def _stem(river: int, admin: int, alt: int, date: str, line: int, seq: int) -> str:
    return f"L{river:02d}_{admin:05d}_{alt}_{date}_N{line:02d}_{seq:05d}"


def _write_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _rgb_pattern(h: int, w: int, seed: int) -> np.ndarray:
    """무채색이 아닌(=A1 위반을 만들지 않는) 결정론적 컬러 패턴."""
    rs = np.random.RandomState(seed)
    yy = np.linspace(0, 255, h, dtype=np.float64)[:, None]
    xx = np.linspace(0, 255, w, dtype=np.float64)[None, :]
    r = (0.6 * yy + 0.2 * xx) % 256
    g = (0.2 * yy + 0.7 * xx + 40) % 256
    b = (0.4 * yy + 0.4 * xx + 90) % 256
    out = np.stack([r, g, b], axis=-1) + rs.randint(0, 8, size=(h, w, 3))
    return np.clip(out, 0, 255).astype(np.uint8)


def make_aihub092_tree(
    root: Path,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    n_per_split: int = 4,
    size: int = 64,
    num_classes: int = 4,
) -> Path:
    """aihub092 와 동일한 디렉터리 계약을 갖는 소형 합성 트리."""
    for si, split in enumerate(splits):
        for scene in ("01.한강", "02.낙동강"):
            for i in range(n_per_split):
                stem = _stem(
                    1 if scene.startswith("01") else 2, 11680 + i, 130, "20230801", si + 1, i
                )
                _write_png(
                    root / "images" / split / scene / f"{stem}.png", _rgb_pattern(size, size, i)
                )
                lab = np.zeros((size, size), dtype=np.uint8)
                lab[: size // 2] = (i % num_classes)
                lab[size // 2 :] = ((i + 1) % num_classes)
                _write_png(root / "labels" / split / scene / f"{stem}{LABEL_SUFFIX}", lab)
    return root


def make_group_tree(
    root: Path, *, num_classes: int = 4, size: int = 16, seed: int = 7
) -> Path:
    """군 크기 편중 + 클래스 편중을 갖는 합성 트리 (정정 A-1 계약 검증용).

    군 = (scene, admin, date, line). 30군 × 2 scene, 크기 min 3 / max 60 으로 편중을 준다.
    """
    rs = np.random.RandomState(seed)
    sizes = [
        3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20,
        22, 24, 26, 28, 30, 34, 38, 42, 46, 50, 54, 58, 60,
    ]
    for scene_id, scene in ((1, "01.한강"), (2, "02.낙동강")):
        for gi, n in enumerate(sizes):
            admin = 11000 + gi
            line = (gi % 20) + 1
            date = f"2023{(gi % 12) + 1:02d}01"
            for j in range(n):
                stem = _stem(scene_id, admin, 130, date, line, j)
                _write_png(
                    root / "images" / "train" / scene / f"{stem}.png", _rgb_pattern(size, size, j)
                )
                lab = np.zeros((size, size), dtype=np.uint8)
                # class 1: 군의 2/3 / class 2: 군의 1/3 / class 3: 소수 군에만 (→ EXEMPT)
                if (gi + scene_id) % 3 != 0:
                    lab[0, 0] = 1
                if (gi + scene_id) % 3 == 1:
                    lab[0, 1] = 2
                if gi < 3 and scene_id == 1:
                    lab[0, 2] = 3
                if rs.rand() < 0.05:
                    lab[1, 1] = IGNORE_INDEX
                _write_png(root / "labels" / "train" / scene / f"{stem}{LABEL_SUFFIX}", lab)
    return root


# ── 235 합성 zip ──────────────────────────────────────────────────────────
def encode_chip_tiff(arr: np.ndarray) -> bytes:
    """``(3,3,5) float32`` → 235 규격 리틀엔디안 TIFF 바이트."""
    h, w, c = arr.shape
    entries = [
        (256, 3, 1, w),
        (257, 3, 1, h),
        (258, 3, c, None),  # BitsPerSample -> 오프셋
        (259, 3, 1, 1),
        (262, 3, 1, 2),
        (273, 4, 1, None),  # StripOffsets -> 오프셋(값)
        (277, 3, 1, c),
        (278, 3, 1, h),
        (279, 4, 1, h * w * c * 4),
        (284, 3, 1, 1),
        (339, 3, c, None),  # SampleFormat -> 오프셋
    ]
    n = len(entries)
    ifd_size = 2 + n * 12 + 4
    bits_off = 8 + ifd_size
    fmt_off = bits_off + 2 * c
    data_off = fmt_off + 2 * c
    out = bytearray(b"II" + struct.pack("<HI", 42, 8))
    out += struct.pack("<H", n)
    for tag, typ, num, val in entries:
        if tag == 258:
            payload = struct.pack("<I", bits_off)
        elif tag == 339:
            payload = struct.pack("<I", fmt_off)
        elif tag == 273:
            payload = struct.pack("<I", data_off)
        elif typ == 3:
            payload = struct.pack("<HH", int(val), 0)
        else:
            payload = struct.pack("<I", int(val))
        out += struct.pack("<HHI", tag, typ, num) + payload
    out += struct.pack("<I", 0)
    out += struct.pack("<" + "H" * c, *([32] * c))
    out += struct.pack("<" + "H" * c, *([3] * c))
    assert len(out) == data_off, (len(out), data_off)
    out += np.ascontiguousarray(arr, dtype="<f4").tobytes()
    return bytes(out)


def make_k235_source(root: Path, *, n_chips: int = 40, site: str = "ABC") -> Path:
    """235 원본 zip 쌍(원천/라벨)을 흉내 낸 합성 소스 트리."""
    rs = np.random.RandomState(3)
    src_dir = root / "Training" / "01.원천데이터"
    lbl_dir = root / "Training" / "02.라벨링데이터"
    src_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    name = f"2.녹조데이터_1.다중분광_1.테스트({site}).zip"
    with zipfile.ZipFile(src_dir / f"TS_{name}", "w") as zs, zipfile.ZipFile(
        lbl_dir / f"TL_{name}", "w"
    ) as zl:
        for i in range(n_chips):
            stem = f"M_220728_{site}_01_{i:04d}"
            veg = i < max(2, n_chips // 20)  # 식생 칩: RE 상승 + NIR 고원
            base = (
                np.array([0.04, 0.11, 0.04, 0.25, 0.45])
                if veg
                else np.array([0.05, 0.06, 0.035, 0.033, 0.045])
            )
            chip = (base[None, None, :] * (1.0 + 0.02 * rs.randn(3, 3, 5))).astype(np.float32)
            zs.writestr(f"/{stem}.tif", encode_chip_tiff(chip))
            props = {
                "Latitude": 36.3 + 1e-4 * i,
                "Longitude": 127.6,
                "Date": "20220728",
                "Time": f"{10 + i % 8:02d}0000",
                "Sensor": "RedEdge",
                "ODOMg": 95.0 if i % 10 == 0 else 11.4,  # 10 % 는 QC 로 제거되어야 한다
                "ODOS": 154.9,
                "EC": 200.3,
                "Chla": 0.5 + 0.5 * i,
                "Phyco": 0.01,
                "WaterTemp": 88.8 if i % 10 == 1 else 25.0,
                "Temp": 32.5,
                "Humidity": 53.4,
                "Windspeed": 2.2,
            }
            zl.writestr(
                f"/{stem}.geojson",
                json.dumps({"type": "FeatureCollection", "features": [{"properties": props}]}),
            )
    return root


# ── M3M 합성 TIFF ─────────────────────────────────────────────────────────
_XMP_TEMPLATE = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
    'drone-dji:BandName="{band}" drone-dji:BandFreq="{freq}" '
    'drone-dji:Irradiance="{irr}" drone-dji:ExposureTime="1000" '
    'drone-dji:SensorGain="1.000" drone-dji:SensorGainAdjustment="0.5" '
    'drone-dji:BlackLevel="3200" drone-dji:CalibratedOpticalCenterX="{cx}" '
    'drone-dji:CalibratedOpticalCenterY="{cy}" drone-dji:RelativeOpticalCenterX="0.0" '
    'drone-dji:RelativeOpticalCenterY="0.0" drone-dji:RelativeAltitude="+80.0" '
    'drone-dji:CalibratedFocalLength="2170.0" '
    'drone-dji:VignettingData="0.0, 0.0, 0.0, 0.0, 0.0, 0.0" '
    'drone-dji:CalibratedHMatrix="1,0,0,0,1,0,0,0,1" '
    'drone-dji:UTCAtExposure="2026-05-15T06:00:01.489029"/></rdf:RDF></x:xmpmeta>'
)


def encode_ms_tiff(arr: np.ndarray, xmp: str) -> bytes:
    """``(H,W) uint16`` + XMP → 압축 없는 단일밴드 리틀엔디안 TIFF."""
    h, w = arr.shape
    entries = [
        (256, 3, 1, w),
        (257, 3, 1, h),
        (258, 3, 1, 16),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (273, 4, 1, None),
        (277, 3, 1, 1),
        (278, 3, 1, h),
        (279, 4, 1, h * w * 2),
        (284, 3, 1, 1),
    ]
    n = len(entries)
    xmp_b = xmp.encode("utf-8")
    ifd_size = 2 + n * 12 + 4
    xmp_off = 8 + ifd_size
    data_off = xmp_off + len(xmp_b)
    out = bytearray(b"II" + struct.pack("<HI", 42, 8))
    out += struct.pack("<H", n)
    for tag, typ, num, val in entries:
        if tag == 273:
            payload = struct.pack("<I", data_off)
        elif typ == 3:
            payload = struct.pack("<HH", int(val), 0)
        else:
            payload = struct.pack("<I", int(val))
        out += struct.pack("<HHI", tag, typ, num) + payload
    out += struct.pack("<I", 0)
    out += xmp_b
    out += np.ascontiguousarray(arr, dtype="<u2").tobytes()
    return bytes(out)


def make_m3m_frame(root: Path, *, size: int = 32, seq: str = "0001") -> M3MFrame:
    from PIL import Image

    flight = root / "고도80"
    flight.mkdir(parents=True, exist_ok=True)
    rs = np.random.RandomState(11)
    bands = {"G": (560, 20000.0), "R": (650, 16000.0), "RE": (730, 13000.0), "NIR": (860, 11000.0)}
    for suffix, (nm, irr) in bands.items():
        dn = (3200 + rs.randint(4000, 9000, size=(size, size))).astype(np.uint16)
        xmp = _XMP_TEMPLATE.format(
            band=suffix, freq=f"{nm}(+/-16)nm", irr=irr, cx=size / 2, cy=size / 2
        )
        (flight / f"DJI_20260515145940_{seq}_MS_{suffix}.TIF").write_bytes(
            encode_ms_tiff(dn, xmp)
        )
    Image.fromarray(_rgb_pattern(size, size, 5)).save(
        flight / f"DJI_20260515145940_{seq}_D.JPG"
    )
    return find_m3m_frames(root)[0]


# ═══════════════════════════════════════════════════════════════════════════
#  A. aihub092 — 페어링 / 파이프라인 (T08)
# ═══════════════════════════════════════════════════════════════════════════
def test_parse_stem_ok_and_group_key() -> None:
    g = parse_stem("L01_11680_65_20231025_N03_00001")
    assert g == {
        "river": "01",
        "admin": "11680",
        "alt": "65",
        "date": "20231025",
        "line": "N03",
        "seq": "00001",
    }
    assert group_key("01.한강", "L01_11680_65_20231025_N03_00001") == (
        "01.한강",
        "11680",
        "20231025",
        "N03",
    )


@pytest.mark.parametrize(
    "bad",
    [
        "L1_11680_65_20231025_N03_00001",  # river 2자리 아님
        "L01_1168_65_20231025_N03_00001",  # admin 5자리 아님
        "L01_11680_65_2023102_N03_00001",  # date 8자리 아님
        "L01_11680_65_20231025_M03_00001",  # line 접두 N 아님
        "L01_11680_65_20231025_N03_00001_windshield_vis",  # ★ sensor-suffix 는 이 데이터에 없다
        "",
    ],
)
def test_parse_stem_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_stem(bad)


def test_find_pairs_missing_label_is_runtime_error(kw_tmp: Path) -> None:
    root = make_aihub092_tree(kw_tmp / "tree", splits=("train",), n_per_split=3)
    victim = sorted((root / "labels" / "train" / "01.한강").glob("*.png"))[0]
    victim.unlink()
    with pytest.raises(RuntimeError, match="pairing failed"):
        find_pairs(root, "train")


def test_find_pairs_orphan_label_is_runtime_error(kw_tmp: Path) -> None:
    root = make_aihub092_tree(kw_tmp / "tree", splits=("train",), n_per_split=3)
    _write_png(
        root / "labels" / "train" / "01.한강" / f"L01_99999_130_20230801_N01_00099{LABEL_SUFFIX}",
        np.zeros((8, 8), dtype=np.uint8),
    )
    with pytest.raises(RuntimeError, match="pairing failed"):
        find_pairs(root, "train")


def _make_dataset(root: Path, **kw: Any) -> AIHub092Dataset:
    params: Dict[str, Any] = dict(num_classes=4, boundary_radius=1, boundary_stride=4)
    params.update(kw)
    return AIHub092Dataset(str(root), "train", **params)


def test_sample_schema_is_complete(kw_tmp: Path) -> None:
    root = make_aihub092_tree(kw_tmp / "tree", splits=("train",), n_per_split=2, size=64)
    ds = _make_dataset(root)
    s = ds[0]
    assert s["rgb"].shape == (3, 64, 64) and s["rgb"].dtype == torch.float32
    assert s["avail"].shape == (len(MODALITY_ORDER),) and float(s["avail"][0]) == 1.0
    assert s["y_seg"].shape == (64, 64) and s["y_seg"].dtype == torch.int64
    assert s["y_edge"].shape == (1, 16, 16) and s["y_edge"].dtype == torch.float32
    assert s["y_edge_valid"].shape == (1, 16, 16) and s["y_edge_valid"].dtype == torch.bool
    assert s["y_chl"].shape == (1, 64, 64) and s["y_chl_valid"].dtype == torch.bool
    assert s["y_chl_scalar"].shape == () and s["y_chl_scalar_valid"].shape == ()
    assert not bool(s["y_chl_valid"].any()) and not bool(s["y_chl_scalar_valid"])
    for key in SAMPLE_REQUIRED_META:
        assert key in s["meta"], f"meta missing required key {key!r}"
    assert s["meta"]["chl_space"] == "log1p"
    assert "msi" not in s and "ir" not in s and "pol" not in s  # A4


def test_boundary_source_criterion_leaves_targets_empty(kw_tmp: Path) -> None:
    """정정 A-28 — source == criterion 이면 y_edge 소유권은 criterion 에 있다."""
    root = make_aihub092_tree(kw_tmp / "tree", splits=("train",), n_per_split=2)
    ds = _make_dataset(root, boundary_source="criterion")
    s = ds[0]
    assert float(s["y_edge"].abs().sum()) == 0.0
    assert not bool(s["y_edge_valid"].any())
    ds2 = _make_dataset(root, boundary_source="dataset")
    assert float(ds2[0]["y_edge"].sum()) > 0.0


def test_rgb_proxy_sets_bio_kind_and_avail(kw_tmp: Path) -> None:
    """V17 / 정정 A-11 — source == rgb_proxy 면 meta['bio_kind'] 도 rgb_proxy 여야 한다."""
    root = make_aihub092_tree(kw_tmp / "tree", splits=("train",), n_per_split=2)
    ds = _make_dataset(root, bio_source="rgb_proxy", bio_kind="rgb_proxy")
    s = ds[0]
    assert s["bio"].shape == (2, 64, 64)
    assert float(s["avail"][MODALITY_ORDER.index("bio")]) == 1.0
    assert float(s["avail"][MODALITY_ORDER.index("msi")]) == 0.0  # A3 완화 경로
    assert s["meta"]["bio_kind"] == "rgb_proxy"


def test_bio_is_computed_after_geometric_transform(kw_tmp: Path) -> None:
    """T08 파이프라인 순서 — 지수의 평균 ≠ 평균의 지수 (01 §7.4.3 step 4).

    scale=2 확대 후 계산한 지수(정본)와, 원본에서 계산한 뒤 같은 기하변환을 먹인 지수는
    **달라야 한다**. 같다면 step 4 가 step 3 앞으로 잘못 이동한 것이다.
    """
    root = make_aihub092_tree(kw_tmp / "tree", splits=("train",), n_per_split=1, size=64)
    geo = JointGeometricTransform(
        crop_size=(64, 64), scale_range=(2.0, 2.0), hflip_p=0.0, vflip_p=0.0, rot90=False
    )
    ds = _make_dataset(root, bio_source="rgb_proxy", bio_kind="rgb_proxy", geometric=geo)
    s = ds[0]

    # 같은 crop 좌표를 재현해 "지수 먼저 → 리샘플" 경로를 만든다
    box = s["meta"]["aug"]["crop_box"]
    top, left, ch, cw = box
    from bloomnet.data.aihub092 import _load_rgb01

    raw = _load_rgb01(sorted((root / "images" / "train" / "01.한강").glob("*.png"))[0])
    bio_first = compute_rgb_proxy(raw[None], eps=1e-8)
    bio_first = torch.nn.functional.interpolate(
        bio_first, size=(128, 128), mode="bilinear", align_corners=False
    )[0][:, top : top + ch, left : left + cw]

    diff = (s["bio"] - bio_first).abs().max().item()
    assert diff > 1e-4, (
        "bio appears to be computed BEFORE the geometric transform "
        f"(max|Δ| = {diff:.3e}); 01 §7.4.3 requires step 4 after step 3"
    )


def test_class_presence_and_collate(kw_tmp: Path) -> None:
    root = make_aihub092_tree(kw_tmp / "tree", splits=("train",), n_per_split=3, num_classes=4)
    ds = _make_dataset(root)
    pres = ds.class_presence
    assert pres.shape == (len(ds), 4) and pres.dtype == np.bool_
    assert pres.any()
    batch = bloom_collate([ds[0], ds[1]])
    assert batch["rgb"].shape[0] == 2
    assert batch["avail"].shape == (2, len(MODALITY_ORDER))
    assert isinstance(batch["meta"], list) and len(batch["meta"]) == 2
    assert batch["y_seg"].dtype == torch.int64
    assert batch["all_missing"]["rgb"] is False


def test_rare_class_crop_prefers_rare_pixels(kw_tmp: Path) -> None:
    root = kw_tmp / "rare"
    size = 64
    stem = _stem(1, 11680, 130, "20230801", 1, 0)
    _write_png(root / "images" / "train" / "01.한강" / f"{stem}.png", _rgb_pattern(size, size, 1))
    lab = np.zeros((size, size), dtype=np.uint8)
    lab[:16, :16] = 3  # 희소 클래스가 좌상단 1/16 면적에만 있다
    _write_png(root / "labels" / "train" / "01.한강" / f"{stem}{LABEL_SUFFIX}", lab)
    geo = JointGeometricTransform(
        crop_size=(16, 16), scale_range=(1.0, 1.0), hflip_p=0.0, vflip_p=0.0, rot90=False
    )

    def _hits(crop_prob: float) -> int:
        n = 0
        for i in range(20):
            ds = _make_dataset(
                root,
                geometric=geo,
                rare_class_ids=(3,),
                rare_class_crop_prob=crop_prob,
                rare_class_min_pixels=1,
                rare_class_crop_attempts=30,
                seed=100 + i,
            )
            n += int((ds[0]["y_seg"] == 3).any())
        return n

    on, off = _hits(1.0), _hits(0.0)
    assert on >= 18, f"rare-class aware crop found the class only {on}/20 times"
    assert on > off, f"rare-class crop is not better than plain crop ({on} vs {off})"


# ── 실데이터 ──────────────────────────────────────────────────────────────
@needs_aihub092
def test_real_aihub092_pairing_counts() -> None:
    pairs = find_pairs(AIHUB_ROOT, "test")
    assert len(pairs) == 3212, f"aihub092 test split changed: {len(pairs)}"
    for img, lbl in pairs[:5]:
        assert img.exists() and lbl.exists()
        assert lbl.name == img.stem + LABEL_SUFFIX
    assert pairs == find_pairs(AIHUB_ROOT, "test")  # 결정론적 순서


@needs_aihub092
def test_real_aihub092_sample_shapes_and_ignore() -> None:
    ds = AIHub092Dataset(
        str(AIHUB_ROOT),
        "test",
        num_classes=12,
        geometric=JointGeometricTransform(crop_size=(256, 256), scale_range=(1.0, 1.0)),
        boundary_radius=1,
        boundary_stride=4,
    )
    assert len(ds) == 3212
    for idx in (0, 1000, 3211):
        s = ds[idx]
        assert s["rgb"].shape == (3, 256, 256)
        assert s["y_seg"].shape == (256, 256)
        vals = torch.unique(s["y_seg"]).tolist()
        assert all(0 <= v < 12 or v == IGNORE_INDEX for v in vals), vals
        assert s["y_edge"].shape == (1, 64, 64)
        assert torch.isfinite(s["rgb"]).all()
        assert s["meta"]["group_key"][0] in ("01.한강", "02.낙동강", "03.금강", "04.영산강", "05.새만금")


@needs_aihub092
def test_real_aihub092_labels_have_no_ignore_and_fit_12_classes() -> None:
    """01 §7.3 [M14] '`id>=12` 픽셀 보유 이미지 = 0' 을 표본 40장으로 재확인한다."""
    pairs = find_pairs(AIHUB_ROOT, "test")
    sub = [p[1] for p in pairs[::80][:40]]
    pres13 = scan_class_presence(sub, 13)
    assert not pres13[:, 12].any(), "found label id >= 12 (255 ignore 미사용 가정 위반)"
    assert pres13[:, :12].any()


# ═══════════════════════════════════════════════════════════════════════════
#  B. k235 (T09)
# ═══════════════════════════════════════════════════════════════════════════
def test_read_chip_tiff_roundtrip_synthetic() -> None:
    rs = np.random.RandomState(0)
    chip = rs.rand(3, 3, 5).astype(np.float32)
    out = read_chip_tiff(encode_chip_tiff(chip))
    assert out.shape == (3, 3, 5) and out.dtype == np.float32
    np.testing.assert_allclose(out, chip, rtol=0, atol=0)


def test_read_chip_tiff_rejects_wrong_band_count() -> None:
    chip = np.zeros((3, 3, 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="5 samples"):
        read_chip_tiff(encode_chip_tiff(chip))


def test_assert_band_order_detects_swap() -> None:
    """H1(…, NIR, RE) 로 뒤집으면 식생 칩의 NIR 고원 증거가 무너져야 한다."""
    rs = np.random.RandomState(1)
    n = 400
    spec = np.tile(np.array([0.05, 0.06, 0.035, 0.033, 0.045]), (n, 1))
    spec[:8] = np.array([0.04, 0.11, 0.04, 0.25, 0.45])  # 식생 칩 상위 2 %
    spec = spec * (1 + 0.01 * rs.randn(n, 5))
    ok = {"blue": 0, "green": 1, "red": 2, "rededge1": 3, "nir": 4}
    assert_band_order(spec, ok)  # H2 → 통과
    swapped = {"blue": 0, "green": 1, "red": 2, "rededge1": 4, "nir": 3}
    with pytest.raises(RuntimeError, match="band order"):
        assert_band_order(spec, swapped)


def test_band_order_evidence_reports_tail_sizes() -> None:
    rs = np.random.RandomState(2)
    spec = np.tile(np.array([0.05, 0.06, 0.035, 0.033, 0.045]), (500, 1))
    spec[:5] = np.array([0.04, 0.11, 0.04, 0.25, 0.45])
    spec = spec * (1 + 0.01 * rs.randn(500, 5))
    ev = band_order_evidence(spec, {"blue": 0, "green": 1, "red": 2, "rededge1": 3, "nir": 4})
    assert {"top_0.001", "top_0.01", "top_0.05", "top_n10", "top_n25"} <= set(ev)
    assert ev["top_n10"]["ratio"] > ev["top_0.05"]["ratio"]  # 꼬리가 짧을수록 식생 신호가 강하다


def test_k235_cache_and_dataset_synthetic(kw_tmp: Path) -> None:
    src = make_k235_source(kw_tmp / "k235src", n_chips=60)
    npz = build_k235_cache(src, kw_tmp / "cache" / "k235.npz", band_order_check="strict")
    z = np.load(npz, allow_pickle=False)
    for key in ("spectra", "spectra_sd", "chla", "site", "date", "stem", "band_order", "lat"):
        assert key in z.files, f"npz missing {key}"
    assert z["spectra"].shape == (60, 5)
    assert str(z["band_order"]) == "blue,green,red,rededge1,nir"

    ds = K235ChipDataset(str(npz), ("ABC",), band_order_check="strict")
    # QC: WaterTemp>40 10 % + ODOMg>90 10 % (겹치지 않게 만들었다) → 20 % 제거
    assert ds.n_before_qc == 60 and len(ds) == 48 and ds.n_removed_by_qc == 12
    s = ds[0]
    assert s["x"].shape == (8, 3, 3) and s["x"].dtype == torch.float32
    assert s["m"].shape == (6,)
    assert s["m"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]  # ★X-03 use_blue=False
    assert float(s["m"][0]) == 0.0
    assert 0.0 <= float(s["y_chl_scalar"]) < 8.0  # log1p 공간
    assert bool(s["y_chl_scalar_valid"])
    assert set(s["meta"]) == {"site", "date", "time", "stem", "chla_mgm3", "phyco"}
    # x 의 canonical slot 값이 원 spectra 와 일치하는지 (blue 는 버려진다).
    # ds[0] 은 QC 통과 후 첫 칩이므로 원 인덱스는 ds.index[0] 이다.
    row = z["spectra"][ds.index[0]]
    np.testing.assert_allclose(s["x"][1, 0, 0].item(), row[1], rtol=1e-6)
    np.testing.assert_allclose(s["x"][5, 0, 0].item(), row[4], rtol=1e-6)
    assert float(s["x"][0, 0, 0]) == 0.0 and float(s["x"][4, 0, 0]) == 0.0
    # bio 2채널은 clamp[-1,1]
    assert -1.0 <= float(s["x"][6, 0, 0]) <= 1.0
    assert -1.0 <= float(s["x"][7, 0, 0]) <= 1.0

    ds_noqc = K235ChipDataset(
        str(npz), ("ABC",), apply_quality_filter=False, band_order_check="off"
    )
    assert len(ds_noqc) == 60

    ds_blue = K235ChipDataset(str(npz), ("ABC",), use_blue=True, band_order_check="off")
    assert ds_blue[0]["m"].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]


def test_k235_band_order_mismatch_is_error(kw_tmp: Path) -> None:
    src = make_k235_source(kw_tmp / "k235src", n_chips=40)
    npz = build_k235_cache(src, kw_tmp / "cache" / "k235.npz", band_order_check="off")
    with pytest.raises(RuntimeError, match="band_order mismatch"):
        K235ChipDataset(
            str(npz), ("ABC",), band_order=("blue", "green", "red", "nir", "rededge1")
        )


@needs_k235
def test_real_k235_chip_tiff_parses() -> None:
    zp = next((K235_ROOT / "Training" / "01.원천데이터").glob("*다중분광_1*.zip"))
    lz = K235_ROOT / "Training" / "02.라벨링데이터" / zp.name.replace("TS_", "TL_", 1)
    with zipfile.ZipFile(zp) as zs, zipfile.ZipFile(lz) as zl:
        names = sorted(n for n in zs.namelist() if n.endswith(".tif"))
        assert len(names) > 0
        for n in names[:3]:
            chip = read_chip_tiff(zs.read(n))
            assert chip.shape == (3, 3, 5) and chip.dtype == np.float32
            assert np.isfinite(chip).all() and (chip >= 0).all()
            props = read_chip_geojson(zl.read(n.replace(".tif", ".geojson")))
            assert "Chla" in props and props["Sensor"] == "RedEdge"


@needs_k235
def test_real_k235_partial_cache_and_dataset(kw_tmp: Path) -> None:
    """실 zip 에서 일부만 캐시해 dataset 계약을 확인한다 (전수 로딩 금지)."""
    npz = build_k235_cache(
        K235_ROOT, kw_tmp / "k235_partial.npz", limit_per_zip=40, band_order_check="off"
    )
    z = np.load(npz, allow_pickle=False)
    assert z["spectra"].shape[1] == 5 and z["spectra"].shape[0] == 40 * 16
    assert sorted(set(z["site"].tolist())) == [
        "BES", "CNH", "CSC", "GDK", "HNM", "JAD", "SHC", "YJD",
    ]
    ds = K235ChipDataset(str(npz), ("CSC", "JAD"), band_order_check="off")
    assert len(ds) > 0
    assert 1e-3 < ds.msi_median < 1.0  # [H1] 235 는 절대 반사율 → k_sensor = 1.0 로 통과
    s = ds[0]
    assert s["x"].shape == (8, 3, 3) and s["m"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    assert float(s["y_chl_scalar"]) == pytest.approx(
        float(np.log1p(s["meta"]["chla_mgm3"])), rel=1e-6
    )


# ═══════════════════════════════════════════════════════════════════════════
#  C. drone_m3m
# ═══════════════════════════════════════════════════════════════════════════
def test_parse_xmp_extracts_dji_fields() -> None:
    xmp = _XMP_TEMPLATE.format(band="Green", freq="560(+/-16)nm", irr=1.0, cx=1, cy=1)
    buf = b"\x00\x01" + xmp.encode()
    d = parse_xmp(buf)
    assert d["BandName"] == "Green" and d["BlackLevel"] == "3200"
    assert parse_xmp(b"no xmp here") == {}


def test_synthetic_m3m_frame_pipeline(kw_tmp: Path) -> None:
    frame = make_m3m_frame(kw_tmp / "m3m", size=32)
    assert frame.complete and set(frame.bands) == {"G", "R", "RE", "NIR"}
    dn, xmp = read_ms_tiff(frame.bands["G"])
    assert dn.shape == (32, 32) and dn.dtype == np.uint16 and xmp["BandName"] == "G"
    rho, info = frame_reflectance(frame, downsample=1, k_sensor=1.0, check_scale=False)
    assert rho.shape == (4, 32, 32) and rho.dtype == np.float32
    assert np.isfinite(rho).all() and (rho >= 0).all()
    assert info["bands"]["NIR"]["irradiance"] == pytest.approx(11000.0)


@needs_m3m
def test_real_m3m_frame_index_matches_analysis() -> None:
    frames = find_m3m_frames(M3M_CLEAR)
    assert len(frames) == 1119, f"analysis §C.1 counted 1,119 M3M frames, got {len(frames)}"
    assert all(f.complete for f in frames)
    overcast = Path("<DATA_ROOT>/flight_overcast_m3m")
    if overcast.is_dir():
        assert len(find_m3m_frames(overcast)) == 1118
    if M300_CLEAR.is_dir():
        assert len(find_m300_frames(M300_CLEAR)) == 2712  # P1 = RGB 전용


@needs_m3m
def test_real_m3m_radiometry_and_h1_contract() -> None:
    """★ 회귀 고정 — 01 §2.6 [M13] 의 k_sensor 잠정값이 [H1] 을 위반한다.

    R1~R4 를 **전부** (특히 R3: ExposureTime[µs] → 초) 적용하면 M3M ``rho_rel`` 중앙값은
    ``O(1e−2)`` 로 235 절대 반사율과 이미 같은 자릿수다. 여기에 ``K_SENSOR["m3m"]=3.5e3``
    을 곱하면 median ≈ 43 이 되어 [H1] 상한(1.0)을 위반한다.
    """
    frame = find_m3m_frames(M3M_CLEAR)[0]
    rho, info = frame_reflectance(frame, downsample=16, k_sensor=1.0, check_scale=True)
    assert rho.shape[0] == 4
    assert 1e-3 < info["msi_median"] < 1.0
    assert 5e-3 < info["msi_median"] < 5e-2, info["msi_median"]
    for suffix in ("G", "R", "RE", "NIR"):
        assert 1e-3 < info["bands"][suffix]["rho_rel_median"] < 1e-1

    with pytest.raises(RuntimeError, match="radiometric scale contract"):
        frame_reflectance(frame, downsample=16, check_scale=True)  # k_sensor = K_SENSOR['m3m']


# ═══════════════════════════════════════════════════════════════════════════
#  D. group 분할 (정정 A-1, T08 계약)
# ═══════════════════════════════════════════════════════════════════════════
def _plan_from_tree(root: Path, **kw: Any) -> Tuple[Any, Any]:
    records, presence = scan_dataset(root, num_classes=4)
    table = build_group_table(records, presence)
    return table, plan_group_split(table, **kw)


def test_group_split_hard_contracts_on_synthetic_tree(kw_tmp: Path) -> None:
    root = make_group_tree(kw_tmp / "grp")
    table, plan = _plan_from_tree(root)
    # (a) V-A 비율 허용오차
    assert plan.max_abs_dev <= 0.02, plan.achieved_ratio
    # (b) V-B 군 배타
    sets = [set(np.nonzero(plan.assign == s)[0].tolist()) for s in range(3)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(s) for s in sets) == len(table.keys)
    # (c) 비면제 클래스 출현율 배율 ∈ [0.5, 1.5] & split 당 독립 군 >= 3
    for row in plan.per_split_class:
        if row["class_id"] in plan.exempt:
            continue
        if row["split"] in ("val", "test"):
            assert 0.5 <= row["ratio_to_global"] <= 1.5, row
        assert row["n_groups"] >= 3, row
    assert plan.violations == []


def test_group_split_is_bit_identical_for_same_seed(kw_tmp: Path) -> None:
    root = make_group_tree(kw_tmp / "grp")
    _, p1 = _plan_from_tree(root)
    _, p2 = _plan_from_tree(root)
    assert np.array_equal(p1.assign, p2.assign)
    assert (p1.seed_used, p1.exempt, p1.n_images_per_split) == (
        p2.seed_used,
        p2.exempt,
        p2.n_images_per_split,
    )
    _, p3 = _plan_from_tree(root, seed=999)
    assert p3.max_abs_dev <= 0.02  # 다른 seed 로도 V-A 는 지켜져야 한다


def test_group_split_raises_when_v_a_unsatisfiable(kw_tmp: Path) -> None:
    """군 하나가 split 목표보다 크면 V-A 는 EXEMPT 강등으로 복구되지 않는다 (01 §7.3 S4)."""
    root = kw_tmp / "skew"
    for gi, n in enumerate((100, 5)):
        for j in range(n):
            stem = _stem(1, 12000 + gi, 130, "20230801", gi + 1, j)
            _write_png(root / "images" / "train" / "01.한강" / f"{stem}.png", _rgb_pattern(8, 8, j))
            _write_png(
                root / "labels" / "train" / "01.한강" / f"{stem}{LABEL_SUFFIX}",
                np.zeros((8, 8), dtype=np.uint8),
            )
    with pytest.raises(RuntimeError, match="V-A violated"):
        _plan_from_tree(root)


def test_make_group_split_writes_tree_and_manifest(kw_tmp: Path) -> None:
    root = make_group_tree(kw_tmp / "grp")
    out = kw_tmp / "grp_split"
    manifest_path = make_group_split(root, out, num_classes=4)
    assert manifest_path.exists()
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "algorithm_version",
        "seed_used",
        "retries_used",
        "achieved_ratio",
        "max_abs_dev",
        "n_images_per_split",
        "n_groups_per_split",
        "per_split_class",
        "exempt_classes",
        "zero_group_pairs",
        "class_stats_global",
        "group_key_def",
        "n_files_scanned",
        "git_rev",
        "created_at",
    }
    missing = required - set(man)
    assert not missing, f"manifest missing keys: {sorted(missing)}"
    assert man["algorithm_version"] == ALGORITHM_VERSION
    assert len(man["achieved_ratio"]) == 3
    assert man["n_files_scanned"] == sum(man["n_images_per_split"])

    # 링크 트리가 실제로 원본을 가리킨다
    total = 0
    for split in SPLIT_NAMES:
        imgs = sorted((out / "images" / split).rglob("*.png"))
        total += len(imgs)
        for p in imgs[:3]:
            assert p.is_symlink()
            assert Path(p.resolve()).is_relative_to(root.resolve())
            lbl = out / "labels" / split / p.parent.name / (p.stem + LABEL_SUFFIX)
            assert lbl.is_symlink() and lbl.exists()
    assert total == man["n_files_scanned"]

    # 재실행은 덮어쓰지 않는다
    with pytest.raises(FileExistsError):
        make_group_split(root, out, num_classes=4)

    # 산출 트리를 그대로 다시 로드할 수 있다 (페어링 계약 유지)
    for split in SPLIT_NAMES:
        if (out / "images" / split).is_dir():
            assert len(find_pairs(out, split)) > 0


def test_make_group_split_rolls_back_on_failure(kw_tmp: Path) -> None:
    root = kw_tmp / "skew"
    for gi, n in enumerate((100, 5)):
        for j in range(n):
            stem = _stem(1, 12000 + gi, 130, "20230801", gi + 1, j)
            _write_png(root / "images" / "train" / "01.한강" / f"{stem}.png", _rgb_pattern(8, 8, j))
            _write_png(
                root / "labels" / "train" / "01.한강" / f"{stem}{LABEL_SUFFIX}",
                np.zeros((8, 8), dtype=np.uint8),
            )
    out = kw_tmp / "never"
    with pytest.raises(RuntimeError):
        make_group_split(root, out, num_classes=4)
    assert not out.exists(), "failed run must leave no partial tree behind"


@pytest.mark.data
@pytest.mark.skipif(
    not (VERIFY_DIR / "cache_A_presence.npy").exists(),
    reason="methods/_verify 전수 스캔 캐시 없음",
)
def test_group_split_reproduces_measured_m14_on_real_data() -> None:
    """★ 실데이터 회귀 — 정정 A-1 `[M14]` 수치를 그대로 재현해야 한다.

    96,340장 라벨을 다시 스캔하지 않고 `methods/_verify` 전수 스캔 캐시를 쓴다
    (`scan_groups_A.py` 산출물, 읽기 전용).
    """
    z = np.load(VERIFY_DIR / "cache_A_index.npz", allow_pickle=False)
    presence = np.load(VERIFY_DIR / "cache_A_presence.npy")[:, :12].astype(bool)
    records = [
        {"group": str(g), "scene": str(s)} for g, s in zip(z["group"], z["scene"])
    ]
    table = build_group_table(records, presence)
    assert len(table.keys) == 174 and table.n_files_scanned == 96340

    plan = plan_group_split(table)
    assert plan.n_images_per_split == (67438, 14454, 14448)
    assert [round(v * 100, 3) for v in plan.achieved_ratio] == [70.0, 15.003, 14.997]
    assert plan.max_abs_dev < 1e-4
    assert plan.n_groups_per_split == (129, 21, 24)
    assert plan.exempt == (8,)  # 목장 = 8군 < 3·MIN_GROUPS
    assert plan.retries_used == 0 and plan.seed_used == 20260731
    assert plan.violations == []


# ═══════════════════════════════════════════════════════════════════════════
#  E. build.py
# ═══════════════════════════════════════════════════════════════════════════
def test_assert_resolution_contract() -> None:
    """config V1/V10 과 **이중 방어**다. 여기서는 config 검증을 우회해 직접 주입한다."""
    cfg = default_config()
    assert_resolution_contract(cfg)  # 512 / 1024 는 통과

    bad = default_config()
    bad.data.train_size = [500, 500]  # __post_init__ 이후 주입 (V1 우회)
    with pytest.raises(ValueError, match="multiple of 32"):
        assert_resolution_contract(bad)

    missing_radius = default_config()
    missing_radius.data.train_size = [256, 256]  # radius 매핑에 256 항목이 없다
    with pytest.raises(ValueError, match="boundary.radius"):
        assert_resolution_contract(missing_radius)


def test_build_datasets_and_dataloaders_aihub092(kw_tmp: Path) -> None:
    root = make_aihub092_tree(kw_tmp / "tree", n_per_split=4, size=64)
    cfg = from_dict(
        {
            "seed": 7,
            "data": {
                "dataset": "aihub092",
                "root": str(root),
                "num_classes": 4,
                "train_size": [64, 64],
                "eval_size": [64, 64],
                "num_workers": 0,
                "pin_memory": False,
                "boundary": {"radius": {64: 1}},
                "augment": {"geometric": {"scale_range": [1.0, 1.0]}},
            },
            "schedule": {"batch_size": 2},
        }
    )
    ds = build_datasets(cfg)
    assert set(ds) == {"train", "val", "test"}
    assert len(ds["train"]) == 8
    # 증강은 train 에만
    assert ds["train"].geometric is not None and ds["train"].photometric is not None
    assert ds["val"].geometric is None and ds["val"].photometric is None

    loaders = build_dataloaders(cfg, ds)
    batch = next(iter(loaders["train"]))
    assert batch["rgb"].shape == (2, 3, 64, 64)
    assert batch["y_seg"].shape == (2, 64, 64)
    assert batch["y_edge"].shape == (2, 1, 16, 16)
    assert len(batch["meta"]) == 2
    assert loaders["train"].drop_last is True and loaders["val"].drop_last is False


def test_build_dataloaders_rfs_sampler(kw_tmp: Path) -> None:
    root = make_aihub092_tree(kw_tmp / "tree", n_per_split=4, size=32, num_classes=4)
    cfg = from_dict(
        {
            "data": {
                "dataset": "aihub092",
                "root": str(root),
                "num_classes": 4,
                "train_size": [32, 32],
                "eval_size": [32, 32],
                "num_workers": 0,
                "pin_memory": False,
                "boundary": {"radius": {32: 1}},
                "augment": {"geometric": {"scale_range": [1.0, 1.0]}},
                "sampler": {"kind": "rfs", "repeat_t": 0.5},
            },
            "schedule": {"batch_size": 2},
        }
    )
    loaders = build_dataloaders(cfg, build_datasets(cfg))
    from bloomnet.data.samplers import RepeatFactorSampler

    assert isinstance(loaders["train"].sampler, RepeatFactorSampler)
    assert loaders["val"].sampler is not None  # SequentialSampler
    assert len(loaders["train"].sampler) >= 8


def test_build_datasets_k235(kw_tmp: Path) -> None:
    src = make_k235_source(kw_tmp / "k235src", n_chips=60)
    npz = build_k235_cache(src, kw_tmp / "cache" / "k235.npz", band_order_check="off")
    cfg = from_dict(
        {
            "mode": "s0_spec",
            "data": {"dataset": "k235", "num_workers": 0, "pin_memory": False},
            "spec": {
                "npz": str(npz),
                "train_sites": ["ABC"],
                "val_sites": ["ABC"],
                "band_order": ["blue", "green", "red", "rededge1", "nir"],
            },
            "schedule": {"batch_size": 4},
        }
    )
    ds = build_datasets(cfg)
    assert set(ds) == {"train", "val", "test"}
    loaders = build_dataloaders(cfg, ds)
    batch = next(iter(loaders["train"]))
    assert batch["x"].shape == (4, 8, 3, 3)
    assert batch["m"].shape == (4, 6)
    assert batch["y_chl_scalar"].shape == (4,)
    assert len(batch["meta"]) == 4


def test_build_datasets_rejects_unknown_kind() -> None:
    cfg = default_config()
    cfg.data.dataset = "nope"
    with pytest.raises(ValueError, match="unknown data.dataset"):
        build_datasets(cfg)
