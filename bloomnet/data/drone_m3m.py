"""M3M / M300 실비행 어댑터 — **라벨 없음, 진단 전용** (01 §8.4 D1~D3, 06 §2.1.3, 레벨 L2).

이 데이터에는 어떤 종류의 라벨도 없다(분석 §C.3: 4,949 프레임 / 주석 0건).
따라서 이 모듈은 **학습 로더가 아니라 진단 로더**다. 제공하는 것:

D1  4밴드에 R1~R4(+R4′, R6)를 적용한 뒤 NDCI / MCI_norm 분포 산출
    → 235(RedEdge 717) 대비 M3M(RE 730)의 분포 이동량 (01 §4.2.4 한계의 정량화)
D2  RGB 프레임의 "밝고 무채색" 비율 분포 (01 §5.2 specular proxy)
    → 슬라이드 16의 "glint 3~15 %" 주장을 실제 수역·태양각에서 검증
D3  2026-05-15(맑음) vs 2026-05-28(흐림) 의 위 두 분포 차이 → 조도 도메인 시프트 크기

★ 실측 발견 (본 구현 중 CPU 재검산, 아래 :data:`EXPOSURE_TIME_SCALE` 주석 참조)
    01 §2.6 [M13] 의 ``rho_rel`` 표는 **R3(노출/게인)을 적용하지 않은 값**이다.
    XMP ``ExposureTime`` 은 마이크로초이므로 R3 까지 정직하게 적용하면
    ``rho_rel`` 중앙값은 ``O(1e−2)`` 가 되어 235 절대 반사율과 이미 같은 자릿수다
    (실측 0.0122). 즉 ``K_SENSOR["m3m"] = 3.5e3`` 은 R3 누락에서 온 값이며,
    그대로 곱하면 [H1] 하드 검증(1e−3 < median < 1.0)을 **상한에서** 위반한다.
    본 모듈은 계약대로 `constants.K_SENSOR` 를 기본값으로 쓰고 위반 시 죽는다
    (01 §2.6 [H1]: "조용한 열화를 허용하지 않는 유일한 예외"). 재산출은 [U-11] /
    ``tools/calibrate_k_sensor.py`` 소관이다.

레벨 L2 — L−1(`constants`), L1(`data.bundle`/`data.indices`) 만 import 한다.
"""

from __future__ import annotations

import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from bloomnet.constants import (
    BAND_CENTERS_NM,
    IGNORE_INDEX,
    K_SENSOR,
    MODALITY_ORDER,
    MSI_SLOTS,
    SENSOR_BAND_IDS,
)
from bloomnet.data.bundle import assert_availability_contract
from bloomnet.data.indices import (
    apply_k_sensor,
    assert_msi_scale_contract,
    canonical_scatter_np,
    compute_bio_canonical,
    coregister_m3m,
    mci_coefficient,
    normalize_imagenet,
    specular_proxy,
)

__all__ = [
    "M3M_BAND_SUFFIXES",
    "M3M_BAND_TO_SLOT",
    "EXPOSURE_TIME_SCALE",
    "M3MFrame",
    "parse_xmp",
    "read_ms_tiff",
    "find_m3m_frames",
    "find_m300_frames",
    "frame_reflectance",
    "DroneM3MDataset",
    "bio_index_stats",
    "glint_stats",
    "illumination_shift_report",
]

# 파일 접미사 → canonical slot 이름 (분석 §C.1 XMP BandName/BandFreq 실측)
M3M_BAND_SUFFIXES: Tuple[str, ...] = ("G", "R", "RE", "NIR")
M3M_BAND_TO_SLOT: Dict[str, str] = {
    "G": "green",
    "R": "red",
    "RE": "rededge1",
    "NIR": "nir",
}

#: XMP ``ExposureTime`` 의 단위 → 초.  DJI M3M 은 **마이크로초**로 기록한다
#: (실측 1053 → 1.053 ms, 나디르 주광 촬영으로 타당). 이 환산을 빼면 R3 가
#: 1e6 배 어긋나 ``rho_rel`` 이 ``O(1e−8)`` 이 된다.
EXPOSURE_TIME_SCALE: float = 1.0e-6

# ★ RGB 와 MS 의 **타임스탬프가 1초 어긋나는 프레임이 존재한다** (2026-05-28 흐림 세트에서
#   1,118장 중 48장 실측: `DJI_20260528122323_0090_D.JPG` ↔ `DJI_20260528122324_0090_MS_G.TIF`).
#   따라서 페어링 키는 전체 stem 이 아니라 **비행 폴더 + 시퀀스 번호**여야 한다.
_RGB_RE = re.compile(r"^DJI_(?P<ts>\d+)_(?P<seq>\d+)_D\.JPG$", re.IGNORECASE)
_MS_RE = re.compile(
    r"^DJI_(?P<ts>\d+)_(?P<seq>\d+)_MS_(?P<band>G|R|RE|NIR)\.TIF$", re.IGNORECASE
)

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_TAG = {
    "width": 256,
    "length": 257,
    "bits": 258,
    "compression": 259,
    "strip_offsets": 273,
    "samples": 277,
    "strip_bytes": 279,
    "planar": 284,
    "sample_format": 339,
}


# ─────────────────────────────────────────────────────────────────────────────
# XMP / TIFF
# ─────────────────────────────────────────────────────────────────────────────
def parse_xmp(buf: bytes) -> Dict[str, str]:
    """DJI XMP 패킷에서 ``drone-dji:*`` 속성을 뽑는다 (분석 §C.1 실측 키).

    XML 파서를 쓰지 않는 이유: DJI 패킷은 네임스페이스 선언이 rdf:Description 속성에
    섞여 있고 파일마다 미묘하게 다르다. 우리가 쓰는 것은 평면 속성 몇 개뿐이므로
    정규식이 더 견고하다.
    """
    i = buf.find(b"<x:xmpmeta")
    j = buf.find(b"</x:xmpmeta>")
    if i < 0 or j < 0:
        return {}
    text = buf[i : j + 12].decode("utf-8", errors="replace")
    out = dict(re.findall(r'drone-dji:(\w+)="([^"]*)"', text))
    out.update(re.findall(r"<drone-dji:(\w+)>([^<]*)</drone-dji:\1>", text))
    return out


def _read_ifd(buf: bytes) -> Tuple[str, Dict[int, Tuple[int, int, bytes]]]:
    bo = buf[:2]
    if bo == b"II":
        end = "<"
    elif bo == b"MM":
        end = ">"
    else:
        raise RuntimeError(f"not a TIFF: byte order {bo!r}")
    magic, ifd_off = struct.unpack(end + "HI", buf[2:8])
    if magic != 42:
        raise RuntimeError(f"not a classic TIFF: magic={magic}")
    (count,) = struct.unpack(end + "H", buf[ifd_off : ifd_off + 2])
    tags: Dict[int, Tuple[int, int, bytes]] = {}
    for i in range(count):
        off = ifd_off + 2 + i * 12
        tag, typ, num = struct.unpack(end + "HHI", buf[off : off + 8])
        size = _TYPE_SIZE.get(typ, 1) * num
        if size <= 4:
            raw = buf[off + 8 : off + 8 + size]
        else:
            (voff,) = struct.unpack(end + "I", buf[off + 8 : off + 12])
            raw = buf[voff : voff + size]
        tags[tag] = (typ, num, raw)
    return end, tags


def _ints(end: str, entry: Optional[Tuple[int, int, bytes]]) -> List[int]:
    if entry is None:
        return []
    typ, num, raw = entry
    fmt = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i"}.get(typ)
    if fmt is None:
        raise RuntimeError(f"unsupported integer TIFF type {typ}")
    return list(struct.unpack(end + fmt * num, raw[: _TYPE_SIZE[typ] * num]))


def read_ms_tiff(path: Path) -> Tuple[np.ndarray, Dict[str, str]]:
    """M3M MS TIFF → ``((H,W) uint16 DN, XMP dict)``. 압축 없는 단일 밴드만 지원한다."""
    buf = Path(path).read_bytes()
    end, tags = _read_ifd(buf)
    w = _ints(end, tags.get(_TAG["width"]))
    h = _ints(end, tags.get(_TAG["length"]))
    spp = _ints(end, tags.get(_TAG["samples"])) or [1]
    bits = _ints(end, tags.get(_TAG["bits"])) or [16]
    comp = _ints(end, tags.get(_TAG["compression"])) or [1]
    offs = _ints(end, tags.get(_TAG["strip_offsets"]))
    cnts = _ints(end, tags.get(_TAG["strip_bytes"]))
    if comp[0] != 1:
        raise RuntimeError(f"M3M MS TIFF must be uncompressed, got Compression={comp[0]}")
    if spp[0] != 1 or bits[0] != 16:
        raise RuntimeError(f"M3M MS TIFF must be 1×uint16, got spp={spp[0]} bits={bits[0]}")
    need = int(w[0]) * int(h[0]) * 2
    data = b"".join(buf[o : o + c] for o, c in zip(offs, cnts or [need] * len(offs)))
    if len(data) < need:
        raise RuntimeError(f"M3M MS TIFF: strip data {len(data)} B < required {need} B")
    dt = "<u2" if end == "<" else ">u2"
    arr = np.frombuffer(data[:need], dtype=dt).reshape(int(h[0]), int(w[0]))
    return np.ascontiguousarray(arr), parse_xmp(buf)


# ─────────────────────────────────────────────────────────────────────────────
# 프레임 인덱싱
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class M3MFrame:
    """RGB 1장 + MS 4밴드 1세트 (분석 §C.1: 모든 RGB 프레임이 완전한 4밴드 세트를 갖는다)."""

    seq: str
    flight: str
    rgb: Optional[Path] = None
    bands: Dict[str, Path] = field(default_factory=dict)
    stem: str = ""

    @property
    def complete(self) -> bool:
        return all(b in self.bands for b in M3M_BAND_SUFFIXES) and self.rgb is not None


def find_m3m_frames(root: Path, *, require_complete: bool = True) -> List[M3MFrame]:
    """``root`` 아래 M3M 프레임을 ``(flight, seq)`` 사전순으로 인덱싱한다.

    페어링 키가 전체 stem 이 아니라 ``(비행 폴더, 시퀀스 번호)`` 인 이유는 :data:`_RGB_RE`
    주석 참조 — RGB/MS 타임스탬프가 1초 어긋나는 프레임이 실제로 있다.

    Raises:
        RuntimeError: ``require_complete`` 인데 4밴드 또는 RGB 가 빠진 프레임이 있을 때
            (분석 §C.1 은 ``TIF = 4 × JPG`` 를 실측했다 — 어긋나면 데이터 이상이다).
    """
    root = Path(root)
    # (flight, seq) 안에서도 시퀀스 번호가 **재시작**하는 비행이 있다 (2026-05-28 정자앞:
    # JPG 199장 / 고유 seq 145 → 54장 중복). 따라서 같은 seq 안에서는 타임스탬프 순으로
    # 등장 순서를 맞춰 짝짓는다. RGB/MS 의 ts 차이는 1초 이내이고 세션 간격은 분 단위다.
    rgb_by: Dict[Tuple[str, str], List[Tuple[str, Path]]] = defaultdict(list)
    ms_by: Dict[Tuple[str, str, str], List[Tuple[str, Path]]] = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        flight = p.parent.name
        m = _MS_RE.match(p.name)
        if m is not None:
            ms_by[(flight, m.group("seq"), m.group("band").upper())].append((m.group("ts"), p))
            continue
        m = _RGB_RE.match(p.name)
        if m is not None:
            rgb_by[(flight, m.group("seq"))].append((m.group("ts"), p))

    keys = sorted({(f, s) for (f, s) in rgb_by} | {(f, s) for (f, s, _b) in ms_by})
    out: List[M3MFrame] = []
    for flight, seq in keys:
        rgbs = sorted(rgb_by.get((flight, seq), []))
        per_band = {b: sorted(ms_by.get((flight, seq, b), [])) for b in M3M_BAND_SUFFIXES}
        n = max([len(rgbs)] + [len(v) for v in per_band.values()])
        for i in range(n):
            fr = M3MFrame(seq=seq, flight=flight)
            if i < len(rgbs):
                fr.rgb = rgbs[i][1]
                fr.stem = fr.rgb.stem[:-2] if fr.rgb.stem.endswith("_D") else fr.rgb.stem
            for b, lst in per_band.items():
                if i < len(lst):
                    fr.bands[b] = lst[i][1]
                    if not fr.stem:
                        fr.stem = lst[i][1].name.split("_MS_")[0]
            out.append(fr)
    if require_complete:
        bad = [f"{f.flight}/{f.seq}" for f in out if not f.complete]
        if bad:
            raise RuntimeError(
                f"M3M frame index incomplete: {len(bad)} frame(s) missing RGB or MS bands, "
                f"first={bad[:3]} (analysis §C.1 measured TIF == 4 × JPG)"
            )
    return out


def find_m300_frames(root: Path) -> List[Path]:
    """M300(Zenmuse P1)은 **RGB 전용**이다 (분석 §C.1: TIF 0개, 열화상 없음)."""
    return sorted(p for p in Path(root).rglob("*.JPG") if p.is_file())


# ─────────────────────────────────────────────────────────────────────────────
# R1~R4 (+R4′, R6)
# ─────────────────────────────────────────────────────────────────────────────
def _xmp_floats(xmp: Dict[str, str], key: str) -> List[float]:
    raw = xmp.get(key, "")
    return [float(v) for v in re.split(r"[,\s;]+", raw.strip()) if v]


def _band_rho_rel(
    dn: np.ndarray, xmp: Dict[str, str], *, exposure_time_scale: float
) -> np.ndarray:
    """단일 밴드 R1→R4. ``indices.m3m_dn_to_relative_reflectance`` 단일 구현에 위임한다."""
    from bloomnet.data.indices import m3m_dn_to_relative_reflectance

    vg = _xmp_floats(xmp, "VignettingData")
    cx = float(xmp.get("CalibratedOpticalCenterX", dn.shape[1] / 2.0))
    cy = float(xmp.get("CalibratedOpticalCenterY", dn.shape[0] / 2.0))
    cx += float(xmp.get("RelativeOpticalCenterX", 0.0))
    cy += float(xmp.get("RelativeOpticalCenterY", 0.0))
    return m3m_dn_to_relative_reflectance(
        dn,
        black_level=int(float(xmp.get("BlackLevel", 3200))),
        vignetting_coeffs=vg if len(vg) == 6 else [0.0] * 6,
        optical_center=(cx, cy),
        exposure_time=float(xmp.get("ExposureTime", 1.0)) * float(exposure_time_scale),
        sensor_gain=float(xmp.get("SensorGain", 1.0)),
        sensor_gain_adj=float(xmp.get("SensorGainAdjustment", 1.0)),
        irradiance=float(xmp.get("Irradiance", 1.0)),
    )


def _downsample(a: np.ndarray, factor: int) -> np.ndarray:
    """블록 평균 축소 (안티에일리어싱). ``factor==1`` 이면 그대로."""
    f = int(factor)
    if f <= 1:
        return a
    h, w = a.shape[-2:]
    hh, ww = (h // f) * f, (w // f) * f
    return a[..., :hh, :ww].reshape(a.shape[:-2] + (hh // f, f, ww // f, f)).mean(axis=(-3, -1))


def _scaled_homography(hm: np.ndarray, factor: int) -> np.ndarray:
    """축소 격자용 호모그래피 ``S H S⁻¹`` (S = diag(1/f, 1/f, 1))."""
    f = float(factor)
    if f == 1.0:
        return hm
    s = np.diag([1.0 / f, 1.0 / f, 1.0])
    s_inv = np.diag([f, f, 1.0])
    return s @ hm @ s_inv


def frame_reflectance(
    frame: M3MFrame,
    *,
    downsample: int = 4,
    exposure_time_scale: float = EXPOSURE_TIME_SCALE,
    apply_r6: bool = True,
    apply_r4p: bool = True,
    k_sensor: Optional[float] = None,
    check_scale: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """M3M 4밴드 → ``(4, H, W) float32`` 반사율 (순서 = :data:`M3M_BAND_SUFFIXES`).

    R1 dark+range → R2 vignetting → R3 노출/게인 → R4 일사 정규화
    → (선택) R6 밴드 정합 → (선택) R4′ ``× k_sensor`` → [H1] 스케일 하드 검증.

    Args:
        downsample: 블록 평균 축소 배율. R1~R4 는 **원 해상도**에서 계산한 뒤 축소한다
            (R2 의 반경이 픽셀 좌표에 종속되므로 먼저 축소하면 틀린다).
        apply_r6: ``CalibratedHMatrix`` 로 4밴드를 공통 보정 격자에 warp.
            ★ 이 행렬의 방향은 DJI 문서 부재로 **가정**이다(band→common 으로 보고 역행렬을 쓴다).
        k_sensor: ``None`` → ``K_SENSOR[sensor]``. 정정 A-2 / X-28.
        check_scale: [H1] ``1e-3 < median(msi[msi>0]) < 1.0`` 하드 검증.

    Returns:
        ``(rho (4,H,W) float32, info dict)``. ``info`` 는 밴드별 XMP 요약 + 측정 median.
    """
    bands: List[np.ndarray] = []
    homs: List[np.ndarray] = []
    info: Dict[str, Any] = {"stem": frame.stem, "flight": frame.flight, "bands": {}}
    for suffix in M3M_BAND_SUFFIXES:
        path = frame.bands.get(suffix)
        if path is None:
            raise RuntimeError(f"frame {frame.stem}: band {suffix} missing")
        dn, xmp = read_ms_tiff(path)
        rho = _band_rho_rel(dn, xmp, exposure_time_scale=exposure_time_scale)
        rho = _downsample(rho, downsample)
        bands.append(rho)
        hm = _xmp_floats(xmp, "CalibratedHMatrix")
        homs.append(
            np.linalg.inv(np.asarray(hm, dtype=np.float64).reshape(3, 3))
            if len(hm) == 9
            else np.eye(3)
        )
        info["bands"][suffix] = {
            "band_name": xmp.get("BandName"),
            "band_freq": xmp.get("BandFreq"),
            "irradiance": float(xmp.get("Irradiance", float("nan"))),
            "exposure_time": float(xmp.get("ExposureTime", float("nan"))),
            "sensor_gain": float(xmp.get("SensorGain", float("nan"))),
            "sensor_gain_adj": float(xmp.get("SensorGainAdjustment", float("nan"))),
            "dn_mean": float(dn.mean()),
            "rho_rel_median": float(np.median(rho[rho > 0])) if np.any(rho > 0) else float("nan"),
        }
        info.setdefault("relative_altitude_m", _safe_float(xmp.get("RelativeAltitude")))
        info.setdefault("utc_at_exposure", xmp.get("UTCAtExposure"))
        info.setdefault("calibrated_focal_length", _safe_float(xmp.get("CalibratedFocalLength")))

    if apply_r6:
        homs = [_scaled_homography(h, downsample) for h in homs]
        arr = coregister_m3m(bands, homs)
    else:
        arr = np.stack(bands).astype(np.float32)

    if apply_r4p:
        arr = np.asarray(apply_k_sensor(arr, "m3m", k_sensor=k_sensor), dtype=np.float32)
    _k = float(K_SENSOR["m3m"] if k_sensor is None else k_sensor)
    info["k_sensor"] = _k if apply_r4p else 1.0
    pos = arr[arr > 0]
    info["msi_median"] = float(np.median(pos)) if pos.size else float("nan")
    if check_scale:
        assert_msi_scale_contract(arr)
    return arr, info


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _gsd_m(relative_altitude_m: Optional[float], focal_px: Optional[float]) -> Optional[float]:
    """GSD = RelativeAltitude / CalibratedFocalLength (01 §6.7 T3 근거식)."""
    if not relative_altitude_m or not focal_px:
        return None
    return float(relative_altitude_m) / float(focal_px)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (진단 전용)
# ─────────────────────────────────────────────────────────────────────────────
class DroneM3MDataset(torch.utils.data.Dataset):
    """M3M 실비행 프레임 — **라벨 없음**. 01 §8.4 D1~D3 진단용 Sample 스키마 어댑터.

    ``y_seg`` 는 전부 ``ignore_index`` 이고 모든 ``*_valid`` 는 False 다 (A6).
    학습에 쓰면 손실이 전부 0 이 되므로, 이 클래스는 평가/진단 경로에서만 쓴다.

    Args:
        root: 비행 폴더 루트 (예: ``…/20260515_사전테스트(M3M)``).
        out_hw: RGB·MSI 를 맞출 공통 출력 크기. MS(2592×1944)와 RGB(5280×3956)는
            해상도도 FOV 도 다르므로 정합이 아니라 **리사이즈**다 — 진단 전용임을 뜻한다.
        downsample: MS 밴드 R1~R4 이후 블록 평균 배율.
    """

    def __init__(
        self,
        root: str,
        *,
        out_hw: Tuple[int, int] = (512, 512),
        downsample: int = 4,
        sensor: str = "m3m",
        active_modalities: Sequence[str] = ("rgb", "msi"),
        bio_source: str = "msi",
        bio_kind: str = "mci",
        ignore_index: int = IGNORE_INDEX,
        k_sensor: Optional[float] = None,
        apply_r6: bool = True,
        check_scale: bool = True,
        strict_contract: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        if "rgb" not in active_modalities:
            raise ValueError("A2: 'rgb' must always be active")
        if bio_source == "msi" and "msi" not in active_modalities:
            raise ValueError(
                "A3: bio_source='msi' requires 'msi' in active_modalities "
                "(bio 는 msi 의 부속 채널이다, 01 §2.2)"
            )
        self.root = Path(root)
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.downsample = int(downsample)
        self.sensor = str(sensor)
        self.active_modalities = tuple(active_modalities)
        self.bio_source = str(bio_source)
        self.bio_kind = str(bio_kind)
        self.ignore_index = int(ignore_index)
        self.k_sensor = k_sensor
        self.apply_r6 = bool(apply_r6)
        self.check_scale = bool(check_scale)
        self.strict_contract = bool(strict_contract)
        self.frames = find_m3m_frames(self.root)
        if limit is not None:
            self.frames = self.frames[: int(limit)]
        if not self.frames:
            raise RuntimeError(f"no M3M frame under {self.root}")
        self.band_ids: Tuple[int, ...] = tuple(SENSOR_BAND_IDS[self.sensor])
        centers = BAND_CENTERS_NM[self.sensor]
        self.mci_c = mci_coefficient(centers["red"], centers["rededge1"], centers["nir"])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        frame = self.frames[index]
        rho, info = frame_reflectance(
            frame,
            downsample=self.downsample,
            apply_r6=self.apply_r6,
            k_sensor=self.k_sensor,
            check_scale=self.check_scale,
        )
        msi = _resize_np(rho, self.out_hw)  # (4,H,W)
        rgb01 = self._load_rgb(frame)

        s, m = canonical_scatter_np(msi, self.band_ids, len(MSI_SLOTS))
        bio = compute_bio_canonical(
            torch.from_numpy(s)[None], torch.from_numpy(m)[None], mci_c=self.mci_c, kind="mci"
        )[0]

        h, w = self.out_hw
        avail = torch.zeros(len(MODALITY_ORDER), dtype=torch.float32)
        avail[0] = 1.0  # A2
        sample: Dict[str, Any] = {"rgb": normalize_imagenet(rgb01).contiguous()}
        if "msi" in self.active_modalities:
            sample["msi"] = torch.from_numpy(msi).contiguous()
            avail[MODALITY_ORDER.index("msi")] = 1.0
        if self.bio_source != "none":
            sample["bio"] = bio.contiguous()
            avail[MODALITY_ORDER.index("bio")] = 1.0
        sample["avail"] = avail

        sample["y_seg"] = torch.full((h, w), self.ignore_index, dtype=torch.int64)
        sample["y_edge"] = torch.zeros(1, h // 4, w // 4, dtype=torch.float32)
        sample["y_edge_valid"] = torch.zeros(1, h // 4, w // 4, dtype=torch.bool)
        sample["y_chl"] = torch.zeros(1, h, w, dtype=torch.float32)
        sample["y_chl_valid"] = torch.zeros(1, h, w, dtype=torch.bool)
        sample["y_chl_scalar"] = torch.zeros((), dtype=torch.float32)
        sample["y_chl_scalar_valid"] = torch.zeros((), dtype=torch.bool)
        sample["meta"] = {
            "stem": frame.stem,
            "scene": frame.flight,
            "group_key": (frame.flight, "", "", ""),
            "split": "diagnostic",
            "site": frame.flight,
            "date": str(info.get("utc_at_exposure") or "")[:10].replace("-", ""),
            "flight_line": frame.flight,
            "alt_m": info.get("relative_altitude_m"),
            "gsd_m": _gsd_m(
                info.get("relative_altitude_m"), info.get("calibrated_focal_length")
            ),
            "sensor": self.sensor,
            "band_ids": tuple(self.band_ids),
            "phys_slot_ids": (),
            "band_centers_nm": dict(BAND_CENTERS_NM[self.sensor]),
            "bio_kind": self.bio_kind,
            "chl_space": "log1p",
            "aug": {},
            "msi_median": info["msi_median"],
        }
        if self.strict_contract:
            assert_availability_contract(
                sample, bio_source=self.bio_source, active_modalities=self._sample_modalities()
            )
        return sample

    def _sample_modalities(self) -> Tuple[str, ...]:
        mods = set(self.active_modalities)
        if self.bio_source != "none":
            mods.add("bio")
        return tuple(m for m in MODALITY_ORDER if m in mods)

    def _load_rgb(self, frame: M3MFrame) -> Tensor:
        from PIL import Image

        if frame.rgb is None:
            raise RuntimeError(f"frame {frame.stem} has no RGB (_D.JPG)")
        with Image.open(frame.rgb) as im:
            im = im.convert("RGB").resize((self.out_hw[1], self.out_hw[0]), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
        t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return t.to(torch.float32) / 255.0


def _resize_np(a: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """``(C,H,W)`` 이중선형 리사이즈 (torch 경유, CPU)."""
    t = torch.from_numpy(np.ascontiguousarray(a)).to(torch.float32)[None]
    out = torch.nn.functional.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return out[0].numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 진단 D1 / D2 / D3
# ─────────────────────────────────────────────────────────────────────────────
def _percentiles(x: np.ndarray) -> Dict[str, float]:
    q = np.percentile(x, [1, 25, 50, 75, 99])
    return {
        "p01": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p99": float(q[4]),
        "mean": float(x.mean()),
        "iqr": float(q[3] - q[1]),
    }


def bio_index_stats(
    frames: Sequence[M3MFrame],
    *,
    downsample: int = 8,
    k_sensor: Optional[float] = None,
    check_scale: bool = False,
    sensor: str = "m3m",
) -> Dict[str, Any]:
    """D1 — NDCI / MCI_norm 분포. 235(RedEdge 717) 대비 이동량 정량화용.

    ``check_scale=False`` 가 기본인 이유: 지수는 밴드 비율이라 ``k_sensor`` 에 불변이므로
    D1 은 스케일 정합 [U-11] 이 끝나기 전에도 유효하다 (01 §2.6 [J] NDCI 불변성).

    ★ R6 워프의 격자 밖 화소는 0 으로 채워지므로 ``(0−R)/(0+R) = −1`` 같은 인공 극단값이
      테두리에 생긴다. 4밴드가 모두 양수인 화소만 집계한다.
    """
    centers = BAND_CENTERS_NM[sensor]
    c = mci_coefficient(centers["red"], centers["rededge1"], centers["nir"])
    band_ids = SENSOR_BAND_IDS[sensor]
    ndci_all: List[np.ndarray] = []
    mci_all: List[np.ndarray] = []
    for fr in frames:
        rho, _ = frame_reflectance(
            fr, downsample=downsample, k_sensor=k_sensor, check_scale=check_scale
        )
        s, m = canonical_scatter_np(rho, band_ids, len(MSI_SLOTS))
        bio = compute_bio_canonical(
            torch.from_numpy(s)[None], torch.from_numpy(m)[None], mci_c=c, kind="mci"
        )[0].numpy()
        valid = (rho > 0).all(axis=0).ravel()
        ndci_all.append(bio[0].ravel()[valid])
        mci_all.append(bio[1].ravel()[valid])
    ndci = np.concatenate(ndci_all)
    mci = np.concatenate(mci_all)
    return {
        "n_frames": len(frames),
        "n_pixels": int(ndci.size),
        "mci_c": c,
        "ndci": _percentiles(ndci),
        "mci_norm": _percentiles(mci),
    }


def glint_stats(
    rgb_paths: Sequence[Path],
    *,
    out_hw: Tuple[int, int] = (512, 512),
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """D2 — "밝고 무채색" 픽셀 비율 (01 §5.2.1 specular proxy ``Vmax·(1−Sat)``).

    슬라이드 16 의 "glint 3~15 %" 주장에 대응하는 실측 분포를 만든다.
    ★ 물리적 glint 가 아니라 이색성 반사 모형의 무채색 성분 대리값이다.
    """
    from PIL import Image

    ratios: List[float] = []
    values: List[np.ndarray] = []
    for p in rgb_paths:
        with Image.open(p) as im:
            im = im.convert("RGB").resize((out_hw[1], out_hw[0]), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None]
        sp = specular_proxy(t)[0, 0].numpy()
        ratios.append(float((sp > threshold).mean()))
        values.append(sp.ravel()[::37])  # 희석 표본 (분포 추정에 충분)
    val = np.concatenate(values) if values else np.zeros(1, dtype=np.float32)
    return {
        "n_frames": len(rgb_paths),
        "threshold": float(threshold),
        "bright_achromatic_ratio": _percentiles(np.asarray(ratios, dtype=np.float64))
        if ratios
        else {},
        "specular_proxy": _percentiles(val),
    }


def illumination_shift_report(
    clear: Dict[str, Any], overcast: Dict[str, Any], *, key: str = "ndci"
) -> Dict[str, float]:
    """D3 — 맑음/흐림 두 세트의 분포 차이 (중앙값·IQR 이동량)."""
    a, b = clear[key], overcast[key]
    return {
        "delta_median": float(b["p50"] - a["p50"]),
        "delta_iqr": float(b["iqr"] - a["iqr"]),
        "clear_median": float(a["p50"]),
        "overcast_median": float(b["p50"]),
    }
