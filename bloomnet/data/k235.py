"""235 상수원·녹조 5밴드 칩 어댑터 (S0-Spec) — 01 §6.2/§6.3/§7.6, 06 §3.2.5 (레벨 L2).

담당:
    * :func:`read_chip_tiff`   — **의존성 없는 raw TIFF IFD 파서** (3×3×5 float32).
    * :func:`assert_band_order` — [U-1] H2 가정의 자기정합성 검사 (식생 칩 NIR > 1.3·RE1).
    * :func:`build_k235_cache` — 16개 zip → 단일 ``.npz`` (~1 MB). zip 압축 해제 금지.
    * :class:`K235ChipDataset` — canonical 8ch(6 slot + [NDCI, MCI_norm]) 입력.

★ 밴드 순서는 확정이 아니다. ``K235_BAND_ORDER_DEFAULT`` 는 가설 H2 이며,
  06 V18 이 ``spec.band_order_confirmed`` 없이 가중치 이식을 금지한다.
  :func:`assert_band_order` 는 H2 가정 하의 **자기정합성** 검사일 뿐 외부 문서 확정이 아니다.

레벨 L2 — L−1(`constants`), L1(`data.indices`) 만 import 한다.
"""

from __future__ import annotations

import json
import re
import struct
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from bloomnet.constants import (
    K235_BAND_ORDER_DEFAULT,
    MCI_C,
    MSI_SLOT_ID,
    MSI_SLOTS,
)
from bloomnet.data.indices import assert_msi_scale_contract, canonical_scatter_np

__all__ = [
    "CHIP_HW",
    "N_BANDS",
    "QC_WATERTEMP_MAX",
    "QC_ODOMG_MAX",
    "NPZ_SCALAR_FIELDS",
    "read_chip_tiff",
    "read_chip_geojson",
    "assert_band_order",
    "band_order_evidence",
    "build_k235_cache",
    "K235ChipDataset",
]

CHIP_HW: Tuple[int, int] = (3, 3)
N_BANDS: int = 5

# 01 §6.3 [L1] 하드 필터.  K235_QC = "WaterTemp <= 40.0 and ODOMg <= 90.0" 의 실행형.
QC_WATERTEMP_MAX: float = 40.0
QC_ODOMG_MAX: float = 90.0

NPZ_SCALAR_FIELDS: Tuple[str, ...] = (
    "chla",
    "phyco",
    "watertemp",
    "odomg",
    "ec",
    "odos",
    "temp",
    "humidity",
    "windspeed",
)
_GEOJSON_KEY = {
    "chla": "Chla",
    "phyco": "Phyco",
    "watertemp": "WaterTemp",
    "odomg": "ODOMg",
    "ec": "EC",
    "odos": "ODOS",
    "temp": "Temp",
    "humidity": "Humidity",
    "windspeed": "Windspeed",
}

_MS_ZIP_RE = re.compile(r"_2\.[^_]*_1\.다중분광_\d+\.[^(]*\(([A-Z]{3})\)\.zip$")

# ─── TIFF 태그 (01 §6.2 [P1] 이 명시한 것만) ────────────────────────────────
_TAG_WIDTH = 256
_TAG_LENGTH = 257
_TAG_BITS = 258
_TAG_COMPRESSION = 259
_TAG_STRIP_OFFSETS = 273
_TAG_SAMPLES = 277
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_BYTES = 279
_TAG_PLANAR = 284
_TAG_SAMPLE_FORMAT = 339
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


# ─────────────────────────────────────────────────────────────────────────────
# raw TIFF 파서
# ─────────────────────────────────────────────────────────────────────────────
def _read_ifd(buf: bytes) -> Tuple[str, Dict[int, Tuple[int, int, bytes]]]:
    if len(buf) < 8:
        raise RuntimeError("not a TIFF: buffer shorter than 8 bytes")
    bo = buf[:2]
    if bo == b"II":
        end = "<"
    elif bo == b"MM":
        end = ">"
    else:
        raise RuntimeError(f"not a TIFF: byte order marker {bo!r}")
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


def read_chip_tiff(buf: bytes) -> np.ndarray:
    """235 MS 칩 TIFF 바이트 → ``(3, 3, 5) float32`` (01 §6.2 [P1]).

    IFD 계약 (실측 확인):
        ``ImageWidth=3, ImageLength=3, SamplesPerPixel=5, BitsPerSample=[32]*5,
        SampleFormat=[3]*5 (IEEE float), Compression=1(none), PlanarConfiguration=1(chunky),
        StripByteCounts=180``.

    Raises:
        RuntimeError: 위 계약을 하나라도 어길 때. ★ 조용히 다른 shape 을 만들지 않는다.
    """
    end, tags = _read_ifd(buf)
    w = _ints(end, tags.get(_TAG_WIDTH))
    h = _ints(end, tags.get(_TAG_LENGTH))
    spp = _ints(end, tags.get(_TAG_SAMPLES)) or [1]
    bits = _ints(end, tags.get(_TAG_BITS))
    fmts = _ints(end, tags.get(_TAG_SAMPLE_FORMAT))
    comp = _ints(end, tags.get(_TAG_COMPRESSION)) or [1]
    planar = _ints(end, tags.get(_TAG_PLANAR)) or [1]
    offs = _ints(end, tags.get(_TAG_STRIP_OFFSETS))
    cnts = _ints(end, tags.get(_TAG_STRIP_BYTES))

    if not w or not h or not offs:
        raise RuntimeError("235 chip TIFF: missing ImageWidth/ImageLength/StripOffsets")
    if comp[0] != 1:
        raise RuntimeError(f"235 chip TIFF must be uncompressed, got Compression={comp[0]}")
    if planar[0] != 1:
        raise RuntimeError(f"235 chip TIFF must be chunky (Planar=1), got {planar[0]}")
    if spp[0] != N_BANDS:
        raise RuntimeError(f"235 chip TIFF must have {N_BANDS} samples, got {spp[0]}")
    if bits and set(bits) != {32}:
        raise RuntimeError(f"235 chip TIFF must be 32-bit, got BitsPerSample={bits}")
    if fmts and set(fmts) != {3}:
        raise RuntimeError(f"235 chip TIFF must be IEEE float (SampleFormat=3), got {fmts}")

    need = int(w[0]) * int(h[0]) * N_BANDS * 4
    data = b"".join(
        buf[o : o + c] for o, c in zip(offs, cnts or [need] * len(offs))
    )
    if len(data) < need:
        raise RuntimeError(f"235 chip TIFF: strip data {len(data)} B < required {need} B")
    dt = np.dtype(("<f4" if end == "<" else ">f4"))
    arr = np.frombuffer(data[:need], dtype=dt).reshape(int(h[0]), int(w[0]), N_BANDS)
    return np.ascontiguousarray(arr, dtype=np.float32)


def read_chip_geojson(buf: bytes) -> Dict[str, Any]:
    """칩 geojson → ``properties`` dict (Chla 등 스칼라 라벨 + 관측 메타)."""
    obj = json.loads(buf.decode("utf-8"))
    feats = obj.get("features") or []
    if not feats:
        raise RuntimeError("235 geojson has no features")
    return dict(feats[0].get("properties") or {})


# ─────────────────────────────────────────────────────────────────────────────
# 밴드 순서 자기정합성 검사 (01 §4.3 [M10], [U-1])
# ─────────────────────────────────────────────────────────────────────────────
def _band_order_ratio(
    arr: np.ndarray, band_index: Dict[str, int], n_top: int
) -> Tuple[float, float, float]:
    r = arr[:, band_index["red"]]
    re1 = arr[:, band_index["rededge1"]]
    nir = arr[:, band_index["nir"]]
    nd = (re1 - r) / (re1 + r + 1e-12)
    top = np.argsort(nd)[-max(1, n_top) :]
    m_nir = float(np.median(nir[top]))
    m_re1 = float(np.median(re1[top]))
    return m_nir, m_re1, (m_nir / m_re1 if m_re1 else float("inf"))


def band_order_evidence(
    spectra: np.ndarray,
    band_index: Dict[str, int],
    *,
    top_fracs: Sequence[float] = (0.001, 0.01, 0.05),
    top_ns: Sequence[int] = (10, 25),
) -> Dict[str, Dict[str, float]]:
    """밴드 순서 가설 증거를 꼬리 크기별로 계량한다 (01 §4.3 [M10] 재현 + 재보정 근거).

    ★ 실측(23,790 칩 전수, CPU): ``top 10 → 1.710``, ``top 25 → 1.694``,
    ``top 0.1 %(24) → 1.585``, **``top 1 %(238) → 0.930``**, ``top 5 % → 0.751``.
    즉 [M10] 이 근거로 든 "식생 스펙트럼" 은 **극단 꼬리 수십 개**이며,
    06 §3.2.5 가 동결한 "상위 1 %" 는 탁수 칩이 섞여 임계 1.3 을 통과하지 못한다.
    """
    arr = np.asarray(spectra, dtype=np.float64)
    n = arr.shape[0]
    out: Dict[str, Dict[str, float]] = {}
    for f in top_fracs:
        k = max(1, int(round(n * float(f))))
        m_nir, m_re1, ratio = _band_order_ratio(arr, band_index, k)
        out[f"top_{f:g}"] = {
            "n": float(k),
            "median_nir": m_nir,
            "median_re1": m_re1,
            "ratio": ratio,
        }
    for k in top_ns:
        m_nir, m_re1, ratio = _band_order_ratio(arr, band_index, int(k))
        out[f"top_n{int(k)}"] = {
            "n": float(min(int(k), n)),
            "median_nir": m_nir,
            "median_re1": m_re1,
            "ratio": ratio,
        }
    return out


def assert_band_order(
    spectra: np.ndarray,
    band_index: Dict[str, int],
    *,
    top_frac: float = 0.01,
    min_ratio: float = 1.3,
) -> None:
    """식생 칩에서 ``median(NIR) > 1.3 · median(RE1)`` 을 확인한다 (06 §3.2.5 동결식).

    식생 칩 = ``ND(rededge1, red) = (RE1−R)/(RE1+R)`` 상위 ``top_frac``.
    적색 경계 이후 NIR 고원은 밴드 순서 가설(H1 vs H2)을 가르는 가장 강한 증거다 (01 §4.3 [M10]).

    ★ 기본값은 06 동결값(1 %, 1.3)을 그대로 쓴다. 그러나 전수 실측에서 이 조합은
    **통과하지 못한다**(ratio 0.930) — :func:`band_order_evidence` 참조.
    호출자(:class:`K235ChipDataset` / :func:`build_k235_cache`)가 기본적으로
    ``band_order_check="warn"`` 을 쓰는 이유가 이것이며, 결과에 영향을 주는 결정
    (가중치 이식)은 06 V18(`spec.band_order_confirmed`)이 이미 별도로 막고 있다.

    Raises:
        RuntimeError: 위반 시.
    """
    arr = np.asarray(spectra, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"spectra must be (N,B), got {arr.shape}")
    for name in ("red", "rededge1", "nir"):
        if name not in band_index:
            raise ValueError(f"band_index must contain '{name}', got {sorted(band_index)}")
    n_top = max(1, int(round(arr.shape[0] * float(top_frac))))
    m_nir, m_re1, ratio = _band_order_ratio(arr, band_index, n_top)
    if not m_nir > float(min_ratio) * m_re1:
        raise RuntimeError(
            "235 band order self-consistency failed (01 §4.3 [M10] / [U-1]): "
            f"vegetation chips (top {n_top} = {top_frac:g}) median NIR={m_nir:.6g} "
            f"is not > {min_ratio:g} * median RE1={m_re1:.6g} (ratio={ratio:.3f}). "
            "band_order hypothesis H2 may be wrong — do NOT transplant (06 V18)."
        )


def _run_band_order_check(
    spectra: np.ndarray, band_index: Dict[str, int], policy: Any, *, where: str
) -> Dict[str, Dict[str, float]]:
    """``policy ∈ {"strict","warn","off"}`` (bool 도 허용: True→strict, False→off)."""
    if policy is True:
        policy = "strict"
    elif policy is False:
        policy = "off"
    if policy not in ("strict", "warn", "off"):
        raise ValueError(f"band_order_check must be strict|warn|off, got {policy!r}")
    evidence = band_order_evidence(spectra, band_index)
    if policy == "off":
        return evidence
    try:
        assert_band_order(spectra, band_index)
    except RuntimeError as exc:
        if policy == "strict":
            raise
        warnings.warn(
            f"{where}: {exc}  |  tail evidence = "
            + ", ".join(f"{k} ratio={v['ratio']:.3f}" for k, v in evidence.items())
            + "  |  [U-1] unresolved; band_order_check='warn' (06 V18 blocks transplant).",
            RuntimeWarning,
            stacklevel=3,
        )
    return evidence


# ─────────────────────────────────────────────────────────────────────────────
# npz 캐시 빌드 (01 §7.6)
# ─────────────────────────────────────────────────────────────────────────────
def _iter_ms_zip_pairs(src_root: Path) -> Iterable[Tuple[str, str, Path, Path]]:
    """``(split_src, site, source_zip, label_zip)`` 를 사전순으로 낸다."""
    for split_src, s_tag, l_tag in (("Training", "TS", "TL"), ("Validation", "VS", "VL")):
        s_dir = src_root / split_src / "01.원천데이터"
        l_dir = src_root / split_src / "02.라벨링데이터"
        if not s_dir.is_dir():
            continue
        for zp in sorted(s_dir.glob("*.zip")):
            m = _MS_ZIP_RE.search(zp.name)
            if m is None:
                continue
            site = m.group(1)
            lzp = l_dir / zp.name.replace(f"{s_tag}_", f"{l_tag}_", 1)
            if not lzp.exists():
                raise RuntimeError(f"235 label zip missing for {zp.name}: {lzp}")
            yield split_src, site, zp, lzp


def build_k235_cache(
    src_root: Path,
    out_npz: Path,
    *,
    band_order: Sequence[str] = K235_BAND_ORDER_DEFAULT,
    # ── 06 동결표에 없는 keyword-only 추가분 (진단·테스트용, 기본값은 전수 처리) ──
    sites: Optional[Sequence[str]] = None,
    limit_per_zip: Optional[int] = None,
    band_order_check: Any = "warn",
) -> Path:
    """16개 zip → 단일 ``.npz`` 캐시 (01 §7.6). **zip 을 풀지 않는다** (원본 트리 불변).

    Args:
        src_root: ``…/01-1.정식개방데이터`` (Training/Validation 을 자식으로 갖는 디렉터리).
        out_npz: 산출 경로. 부모 디렉터리는 자동 생성한다.
        band_order: 5밴드 이름 순서 (기본 = 가설 H2).
        sites / limit_per_zip: 부분 빌드 (스모크·테스트용).
        band_order_check: ``{"strict","warn","off"}``. 기본 ``"warn"`` 인 이유는
            :func:`assert_band_order` docstring 참조 (동결식이 전수 실측에서 통과하지 못한다).

    Returns:
        ``out_npz`` 경로.
    """
    src_root = Path(src_root)
    out_npz = Path(out_npz)
    if len(band_order) != N_BANDS:
        raise ValueError(f"band_order must have {N_BANDS} entries, got {list(band_order)}")

    spectra: List[np.ndarray] = []
    spectra_sd: List[np.ndarray] = []
    scal: Dict[str, List[float]] = {k: [] for k in NPZ_SCALAR_FIELDS}
    site_l: List[str] = []
    date_l: List[str] = []
    time_l: List[str] = []
    stem_l: List[str] = []
    split_l: List[str] = []
    lat_l: List[float] = []
    lon_l: List[float] = []

    for split_src, site, zp, lzp in _iter_ms_zip_pairs(src_root):
        if sites is not None and site not in sites:
            continue
        with zipfile.ZipFile(zp) as zs, zipfile.ZipFile(lzp) as zl:
            tif = {Path(n).stem: n for n in zs.namelist() if n.lower().endswith(".tif")}
            gjs = {Path(n).stem: n for n in zl.namelist() if n.lower().endswith(".geojson")}
            missing = sorted(set(tif) ^ set(gjs))
            if missing:
                raise RuntimeError(
                    f"235 pairing failed in {zp.name}: {len(missing)} unpaired stem(s), "
                    f"first={missing[:3]}"
                )
            stems = sorted(tif)
            if limit_per_zip is not None:
                stems = stems[: int(limit_per_zip)]
            for stem in stems:
                chip = read_chip_tiff(zs.read(tif[stem]))
                flat = chip.reshape(-1, N_BANDS)
                spectra.append(flat.mean(axis=0))  # [P2] 9픽셀 평균
                spectra_sd.append(flat.std(axis=0))
                props = read_chip_geojson(zl.read(gjs[stem]))
                for key in NPZ_SCALAR_FIELDS:
                    scal[key].append(_as_float(props.get(_GEOJSON_KEY[key])))
                site_l.append(site)
                date_l.append(str(props.get("Date", "")))
                time_l.append(str(props.get("Time", "")))
                stem_l.append(stem)
                split_l.append(split_src)
                lat_l.append(_as_float(props.get("Latitude")))
                lon_l.append(_as_float(props.get("Longitude")))

    if not spectra:
        raise RuntimeError(f"235 cache build found no chip under {src_root}")

    spec_arr = np.stack(spectra).astype(np.float32)
    _run_band_order_check(
        spec_arr,
        {n: i for i, n in enumerate(band_order)},
        band_order_check,
        where="build_k235_cache",
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "spectra": spec_arr,
        "spectra_sd": np.stack(spectra_sd).astype(np.float32),
        "site": np.array(site_l, dtype="<U3"),
        "date": np.array(date_l, dtype="<U8"),
        "time": np.array(time_l, dtype="<U6"),
        "stem": np.array(stem_l, dtype="<U32"),
        "split_src": np.array(split_l, dtype="<U10"),
        "lat": np.asarray(lat_l, dtype=np.float64),
        "lon": np.asarray(lon_l, dtype=np.float64),
        "band_order": np.array(",".join(band_order), dtype="<U40"),
    }
    for key in NPZ_SCALAR_FIELDS:
        payload[key] = np.asarray(scal[key], dtype=np.float32)
    np.savez_compressed(out_npz, **payload)
    return out_npz


def _as_float(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class K235ChipDataset(torch.utils.data.Dataset):
    """235 MS 칩 → ``x (8,3,3)`` + ``m (6,)`` + ``log1p(Chla)`` (06 §3.2.5, ★X-03).

    canonical 조립 (01 §6.2 [P5], 정정 A-9):
        ``x = [S(6 canonical slot), NDCI, clamp(MCI_norm, −1, 1)]``.
        ``use_blue=False`` (기본) 면 235 도 M3M 과 동일하게 slot (1,2,3,5) 만 채워
        ``m = [0,1,1,1,0,1]`` 이 된다 — ``C_abs`` 행이 학습/배포에서 같은 취급을 받게 하려는 것이다.

    Note:
        npz 는 칩당 **9픽셀 평균 (5,)** 만 보관한다(01 §7.6 캐시 규격). 따라서 ``(8,3,3)`` 은
        그 벡터의 공간 브로드캐스트다. 이는 T3(“3×3 dw conv 는 이식 금지”)의 근거와 정합한다 —
        3×3 칩의 공간 구조는 애초에 이식 가치가 없다.
    """

    def __init__(
        self,
        npz_path: str,
        sites: Sequence[str],
        *,
        band_order: Sequence[str] = K235_BAND_ORDER_DEFAULT,
        use_blue: bool = False,
        mci_c: float = MCI_C["rededge_mx"],
        apply_quality_filter: bool = True,
        target: str = "log1p",
        # ── 06 동결표에 없는 keyword-only 추가분 ────────────────────────────
        band_order_check: Any = "warn",
        check_msi_scale: bool = True,
    ) -> None:
        if target != "log1p":
            raise ValueError(f"target must be 'log1p' (X-14 고정), got {target!r}")
        if len(band_order) != N_BANDS:
            raise ValueError(f"band_order must have {N_BANDS} entries, got {list(band_order)}")
        self.npz_path = str(npz_path)
        self.band_order = tuple(str(b) for b in band_order)
        self.use_blue = bool(use_blue)
        self.mci_c = float(mci_c)
        self.apply_quality_filter = bool(apply_quality_filter)
        self.target = target

        z = np.load(self.npz_path, allow_pickle=False)
        spectra = np.asarray(z["spectra"], dtype=np.float64)
        cached_order = str(z["band_order"]) if "band_order" in z else ""
        if cached_order and tuple(cached_order.split(",")) != self.band_order:
            raise RuntimeError(
                f"band_order mismatch: cache={cached_order!r} vs requested={self.band_order}"
            )
        self.band_index: Dict[str, int] = {n: i for i, n in enumerate(self.band_order)}
        self.band_order_evidence = _run_band_order_check(
            spectra, self.band_index, band_order_check, where="K235ChipDataset"
        )
        if check_msi_scale:
            # [H1] 01 §2.6 — 235 는 절대 반사율이라 k_sensor = 1.0 이며 그대로 통과해야 한다.
            self.msi_median = assert_msi_scale_contract(spectra)
        else:
            self.msi_median = float("nan")

        site_arr = np.asarray(z["site"]).astype(str)
        keep = np.isin(site_arr, np.asarray(list(sites), dtype=object).astype(str))
        self.n_before_qc = int(keep.sum())
        if self.apply_quality_filter:  # 01 §6.3 [L1]
            wt = np.asarray(z["watertemp"], dtype=np.float64)
            do = np.asarray(z["odomg"], dtype=np.float64)
            keep = keep & (wt <= QC_WATERTEMP_MAX) & (do <= QC_ODOMG_MAX)
        chla = np.asarray(z["chla"], dtype=np.float64)
        keep = keep & np.isfinite(chla)
        self.index = np.nonzero(keep)[0].astype(np.int64)
        self.n_removed_by_qc = self.n_before_qc - int(self.index.size)
        if self.index.size == 0:
            raise RuntimeError(f"K235ChipDataset: no chip left for sites={list(sites)}")

        self.spectra = spectra[self.index]
        self.chla = chla[self.index]
        self.meta_fields = {
            "site": site_arr[self.index],
            "date": np.asarray(z["date"]).astype(str)[self.index],
            "time": np.asarray(z["time"]).astype(str)[self.index],
            "stem": np.asarray(z["stem"]).astype(str)[self.index],
            "phyco": np.asarray(z["phyco"], dtype=np.float64)[self.index],
        }
        self._x, self._m = self._build_canonical(self.spectra)

    # ── API ──────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return int(self.index.size)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        x = torch.from_numpy(np.ascontiguousarray(self._x[i]))
        y = float(np.log1p(self.chla[i]))
        return {
            "x": x,
            "m": torch.from_numpy(np.ascontiguousarray(self._m)),
            "y_chl_scalar": torch.tensor(y, dtype=torch.float32),
            "y_chl_scalar_valid": torch.tensor(True),
            "meta": {
                "site": str(self.meta_fields["site"][i]),
                "date": str(self.meta_fields["date"][i]),
                "time": str(self.meta_fields["time"][i]),
                "stem": str(self.meta_fields["stem"][i]),
                "chla_mgm3": float(self.chla[i]),
                "phyco": float(self.meta_fields["phyco"][i]),
            },
        }

    @property
    def sites(self) -> np.ndarray:
        return self.meta_fields["site"]

    @property
    def group_ids(self) -> np.ndarray:
        """``(site, date)`` 군 라벨 — 01 [M8] 군간 분산 93.41 % 의 단위."""
        return np.array(
            [f"{s}|{d}" for s, d in zip(self.meta_fields["site"], self.meta_fields["date"])]
        )

    # ── internals ────────────────────────────────────────────────────────────
    def _active_bands(self) -> Tuple[List[int], List[int]]:
        """``(사용할 spectra 열, 대응 canonical slot)``. ``use_blue=False`` → blue 제외."""
        cols: List[int] = []
        slots: List[int] = []
        for name in self.band_order:
            if name == "blue" and not self.use_blue:
                continue
            if name not in MSI_SLOT_ID:
                raise ValueError(f"band '{name}' is not a canonical slot ({MSI_SLOTS})")
            cols.append(self.band_index[name])
            slots.append(MSI_SLOT_ID[name])
        return cols, slots

    def _build_canonical(self, spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cols, slots = self._active_bands()
        n = spectra.shape[0]
        raw = spectra[:, cols].astype(np.float32)  # (N, K)
        # (K,1,1) 로 세운 뒤 canonical_scatter_np 를 그대로 쓴다 (지수식 단일 구현 유지)
        s_list = []
        for i in range(n):
            s, m = canonical_scatter_np(raw[i][:, None, None], slots, len(MSI_SLOTS))
            s_list.append(s[:, 0, 0])
        s_arr = np.stack(s_list)  # (N, 6)
        m_vec = np.zeros(len(MSI_SLOTS), dtype=np.float32)
        m_vec[slots] = 1.0

        red = s_arr[:, MSI_SLOT_ID["red"]]
        re1 = s_arr[:, MSI_SLOT_ID["rededge1"]]
        nir = s_arr[:, MSI_SLOT_ID["nir"]]
        eps = 1.0e-6
        ndci = (re1 - red) / (re1 + red + eps)
        mci = (re1 - red - self.mci_c * (nir - red)) / (red + re1 + nir + eps)
        bio = np.clip(np.stack([ndci, mci], axis=1), -1.0, 1.0).astype(np.float32)

        vec = np.concatenate([s_arr, bio], axis=1).astype(np.float32)  # (N, 8)
        x = np.repeat(np.repeat(vec[:, :, None, None], CHIP_HW[0], axis=2), CHIP_HW[1], axis=3)
        return np.ascontiguousarray(x), m_vec
