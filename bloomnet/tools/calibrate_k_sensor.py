"""``k_sensor`` 1회 산출 (L3, CLI) — 01 §2.6 R4′, 정정 A-2/A-40, X-28.

**M-13 blocker 의 해소 경로다.** 현재 ``constants.K_SENSOR["m3m"] = 3.5e3`` 은 잠정값이며,
``data/k235.py``·``data/drone_m3m.py`` 담당자의 전수 실측(01 §2.6 [M13] 표가 R3 노출/게인
단계를 빼고 계산됐다)에 따르면 R1~R4 전부를 적용한 ``rho_rel`` median 은 이미 0.0122 로
235 절대 반사율과 같은 자릿수다 → 실제 ``k ≈ 1``. ``3.5e3`` 을 곱하면 [H1]
``1e-3 < median < 1.0`` 을 **상한에서** 위반한다.

산식 (01 §2.6)::

    k = median_b( median(rho_235_water[b]) / median(rho_rel[b, water px]) )

Note:
    **자동 반영 금지.** 산출값을 ``constants.K_SENSOR`` 에 자동으로 쓰지 않는다.
    리뷰 1회를 거쳐 사람이 리터럴을 갱신하고, manifest 를 PR 에 첨부한다 (06 §3.6).

실행::

    CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.calibrate_k_sensor \\
        --frames <M3M 비행 루트> --sensor m3m \\
        --k235_npz bloomnet/data/cache/k235_ms.npz \\
        [--water_roi roi.json] [--out k_sensor_manifest.json]

``--water_roi`` 미지정 시 **수면 대리 마스크**(NIR 저반사 + 저채도)를 자동 생성하고
manifest 에 ``water_roi: "auto:nir_percentile"`` 로 기록한다. 실측 ROI 가 있으면 반드시
주는 편이 낫다 — 자동 마스크는 부유물·그림자를 물로 오인할 수 있다.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from bloomnet.constants import K_SENSOR, MSI_MEDIAN_RANGE
from bloomnet.data.drone_m3m import (
    M3M_BAND_TO_SLOT,
    M3M_BAND_SUFFIXES,
    find_m3m_frames,
    frame_reflectance,
)
from bloomnet.version import __version__, git_revision

__all__ = ["calibrate_k_sensor", "auto_water_mask", "main"]

#: 자동 수면 마스크의 NIR 백분위 하한/상한 (물은 NIR 을 강하게 흡수한다).
AUTO_NIR_PCTL: float = 30.0


def auto_water_mask(
    rho: np.ndarray, *, nir_index: int = 3, pctl: float = AUTO_NIR_PCTL
) -> np.ndarray:
    """``(B,H,W)`` 반사율 → 수면 대리 마스크 ``(H,W) bool``.

    NIR 이 하위 ``pctl`` 백분위이고 전 밴드가 양수인 픽셀. 실측 ROI 의 대용이며
    **정밀 마스크가 아니다** — manifest 에 자동 생성 사실을 남긴다.
    """
    nir = rho[nir_index]
    finite = np.isfinite(rho).all(axis=0) & (rho > 0).all(axis=0)
    if not finite.any():
        return finite
    thr = float(np.percentile(nir[finite], float(pctl)))
    return finite & (nir <= thr)


def _load_roi(water_roi: Optional[Path], shape: Sequence[int]) -> Optional[np.ndarray]:
    """ROI 파일 → ``(H,W) bool``. ``.npy`` 마스크 또는 ``.json`` 사각형 목록을 받는다."""
    if water_roi is None:
        return None
    p = Path(water_roi)
    if p.suffix == ".npy":
        m = np.load(p).astype(bool)
    else:
        boxes = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(boxes, dict):
            boxes = boxes.get("boxes", [])
        m = np.zeros(tuple(shape), dtype=bool)
        for b in boxes:  # [y0, x0, y1, x1] (원 해상도 좌표가 아니라 축소 후 좌표)
            y0, x0, y1, x1 = (int(v) for v in b)
            m[y0:y1, x0:x1] = True
    if m.shape != tuple(shape):
        raise ValueError(f"water_roi shape {m.shape} != frame shape {tuple(shape)}")
    return m


def calibrate_k_sensor(
    frames: Sequence[Path],
    water_roi: Optional[Path],
    *,
    sensor: str,
    k235_npz: Path,
    downsample: int = 8,
    max_frames: int = 50,
) -> Dict[str, float]:
    """수면 픽셀 median 기반 ``k_sensor`` 산출 (01 §2.6 R4′).

    Args:
        frames: M3M 비행 폴더 경로들 (각 폴더를 :func:`find_m3m_frames` 로 스캔한다).
        water_roi: 수면 ROI. ``None`` 이면 자동 대리 마스크.
        sensor: ``"m3m"`` 등. manifest 기록용.
        k235_npz: 235 캐시 (``spectra``·``band_order``). 절대 반사율 기준.
        downsample: 프레임 축소 배율 (R1~R4 는 원 해상도에서 계산된다).
        max_frames: 사용할 최대 프레임 수 (median 은 소수 프레임으로도 안정적이다).

    Returns:
        ``{"k_sensor": …, "k_per_band": {...}, "n_frames": …, ...}``.
        결과는 **반환만** 하며 ``constants.K_SENSOR`` 를 수정하지 않는다.
    """
    with np.load(Path(k235_npz), allow_pickle=True) as z:
        spectra = np.asarray(z["spectra"], dtype=np.float64)
        order = [s.strip() for s in str(z["band_order"]).split(",")]
    k235_median = {name: float(np.median(spectra[:, i])) for i, name in enumerate(order)}

    frame_objs = []
    for root in frames:
        frame_objs.extend(find_m3m_frames(Path(root)))
        if len(frame_objs) >= max_frames:
            break
    frame_objs = frame_objs[: int(max_frames)]
    if not frame_objs:
        raise FileNotFoundError(f"no complete M3M frames under {[str(f) for f in frames]}")

    per_band_ratios: Dict[str, List[float]] = {b: [] for b in M3M_BAND_SUFFIXES}
    rho_medians: Dict[str, List[float]] = {b: [] for b in M3M_BAND_SUFFIXES}
    used = 0
    for fr in frame_objs:
        # k_sensor 를 구하려는 중이므로 R4′ 와 [H1] 검증을 끈다 (닭-달걀 회피).
        rho, _info = frame_reflectance(
            fr, downsample=downsample, apply_r4p=False, check_scale=False
        )
        mask = _load_roi(water_roi, rho.shape[1:])
        if mask is None:
            mask = auto_water_mask(rho)
        if not mask.any():
            continue
        used += 1
        for bi, suffix in enumerate(M3M_BAND_SUFFIXES):
            px = rho[bi][mask]
            px = px[np.isfinite(px) & (px > 0)]
            if px.size == 0:
                continue
            med = float(np.median(px))
            rho_medians[suffix].append(med)
            slot = M3M_BAND_TO_SLOT[suffix]
            ref = k235_median.get(slot)
            if ref is None or med <= 0:
                continue
            per_band_ratios[suffix].append(ref / med)

    if used == 0:
        raise RuntimeError("수면 픽셀을 하나도 찾지 못했다 — --water_roi 를 지정하라")

    k_per_band = {
        M3M_BAND_TO_SLOT[b]: float(np.median(v)) for b, v in per_band_ratios.items() if v
    }
    if not k_per_band:
        raise RuntimeError("밴드별 비율을 산출하지 못했다 (235 캐시의 band_order 를 확인하라)")
    k = float(np.median(list(k_per_band.values())))

    rho_med_all = float(np.median([m for v in rho_medians.values() for m in v]))
    lo, hi = MSI_MEDIAN_RANGE
    k_current = float(K_SENSOR.get(str(sensor), float("nan")))
    return {
        "sensor": str(sensor),
        "k_sensor": k,
        "k_per_band": k_per_band,
        "k235_median_per_slot": k235_median,
        "rho_rel_median": rho_med_all,
        "median_after_k": rho_med_all * k,
        "h1_range": [float(lo), float(hi)],
        "h1_ok": bool(lo < rho_med_all * k < hi),
        "current_constant": k_current,
        "median_with_current_constant": rho_med_all * k_current,
        "n_frames": int(used),
        "downsample": int(downsample),
        "water_roi": str(water_roi) if water_roi else "auto:nir_percentile",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="k_sensor 1회 산출 (01 §2.6 R4′ / M-13)")
    ap.add_argument("--frames", nargs="+", required=True, type=Path, help="M3M 비행 폴더")
    ap.add_argument("--water_roi", type=Path, default=None, help=".npy 마스크 또는 .json 박스")
    ap.add_argument("--sensor", default="m3m")
    ap.add_argument("--k235_npz", type=Path, default=Path("bloomnet/data/cache/k235_ms.npz"))
    ap.add_argument("--downsample", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None, help="manifest JSON 경로")
    args = ap.parse_args(argv)

    res: Dict[str, Any] = calibrate_k_sensor(
        args.frames,
        args.water_roi,
        sensor=args.sensor,
        k235_npz=args.k235_npz,
        downsample=args.downsample,
        max_frames=args.max_frames,
    )
    res.update(
        {
            "tool": "calibrate_k_sensor",
            "bloomnet_version": __version__,
            "git_rev": git_revision(),
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        }
    )
    text = json.dumps(res, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(
        "\n★ 이 값을 constants.K_SENSOR 에 **자동 반영하지 않는다** (06 §3.6). "
        "리뷰 1회 후 사람이 리터럴을 갱신하고 이 manifest 를 PR 에 첨부한다."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
