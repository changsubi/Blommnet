"""235 칩 npz 캐시 빌더 (L3, CLI) — 01 §7.6.

알고리즘 본체는 :func:`bloomnet.data.k235.build_k235_cache` 가 유일 구현이다.
이 파일은 **CLI 배선과 리포트만** 한다 (로직 복제 금지, 06 §2.1).

실행::

    CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.build_k235_cache \\
        --src "/.../01-1.정식개방데이터" --out bloomnet/data/cache/k235_ms.npz

원본 zip 을 **풀지 않는다** (읽기 전용 스트리밍). 산출 npz 만 쓴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from bloomnet.constants import K235_BAND_ORDER_DEFAULT, MSI_MEDIAN_RANGE
from bloomnet.data.k235 import band_order_evidence, build_k235_cache

__all__ = ["main"]

DEFAULT_OUT = Path("bloomnet/data/cache/k235_ms.npz")


def _report(npz_path: Path) -> dict:
    """캐시 요약 — [H1] 스케일 계약과 (site,date) 군 수를 함께 낸다."""
    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.keys())
        x = z["spectra"] if "spectra" in keys else None
        site = z["site"] if "site" in keys else None
        date = z["date"] if "date" in keys else None
        n = int(x.shape[0]) if x is not None else 0
        med = float(np.median(x[x > 0])) if x is not None and x.size else float("nan")
        groups = (
            len({(str(a), str(b)) for a, b in zip(site.tolist(), date.tolist())})
            if site is not None and date is not None
            else None
        )
    lo, hi = MSI_MEDIAN_RANGE
    return {
        "npz": str(npz_path),
        "n_chips": n,
        "arrays": keys,
        "median_positive": med,
        "h1_contract": f"{lo} < median < {hi}",
        "h1_ok": bool(lo < med < hi) if med == med else False,
        "n_site_date_groups": groups,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="235 칩 npz 캐시 (01 §7.6)")
    ap.add_argument("--src", required=True, type=Path, help="01-1.정식개방데이터 루트")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--band_order", nargs=5, default=list(K235_BAND_ORDER_DEFAULT))
    ap.add_argument("--sites", nargs="*", default=None, help="부분 빌드 (기본 전수)")
    ap.add_argument("--limit_per_zip", type=int, default=None, help="스모크용 상한")
    ap.add_argument(
        "--band_order_check",
        choices=("strict", "warn", "off"),
        default="warn",
        help="동결식이 전수 실측에서 통과하지 못한다 — 기본 warn (data/k235.py docstring 참조)",
    )
    ap.add_argument("--evidence", action="store_true", help="꼬리 크기별 NIR/RE1 비율 재보정 근거 출력")
    args = ap.parse_args(argv)

    out = build_k235_cache(
        args.src,
        args.out,
        band_order=tuple(args.band_order),
        sites=args.sites,
        limit_per_zip=args.limit_per_zip,
        band_order_check=args.band_order_check,
    )
    rep = _report(Path(out))
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if not rep["h1_ok"]:
        print("  ⚠ [H1] msi 스케일 계약 위반 — 01 §2.6 R4′ / k_sensor 를 확인하라")

    if args.evidence:
        with np.load(Path(out), allow_pickle=True) as z:
            spectra = np.asarray(z["spectra"])
            order = str(z["band_order"]).split(",")
        band_index = {name.strip(): i for i, name in enumerate(order)}
        ev = band_order_evidence(spectra, band_index)
        print("\nband_order evidence (median NIR / median RE1):")
        print(json.dumps(ev, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
