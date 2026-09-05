"""라벨 id 마스크 → 사람이 볼 수 있는 RGB 사본 (L3, CLI).

[분석] §E.1.13 ``이전 구현/tools/colorize_labels.py`` copy-as-is
(+ ignore_index 처리 1건 adapt).

**학습 라벨을 절대 건드리지 않는다.** ``labels/`` 의 PNG 는 화소값이 곧 클래스 id 라
팔레트를 입히면 id 가 깨져 학습이 망가진다. 그래서 병렬 트리 ``labels_color/`` 에
RGB 사본을 쓴다 (로더는 이 트리를 보지 않는다).

원본 대비 변경 1건
------------------
원본은 ``np.clip(mask, 0, 11)`` 로 잘라 **ignore(255)가 클래스 11(연못)로 둔갑**한다.
BloomNet 은 ``IGNORE_INDEX=255`` 를 실제로 쓰므로 ignore 를 회색으로 별도 렌더한다.

실행::

    python -m bloomnet.tools.colorize_labels --root bloomnet/data/aihub092_group \\
        [--out <dir>] [--workers 8] [--limit 0]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from bloomnet.constants import FINE_CLASS_NAMES, IGNORE_INDEX

__all__ = ["PALETTE", "IGNORE_COLOR", "colorize_labels", "main"]

#: 12클래스 팔레트 (이전 구현 원본과 **bit-identical**).
#: 시각화 전용 리터럴이라 ``constants.py`` 에 두지 않는다 — 어떤 모델·손실 경로도 읽지 않는다.
PALETTE: np.ndarray = np.array(
    [
        (0, 0, 0),  # 0  background          black
        (240, 220, 0),  # 1  밭_논                yellow
        (204, 80, 160),  # 2  잔재물               magenta
        (0, 158, 115),  # 3  배수로                teal
        (145, 70, 255),  # 4  비닐하우스           violet
        (110, 210, 0),  # 5  과수원                lime
        (230, 120, 0),  # 6  축사                  orange
        (135, 70, 20),  # 7  야적퇴비_가축분뇨     brown
        (255, 105, 180),  # 8  목장                pink
        (210, 40, 40),  # 9  분뇨개별처리시설      red
        (0, 90, 220),  # 10 부유쓰레기             blue
        (0, 200, 230),  # 11 연못                  cyan
    ],
    dtype=np.uint8,
)

#: ignore(255) 전용 색. 클래스 색과 겹치지 않는 중성 회색.
IGNORE_COLOR: Tuple[int, int, int] = (128, 128, 128)

LABEL_GLOB = "labels/*/*/*_labelids.png"


def _colorize_one(args: Tuple[str, str]) -> bool:
    from PIL import Image

    src, dst = args
    mask = np.array(Image.open(src).convert("L"))
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    known = mask < len(PALETTE)
    rgb[known] = PALETTE[mask[known]]
    rgb[mask == IGNORE_INDEX] = IGNORE_COLOR  # ★ 원본의 clip 을 대체 (id 둔갑 방지)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(dst)
    return True


def colorize_labels(
    root: Path,
    out_root: Optional[Path] = None,
    *,
    workers: int = 8,
    limit: int = 0,
) -> int:
    """``root/labels/**/*_labelids.png`` → ``out_root/**/*_color.png``.

    Args:
        root: 데이터셋 루트 (``labels/`` 를 자식으로 갖는다).
        out_root: 기본 ``root/labels_color``.
        workers: 프로세스 수. 1 이하면 직렬.
        limit: >0 이면 그 개수만 처리 (스모크).

    Returns:
        처리한 파일 수.
    """
    root = Path(root)
    out_root = Path(out_root) if out_root is not None else root / "labels_color"
    srcs = sorted(root.glob(LABEL_GLOB))
    if limit > 0:
        srcs = srcs[: int(limit)]
    if not srcs:
        raise FileNotFoundError(f"no labels matched {root / LABEL_GLOB}")

    jobs: List[Tuple[str, str]] = []
    for s in srcs:
        rel = s.relative_to(root / "labels")
        dst = out_root / rel.parent / (s.name.replace("_labelids.png", "_color.png"))
        jobs.append((str(s), str(dst)))

    if workers <= 1:
        for j in jobs:
            _colorize_one(j)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            list(ex.map(_colorize_one, jobs, chunksize=32))
    return len(jobs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="라벨 팔레트 시각화 ([분석] §E.1.13)")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    n = colorize_labels(args.root, args.out, workers=args.workers, limit=args.limit)
    print(f"colorized {n} labels -> {args.out or (args.root / 'labels_color')}")
    for i, name in enumerate(FINE_CLASS_NAMES):
        print(f"  {i:>2} {name:<20} rgb{tuple(int(v) for v in PALETTE[i])}")
    print(f"  {IGNORE_INDEX:>2} (ignore){'':<13} rgb{IGNORE_COLOR}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
