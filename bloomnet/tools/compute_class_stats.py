"""12클래스 픽셀/이미지 분포 전수 스캔 (L3, CLI) — 05 §4.1, U-9 해소.

산출 JSON 은 두 소비자를 동시에 만족한다.

* ``class_presence`` ``(N,K)`` — :class:`~bloomnet.data.samplers.RepeatFactorSampler` 입력.
  ``data/build.py::load_class_presence`` 와 ``data/aihub092.py::_load_presence_file`` 이
  둘 다 이 키를 읽는다.
* ``per_class`` — 05 §4.1 표(픽셀 비율·이미지 출현율·RFS 배율)의 **전수 갱신**.
  05 §4.1 표는 3,000장 표본이며 정정 B-25 가 전수 확정 전 배치 확률 사용을 금지했다.

``.npy`` 로도 함께 저장한다 (96k×12 JSON 은 파싱이 느리다). config 에는 둘 중
아무 경로나 넣으면 된다.

실행::

    CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.compute_class_stats \\
        --root bloomnet/data/aihub092_group --split train \\
        --out bloomnet/data/cache/class_stats_train.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from bloomnet.constants import FINE_CLASS_NAMES, IGNORE_INDEX
from bloomnet.data.aihub092 import _read_label_array, find_pairs, scan_class_presence
from bloomnet.version import __version__, git_revision

__all__ = ["compute_class_stats", "main"]

#: 05 §4.3 채택 RFS 임계값.
RFS_T: float = 0.05


def compute_class_stats(
    root: Path,
    split: str,
    out_json: Path,
    num_classes: int = 12,
    *,
    pixels: bool = True,
) -> Path:
    """``root``/``split`` 전수 스캔 → ``out_json`` (+ 같은 stem 의 ``.npy``).

    Args:
        root: ``images/{split}/<scene>/*.png`` 구조의 루트 (asis 또는 group 트리).
        split: ``train`` / ``val`` / ``test``.
        out_json: 산출 JSON.
        num_classes: 12 (aihub092 fine).
        pixels: False 면 픽셀 카운트를 건너뛰고 출현행렬만 만든다 (2배 빠르다).

    Returns:
        ``out_json`` 경로.
    """
    root = Path(root)
    pairs = find_pairs(root, split)  # 라벨 1개라도 없으면 RuntimeError (조용한 스킵 금지)
    labels = [lab for _, lab in pairs]
    presence = scan_class_presence(labels, num_classes=num_classes)

    pix = np.zeros(int(num_classes) + 1, dtype=np.int64)  # 마지막 칸 = ignore
    if pixels:
        for p in labels:
            arr = _read_label_array(p)
            vals, cnts = np.unique(arr, return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                iv = int(v)
                if 0 <= iv < num_classes:
                    pix[iv] += int(c)
                elif iv == IGNORE_INDEX:
                    pix[-1] += int(c)

    n = int(presence.shape[0])
    f_img = presence.mean(axis=0)  # 이미지 출현율
    total_px = int(pix[:num_classes].sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r_c = np.where(f_img > 0, np.maximum(1.0, np.sqrt(RFS_T / np.maximum(f_img, 1e-12))), 1.0)

    per_class = []
    for c in range(int(num_classes)):
        per_class.append(
            {
                "class_id": c,
                "name": FINE_CLASS_NAMES[c] if c < len(FINE_CLASS_NAMES) else str(c),
                "n_images": int(presence[:, c].sum()),
                "image_rate": float(f_img[c]),
                "n_pixels": int(pix[c]),
                "pixel_rate": float(pix[c] / total_px) if total_px else 0.0,
                "rfs_repeat_factor": float(r_c[c]),
            }
        )

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    npy_path = out_json.with_suffix(".npy")
    np.save(npy_path, presence)

    payload = {
        "tool": "compute_class_stats",
        "root": str(root.resolve()),
        "split": split,
        "num_classes": int(num_classes),
        "n_images": n,
        "n_pixels_labeled": total_px,
        "n_pixels_ignore": int(pix[-1]),
        "rfs_t": RFS_T,
        "per_class": per_class,
        "class_presence_npy": str(npy_path),
        "class_presence": presence.astype(np.uint8).tolist(),
        "bloomnet_version": __version__,
        "git_rev": git_revision(),
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out_json


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="클래스 분포 전수 스캔 (05 §4.1 / U-9)")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num_classes", type=int, default=12)
    ap.add_argument("--no_pixels", action="store_true", help="픽셀 카운트 생략 (출현행렬만)")
    args = ap.parse_args(argv)

    path = compute_class_stats(
        args.root, args.split, args.out, args.num_classes, pixels=not args.no_pixels
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path}  (n_images={data['n_images']})")
    print(f"{'id':>2} {'name':<20} {'img_rate':>9} {'px_rate':>9} {'rfs_r':>6}")
    for row in data["per_class"]:
        print(
            f"{row['class_id']:>2} {row['name']:<20} {row['image_rate']:>9.5f} "
            f"{row['pixel_rate']:>9.6f} {row['rfs_repeat_factor']:>6.3f}"
        )
    print("\n※ 05 §4.1 표는 3,000장 표본이다. 전수값 확정 전에는 배치 확률에 표본값을 쓰지 말 것 (정정 B-25).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
