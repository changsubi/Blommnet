"""aihub092 심볼릭 링크 트리 생성 (L3, CLI) — 01 §7.1.

52 GB 를 복사하지 않는다. **BloomNet 트리 안에만** 쓰며 원본
``이전 구현/data/aihub092`` 에는 아무것도 만들지 않는다.

산출::

    <out_root>/
        images       -> <src_root>/images         (디렉터리 링크)
        labels       -> <src_root>/labels
        labels_color -> <src_root>/labels_color   (있을 때만)
        manifest.json

실행::

    CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.link_aihub092 \\
        --src <AIHUB092_ROOT> \\
        --out <repo_root>/bloomnet/data/aihub092_asis
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from bloomnet.version import __version__, git_revision

__all__ = ["link_aihub092", "main", "DEFAULT_KINDS"]

DEFAULT_KINDS: Sequence[str] = ("images", "labels", "labels_color")

#: 원본 무결성 표본 검사 개수 (01 §7.1 규칙 5).
VERIFY_SAMPLES: int = 100


def _count_files(root: Path) -> Dict[str, int]:
    """``<kind>/<split>/<scene>`` 별 파일 수. 링크를 따라가 실제 파일을 센다."""
    out: Dict[str, int] = {}
    if not root.is_dir():
        return out
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            n = sum(1 for p in scene_dir.iterdir() if p.is_file())
            out[f"{split_dir.name}/{scene_dir.name}"] = n
    return out


def link_aihub092(
    src_root: Path,
    out_root: Path,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> Path:
    """``src_root`` 의 kind 디렉터리들을 ``out_root`` 아래로 심볼릭 링크한다.

    Args:
        src_root: ``images/``·``labels/`` 를 자식으로 갖는 원본 루트.
        out_root: 산출 루트. **이미 존재하면 즉시 중단**한다 (덮어쓰기 금지, 01 §7.1 규칙 1).
        kinds: 링크할 하위 디렉터리. ``labels_color`` 는 원본에 없으면 조용히 건너뛴다.

    Returns:
        ``out_root/manifest.json`` 경로.

    Raises:
        FileNotFoundError: ``src_root`` 또는 필수 kind(``images``/``labels``) 부재.
        FileExistsError: ``out_root`` 가 이미 존재.
    """
    src_root = Path(src_root).resolve()
    out_root = Path(out_root)
    if not src_root.is_dir():
        raise FileNotFoundError(f"source root not found: {src_root}")
    if out_root.exists():
        raise FileExistsError(
            f"output root already exists (overwrite forbidden, 01 §7.1): {out_root}"
        )
    for required in ("images", "labels"):
        if required in kinds and not (src_root / required).is_dir():
            raise FileNotFoundError(f"required kind missing in source: {src_root / required}")

    created: List[Path] = []
    linked: List[str] = []
    try:
        out_root.mkdir(parents=True)
        for kind in kinds:
            src = src_root / kind
            if not src.is_dir():
                continue  # labels_color 는 선택 (01 §7.1)
            dst = out_root / kind
            dst.symlink_to(src.resolve(), target_is_directory=True)  # 반드시 resolve()
            created.append(dst)
            linked.append(kind)

        _verify_realpaths(out_root, src_root, linked)

        counts = {k: _count_files(out_root / k) for k in linked}
        manifest = {
            "tool": "link_aihub092",
            "algorithm_version": "01-§7.1",
            "source_root": str(src_root),
            "output_root": str(out_root.resolve()),
            "kinds": linked,
            "file_counts": counts,
            "n_files_total": {k: sum(v.values()) for k, v in counts.items()},
            "bloomnet_version": __version__,
            "git_rev": git_revision(),
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "seed": None,  # 링크 생성에 난수를 쓰지 않는다 (검증 표본만 seed 고정)
            "verify_samples": VERIFY_SAMPLES,
        }
        manifest_path = out_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except BaseException:
        for p in reversed(created):  # 자기가 만든 것만 bottom-up 으로 되돌린다 (규칙 2)
            try:
                p.unlink()
            except OSError:
                pass
        shutil.rmtree(out_root, ignore_errors=True)
        raise
    return manifest_path


def _verify_realpaths(out_root: Path, src_root: Path, kinds: Sequence[str]) -> None:
    """01 §7.1 규칙 5 — 링크 하위 무작위 N개의 ``realpath`` 가 원본 하위인지 assert."""
    files: List[Path] = []
    for kind in kinds:
        d = out_root / kind
        for p in d.rglob("*"):
            if p.is_file():
                files.append(p)
            if len(files) >= 20_000:  # 52 GB 트리 전수 나열 방지 — 표본이면 충분하다
                break
        if len(files) >= 20_000:
            break
    if not files:
        return
    rs = random.Random(0)
    root_str = str(src_root)
    for p in rs.sample(files, min(VERIFY_SAMPLES, len(files))):
        real = os.path.realpath(p)
        if not real.startswith(root_str):
            raise RuntimeError(
                f"link escapes source tree: {p} -> {real} (expected under {root_str})"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="aihub092 심볼릭 링크 트리 (01 §7.1)")
    ap.add_argument("--src", required=True, type=Path, help="원본 aihub092 루트")
    ap.add_argument("--out", required=True, type=Path, help="산출 루트 (이미 있으면 중단)")
    ap.add_argument("--kinds", nargs="+", default=list(DEFAULT_KINDS))
    args = ap.parse_args(argv)

    path = link_aihub092(args.src, args.out, kinds=args.kinds)
    print(f"manifest: {path}")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
