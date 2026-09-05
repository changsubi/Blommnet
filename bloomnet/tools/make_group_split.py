"""flight-line group 분할 + manifest — 01 §7.3 (★정정 A-1), 06 §3.6 (레벨 L3, CLI).

기본 split 의 누수는 실측되었다 `[M3]`: val 7,493장·test 3,212장이 **100 %** train 에도
등장하는 flight-line 군에 속한다. 따라서 `group` 분할이 **주 지표**이고 `asis` 는 비교용이다.

알고리즘 (정정 A-1 — 초판은 자기 자신의 하드 검증을 통과하지 못해 폐기됨):
    S1  river 층화 LPT greedy   — 큰 군부터 '최대 상대 결손' split 에 배정 → 크기(V-A)를 먼저 맞춘다
    S2  결정론적 스왑/이동 보정 — 목적함수 P 를 줄이는 단일 이동/2군 스왑을 하나씩 적용
    S3  하드 검증 V-A~V-D
    S4  실패 시 seed += 1 재시도(최대 ``retries``), 그래도 실패하면 위반 클래스만 EXEMPT 강등.
        단 **V-A 실패는 강등으로 해소되지 않으므로 RuntimeError**.

    `[J]` 순서가 "크기 먼저, 클래스 나중" 인 이유는 실측이다: rare 클래스 선배치 안은
    64.4/16.2/19.4 %(dev 0.056)로 무너졌다. 크기 제약은 사후 복구가 불가능한 반면
    클래스 균형은 스왑으로 복구 가능하다 (0.056 → 0.0000, 위반 5건 → 0건).

실측 재현 `[M14]` (SEED=20260731, 전수 96,340장):
    67,438 / 14,454 / 14,448 = 70.000 / 15.003 / 14.997 %, max_abs_dev < 1e-4,
    위반 0건, exempt = {8}(목장, 8군), retries_used = 0.

레벨 L3 — L2(`data.aihub092`) 이하만 import 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bloomnet.constants import (
    SCENES,
    SPLIT_BAND,
    SPLIT_MIN_GROUPS,
    SPLIT_RETRIES,
    SPLIT_SEED,
    SPLIT_TARGET,
    SPLIT_TOL,
)
from bloomnet.data.aihub092 import (
    LABEL_SUFFIX,
    find_pairs,
    group_key,
    scan_class_presence,
)
from bloomnet.version import git_revision

__all__ = [
    "ALGORITHM_VERSION",
    "SPLIT_NAMES",
    "GROUP_KEY_DEF",
    "SplitPlan",
    "GroupTable",
    "scan_dataset",
    "build_group_table",
    "plan_group_split",
    "make_group_split",
    "main",
]

ALGORITHM_VERSION: str = "A-1"
SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")
GROUP_KEY_DEF: str = "(scene, admin, date, line)  # 01 §7.3 [M3] flight-line"

_SOURCE_SPLITS: Tuple[str, ...] = ("train", "val", "test")


def _h(key: str) -> str:
    """결정론적 타이브레이크 키. 모든 정렬·선택의 유일한 난수원이다."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# S0. 전처리 — 스캔 및 군 집계
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GroupTable:
    """군 단위 집계 (S0). ``keys`` 는 사전순 정렬되어 있으며 이것이 결정론의 기반이다."""

    keys: List[str]
    scenes: List[str]
    n_images: np.ndarray  # (G,)      int64
    class_counts: np.ndarray  # (G, K) int64 — 클래스 k 를 포함한 이미지 수
    global_presence: np.ndarray  # (K,) float64 — 전역 이미지 출현율
    n_files_scanned: int

    @property
    def n_groups_with_class(self) -> np.ndarray:
        return (self.class_counts > 0).sum(axis=0)


def scan_dataset(
    root: Path,
    *,
    num_classes: int = 12,
    presence_cache: Optional[Path] = None,
    scan_workers: int = 0,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """``root`` 의 전 split 을 스캔해 ``(records, presence (N,K) bool)`` 을 만든다.

    ``records[i]`` = ``{"src_split","scene","stem","image","label","group"}``.
    ``presence_cache`` 가 있으면 라벨 재스캔을 건너뛴다 (전수 스캔은 96,340장 = 수십 초).
    """
    root = Path(root)
    records: List[Dict[str, Any]] = []
    for split in _SOURCE_SPLITS:
        if not (root / "images" / split).is_dir():
            continue
        for img, lbl in find_pairs(root, split):
            scene = img.parent.name
            stem = img.stem
            gk = group_key(scene, stem)
            records.append(
                {
                    "src_split": split,
                    "scene": scene,
                    "stem": stem,
                    "image": img,
                    "label": lbl,
                    "group": "|".join(gk),
                }
            )
    if not records:
        raise RuntimeError(f"make_group_split: nothing scanned under {root}")

    if presence_cache is not None and Path(presence_cache).exists():
        pres = np.load(Path(presence_cache))
        pres = np.asarray(pres)[:, :num_classes].astype(bool)
        if pres.shape[0] != len(records):
            raise RuntimeError(
                f"presence cache rows {pres.shape[0]} != scanned images {len(records)}"
            )
        return records, pres

    labels = [r["label"] for r in records]
    if scan_workers and scan_workers > 1:
        from multiprocessing import Pool

        with Pool(int(scan_workers)) as pool:
            chunks = pool.map(
                _presence_worker,
                [(labels[i : i + 256], num_classes) for i in range(0, len(labels), 256)],
            )
        pres = np.concatenate(chunks) if chunks else np.zeros((0, num_classes), dtype=bool)
    else:
        pres = scan_class_presence(labels, num_classes)
    if presence_cache is not None:
        Path(presence_cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(Path(presence_cache), pres)
    return records, pres


def _presence_worker(arg: Tuple[Sequence[Path], int]) -> np.ndarray:
    paths, k = arg
    return scan_class_presence(paths, k)


def build_group_table(
    records: Sequence[Dict[str, Any]], presence: np.ndarray
) -> GroupTable:
    """이미지 레코드 → 군 집계 (S0)."""
    pres = np.asarray(presence).astype(np.int64)
    by: Dict[str, Dict[str, Any]] = {}
    for i, r in enumerate(records):
        slot = by.setdefault(r["group"], {"scene": r["scene"], "idx": []})
        slot["idx"].append(i)
    keys = sorted(by)
    n_images = np.array([len(by[k]["idx"]) for k in keys], dtype=np.int64)
    class_counts = np.stack([pres[by[k]["idx"]].sum(axis=0) for k in keys]).astype(np.int64)
    scenes = [by[k]["scene"] for k in keys]
    return GroupTable(
        keys=keys,
        scenes=scenes,
        n_images=n_images,
        class_counts=class_counts,
        global_presence=pres.mean(axis=0).astype(np.float64),
        n_files_scanned=len(records),
    )


# ─────────────────────────────────────────────────────────────────────────────
# S1~S4. 알고리즘
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SplitPlan:
    """분할 결과 + manifest 에 들어갈 통계 전부."""

    assign: np.ndarray
    exempt: Tuple[int, ...]
    seed_used: int
    retries_used: int
    achieved_ratio: Tuple[float, float, float]
    max_abs_dev: float
    n_images_per_split: Tuple[int, int, int]
    n_groups_per_split: Tuple[int, int, int]
    per_split_class: List[Dict[str, Any]] = field(default_factory=list)
    zero_group_pairs: List[Tuple[str, int]] = field(default_factory=list)
    violations: List[Tuple[str, float]] = field(default_factory=list)
    log: List[Tuple[int, float, int]] = field(default_factory=list)


def _aggregate(
    assign: np.ndarray, n: np.ndarray, cls: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = cls.shape[1]
    n_img = np.zeros(3, dtype=np.int64)
    counts = np.zeros((3, k), dtype=np.int64)
    groups = np.zeros((3, k), dtype=np.int64)
    for s in range(3):
        m = assign == s
        n_img[s] = n[m].sum()
        counts[s] = cls[m].sum(axis=0)
        groups[s] = (cls[m] > 0).sum(axis=0)
    return n_img, counts, groups


def _penalty(
    assign: np.ndarray,
    n: np.ndarray,
    cls: np.ndarray,
    glob: np.ndarray,
    exempt: set,
    *,
    target: np.ndarray,
    tol: float,
    band: float,
    min_groups: int,
) -> Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """S2 목적함수 (01 §7.3 S2 의 식 그대로).

    hinge(클래스 출현율 배율) + 0.5·(군 수 부족) + 20·(V-A 초과, 하드) + 3·|frac−target| (소프트)
    """
    n_img, counts, groups = _aggregate(assign, n, cls)
    rate = counts / np.maximum(n_img[:, None], 1)
    ratio = rate / np.maximum(glob[None, :], 1e-12)
    p = 0.0
    for k in range(cls.shape[1]):
        if k in exempt:
            continue
        for s in (1, 2):  # val, test 만 배율 검사 (train 은 잔여)
            p += max(0.0, (1 - band) - ratio[s, k]) + max(0.0, ratio[s, k] - (1 + band))
        for s in range(3):
            p += 0.5 * max(0, min_groups - int(groups[s, k]))
    frac = n_img / max(int(n_img.sum()), 1)
    p += 20.0 * max(0.0, float(np.abs(frac - target).max()) - tol)
    p += 3.0 * float(np.abs(frac - target).sum())
    return p, (n_img, frac, ratio, groups)


def _violations(
    assign: np.ndarray,
    n: np.ndarray,
    cls: np.ndarray,
    glob: np.ndarray,
    exempt: set,
    *,
    target: np.ndarray,
    tol: float,
    band: float,
    min_groups: int,
) -> List[Tuple[str, float]]:
    """S3 하드 검증 V-A / V-C / V-D. (V-B 는 배정이 분할이라 구조적으로 성립하며 별도 assert 한다.)"""
    _, (_, frac, ratio, groups) = _penalty(
        assign, n, cls, glob, exempt, target=target, tol=tol, band=band, min_groups=min_groups
    )
    out: List[Tuple[str, float]] = []
    dev = float(np.abs(frac - target).max())
    if dev > tol:
        out.append(("V-A", dev))
    for k in range(cls.shape[1]):
        if k in exempt:
            continue
        for s in (1, 2):
            if not (1 - band <= ratio[s, k] <= 1 + band):
                out.append((f"V-C c{k}/{SPLIT_NAMES[s]}", round(float(ratio[s, k]), 3)))
        for s in range(3):
            if int(groups[s, k]) < min_groups:
                out.append((f"V-D c{k}/{SPLIT_NAMES[s]}", float(groups[s, k])))
    return out


def _lpt_greedy(
    table: GroupTable, seed: int, target: np.ndarray
) -> np.ndarray:
    """S1 — river(scene) 층화 LPT greedy. 군은 절대 쪼개지 않는다."""
    assign = np.full(len(table.keys), -1, dtype=np.int64)
    by_scene: Dict[str, List[int]] = defaultdict(list)
    for i, sc in enumerate(table.scenes):
        by_scene[sc].append(i)
    for sc in sorted(by_scene):
        idxs = by_scene[sc]
        tgt = target * float(table.n_images[idxs].sum())
        cur = np.zeros(3, dtype=np.float64)
        order = sorted(idxs, key=lambda i: (-int(table.n_images[i]), _h(f"{table.keys[i]}|{seed}")))
        for i in order:
            s = int(np.argmax((tgt - cur) / target))
            assign[i] = s
            cur[s] += float(table.n_images[i])
    return assign


def _repair(
    assign: np.ndarray,
    table: GroupTable,
    exempt: set,
    seed: int,
    *,
    target: np.ndarray,
    tol: float,
    band: float,
    min_groups: int,
    max_iter: int = 400,
) -> Tuple[np.ndarray, float]:
    """S2 — 개선되는 '군 1개 이동' 과 '군 2개 스왑' 중 P 최소인 것을 1개씩 적용."""
    n, cls, glob = table.n_images, table.class_counts, table.global_presence
    kw = dict(target=target, tol=tol, band=band, min_groups=min_groups)
    p, _ = _penalty(assign, n, cls, glob, exempt, **kw)  # type: ignore[arg-type]
    for _ in range(max_iter):
        best: Optional[Tuple[Tuple[float, str], Tuple[str, int, int]]] = None
        for i in range(len(assign)):
            for s in range(3):
                if s == assign[i]:
                    continue
                old = assign[i]
                assign[i] = s
                q, _ = _penalty(assign, n, cls, glob, exempt, **kw)  # type: ignore[arg-type]
                assign[i] = old
                if q < p - 1e-9:
                    key = (q, _h(f"move|{table.keys[i]}|{s}|{seed}"))
                    if best is None or key < best[0]:
                        best = (key, ("move", i, s))
        for i in range(len(assign)):
            for j in range(i + 1, len(assign)):
                if assign[i] == assign[j]:
                    continue
                assign[i], assign[j] = assign[j], assign[i]
                q, _ = _penalty(assign, n, cls, glob, exempt, **kw)  # type: ignore[arg-type]
                assign[i], assign[j] = assign[j], assign[i]
                if q < p - 1e-9:
                    key = (q, _h(f"swap|{table.keys[i]}|{table.keys[j]}|{seed}"))
                    if best is None or key < best[0]:
                        best = (key, ("swap", i, j))
        if best is None:
            break
        kind, i, j = best[1]
        if kind == "move":
            assign[i] = j
        else:
            assign[i], assign[j] = assign[j], assign[i]
        p = best[0][0]
    return assign, p


def plan_group_split(
    table: GroupTable,
    *,
    seed: int = SPLIT_SEED,
    ratios: Tuple[float, float, float] = SPLIT_TARGET,
    tol: float = SPLIT_TOL,
    band: float = SPLIT_BAND,
    min_groups: int = SPLIT_MIN_GROUPS,
    retries: int = SPLIT_RETRIES,
) -> SplitPlan:
    """S1~S4 를 수행하고 통계를 채운 :class:`SplitPlan` 을 낸다.

    Raises:
        RuntimeError: 최종적으로 **V-A** 를 만족하지 못할 때 (군 크기 편중이 원인이라
            EXEMPT 강등으로는 해소되지 않는다 — 01 §7.3 S4).
    """
    target = np.asarray(ratios, dtype=np.float64)
    if target.shape != (3,) or abs(float(target.sum()) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be 3 numbers summing to 1, got {ratios}")
    n, cls, glob = table.n_images, table.class_counts, table.global_presence
    ngw = table.n_groups_with_class
    # SPLIT_EXEMPT = { k : n_groups(k) < 3 · MIN_GROUPS }
    exempt = {int(k) for k in range(cls.shape[1]) if int(ngw[k]) < 3 * min_groups}
    kw = dict(target=target, tol=tol, band=band, min_groups=min_groups)

    log: List[Tuple[int, float, int]] = []
    assign: Optional[np.ndarray] = None
    used_seed = int(seed)
    retries_used = 0
    viol: List[Tuple[str, float]] = []
    for t in range(max(1, int(retries))):
        cur_seed = int(seed) + t
        a = _lpt_greedy(table, cur_seed, target)
        a, p = _repair(a, table, exempt, cur_seed, **kw)  # type: ignore[arg-type]
        v = _violations(a, n, cls, glob, exempt, **kw)  # type: ignore[arg-type]
        log.append((cur_seed, round(float(p), 4), len(v)))
        assign, used_seed, retries_used, viol = a, cur_seed, t, v
        if not v:
            break

    assert assign is not None
    if viol:  # S4 폴백 — 위반이 남은 클래스만 강등
        _, (_, _, ratio, groups) = _penalty(  # type: ignore[arg-type]
            assign, n, cls, glob, exempt, **kw
        )
        for k in range(cls.shape[1]):
            if k in exempt:
                continue
            bad = any(not (1 - band <= ratio[s, k] <= 1 + band) for s in (1, 2)) or bool(
                (groups[:, k] < min_groups).any()
            )
            if bad:
                exempt.add(int(k))
        viol = _violations(assign, n, cls, glob, exempt, **kw)  # type: ignore[arg-type]

    # V-B — 군 집합 배타성 (배정이 partition 이므로 구조적으로 참이지만 명시적으로 검사한다)
    sets = [set(np.nonzero(assign == s)[0].tolist()) for s in range(3)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("V-B violated: split group sets are not mutually exclusive")

    n_img, counts, groups = _aggregate(assign, n, cls)
    frac = n_img / max(int(n_img.sum()), 1)
    max_dev = float(np.abs(frac - target).max())
    if max_dev > tol:
        raise RuntimeError(
            "V-A violated and not recoverable by SPLIT_EXEMPT demotion "
            f"(01 §7.3 S4): max|achieved-target| = {max_dev:.4f} > tol {tol}. "
            f"achieved = {np.round(frac, 5).tolist()}, groups = {len(table.keys)}, "
            "cause is usually extreme group-size skew (a single group larger than a split target)."
        )

    rate = counts / np.maximum(n_img[:, None], 1)
    ratio = rate / np.maximum(glob[None, :], 1e-12)
    per_split_class: List[Dict[str, Any]] = []
    zero_pairs: List[Tuple[str, int]] = []
    for s in range(3):
        for k in range(cls.shape[1]):
            per_split_class.append(
                {
                    "split": SPLIT_NAMES[s],
                    "class_id": int(k),
                    "n_groups": int(groups[s, k]),
                    "n_images": int(counts[s, k]),
                    "presence_rate": float(rate[s, k]),
                    "ratio_to_global": float(ratio[s, k]),
                }
            )
            if int(groups[s, k]) == 0:
                zero_pairs.append((SPLIT_NAMES[s], int(k)))

    return SplitPlan(
        assign=assign,
        exempt=tuple(sorted(exempt)),
        seed_used=used_seed,
        retries_used=retries_used,
        achieved_ratio=(float(frac[0]), float(frac[1]), float(frac[2])),
        max_abs_dev=max_dev,
        n_images_per_split=(int(n_img[0]), int(n_img[1]), int(n_img[2])),
        n_groups_per_split=tuple(  # type: ignore[arg-type]
            int((assign == s).sum()) for s in range(3)
        ),
        per_split_class=per_split_class,
        zero_group_pairs=zero_pairs,
        violations=viol,
        log=log,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 링크 트리 + manifest
# ─────────────────────────────────────────────────────────────────────────────
def _link(src: Path, dst: Path, created: List[Path]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())  # 반드시 resolve() — 상대 링크는 트리 이동에 취약
    created.append(dst)


def make_group_split(
    root: Path,
    out_root: Path,
    *,
    seed: int = SPLIT_SEED,
    ratios: Tuple[float, float, float] = SPLIT_TARGET,
    tol: float = SPLIT_TOL,
    band: float = SPLIT_BAND,
    min_groups: int = SPLIT_MIN_GROUPS,
    retries: int = SPLIT_RETRIES,
    # ── 06 동결표에 없는 keyword-only 추가분 ────────────────────────────────
    num_classes: int = 12,
    presence_cache: Optional[Path] = None,
    scan_workers: int = 0,
    link: bool = True,
    verify_samples: int = 100,
) -> Path:
    """``root`` (asis 링크 트리) → ``out_root`` (누수 없는 group 분할 링크 트리).

    Args:
        root: ``images/{train,val,test}/<scene>/*.png`` 구조의 원본(또는 asis 링크) 루트.
        out_root: 산출 루트. **이미 존재하면 즉시 중단한다** (덮어쓰기 금지, 01 §7.1).
        presence_cache: 라벨 전수 스캔 결과 ``.npy`` 캐시 (재실행 가속).
        link: False 면 manifest 만 쓴다 (드라이런).
        verify_samples: 링크 생성 후 무작위 N개의 ``realpath`` 가 원본 하위인지 검사 (01 §7.1 규칙 5).

    Returns:
        ``out_root/group_split_manifest.json`` 경로.
    """
    root = Path(root).resolve()
    out_root = Path(out_root)
    if out_root.exists():
        raise FileExistsError(
            f"output root already exists (overwrite forbidden, 01 §7.1): {out_root}"
        )

    records, presence = scan_dataset(
        root,
        num_classes=num_classes,
        presence_cache=Path(presence_cache) if presence_cache else None,
        scan_workers=scan_workers,
    )
    table = build_group_table(records, presence)
    plan = plan_group_split(
        table, seed=seed, ratios=ratios, tol=tol, band=band, min_groups=min_groups, retries=retries
    )
    assign_of = {k: int(plan.assign[i]) for i, k in enumerate(table.keys)}

    created: List[Path] = []
    made_root = False
    try:
        out_root.mkdir(parents=True)
        made_root = True
        if link:
            for r in records:
                split = SPLIT_NAMES[assign_of[r["group"]]]
                _link(
                    r["image"],
                    out_root / "images" / split / r["scene"] / r["image"].name,
                    created,
                )
                _link(
                    r["label"],
                    out_root / "labels" / split / r["scene"] / (r["stem"] + LABEL_SUFFIX),
                    created,
                )
            _verify_realpaths(created, root, verify_samples, seed)
        manifest_path = out_root / "group_split_manifest.json"
        manifest_path.write_text(
            json.dumps(
                _manifest(
                    root, out_root, table, plan, seed, ratios, tol, band, min_groups, retries
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except BaseException:
        for p in reversed(created):  # 자기가 만든 것만 bottom-up 으로 되돌린다
            try:
                p.unlink()
            except OSError:
                pass
        if made_root:
            shutil.rmtree(out_root, ignore_errors=True)
        raise
    return manifest_path


def _verify_realpaths(created: Sequence[Path], src_root: Path, n: int, seed: int) -> None:
    """01 §7.1 규칙 5 — 링크가 실제로 원본 트리를 가리키는지 표본 검사."""
    if not created or n <= 0:
        return
    rs = np.random.RandomState(int(seed) % (2**32))
    idx = rs.choice(len(created), size=min(int(n), len(created)), replace=False)
    # ★ src_root(asis) 자체가 원본으로의 심볼릭 링크 트리다. 링크는 src.resolve() 로 만들므로
    #   realpath 는 언제나 asis 밖(원본)을 가리킨다 — "root 안에 있는가"는 잘못된 검사였다.
    #   올바른 검사: 만든 링크가 asis 트리의 **같은 scene/파일명 항목과 동일한 실파일**을 가리키는가.
    for i in idx:
        dst = created[int(i)]
        real = os.path.realpath(dst)
        if not os.path.isfile(real):
            raise RuntimeError(f"link integrity check failed: {dst} -> {real} is not a file")
        # dst = out_root/<kind>/<split>/<scene>/<name> ; 원본 split 은 모르므로 세 split 을 모두 찾는다
        kind, _split, scene, name = dst.parts[-4], dst.parts[-3], dst.parts[-2], dst.name
        candidates = [src_root / kind / sp / scene / name for sp in ("train", "val", "test")]
        matches = [c for c in candidates if c.exists() and os.path.realpath(c) == real]
        if not matches:
            raise RuntimeError(
                f"link integrity check failed: {dst} -> {real} does not match any asis entry "
                f"{[str(c) for c in candidates if c.exists()]}"
            )


def _manifest(
    root: Path,
    out_root: Path,
    table: GroupTable,
    plan: SplitPlan,
    seed: int,
    ratios: Tuple[float, float, float],
    tol: float,
    band: float,
    min_groups: int,
    retries: int,
) -> Dict[str, Any]:
    ngw = table.n_groups_with_class
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "source_root": str(root),
        "output_root": str(out_root.resolve()),
        "seed": int(seed),
        "seed_used": int(plan.seed_used),
        "retries_used": int(plan.retries_used),
        "params": {
            "target": list(ratios),
            "tol": float(tol),
            "band": float(band),
            "min_groups": int(min_groups),
            "retries": int(retries),
        },
        "achieved_ratio": list(plan.achieved_ratio),
        "max_abs_dev": float(plan.max_abs_dev),
        "n_images_per_split": list(plan.n_images_per_split),
        "n_groups_per_split": list(plan.n_groups_per_split),
        "n_groups_total": len(table.keys),
        "per_split_class": plan.per_split_class,
        "exempt_classes": [
            {
                "id": int(k),
                "n_groups": int(ngw[k]),
                "reason": (
                    f"n_groups {int(ngw[k])} < 3 * MIN_GROUPS ({3 * min_groups})"
                    if int(ngw[k]) < 3 * min_groups
                    else "demoted after retries exhausted (01 §7.3 S4)"
                ),
            }
            for k in plan.exempt
        ],
        "zero_group_pairs": [list(p) for p in plan.zero_group_pairs],
        "class_stats_global": [float(v) for v in table.global_presence],
        "n_groups_with_class": [int(v) for v in ngw],
        "group_key_def": GROUP_KEY_DEF,
        "group_assignment": {k: SPLIT_NAMES[int(plan.assign[i])] for i, k in enumerate(table.keys)},
        "n_files_scanned": int(table.n_files_scanned),
        "remaining_violations": [[v[0], float(v[1])] for v in plan.violations],
        "search_log": [[int(a), float(b), int(c)] for a, b, c in plan.log],
        "scenes_expected": list(SCENES),
        "git_rev": git_revision(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="asis 링크 트리 루트")
    ap.add_argument("--out_root", required=True, help="group 분할 산출 루트 (기존 존재 시 중단)")
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--num_classes", type=int, default=12)
    ap.add_argument("--presence_cache", default=None)
    ap.add_argument("--scan_workers", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true", help="링크를 만들지 않고 manifest 만 쓴다")
    a = ap.parse_args(argv)
    path = make_group_split(
        Path(a.root),
        Path(a.out_root),
        seed=a.seed,
        num_classes=a.num_classes,
        presence_cache=Path(a.presence_cache) if a.presence_cache else None,
        scan_workers=a.scan_workers,
        link=not a.dry_run,
    )
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
