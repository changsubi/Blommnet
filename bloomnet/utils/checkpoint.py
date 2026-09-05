"""체크포인트 — 원자적 save/load · shape-tolerant partial load · train-only 키 제거.

정책은 05 §6.3, shape-filtered load 는 04 §8.1 을 따른다.

* **원자적 쓰기**: ``.tmp`` 에 쓰고 ``os.replace`` — 저장 중 중단으로 파일이 깨지지 않는다.
* **누적 금지**: ``best_epoch_XXX_miou_YYYY.pt`` 형태로 쌓는 이전 구현 패턴을 복사하지
  않는다 ([분석] §E.1.9). ``best.pt`` / ``last.pt`` 각 1개만 덮어쓴다.
* **shape-tolerant partial load**: 헌법 C-3(정정 A-17) 의 모드 전환 규약 —
  단일 아키텍처 정의가 모든 모드를 커버하고, 전환은 shape-tolerant partial load +
  신규 모듈 랜덤 초기화로 수행한다. K=12 → K=2 전이에서 걸러지는 키는
  ``seg_head.cls.{weight,bias}`` (+ aux head 의 cls 키) 뿐이다.

레벨 L1.
"""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from bloomnet.constants import TRAIN_ONLY_MODULES  # L−1 레벨 예외 (정정 A-23)

__all__ = [
    "TRAIN_ONLY_MODULES",
    "CHECKPOINT_REQUIRED_KEYS",
    "CHECKPOINT_RECOMMENDED_KEYS",
    "atomic_save",
    "save_checkpoint",
    "load_checkpoint",
    "load_state_dict_shape_tolerant",
    "prune_train_only_keys",
    "collect_rng_state",
    "restore_rng_state",
    "git_commit",
]

CHECKPOINT_REQUIRED_KEYS: Tuple[str, ...] = (
    "epoch", "global_step", "best_metric", "metric_name", "model_state_dict",
)
#: 05 §6.3 payload 의 나머지. 없으면 재개 재현성이 떨어지므로 경고 대상이다.
CHECKPOINT_RECOMMENDED_KEYS: Tuple[str, ...] = (
    "ema_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
    "args", "rng_state", "early_stop_counter", "class_stats_path", "git_commit",
)


def atomic_save(obj: Any, path: os.PathLike[str] | str) -> Path:
    """``path.tmp`` 에 저장한 뒤 ``os.replace`` 로 교체한다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, p)
    return p


def save_checkpoint(
    path: os.PathLike[str] | str,
    payload: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Path:
    """05 §6.3 payload 를 원자적으로 저장한다.

    Args:
        strict: True 면 :data:`CHECKPOINT_REQUIRED_KEYS` 가 전부 있어야 한다.
            빠진 키가 있으면 ``KeyError``. 조용히 저장해서 재개 시점에 터지는 것보다
            저장 시점에 터지는 편이 싸다.
    """
    if strict:
        missing = [k for k in CHECKPOINT_REQUIRED_KEYS if k not in payload]
        if missing:
            raise KeyError(f"checkpoint payload 필수 키 누락: {missing}")
    return atomic_save(dict(payload), path)


def load_checkpoint(
    path: os.PathLike[str] | str, *, map_location: Any = "cpu", weights_only: bool = False
) -> Dict[str, Any]:
    """체크포인트 로드. 기본 ``map_location="cpu"`` (헌법 C-5.2: GPU 를 잡지 않는다).

    ``weights_only=False`` 가 기본인 이유: payload 에 ``args``(dict)·``rng_state`` 같은
    비텐서 객체가 들어 있다. 신뢰할 수 없는 파일에는 ``weights_only=True`` 를 쓸 것.
    """
    return torch.load(path, map_location=map_location, weights_only=weights_only)


def load_state_dict_shape_tolerant(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    verbose: bool = True,
) -> Dict[str, List[str]]:
    """shape 이 맞는 키만 로드한다 (04 §8.1 ``load_backbone_compatible``).

    Returns:
        ``{"loaded", "missing", "unexpected", "shape_mismatch"}`` — 전부 정렬된 키 목록.
        ``missing`` 은 **로그로 반드시 출력**되어야 한다(04 §8.1). ``verbose=True`` 면
        이 함수가 출력한다.
    """
    own = model.state_dict()
    keep: Dict[str, Tensor] = {}
    shape_mismatch: List[str] = []
    for k, v in state_dict.items():
        tv = own.get(k)
        if tv is None:
            continue
        if tuple(tv.shape) != tuple(v.shape):
            shape_mismatch.append(k)
            continue
        keep[k] = v

    model.load_state_dict(keep, strict=False)
    report = {
        "loaded": sorted(keep),
        "missing": sorted(k for k in own if k not in keep),
        "unexpected": sorted(k for k in state_dict if k not in own),
        "shape_mismatch": sorted(shape_mismatch),
    }
    if verbose:
        print(
            f"[checkpoint] loaded={len(report['loaded'])} "
            f"missing={len(report['missing'])} "
            f"unexpected={len(report['unexpected'])} "
            f"shape_mismatch={len(report['shape_mismatch'])}"
        )
        for name in ("missing", "unexpected", "shape_mismatch"):
            if report[name]:
                print(f"[checkpoint] {name}: {report[name]}")
    return report


def prune_train_only_keys(
    state_dict: Mapping[str, Tensor],
    *,
    modules: Sequence[str] = TRAIN_ONLY_MODULES,
    extra_prefixes: Sequence[str] = (),
) -> Dict[str, Tensor]:
    """학습 전용 모듈의 키를 제거한다 (``best.pt`` 는 teacher/Aux/SIAM 을 담지 않는다).

    dotted 경로의 **세그먼트 단위 접두 일치**로 판정한다. 즉 ``constants.TRAIN_ONLY_MODULES``
    의 이름이 곧 계약이며, 모델은 학습 전용 모듈을 정확히 그 이름으로 등록해야 한다
    (T23 이 ``edge_head|aux_|siam`` 키 0개로 검증). 부분 문자열 일치를 쓰지 않는 이유는
    ``main_edge_head_gate`` 같은 무관한 이름을 조용히 지워 버리기 때문이다.

    Args:
        extra_prefixes: 모델이 다른 이름(예: ``aux_head_s8``)을 쓸 때 명시적으로 추가한다.
    """
    mods = tuple(modules) + tuple(extra_prefixes)

    def drop(key: str) -> bool:
        return any(seg.startswith(m) for seg in key.split(".") for m in mods)

    return {k: v for k, v in state_dict.items() if not drop(k)}


def collect_rng_state() -> Dict[str, Any]:
    """python / numpy / torch(+cuda) RNG 상태 (05 §6.3 — 재개 재현성용, 이전 구현에 없음)."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():  # pragma: no cover - 헌법 C-5.2
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """:func:`collect_rng_state` 가 만든 상태를 되돌린다. 없는 항목은 건너뛴다."""
    if state.get("python") is not None:
        random.setstate(tuple(state["python"]))
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8))
    if state.get("cuda") is not None and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all(state["cuda"])


def git_commit(repo_dir: Optional[os.PathLike[str] | str] = None) -> str:
    """``<sha>`` 또는 ``<sha>-dirty``. 저장소가 아니면 ``"unknown"``.

    05 §7.2 가 요구하는 dirty 표시를 포함한다.
    """
    cwd = str(repo_dir) if repo_dir is not None else str(Path(__file__).resolve().parent)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if sha.returncode != 0:
            return "unknown"
        rev = sha.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return rev + ("-dirty" if status.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"
