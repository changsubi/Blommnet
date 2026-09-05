"""분산 학습 헬퍼 — rank/world 조회와 ``SyncBatchNorm`` 승격 (04 §1.2).

04 §1.2 규약: **effective batch < 8 이면 ``SyncBatchNorm`` 으로 승격**한다
(단일 GPU 의 작은 batch 에서 BN 통계가 무너지는 것을 막는다).

분산이 초기화되지 않은 환경(= 본 개발·테스트 환경, 헌법 C-5.2)에서도 모든 함수가
예외 없이 동작해야 한다. 초기화 전에는 rank 0 / world 1 로 답한다.

레벨 L1.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

__all__ = [
    "SYNCBN_BATCH_THRESHOLD",
    "is_dist_available",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "barrier",
    "all_reduce_mean",
    "all_reduce_sum",
    "convert_syncbn",
    "maybe_convert_syncbn",
]

#: 04 §1.2 — effective batch 가 이 값 **미만**이면 SyncBatchNorm 으로 승격한다.
SYNCBN_BATCH_THRESHOLD: int = 8


def is_dist_available() -> bool:
    """``torch.distributed`` 가 사용 가능하고 process group 이 초기화되었는가."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available() else 1


def is_main_process() -> bool:
    """rank 0 인가. 로깅·체크포인트 저장의 게이트."""
    return get_rank() == 0


def barrier() -> None:
    """분산이 아니면 무연산."""
    if is_dist_available():
        dist.barrier()


def all_reduce_sum(tensor: Tensor) -> Tensor:
    """전 rank 합. 분산이 아니면 입력을 그대로 돌려준다(복사 없음)."""
    if not is_dist_available():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_mean(tensor: Tensor) -> Tensor:
    """전 rank 평균. 분산이 아니면 입력을 그대로 돌려준다."""
    if not is_dist_available():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def convert_syncbn(model: nn.Module) -> nn.Module:
    """조건 없이 모든 BatchNorm 을 ``SyncBatchNorm`` 으로 교체한다.

    정책 판단 없이 변환만 한다. 정책은 :func:`maybe_convert_syncbn` 이 담당한다.
    """
    return nn.SyncBatchNorm.convert_sync_batchnorm(model)


def maybe_convert_syncbn(
    model: nn.Module,
    *,
    effective_batch_size: int,
    threshold: int = SYNCBN_BATCH_THRESHOLD,
    force: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """04 §1.2 정책에 따라 조건부로 SyncBatchNorm 으로 승격한다.

    Args:
        effective_batch_size: rank 당 batch × grad accumulation (per-device 유효 batch).
        force: True 면 분산 여부와 무관하게 변환한다(디버그·테스트 전용).

    Returns:
        ``(model, info)``. ``info`` 는 ``{"converted": bool, "reason": str,
        "world_size": int, "effective_batch_size": int}``.
        **변환하지 않았을 때도 사유 문자열을 남긴다** — 조용히 건너뛰면
        BN 통계 붕괴를 사후에 추적할 수 없다.
    """
    info: Dict[str, Any] = {
        "converted": False,
        "reason": "",
        "world_size": get_world_size(),
        "effective_batch_size": int(effective_batch_size),
    }
    if force:
        info.update(converted=True, reason="force=True")
        return convert_syncbn(model), info
    if not is_dist_available():
        info["reason"] = "distributed not initialized"
        return model, info
    if get_world_size() < 2:
        info["reason"] = "world_size < 2"
        return model, info
    if effective_batch_size >= threshold:
        info["reason"] = f"effective_batch_size {effective_batch_size} >= threshold {threshold}"
        return model, info
    info.update(
        converted=True,
        reason=f"effective_batch_size {effective_batch_size} < threshold {threshold} (04 §1.2)",
    )
    return convert_syncbn(model), info
