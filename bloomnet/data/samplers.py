"""샘플러 — 05 §4.3 / 06 §3.2.6.

* :class:`RepeatFactorSampler` — LVIS(Gupta 2019) RFS. 06 §3.2.6 동결 시그니처.
* :class:`GroupBatchSampler` — 06 §2.1.3 이 이름만 지정하고 시그니처는 동결하지 않았다.
  누수 없는 group split(01 §7.3)에서 "같은 군의 샘플을 한 배치에 모으는" 진단·평가용.

레벨 L1. `bloomnet.constants` 외에는 `bloomnet.*` 를 import 하지 않는다.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

__all__ = ["RepeatFactorSampler", "GroupBatchSampler"]


class RepeatFactorSampler(torch.utils.data.Sampler):
    """이미지별 반복계수 샘플러 (05 §4.3 전략 1).

    ``f_img_c`` = 클래스 c 를 포함하는 이미지 비율,
    ``r_c = max(1, sqrt(t / f_img_c))``, ``r_i = max_{c ∈ i} r_c``.
    에폭마다 ``floor(r_i) + Bernoulli(frac(r_i))`` 회 샘플한다.

    Args:
        class_presence: ``(N, K)`` bool. ``[i, c]`` = 이미지 i 에 클래스 c 가 존재.
        t: 임계값. 05 §4.3 채택값 0.05 (에폭 비용 +4.7 %, class 9 노출 4.66배).
        seed: 기준 seed. 실제 난수는 ``seed + epoch``.
        ignore_class_ids: 반복계수 산정에서 제외할 클래스 (기본 background=0).

    Note:
        05 §4.3 측정치(t=0.05, 3,000장 표본): c7 2.104 / c8 2.397 / c9 4.663 /
        c10 1.046 / c11 1.569. **표본값이므로 배치 확률 계산에 쓰지 않는다** (정정 B-25).
        실제 배율은 선택된 split 의 ``class_stats.json`` 에서 매번 재계산한다 (정정 B-26).
    """

    def __init__(
        self,
        class_presence: np.ndarray,
        *,
        t: float = 0.05,
        seed: int = 1234,
        ignore_class_ids: Sequence[int] = (0,),
    ) -> None:
        presence = np.asarray(class_presence)
        if presence.ndim != 2:
            raise ValueError(f"class_presence must be (N,K), got {presence.shape}")
        if presence.shape[0] == 0:
            raise ValueError("class_presence must have at least one image")
        if not (0.0 <= float(t) <= 1.0):
            raise ValueError(f"t must be in [0,1], got {t}")
        self.presence = presence.astype(bool, copy=False)
        self.t = float(t)
        self.seed = int(seed)
        self.ignore_class_ids = tuple(int(c) for c in ignore_class_ids)

        n, k = self.presence.shape
        f_img = self.presence.mean(axis=0)                       # (K,)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_c = np.where(f_img > 0.0, np.sqrt(self.t / np.maximum(f_img, 1e-12)), 1.0)
        r_c = np.maximum(1.0, r_c)
        for c in self.ignore_class_ids:
            if 0 <= c < k:
                r_c[c] = 1.0
        self.class_repeat_factors = r_c.astype(np.float64)

        masked = np.where(self.presence, self.class_repeat_factors[None, :], 1.0)
        self.repeat_factors = masked.max(axis=1).astype(np.float64)   # (N,)
        self._epoch = 0
        self._indices: List[int] = []
        self._build(0)

    # ── API ──────────────────────────────────────────────────────────────────
    def set_epoch(self, epoch: int) -> None:
        """에폭을 바꾸면 stochastic rounding 을 다시 뽑는다 (재현 가능)."""
        self._epoch = int(epoch)
        self._build(self._epoch)

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def expected_epoch_size(self) -> float:
        """기대 에폭 크기 (= Σ r_i). 실제 길이는 Bernoulli 때문에 매 에폭 조금 달라진다."""
        return float(self.repeat_factors.sum())

    # ── internals ────────────────────────────────────────────────────────────
    def _build(self, epoch: int) -> None:
        rs = np.random.RandomState((self.seed + epoch) % (2**32))
        base = np.floor(self.repeat_factors).astype(np.int64)
        frac = self.repeat_factors - base
        extra = (rs.random_sample(base.shape) < frac).astype(np.int64)
        reps = base + extra
        idx = np.repeat(np.arange(reps.shape[0], dtype=np.int64), reps)
        rs.shuffle(idx)
        self._indices = idx.tolist()


class GroupBatchSampler(torch.utils.data.Sampler):
    """같은 군(group)의 샘플을 한 배치에 모으는 결정론적 배치 샘플러.

    ★ 06 §2.1.3 은 파일 소유만 지정하고 시그니처를 동결하지 않았다. 본 구현은
    01 §7.3 의 group 키(scene, admin, date, line)를 배치 경계로 쓰는 진단·평가용이며,
    기본 학습 경로(``sampler.kind ∈ {shuffle, rfs}``)에서는 사용하지 않는다.

    Args:
        group_ids: 길이 N. 각 샘플의 군 식별자 (hashable).
        batch_size: 배치 크기.
        drop_last: True 면 batch_size 미만 잔여 배치를 버린다.
        shuffle: 군 순서와 군 내부 순서를 섞는다 (``seed + epoch`` 로 재현).
        seed: 기준 seed.
    """

    def __init__(
        self,
        group_ids: Sequence[Any],
        batch_size: int,
        *,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 1234,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.group_ids = list(group_ids)
        if not self.group_ids:
            raise ValueError("group_ids must not be empty")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        buckets: Dict[Any, List[int]] = {}
        for i, g in enumerate(self.group_ids):
            buckets.setdefault(g, []).append(i)
        # 결정론을 위해 최초 등장 순서로 고정
        self._order: List[Any] = []
        seen = set()
        for g in self.group_ids:
            if g not in seen:
                seen.add(g)
                self._order.append(g)
        self._buckets = buckets
        self._epoch = 0
        self._batches: List[List[int]] = []
        self._build(0)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._build(self._epoch)

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches)

    def _build(self, epoch: int) -> None:
        rs: Optional[np.random.RandomState] = None
        order = list(self._order)
        if self.shuffle:
            rs = np.random.RandomState((self.seed + epoch) % (2**32))
            perm = rs.permutation(len(order))
            order = [order[i] for i in perm]
        batches: List[List[int]] = []
        for g in order:
            idx = list(self._buckets[g])
            if rs is not None:
                p = rs.permutation(len(idx))
                idx = [idx[i] for i in p]
            n_full = len(idx) // self.batch_size
            for b in range(n_full):
                batches.append(idx[b * self.batch_size : (b + 1) * self.batch_size])
            rest = idx[n_full * self.batch_size :]
            if rest and not self.drop_last:
                batches.append(rest)
        self._batches = batches

    @property
    def num_groups(self) -> int:
        return len(self._buckets)

    @property
    def expected_num_batches(self) -> int:
        """``drop_last`` 를 반영한 배치 수 (검산용)."""
        total = 0
        for idx in self._buckets.values():
            if self.drop_last:
                total += len(idx) // self.batch_size
            else:
                total += math.ceil(len(idx) / self.batch_size)
        return total
