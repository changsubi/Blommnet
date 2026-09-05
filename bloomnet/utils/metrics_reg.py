"""Chl-a 회귀 지표 — 스트리밍 RMSE/MAE/R²/Spearman/경보 F1 + log1p↔mg/m³ 변환.

지표 정의는 01 §6.8, 리포트 스키마는 05 §6.1 ``val_regression_metrics.csv`` 를 따른다.

**타깃 공간 (★X-14)**: ``y_chl`` 은 이미 ``log1p(mg/m³)`` 공간이다. 이 모듈은 log 를 **다시**
씌우지 않는다. 원단위 지표는 :func:`log1p_to_mgm3` 로 되돌린 뒤 계산한다.

**support=0 경보 구간 (정정 A-30 / B-30)**: 235 에는 ``>= 100 mg/m³`` 표본이 0개다(01 [M5]).
support 가 0 인 구간은 macro-F1 에서 **자동 제외**하고 제외 목록과 사유를 함께 반환한다.
``f1_at_100`` 은 값 대신 ``None`` + 사유 문자열이 된다.

레벨 L0. ``constants.py`` 는 레벨 예외(정정 A-23)이므로 import 할 수 있다.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from bloomnet.constants import (  # L−1 레벨 예외 (정정 A-23)
    ALARM_LEVEL_NAMES,
    ALARM_THRESHOLDS_LOG1P,
    ALARM_THRESHOLDS_MGM3,
)

__all__ = [
    "ALARM_THRESHOLDS_MGM3",
    "ALARM_THRESHOLDS_LOG1P",
    "ALARM_LEVEL_NAMES",
    "LOG1P_SANE_MAX",
    "log1p_to_mgm3",
    "mgm3_to_log1p",
    "alarm_level",
    "spearman_corr",
    "RegressionAccumulator",
]

#: ``log1p(2980) = 8.0``. 이 값을 넘으면 이중 log 변환 의심 (★X-14 하드가드).
LOG1P_SANE_MAX: float = 8.0

_LOG_2PI: float = math.log(2.0 * math.pi)


def _at_least_f32(x: Tensor) -> Tensor:
    """정수·fp16 은 float32 로 올리고, float64 는 **그대로 둔다**(누적 정밀도 보존)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def log1p_to_mgm3(y_log1p: Tensor) -> Tensor:
    """log1p 공간 → mg/m³.

    ``torch.expm1`` 을 쓰지 않는다 — ONNX 에 ``Expm1`` 연산자가 없다(정정 B-13).
    코드베이스 전체가 ``exp(x) - 1.0`` 형태로 통일되어야 배포 경로와 평가 경로가 같은 값을 낸다.
    """
    return torch.exp(_at_least_f32(y_log1p)) - 1.0


def mgm3_to_log1p(y_mgm3: Tensor) -> Tensor:
    """mg/m³ → log1p 공간. 음수 라벨은 0 으로 방어한다 (05 §2.5.3 규칙 N5)."""
    return torch.log1p(_at_least_f32(y_mgm3).clamp_min(0.0))


def alarm_level(
    chl_mgm3: Tensor,
    *,
    thresholds: Sequence[float] = ALARM_THRESHOLDS_MGM3,
) -> Tensor:
    """mg/m³ → 경보 단계 {0 정상, 1 관심, 2 경계, 3 대발생} (헌법 C-1: 후처리, 학습 아님).

    경계값은 **임계값 이상**이 상위 단계다 (``x >= 15`` → 관심). ``torch.bucketize`` 에서
    이 규약은 ``right=True`` (= ``boundaries[i-1] <= x < boundaries[i]``) 다.
    ``right=False`` 를 쓰면 ``x == 15.0`` 이 정상으로 떨어져 경보를 1단계 놓친다.
    """
    x = _at_least_f32(chl_mgm3)
    b = torch.as_tensor(list(thresholds), dtype=x.dtype, device=x.device)
    return torch.bucketize(x, b, right=True)


def spearman_corr(a: Tensor, b: Tensor) -> float:
    """Spearman 순위상관 (동점은 평균 순위). 표본 < 2 또는 순위 분산 0 이면 ``nan``."""
    a = a.detach().reshape(-1).to(torch.float64)
    b = b.detach().reshape(-1).to(torch.float64)
    if a.numel() != b.numel():
        raise ValueError(f"length mismatch: {a.numel()} vs {b.numel()}")
    if a.numel() < 2:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.sqrt((ra * ra).sum() * (rb * rb).sum())
    if float(denom) <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _average_ranks(x: Tensor) -> Tensor:
    """동점에 평균 순위를 부여한다 (1-based, 값 자체는 상관계수에 영향 없음)."""
    n = x.numel()
    order = torch.argsort(x, stable=True)
    sorted_x = x[order]
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and bool(sorted_x[j] == sorted_x[i]):
            j += 1
        ranks[i:j] = (i + j - 1) / 2.0
        i = j
    out = torch.empty(n, dtype=torch.float64)
    out[order] = ranks
    return out


class RegressionAccumulator:
    """Chl-a 회귀 스트리밍 누적기 (01 §6.8 지표 + 05 §6.1 CSV 스키마).

    전 지표를 배치마다 **누적합**으로 갱신하므로 전체 예측을 메모리에 들고 있지 않는다.
    예외는 Spearman 두 개뿐이며, 이들은 고정 크기 저수지 표집(reservoir sampling)을 쓴다.

    Args:
        thresholds_mgm3: 경보 임계값 (사업문서 s5).
        max_rank_samples: Spearman 용 저수지 크기. 0 이면 Spearman 을 계산하지 않는다.
        seed: 저수지 표집 RNG 시드 (동일 입력 순서 → 동일 결과).
        debug_assert: True 면 타깃이 log1p 공간인지 검사한다 (★X-14).
            hot path 동기화를 만들지 않도록 **기본값 False** 이며(정정 A-33 정책),
            평가 루프는 첫 배치에만 켜는 것을 권장한다.
    """

    def __init__(
        self,
        *,
        thresholds_mgm3: Sequence[float] = ALARM_THRESHOLDS_MGM3,
        max_rank_samples: int = 200_000,
        seed: int = 0,
        debug_assert: bool = False,
    ) -> None:
        self.thresholds_mgm3: Tuple[float, ...] = tuple(float(t) for t in thresholds_mgm3)
        self.n_levels: int = len(self.thresholds_mgm3) + 1
        self.max_rank_samples = int(max_rank_samples)
        self.seed = int(seed)
        self.debug_assert = bool(debug_assert)
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        self.n: int = 0
        self._sum_y = 0.0
        self._sum_y2 = 0.0
        self._sum_e = 0.0
        self._sum_e2 = 0.0
        self._sum_abs_e = 0.0
        self._sum_e2_mgm3 = 0.0
        self._sum_abs_e_mgm3 = 0.0
        self._sum_ape = 0.0            # |e| / max(|y|, eps),  y > 0 인 표본만
        self._n_ape = 0
        self._sum_sigma = 0.0
        self._sum_nll = 0.0
        self._n_nll = 0
        # 경보: (n_levels, n_levels) GT×pred 혼동행렬
        self._alarm_cm = torch.zeros((self.n_levels, self.n_levels), dtype=torch.int64)
        self._rng = np.random.default_rng(self.seed)
        self._seen_for_rank = 0
        #: 저수지 (n, 4) float64 — 열 = [pred, true, |err|, sigma]
        self._res = torch.empty((0, 4), dtype=torch.float64)

    # --------------------------------------------------------------- update
    def update(
        self,
        pred_log1p: Tensor,
        target_log1p: Tensor,
        mask: Optional[Tensor] = None,
        log_var: Optional[Tensor] = None,
    ) -> None:
        """유효 픽셀만 누적한다.

        Args:
            pred_log1p: 예측 ``u`` (softplus 출력, log1p 공간). 임의 shape.
            target_log1p: 타깃 ``y_chl`` (**log1p 공간**, ★X-14). ``pred`` 와 broadcast 가능해야 한다.
            mask: 유효 픽셀 bool 마스크. None 이면 전부 유효.
            log_var: ``s = log σ²`` (log1p 공간 기준). 주면 NLL·σ 지표가 활성화된다.
        """
        try:
            pred, true = torch.broadcast_tensors(
                pred_log1p.detach().float(), target_log1p.detach().float()
            )
        except RuntimeError as exc:  # 해상도 불일치는 호출자가 업샘플로 해소해야 한다
            raise ValueError(
                f"pred {tuple(pred_log1p.shape)} 와 target {tuple(target_log1p.shape)} 가 "
                "broadcast 되지 않는다. 로짓/타깃 해상도를 먼저 맞춰라 (04 §8.1)."
            ) from exc

        sel = torch.isfinite(pred) & torch.isfinite(true)
        if mask is not None:
            sel = sel & mask.detach().to(torch.bool).expand_as(pred)
        s_all: Optional[Tensor] = None
        if log_var is not None:
            s_all = log_var.detach().float().expand_as(pred)
            sel = sel & torch.isfinite(s_all)

        # 누적합은 **float64** 로 한다. 픽셀 회귀는 배치당 10⁵~10⁶ 표본이라
        # float32 누산은 유효숫자를 잃고, 배치 분할 방식에 따라 값이 달라진다.
        pred = pred[sel].double()
        true = true[sel].double()
        if pred.numel() == 0:
            return

        if self.debug_assert:
            hi = float(true.max())
            if hi >= LOG1P_SANE_MAX:
                raise AssertionError(
                    f"target max {hi:.4f} >= {LOG1P_SANE_MAX} — "
                    "y_chl 은 log1p(mg/m3) 공간이어야 한다 (X-14). 원단위를 넣은 것 같다."
                )

        err = pred - true
        self.n += int(pred.numel())
        self._sum_y += float(true.sum())
        self._sum_y2 += float((true * true).sum())
        self._sum_e += float(err.sum())
        self._sum_e2 += float((err * err).sum())
        self._sum_abs_e += float(err.abs().sum())

        pred_mg = log1p_to_mgm3(pred)
        true_mg = log1p_to_mgm3(true)
        err_mg = pred_mg - true_mg
        self._sum_e2_mgm3 += float((err_mg * err_mg).sum())
        self._sum_abs_e_mgm3 += float(err_mg.abs().sum())
        pos = true_mg > 1e-6
        if bool(pos.any()):
            self._sum_ape += float((err_mg[pos].abs() / true_mg[pos]).sum())
            self._n_ape += int(pos.sum())

        lv_true = alarm_level(true_mg, thresholds=self.thresholds_mgm3)
        lv_pred = alarm_level(pred_mg, thresholds=self.thresholds_mgm3)
        idx = lv_true.to(torch.int64) * self.n_levels + lv_pred.to(torch.int64)
        self._alarm_cm += torch.bincount(
            idx, minlength=self.n_levels * self.n_levels
        ).reshape(self.n_levels, self.n_levels)

        sigma: Optional[Tensor] = None
        if s_all is not None:
            s = s_all[sel].double()
            sigma = torch.exp(0.5 * s)
            # Gaussian NLL (log1p 공간). 학습 손실의 Huber-NLL(05 §2.5.2)과는 다른,
            # 문헌 비교가 가능한 표준형이다. 상수항 0.5·log(2π) 를 포함한다.
            nll = 0.5 * (s + err * err * torch.exp(-s) + _LOG_2PI)
            self._sum_nll += float(nll.sum())
            self._n_nll += int(nll.numel())
            self._sum_sigma += float(sigma.sum())

        self._reservoir(pred, true, sigma, err.abs())

    def _reservoir(
        self, pred: Tensor, true: Tensor, sigma: Optional[Tensor], abserr: Tensor
    ) -> None:
        """블록 단위 저수지 표집 — 스트림 전체의 균등 ``cap``-부분집합을 유지한다.

        원소별 Algorithm R 을 파이썬 루프로 돌리면 배치당 10⁶ 픽셀에서 못 쓴다.
        대신 "새 블록에서 몇 개가 최종 저수지에 남는가" 를
        **초기하분포 Hypergeometric(seen, m, cap)** 에서 한 번 뽑고 벡터화해 교체한다.
        결과 분포는 원소별 저수지 표집과 동일하다.
        """
        cap = self.max_rank_samples
        if cap <= 0:
            return
        sg = sigma if sigma is not None else torch.full_like(pred, float("nan"))
        block = torch.stack([pred, true, abserr, sg], dim=1)

        room = cap - int(self._res.shape[0])
        if room > 0:
            take = min(room, int(block.shape[0]))
            self._res = torch.cat([self._res, block[:take]], dim=0)
            self._seen_for_rank += take
            block = block[take:]
        m = int(block.shape[0])
        if m == 0:
            return

        # 최종 cap 개 중 새 블록에서 오는 개수
        n_new = int(self._rng.hypergeometric(m, self._seen_for_rank, cap))
        self._seen_for_rank += m
        if n_new == 0:
            return
        keep_old = self._rng.choice(cap, size=cap - n_new, replace=False)
        take_new = self._rng.choice(m, size=n_new, replace=False)
        self._res = torch.cat(
            [
                self._res[torch.from_numpy(np.sort(keep_old)).long()],
                block[torch.from_numpy(np.sort(take_new)).long()],
            ],
            dim=0,
        )

    # -------------------------------------------------------------- compute
    def compute(self) -> Dict[str, Any]:
        """지표 dict. 표본 0 이면 수치 항이 전부 ``nan`` 이고 ``n_valid == 0`` 이다."""
        out: Dict[str, Any] = {"n_valid": self.n}
        if self.n == 0:
            for k in (
                "mae_log", "rmse_log", "r2_log", "bias_log",
                "mae_mgm3", "rmse_mgm3", "mape_mgm3",
                "spearman", "nll", "sigma_mean", "sigma_err_spearman",
                "alarm_acc", "alarm_macro_f1",
            ):
                out[k] = float("nan")
            for t in self.thresholds_mgm3:
                out[_f1_key(t)] = None
            out["n_rank_samples"] = 0
            out["alarm_excluded_levels"] = list(range(self.n_levels))
            out["alarm_exclusion_reasons"] = {
                str(i): "no samples (empty accumulator)" for i in range(self.n_levels)
            }
            return out

        n = float(self.n)
        out["mae_log"] = self._sum_abs_e / n
        out["rmse_log"] = math.sqrt(max(self._sum_e2 / n, 0.0))
        out["bias_log"] = self._sum_e / n
        ss_tot = self._sum_y2 - (self._sum_y * self._sum_y) / n
        out["r2_log"] = float("nan") if ss_tot <= 0.0 else 1.0 - (self._sum_e2 / ss_tot)
        out["mae_mgm3"] = self._sum_abs_e_mgm3 / n
        out["rmse_mgm3"] = math.sqrt(max(self._sum_e2_mgm3 / n, 0.0))
        out["mape_mgm3"] = (self._sum_ape / self._n_ape) if self._n_ape else float("nan")

        if self._res.shape[0] > 0:
            out["spearman"] = spearman_corr(self._res[:, 0], self._res[:, 1])
            sg = self._res[:, 3]
            if bool(torch.isfinite(sg).all()):
                out["sigma_err_spearman"] = spearman_corr(sg, self._res[:, 2])
            else:
                out["sigma_err_spearman"] = float("nan")
            out["n_rank_samples"] = int(self._res.shape[0])
        else:
            out["spearman"] = float("nan")
            out["sigma_err_spearman"] = float("nan")
            out["n_rank_samples"] = 0

        out["nll"] = (self._sum_nll / self._n_nll) if self._n_nll else float("nan")
        out["sigma_mean"] = (self._sum_sigma / self._n_nll) if self._n_nll else float("nan")

        out.update(self._alarm_metrics())
        return out

    def _alarm_metrics(self) -> Dict[str, Any]:
        cm = self._alarm_cm.to(torch.float64)
        total = float(cm.sum())
        res: Dict[str, Any] = {}
        res["alarm_acc"] = float(torch.diag(cm).sum()) / total if total > 0 else float("nan")

        support = cm.sum(dim=1)
        f1s: List[float] = []
        excluded: List[int] = []
        reasons: Dict[str, str] = {}
        for lv in range(self.n_levels):
            if float(support[lv]) <= 0.0:
                excluded.append(lv)
                reasons[str(lv)] = _no_sample_reason(lv, self.thresholds_mgm3)
                continue
            tp = float(cm[lv, lv])
            fn = float(support[lv]) - tp
            fp = float(cm[:, lv].sum()) - tp
            denom = 2.0 * tp + fp + fn
            f1s.append(0.0 if denom <= 0.0 else (2.0 * tp) / denom)
        # (정정 A-30) support=0 구간을 포함하면 그 F1 이 0/NaN 이 되어 평균을 최대 25 % 왜곡한다.
        res["alarm_macro_f1"] = (sum(f1s) / len(f1s)) if f1s else float("nan")
        res["alarm_excluded_levels"] = excluded
        res["alarm_exclusion_reasons"] = reasons

        # 임계값별 이진 F1: GT >= t 를 양성으로.
        for k, t in enumerate(self.thresholds_mgm3):
            hi = slice(k + 1, self.n_levels)
            tp = float(cm[hi, hi].sum())
            fn = float(cm[hi, : k + 1].sum())
            fp = float(cm[: k + 1, hi].sum())
            key = _f1_key(t)
            if tp + fn <= 0.0:
                res[key] = None
                res[key + "_reason"] = _no_sample_reason(k + 1, self.thresholds_mgm3)
                continue
            denom = 2.0 * tp + fp + fn
            res[key] = 0.0 if denom <= 0.0 else (2.0 * tp) / denom
        return res


def _f1_key(threshold_mgm3: float) -> str:
    return f"f1_at_{int(round(threshold_mgm3))}"


def _no_sample_reason(level: int, thresholds: Sequence[float]) -> str:
    if level >= len(thresholds):
        lo = thresholds[-1] if thresholds else 0.0
        return f"no samples >= {lo:g} mg/m3 in k235"
    if level == 0:
        return f"no samples < {thresholds[0]:g} mg/m3"
    return f"no samples in [{thresholds[level - 1]:g}, {thresholds[level]:g}) mg/m3"
