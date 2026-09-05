"""S0-Spec LOSO 평가와 **이식 게이트** — 01 §6.4/§6.8/§6.9, 06 §3.6 (레벨 L3).

이 파일이 답하는 질문은 하나다: **235 분광으로 학습한 가중치를 BloomNet 에 넣어도 되는가.**

01 [M9] 의 사전 실측(ridge, 15특징)은 8개 site 전부에서 ``R² < 0``, 평균 **−1.52** 였고
전역 평균 예측(RMSE_log 0.827)조차 이기지 못했다. 06 R-2 도 "이식 게이트 전원 실패 가능성
**높음**" 으로 사전 등록했다. **게이트 실패는 실험 실패가 아니라 "이식하지 않는다" 는 정상
경로**이며(01 §6.9 / 05 §5.3.6 정정 B-29), 이 모듈은 그 판정을 재현 가능한 수치로 남긴다.

분할 규약 (01 §6.4, ★X-22):
    * ``[S1]`` **보고용**: Leave-One-Site-Out 8-fold. §8.2 의 모든 수치는 이 평균이다.
    * ``[S2]`` **출하 체크포인트**: train = 6 site / val = {CSC, JAD}. G3 판정에만 쓴다.
    * ``[S4]`` 무작위 80/20 **금지** — [M8] 군간 분산 93.41 % 때문에 R² 를 +0.384 로
      5배 이상 과대평가한다.

레벨 L3 — L−1(`constants`), L0(`utils.*`), L2(`data.k235`, `pretrain.spec_mlp`) 만 import 한다.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from bloomnet.constants import (
    ALARM_THRESHOLDS_MGM3,
    K235_ALL_SITES,
    K235_TRAIN_SITES,
    K235_VAL_SITES,
)
from bloomnet.data.k235 import K235ChipDataset
from bloomnet.losses.regression import u_ramp
from bloomnet.pretrain.spec_mlp import SpecMLP, init_spec_mlp, spec_loss
from bloomnet.utils.metrics_reg import RegressionAccumulator
from bloomnet.utils.seed import seed_everything_strict

__all__ = [
    "RUN_DEFAULTS",
    "GATE_DEFAULTS",
    "load_k235_arrays",
    "train_spec_mlp",
    "evaluate_spec_mlp",
    "run_loso",
    "run_holdout",
    "transplant_gate",
    "format_gate_report",
]

#: ``run_loso`` / ``run_holdout`` 의 ``**cfg`` 로 받을 수 있는 키와 기본값.
#: 06 §4.2 ``spec:`` 블록의 키 이름을 그대로 쓴다 (config 값을 ``**cfg`` 로 흘려보낼 수 있게).
#: **미지 키는 조용히 무시하지 않고 TypeError** 를 낸다 — 오타 하나가 학습 설정을 소리 없이
#: 되돌리는 것이 이 프로젝트에서 가장 비싼 실패다(06 §4.2 미지 키 거부 규약과 동일 정신).
RUN_DEFAULTS: Dict[str, Any] = {
    # 데이터
    "band_order": None,  # None -> K235_BAND_ORDER_DEFAULT (가설 H2)
    "use_blue": False,  # X-03
    "quality_filter": True,  # 01 §6.3 [L1]
    "mci_c": None,  # None -> MCI_C["rededge_mx"] (235 = RedEdge)
    "band_order_check": "warn",  # {"strict","warn","off"}
    "check_msi_scale": True,  # [H1] 01 §2.6
    # 모델
    "head_out": 1,  # X-09
    "bio_gain": 4.0,
    "use_aleatoric": False,  # head_out=2 필요
    # 손실 (01 §6.6, X-11)
    "huber_beta": 0.3,
    "rank_weight": 0.2,
    "rank_margin": 0.1,
    # 최적화 (05 §5.3.4)
    "epochs": 300,
    "batch_size": 512,
    "lr": 3.0e-3,
    "weight_decay": 1.0e-4,
    "warmup_epochs": 5,
    "eta_min": 1.0e-5,
    "grad_clip": 1.0,
    "device": "cpu",  # 1,089 params — GPU 불필요 (헌법 C-5.2)
    "seed": 1234,
    # 리포트
    "alarm_thresholds": ALARM_THRESHOLDS_MGM3,
    "verbose": False,
}

#: 이식 게이트 임계 (01 §6.9 / 06 §4.2 ``spec.transplant.gate_*``).
GATE_DEFAULTS: Dict[str, float] = {
    "g1_rmse": 0.750,  # LOSO 8-fold 평균 RMSE_log
    "g2_spearman": 0.30,  # 전 fold 최소 Spearman ρ
    "g3_f1_at15": 0.60,  # [S2] val(CSC,JAD) 15 mg/m³ 임계 F1
}


# ─────────────────────────────────────────────────────────────────────────────
# cfg
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(cfg) - set(RUN_DEFAULTS))
    if unknown:
        raise TypeError(
            f"run_loso/run_holdout: 미지 설정 키 {unknown}. "
            f"허용 키 = {sorted(RUN_DEFAULTS)}"
        )
    out = dict(RUN_DEFAULTS)
    out.update(cfg)
    if out["use_aleatoric"] and int(out["head_out"]) != 2:
        raise ValueError("use_aleatoric=True 는 head_out=2 를 요구한다 (X-09)")
    if str(out["device"]).startswith("cuda"):
        # 헌법 C-5.2. 1,089 파라미터 모델에 GPU 를 쓸 이유가 없고, 요구하면 설정 오류다.
        raise ValueError("spec.device 는 cpu 다 (헌법 C-5.2, 1,089 params)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────────────────────────
def load_k235_arrays(
    npz_path: str,
    sites: Sequence[str] = K235_ALL_SITES,
    **cfg: Any,
) -> Dict[str, Any]:
    """235 npz → 학습에 바로 쓰는 텐서 묶음.

    Returns:
        ``{"x": (N,8,3,3) f32, "m": (6,) f32, "y": (N,) f32 log1p, "site": (N,) <U3,
        "group": (N,) <U12 (site|date), "chla": (N,) f64, "dataset": K235ChipDataset}``.

    Note:
        ``K235ChipDataset`` 을 **한 번만** 열고 fold 마다 index 로 자른다. 전체가
        ``21,511 × 8 × 3 × 3 × 4 B ≈ 6.2 MB`` 라 RAM 상주가 가능하다(01 §6.2 캐시 권고).
    """
    c = _resolve_cfg(cfg)
    kw: Dict[str, Any] = {
        "use_blue": bool(c["use_blue"]),
        "apply_quality_filter": bool(c["quality_filter"]),
        "band_order_check": c["band_order_check"],
        "check_msi_scale": bool(c["check_msi_scale"]),
    }
    if c["band_order"] is not None:
        kw["band_order"] = list(c["band_order"])
    if c["mci_c"] is not None:
        kw["mci_c"] = float(c["mci_c"])
    ds = K235ChipDataset(npz_path, list(sites), **kw)

    # `_x`/`_m` 은 K235ChipDataset 이 생성자에서 한 번에 만들어 두는 canonical 배열이다.
    # 없으면(구현 교체) __getitem__ 으로 폴백한다 — 값은 동일하고 속도만 다르다.
    raw_x = getattr(ds, "_x", None)
    if raw_x is not None:
        x = torch.from_numpy(np.ascontiguousarray(raw_x))  # (N,8,3,3) f32
        m = torch.from_numpy(np.ascontiguousarray(ds._m))  # (6,)
    else:  # pragma: no cover - 방어적 폴백
        items = [ds[i] for i in range(len(ds))]
        x = torch.stack([it["x"] for it in items])
        m = items[0]["m"]
    y = torch.from_numpy(np.log1p(ds.chla).astype(np.float32))  # (N,) log1p
    return {
        "x": x,
        "m": m,
        "y": y,
        "site": np.asarray(ds.sites).astype(str),
        "group": np.asarray(ds.group_ids).astype(str),
        "chla": np.asarray(ds.chla, dtype=np.float64),
        "dataset": ds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 학습 / 평가
# ─────────────────────────────────────────────────────────────────────────────
def _lr_at(epoch: int, c: Dict[str, Any]) -> float:
    """warmup(linear) → CosineAnnealing(eta_min) — 05 §5.3.4."""
    warm = int(c["warmup_epochs"])
    base, eta_min = float(c["lr"]), float(c["eta_min"])
    total = int(c["epochs"])
    if warm > 0 and epoch < warm:
        return base * float(epoch + 1) / float(warm)
    denom = max(total - warm, 1)
    t = min(max(epoch - warm, 0), denom) / denom
    return eta_min + 0.5 * (base - eta_min) * (1.0 + math.cos(math.pi * t))


def train_spec_mlp(
    x: Tensor,
    m: Tensor,
    y: Tensor,
    *,
    cfg: Dict[str, Any],
    seed: Optional[int] = None,
) -> SpecMLP:
    """SpecMLP 1개를 학습해 돌려준다 (05 §5.3.4 레시피).

    Args:
        x: ``(N,8,3,3)`` canonical 입력.
        m: ``(6,)`` 또는 ``(N,6)`` slot presence. 235 는 전 칩이 같은 slot 집합이다.
        y: ``(N,)`` log1p 공간 타깃.
        cfg: :func:`_resolve_cfg` 통과본.
        seed: fold 별 시드. None 이면 ``cfg["seed"]``.

    Note:
        ``no_decay``: 모든 bias 와 norm affine, ``C_abs`` (05 §5.3.4). ``C_abs`` 를 넣는 이유는
        그것이 "결측 slot 의 기대 기여" 라는 **bias 성 파라미터**이기 때문이다.
    """
    seed_everything_strict(int(cfg["seed"] if seed is None else seed))
    dev = torch.device(str(cfg["device"]))
    model = SpecMLP(head_out=int(cfg["head_out"])).to(dev)
    # 생성자는 bio_gain 기본값 4.0 으로 초기화한다 — cfg 값으로 한 번 더 덮어쓴다.
    init_spec_mlp(model, bio_gain=float(cfg["bio_gain"]))

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or name == "c_abs") else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(cfg["lr"]),
        betas=(0.9, 0.999),
    )

    x, y = x.to(dev), y.to(dev)
    m2 = m.to(dev)
    if m2.dim() == 1:
        m2 = m2.unsqueeze(0).expand(x.shape[0], -1)
    n = int(x.shape[0])
    bs = min(int(cfg["batch_size"]), n)
    gen = torch.Generator().manual_seed(int(cfg["seed"] if seed is None else seed))

    model.train()
    total_epochs = int(cfg["epochs"])
    for epoch in range(total_epochs):
        lr = _lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        # aleatoric degenerate-solution guard 램프. warm_frac/ramp_frac 기본값이
        # 300 에폭에서 E_warm=90 / E_ramp=60 을 주어 05 §5.3.4 와 정확히 일치한다.
        u = u_ramp(epoch, total_epochs) if bool(cfg["use_aleatoric"]) else 0.0
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            out = model(x[idx], m2[idx])
            d = spec_loss(
                out,
                y[idx],
                beta=float(cfg["huber_beta"]),
                rank_weight=float(cfg["rank_weight"]),
                rank_margin=float(cfg["rank_margin"]),
                use_aleatoric=bool(cfg["use_aleatoric"]),
                u=u,
            )
            opt.zero_grad(set_to_none=True)
            d["loss"].backward()
            if float(cfg["grad_clip"]) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate_spec_mlp(
    model: SpecMLP,
    x: Tensor,
    m: Tensor,
    y: Tensor,
    *,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """01 §6.8 지표 dict. 계산은 전부 :class:`RegressionAccumulator` 에 위임한다(X-20)."""
    dev = torch.device(str(cfg["device"]))
    model = model.to(dev).eval()
    m2 = m.to(dev)
    if m2.dim() == 1:
        m2 = m2.unsqueeze(0).expand(x.shape[0], -1)

    acc = RegressionAccumulator(
        thresholds_mgm3=tuple(float(t) for t in cfg["alarm_thresholds"]),
        max_rank_samples=max(int(x.shape[0]), 1),  # fold 전수 -> Spearman 이 근사가 아니다
        seed=int(cfg["seed"]),
    )
    bs = max(int(cfg["batch_size"]), 1)
    for i in range(0, int(x.shape[0]), bs):
        out = model(x[i : i + bs].to(dev), m2[i : i + bs])
        d = spec_loss(
            out,
            y[i : i + bs].to(dev),
            beta=float(cfg["huber_beta"]),
            rank_weight=0.0,
            use_aleatoric=False,
        )
        log_var = out[:, 1:2] if out.shape[1] > 1 else None
        acc.update(d["y_hat"], y[i : i + bs].to(dev).reshape(-1, 1), None, log_var)
    return acc.compute()


# ─────────────────────────────────────────────────────────────────────────────
# LOSO
# ─────────────────────────────────────────────────────────────────────────────
_MEAN_KEYS: Tuple[str, ...] = (
    "rmse_log",
    "mae_log",
    "r2_log",
    "bias_log",
    "spearman",
    "rmse_mgm3",
    "mae_mgm3",
    "mape_mgm3",
    "alarm_acc",
    "alarm_macro_f1",
)


def _mean_of(folds: Dict[str, Dict[str, Any]], key: str) -> float:
    vals = [
        float(v[key])
        for v in folds.values()
        if key in v and v[key] is not None and math.isfinite(float(v[key]))
    ]
    return sum(vals) / len(vals) if vals else float("nan")


def run_loso(
    npz_path: str,
    *,
    sites: Sequence[str] = K235_ALL_SITES,
    **cfg: Any,
) -> Dict[str, Any]:
    """Leave-One-Site-Out 8-fold (01 §6.4 [S1], 06 §3.6 동결).

    각 fold: ``train = sites \\ {s}``, ``test = {s}``. **site 는 절대 양쪽에 들어가지 않는다.**

    Returns:
        ``{"folds": {site: metrics}, "mean": {...}, "sites": [...], "n": {...},
        "config": {...}, "elapsed_s": float}``.
        ``metrics`` 는 :meth:`RegressionAccumulator.compute` 의 dict 이며
        ``rmse_log / mae_log / r2_log / spearman / rmse_mgm3 / f1_at_15 / f1_at_25`` 를 포함한다.

    Raises:
        ValueError: ``sites`` 가 2개 미만이면 LOSO 자체가 정의되지 않는다.
    """
    c = _resolve_cfg(cfg)
    site_list = [str(s) for s in sites]
    if len(site_list) < 2:
        raise ValueError(f"LOSO 는 site >= 2 를 요구한다, got {site_list}")

    t0 = time.time()
    data = load_k235_arrays(npz_path, site_list, **cfg)
    x, m, y, site_arr = data["x"], data["m"], data["y"], data["site"]

    folds: Dict[str, Dict[str, Any]] = {}
    n_per: Dict[str, Dict[str, int]] = {}
    for k, hold in enumerate(site_list):
        te = np.nonzero(site_arr == hold)[0]
        tr = np.nonzero(site_arr != hold)[0]
        if te.size == 0:
            raise ValueError(f"site {hold!r} 의 표본이 0개다 (npz 또는 QC 필터 확인)")
        if tr.size == 0:
            raise ValueError(f"site {hold!r} 를 빼면 학습 표본이 0개다")
        ti = torch.from_numpy(tr.astype(np.int64))
        vi = torch.from_numpy(te.astype(np.int64))
        model = train_spec_mlp(x[ti], m, y[ti], cfg=c, seed=int(c["seed"]) + k)
        folds[hold] = evaluate_spec_mlp(model, x[vi], m, y[vi], cfg=c)
        n_per[hold] = {"train": int(tr.size), "test": int(te.size)}
        if c["verbose"]:
            f = folds[hold]
            print(
                f"[LOSO] {hold:<4s} n={te.size:5d}  RMSE_log={f['rmse_log']:.4f}  "
                f"R2={f['r2_log']:+.3f}  rho={f['spearman']:+.3f}"
            )

    mean = {k: _mean_of(folds, k) for k in _MEAN_KEYS}
    mean["min_spearman"] = min(
        (float(v["spearman"]) for v in folds.values() if math.isfinite(float(v["spearman"]))),
        default=float("nan"),
    )
    mean["n_folds"] = len(folds)
    return {
        "folds": folds,
        "mean": mean,
        "sites": site_list,
        "n": n_per,
        "config": c,
        "elapsed_s": time.time() - t0,
    }


def run_holdout(
    npz_path: str,
    *,
    train_sites: Sequence[str] = K235_TRAIN_SITES,
    val_sites: Sequence[str] = K235_VAL_SITES,
    **cfg: Any,
) -> Dict[str, Any]:
    """01 §6.4 [S2] 출하 분할 — G3(``f1_at_15``) 판정용.

    06 §3.6 동결표에 없는 **추가 함수**다. ``transplant_gate(loso, holdout)`` 의 두 번째
    인자를 만들 함수가 동결표에 없어 게이트 계약이 닫히지 않는다.
    """
    c = _resolve_cfg(cfg)
    tr_s, va_s = [str(s) for s in train_sites], [str(s) for s in val_sites]
    overlap = sorted(set(tr_s) & set(va_s))
    if overlap:
        raise ValueError(f"[S2] train/val site 중복: {overlap} — 01 §6.4 누수 금지")

    t0 = time.time()
    data = load_k235_arrays(npz_path, tr_s + va_s, **cfg)
    x, m, y, site_arr = data["x"], data["m"], data["y"], data["site"]
    ti = torch.from_numpy(np.nonzero(np.isin(site_arr, tr_s))[0].astype(np.int64))
    vi = torch.from_numpy(np.nonzero(np.isin(site_arr, va_s))[0].astype(np.int64))
    if ti.numel() == 0 or vi.numel() == 0:
        raise ValueError(f"[S2] 표본 부족: train={ti.numel()} val={vi.numel()}")

    model = train_spec_mlp(x[ti], m, y[ti], cfg=c)
    metrics = evaluate_spec_mlp(model, x[vi], m, y[vi], cfg=c)
    metrics["train_sites"] = tr_s
    metrics["val_sites"] = va_s
    metrics["n_train"] = int(ti.numel())
    metrics["elapsed_s"] = time.time() - t0
    metrics["config"] = c
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 이식 게이트 (01 §6.9)
# ─────────────────────────────────────────────────────────────────────────────
def transplant_gate(
    loso_metrics: Dict[str, Any],
    holdout_metrics: Dict[str, Any],
    *,
    g1_rmse: float = GATE_DEFAULTS["g1_rmse"],
    g2_spearman: float = GATE_DEFAULTS["g2_spearman"],
    g3_f1_at15: float = GATE_DEFAULTS["g3_f1_at15"],
) -> Tuple[bool, Dict[str, bool]]:
    """이식 게이트 G1/G2/G3 판정 (01 §6.9, 06 §3.6 동결).

    ============ ==========================================================
    G1           LOSO 8-fold **평균** ``RMSE_log <= 0.750``
    G2           **모든 fold** 에서 ``Spearman ρ >= 0.30``
    G3           [S2] val(CSC, JAD) 의 ``F1@15 mg/m³ >= 0.60``
    ============ ==========================================================

    Args:
        loso_metrics: :func:`run_loso` 의 반환값. ``{"folds":…, "mean":…}`` 를 기대하지만
            ``{"rmse_log":…, "spearman":…}`` 만 담긴 평탄한 dict 도 받는다(단일 fold 진단용).
        holdout_metrics: :func:`run_holdout` 반환값(또는 ``f1_at_15`` 를 가진 dict).

    Returns:
        ``(all_pass, {"G1": bool, "G2": bool, "G3": bool})``.
        하나라도 False 면 **T1~T5 를 전부 수행하지 않는다** (01 §6.7 T7).

    Note:
        판정 불가(``nan`` / ``None``)는 **실패로 취급**한다. G3 의 ``f1_at_15`` 가 ``None`` 인
        경우는 "양성 표본이 0개" 라는 뜻이고(정정 A-30 / B-30), 그것은 "성능이 좋다" 가
        아니라 "판정할 수 없다" 이므로 이식을 허용해서는 안 된다.
    """
    mean = loso_metrics.get("mean", loso_metrics)
    folds = loso_metrics.get("folds", None)

    g1 = _finite_le(mean.get("rmse_log"), float(g1_rmse))

    if folds:
        rhos = [f.get("spearman") for f in folds.values()]
        g2 = bool(rhos) and all(_finite_ge(r, float(g2_spearman)) for r in rhos)
    else:
        g2 = _finite_ge(mean.get("min_spearman", mean.get("spearman")), float(g2_spearman))

    g3 = _finite_ge(holdout_metrics.get("f1_at_15"), float(g3_f1_at15))

    detail = {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3)}
    return (all(detail.values()), detail)


def _finite_le(v: Any, thr: float) -> bool:
    return v is not None and math.isfinite(float(v)) and float(v) <= thr


def _finite_ge(v: Any, thr: float) -> bool:
    return v is not None and math.isfinite(float(v)) and float(v) >= thr


def format_gate_report(
    loso_metrics: Dict[str, Any],
    holdout_metrics: Dict[str, Any],
    gate: Tuple[bool, Dict[str, bool]],
    *,
    g1_rmse: float = GATE_DEFAULTS["g1_rmse"],
    g2_spearman: float = GATE_DEFAULTS["g2_spearman"],
    g3_f1_at15: float = GATE_DEFAULTS["g3_f1_at15"],
) -> str:
    """사람이 읽는 게이트 리포트. 01 §6.9 는 "실패 사실과 수치를 그대로 기록" 을 요구한다."""
    ok, detail = gate
    mean = loso_metrics.get("mean", loso_metrics)
    folds: Dict[str, Dict[str, Any]] = loso_metrics.get("folds", {})
    lines: List[str] = []
    lines.append("S0-Spec 이식 게이트 (01 §6.9)")
    lines.append(f"  LOSO fold 수 : {mean.get('n_folds', len(folds))}")
    for site, f in folds.items():
        lines.append(
            f"    {site:<4s} n={f.get('n_valid', 0):6d}  "
            f"RMSE_log={f.get('rmse_log', float('nan')):.4f}"
            f"  MAE={f.get('mae_log', float('nan')):.4f}"
            f"  R2={f.get('r2_log', float('nan')):+.4f}"
            f"  rho={f.get('spearman', float('nan')):+.4f}"
        )
    lines.append(
        f"  평균: RMSE_log={mean.get('rmse_log', float('nan')):.4f}"
        f"  MAE={mean.get('mae_log', float('nan')):.4f}"
        f"  R2={mean.get('r2_log', float('nan')):+.4f}"
    )
    lines.append("  베이스라인(01 [M9]): 전역평균 0.827 / ridge 0.810 / SPS호환 6ch 0.877")
    lines.append(
        f"  G1 mean RMSE_log <= {g1_rmse:.3f} : "
        f"{mean.get('rmse_log', float('nan')):.4f} -> {'PASS' if detail['G1'] else 'FAIL'}"
    )
    lines.append(
        f"  G2 all rho >= {g2_spearman:.2f}      : "
        f"min={mean.get('min_spearman', float('nan')):+.4f} -> {'PASS' if detail['G2'] else 'FAIL'}"
    )
    f1 = holdout_metrics.get("f1_at_15")
    f1s = "None" if f1 is None else f"{float(f1):.4f}"
    lines.append(
        f"  G3 val F1@15 >= {g3_f1_at15:.2f}    : {f1s} -> {'PASS' if detail['G3'] else 'FAIL'}"
    )
    lines.append(f"  판정: {'이식 수행' if ok else '이식 금지 — SPS/ChlHead 랜덤 초기화로 진행'}")
    return "\n".join(lines)
