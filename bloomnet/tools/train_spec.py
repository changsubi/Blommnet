"""S0-Spec 진입점 — 235 칩 Chl-a 회귀 + LOSO + 이식 게이트 (01 §6, 05 §5.3).

CPU 로 오늘 실행 가능한 유일한 학습이다 (`SpecMLP` 1,089 params).

★ 이 스크립트는 **이식(transplant)을 수행하지 않는다** (정정 A-19).
  게이트 G1~G3 결과와 `spec.band_order_confirmed` 상태를 JSON 으로 남기는 것까지가 계약이다.
  근거: 02 §12-U2 "밴드 순서 확정 전 235 로 SPS 학습 금지", 05 §5.3.2, 06 V12/V18.
  실제 이식은 게이트 통과 + `band_order_confirmed=true` 확정 후 `pretrain/transplant.py` 로
  별도 수행하고, 그 산출물을 `train.py --init_from` 으로 넘긴다.

사용 예 (cwd = <repo_root>)
    CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.train_spec \\
        --config bloomnet/configs/s0_spec_k235.yaml
    CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.train_spec --config s0_spec_k235 \\
        --build_only          # SpecMLP 만 만들어 파라미터 수(1,089)를 확인하고 종료
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Sequence

from bloomnet.tools._cli import (
    banner,
    build_tool_parser,
    handle_common_flags,
    parse_tool_cli,
    prepare_run_dir,
    require_module,
    resolve_out_path,
    write_json,
)

__all__ = ["main"]

# `run_loso(npz_path, *, sites, **cfg)` 에 넘길 spec 하위 키 (06 §3.6 의 **cfg).
_LOSO_KEYS = (
    "band_order",
    "use_blue",
    "quality_filter",
    "head_out",
    "huber_beta",
    "rank_weight",
    "rank_margin",
    "bio_gain",
    "mci_c",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "device",
)


def _loso_kwargs(cfg: Any) -> Dict[str, Any]:
    kw = {k: getattr(cfg.spec, k) for k in _LOSO_KEYS}
    kw["seed"] = int(cfg.seed)
    return kw


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_tool_parser("BloomNet S0-Spec (235 칩 Chl-a 회귀 + LOSO + 이식 게이트)")
    parser.add_argument("--npz", default=None, help="k235 npz 캐시 경로 (기본 spec.npz)")
    parser.add_argument(
        "--report",
        default=None,
        help="게이트 리포트 JSON 경로 (기본 run_dir/spec_gate.json)",
    )
    cfg, args = parse_tool_cli(parser, argv)

    rc = handle_common_flags(cfg, args)
    if rc is not None:
        return rc

    if cfg.mode != "s0_spec":
        print(
            f"경고: mode='{cfg.mode}' 인데 S0-Spec 러너로 실행됐다. "
            "configs/s0_spec_k235.yaml 을 쓰는 것이 정본이다.",
            file=sys.stderr,
        )

    from bloomnet.utils.seed import seed_everything_strict

    seed_everything_strict(int(cfg.seed))

    # ── 스모크: SpecMLP 만 만들고 종료 ───────────────────────────────────
    if args.build_only:
        spec_mlp = require_module(
            "bloomnet.pretrain.spec_mlp",
            group="구현 대기 (06 §2.2 L2)",
            hint="pretrain/spec_mlp.py 의 `SpecMLP` / `init_spec_mlp` 가 필요하다 (06 §3.6, X-09).",
        )
        model = spec_mlp.SpecMLP(head_out=int(cfg.spec.head_out))
        if hasattr(spec_mlp, "init_spec_mlp"):
            spec_mlp.init_spec_mlp(model, bio_gain=float(cfg.spec.bio_gain))
        n = sum(p.numel() for p in model.parameters())
        print(banner("build_only — SpecMLP 빌드 성공 (학습은 하지 않는다)"))
        print(f"  head_out   : {cfg.spec.head_out}")
        print(f"  params     : {n}  (동결값 1,089 / head_out=2 는 1,122)")
        print(f"  device     : {cfg.spec.device}")
        return 0

    # ── LOSO + 홀드아웃 + 게이트 ─────────────────────────────────────────
    npz = resolve_out_path(args.npz or cfg.spec.npz)
    if not npz.is_file():
        raise SystemExit(
            f"235 npz 캐시가 없다: {npz}\n"
            "  먼저 `python -m bloomnet.tools.build_k235_cache` 로 캐시를 만든다 (01 §7.6)."
        )

    loso_mod = require_module(
        "bloomnet.pretrain.loso",
        group="구현 대기 (06 §2.2 L3)",
        hint="pretrain/loso.py 의 `run_loso` / `transplant_gate` 가 필요하다 (06 §3.6).",
    )
    run_dir = prepare_run_dir(cfg)
    print(banner(f"S0-Spec — npz={npz.name} device={cfg.spec.device}"))

    kwargs = _loso_kwargs(cfg)
    loso_metrics: Dict[str, Any] = {}
    if cfg.spec.loso:
        loso_metrics = loso_mod.run_loso(str(npz), **kwargs)

    # 홀드아웃(val_sites = CSC, JAD). 전용 러너가 있으면 쓰고, 없으면 LOSO fold 에서 집계한다.
    run_holdout = getattr(loso_mod, "run_holdout", None)
    if callable(run_holdout):
        holdout = run_holdout(
            str(npz),
            train_sites=list(cfg.spec.train_sites),
            val_sites=list(cfg.spec.val_sites),
            **kwargs,
        )
        holdout_source = "run_holdout"
    else:
        folds = (loso_metrics or {}).get("folds", {})
        holdout = {s: folds[s] for s in cfg.spec.val_sites if s in folds}
        holdout_source = "loso_folds_for_val_sites"  # ★ 근사임을 리포트에 남긴다

    # 게이트 임계는 **config 값**을 쓴다 (loso.GATE_DEFAULTS 를 조용히 쓰지 않는다)
    tp = cfg.spec.transplant
    passed, detail = loso_mod.transplant_gate(
        loso_metrics,
        holdout,
        g1_rmse=float(tp.gate_g1_rmse),
        g2_spearman=float(tp.gate_g2_spearman),
        g3_f1_at15=float(tp.gate_g3_f1_at15),
    )

    report = {
        "npz": str(npz),
        "mode": cfg.mode,
        "seed": int(cfg.seed),
        "spec": {k: getattr(cfg.spec, k) for k in _LOSO_KEYS},
        "train_sites": list(cfg.spec.train_sites),
        "val_sites": list(cfg.spec.val_sites),
        "loso_metrics": loso_metrics,
        "holdout_metrics": holdout,
        "holdout_source": holdout_source,
        "gate_thresholds": {
            "G1_rmse_log_max": cfg.spec.transplant.gate_g1_rmse,
            "G2_spearman_min": cfg.spec.transplant.gate_g2_spearman,
            "G3_f1_at15_min": cfg.spec.transplant.gate_g3_f1_at15,
        },
        "gate_passed": bool(passed),
        "gate_detail": detail,
        "band_order": list(cfg.spec.band_order),
        "band_order_confirmed": bool(cfg.spec.band_order_confirmed),
        "transplant_performed": False,  # ★ 정정 A-19 — 이 러너는 절대 이식하지 않는다
        "transplant_blocked_by": [
            r
            for r, ok in (
                ("gate_G1G2G3", bool(passed)),
                ("band_order_confirmed(V18)", bool(cfg.spec.band_order_confirmed)),
                ("spec.transplant.enabled(V12)", bool(cfg.spec.transplant.enabled)),
            )
            if not ok
        ],
    }
    out = resolve_out_path(args.report) if args.report else (run_dir / "spec_gate.json")
    write_json(out, report)

    report_fn = getattr(loso_mod, "format_gate_report", None)
    if callable(report_fn):
        print(
            report_fn(
                loso_metrics,
                holdout,
                (passed, detail),
                g1_rmse=float(tp.gate_g1_rmse),
                g2_spearman=float(tp.gate_g2_spearman),
                g3_f1_at15=float(tp.gate_g3_f1_at15),
            )
        )
    print(f"\n  이식 게이트 : {'통과' if passed else '실패'}  {detail}")
    print(f"  band_order  : {cfg.spec.band_order} (confirmed={cfg.spec.band_order_confirmed})")
    print("  이식 수행   : 아니오 (정정 A-19 — pretrain/transplant.py 로 별도 수행)")
    print(f"  리포트      : {out}")
    if not passed:
        print("  → 게이트 실패는 '분광 단독으로는 안 된다'는 정량 근거다 (01 §6.9). 기대 결과에 포함된다.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
