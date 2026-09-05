"""평가 진입점 — seg / boundary / Chl-a 회귀 지표 + JSON 리포트 (05 §6, 01 §8).

사용 예 (cwd = <repo_root>)
    CUDA_VISIBLE_DEVICES=0 python -m bloomnet.tools.eval \\
        --config bloomnet/configs/s0_rgb_aihub092.yaml \\
        --ckpt outputs/<run>/best.pt --split test --out outputs/<run>/eval_test.json

계약
    - mIoU 규약은 `exclude_absent` 고정이다 (X-20, 헌법 C-12). 구현은 `utils/metrics_seg` 단일 출처.
    - 부트스트랩 CI 는 **이미지가 아니라 flight-line 군 단위** 리샘플이다 (정정 A-4).
    - `f1_at_100` 은 235 에 지지 표본이 0개라 값 대신 null + 사유가 기록된다 (정정 A-30).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional, Sequence

from bloomnet.tools._cli import (
    banner,
    build_model,
    build_tool_parser,
    handle_common_flags,
    model_summary,
    parse_tool_cli,
    require_module,
    resolve_device,
    resolve_out_path,
    write_json,
)

__all__ = ["main", "build_criterion"]


def build_criterion(cfg: Any) -> Any:
    """config → `BloomNetCriterion` (06 §3.5). 평가에서도 손실 분해를 함께 보고한다."""
    crit_mod = require_module(
        "bloomnet.losses.criterion",
        group="구현 대기 (06 §2.2 L3)",
        hint="losses/criterion.py 의 `BloomNetCriterion` 이 필요하다 (06 §3.5).",
    )
    lo = cfg.loss
    return crit_mod.BloomNetCriterion(
        num_classes=int(cfg.data.num_classes),
        ignore_index=int(cfg.data.ignore_index),
        lambda_seg=float(lo.lambda_seg),
        lambda_reg=float(lo.lambda_reg),
        lambda_bd=float(lo.lambda_bd),
        lambda_bas=float(lo.lambda_bas),
        lambda_aux=dict(lo.lambda_aux),
        lambda_siam=float(lo.lambda_siam),
        w_dice=float(lo.w_dice),
        ohem_thresh=float(lo.ohem_thresh),
        ohem_keep_frac=float(lo.ohem_keep_frac),
        bas_tau=float(lo.bas_tau),
        huber_beta=float(lo.huber_beta),
        unc_clamp=tuple(lo.unc_clamp),
        u_warm_frac=float(lo.u_warm_frac),
        u_ramp_frac=float(lo.u_ramp_frac),
        boundary_source=cfg.data.boundary.source,
        boundary_radius=cfg.boundary_radius(int(cfg.data.eval_size[0])),
        boundary_stride=int(cfg.data.boundary.stride),
        aux_loss_stride=int(lo.aux_loss_stride),
        aux_decay=bool(lo.aux_decay),
        debug_assert=bool(lo.debug_assert),
    )


def _build_evaluators(cfg: Any) -> Sequence[Any]:
    ev = require_module(
        "bloomnet.engine.evaluator",
        group="구현 대기 (06 §2.2 L3)",
        hint="engine/evaluator.py 의 Seg/Reg/Boundary Evaluator 와 `evaluate` 가 필요하다.",
    )
    from bloomnet.constants import FINE_CLASS_NAMES

    k = int(cfg.data.num_classes)
    names = list(FINE_CLASS_NAMES) if k == len(FINE_CLASS_NAMES) else None
    return [
        ev.SegEvaluator(k, int(cfg.data.ignore_index), names, seed=int(cfg.seed)),
        ev.RegEvaluator(
            thresholds_mgm3=tuple(cfg.eval.alarm_thresholds),
            debug_assert=bool(cfg.loss.debug_assert),
        ),
        ev.BoundaryEvaluator(
            tuple(cfg.eval.boundary_tolerances),
            num_classes=k,
            ignore_index=int(cfg.data.ignore_index),
        ),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_tool_parser("BloomNet 평가 + JSON 리포트")
    parser.add_argument("--ckpt", default=None, help="체크포인트 (.pt). 없으면 랜덤 초기화로 평가")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--out", default=None, help="리포트 JSON (기본 <ckpt 디렉터리>/eval_<split>.json)"
    )
    parser.add_argument("--ema", action="store_true", help="체크포인트의 EMA 가중치를 쓴다")
    cfg, args = parse_tool_cli(parser, argv)

    rc = handle_common_flags(cfg, args)
    if rc is not None:
        return rc

    from bloomnet.utils.seed import seed_everything_strict

    seed_everything_strict(int(cfg.seed))

    model = build_model(cfg)
    if args.build_only:
        summary = model_summary(model, cfg)
        print(banner("build_only — 모델 빌드 성공 (평가는 하지 않는다)"))
        for k, v in summary.items():
            print(f"  {k:20s} {v}")
        return 0

    import torch

    device = resolve_device(cfg.device)
    if args.ckpt:
        from bloomnet.utils.checkpoint import load_state_dict_shape_tolerant

        ck_path = resolve_out_path(args.ckpt)
        if not ck_path.is_file():
            raise SystemExit(f"체크포인트가 없다: {ck_path}")
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
        key = "ema" if args.ema and "ema" in ck else "model"
        state = ck.get(key, ck)
        report = load_state_dict_shape_tolerant(model, state)
        print(f"  ckpt={ck_path} key={key} load={report}")
    else:
        print("  경고: --ckpt 미지정 — 랜덤 초기화 가중치로 평가한다 (배선 확인 용도).")

    model.to(device).eval()

    from bloomnet.data.build import build_dataloaders, build_datasets

    datasets = build_datasets(cfg)
    if args.split not in datasets:
        raise SystemExit(
            f"split '{args.split}' 이 없다 — 가용: {sorted(datasets)} "
            f"(dataset={cfg.data.dataset})"
        )
    loader = build_dataloaders(cfg, {args.split: datasets[args.split]})[args.split]

    ev = require_module(
        "bloomnet.engine.evaluator",
        group="구현 대기 (06 §2.2 L3)",
        hint="engine/evaluator.py 의 `evaluate` 가 필요하다.",
    )
    criterion = build_criterion(cfg)
    print(banner(f"평가 — split={args.split} n={len(loader.dataset)} device={device}"))
    metrics: Dict[str, Any] = ev.evaluate(
        model,
        loader,
        criterion,
        cfg=cfg,
        evaluators=_build_evaluators(cfg),
        device=device,
        amp=(cfg.train.amp != "off"),
    )

    payload = {
        "mode": cfg.mode,
        "split": args.split,
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "ema": bool(args.ema),
        "num_classes": int(cfg.data.num_classes),
        "miou_convention": cfg.eval.miou_convention,
        "metrics": metrics,
    }
    if args.out:
        out = resolve_out_path(args.out)
    elif args.ckpt:
        out = resolve_out_path(args.ckpt).parent / f"eval_{args.split}.json"
    else:
        out = resolve_out_path(cfg.output_dir) / f"eval_{args.split}.json"
    write_json(out, payload)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k:24s} {v:.6f}")
    print(f"\n리포트: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
