"""BloomNet seg 학습 진입점 (S0-RGB / S1 / S2) — 06 §2.1 `scripts/train.py`.

이 파일은 **얇게** 유지한다(로직 금지, 06 §2.1 매니페스트). 하는 일은 4가지다.
    1. config 로드/검증 (06 §4.1 로드 규약 — `--config` + `--set` + 1급 플래그 15종)
    2. `run_name` 확정 + `run_dir/config.resolved.yaml` 덤프 (재현성, §4.1 step 6)
    3. 시드 고정 (05 §7.1)
    4. `engine.trainer.fit(cfg)` 호출

사용 예 (cwd = <repo_root>)
    CUDA_VISIBLE_DEVICES=0 python -m bloomnet.tools.train \\
        --config bloomnet/configs/s0_rgb_aihub092.yaml
    CUDA_VISIBLE_DEVICES=0 python -m bloomnet.tools.train --config s1_rgb_ms4 \\
        --init_from outputs/s0_rgb_.../best.pt --batch_size 8 --set optim.lr=6.0e-4

학습을 시작하지 않고 계약만 확인하려면
    python -m bloomnet.tools.train --config s0_rgb_aihub092 --print_config
    CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.train --config s2_full \\
        --build_only --device cpu
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from bloomnet.tools._cli import (
    banner,
    build_model,
    build_tool_parser,
    handle_common_flags,
    model_summary,
    parse_tool_cli,
    prepare_run_dir,
    require_cuda_or_explicit_cpu,
    require_module,
    write_json,
)

__all__ = ["main"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_tool_parser("BloomNet seg 학습 (S0-RGB / S1 / S2)")
    cfg, args = parse_tool_cli(parser, argv)

    rc = handle_common_flags(cfg, args)
    if rc is not None:
        return rc

    if cfg.mode == "s0_spec":
        raise SystemExit(
            "mode='s0_spec' 은 seg 학습 경로가 아니다 — `python -m bloomnet.tools.train_spec` 를 쓴다 "
            "(01 §6.5, SpecMLP 1,089 params 칩 회귀)."
        )

    from bloomnet.utils.seed import seed_everything_strict

    seed_everything_strict(int(cfg.seed))

    # ── 스모크: config → 모델 빌드까지만 (학습 없음) ──────────────────────
    if args.build_only:
        model = build_model(cfg)
        summary = model_summary(model, cfg)
        print(banner("build_only — 모델 빌드 성공 (학습은 하지 않는다)"))
        for k, v in summary.items():
            print(f"  {k:20s} {v}")
        return 0

    # ── 실제 학습 ────────────────────────────────────────────────────────
    device = require_cuda_or_explicit_cpu(cfg)
    run_dir = prepare_run_dir(cfg)
    print(banner(f"BloomNet 학습 — mode={cfg.mode} device={device}"))
    print(f"  run_dir      : {run_dir}")
    print(f"  active_paths : {list(cfg.active_paths)}")
    print(f"  dataset/root : {cfg.data.dataset} / {cfg.data.root}")
    print(f"  epochs×batch : {cfg.schedule.epochs} × {cfg.schedule.batch_size}")
    print(f"  dry_run      : {cfg.dry_run}")

    trainer = require_module(
        "bloomnet.engine.trainer",
        group="구현 대기 (06 §2.2 L6)",
        hint="engine/trainer.py 의 `fit(cfg)` 가 필요하다 (06 §3.6).",
    )
    result = trainer.fit(cfg)
    write_json(run_dir / "train_summary.json", dict(result or {}))
    print(f"\n완료 — 요약: {run_dir / 'train_summary.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
