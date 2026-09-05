"""latency / throughput / 파라미터 · MAC 측정 (04 §10, 06 §6).

측정 프로토콜은 `이전 구현/benchmark_inference_cost.py` 의 것을 **복제**한다
(04 §9.4 step 6: "해당 스크립트를 실행하지 말고 프로토콜만 복제한다"):

    1회 untimed forward → warmup N회 → device sync → perf_counter × iters → device sync

MAC 은 정정 A-24 를 따른다 — **256² 에서 1회 측정하고 ×4(512²)·×16(1024²) 로 스케일**한다.
근거: 1024² S2-Full 1회 = 62.678 GMAC = 125.4 GFLOP 라 4-스레드 CPU 에서 실측이 비현실적이고,
유일한 비선형 항인 PAPPM global branch 는 30,720 MAC(전체의 5e-5 %)이다.

사용 예 (cwd = <repo_root>)
    CUDA_VISIBLE_DEVICES="" python -m bloomnet.tools.benchmark \\
        --config bloomnet/configs/s0_rgb_aihub092.yaml --device cpu --macs_only
    CUDA_VISIBLE_DEVICES=0 python -m bloomnet.tools.benchmark --config s2_full \\
        --input_hw 1024 1024 --iters 100 --warmup 20
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from bloomnet.tools._cli import (
    banner,
    build_model,
    build_tool_parser,
    handle_common_flags,
    make_dummy_inputs,
    model_summary,
    parse_tool_cli,
    resolve_device,
    resolve_out_path,
    write_json,
)

__all__ = ["main", "measure_latency"]

# 06 §6 / §10 요약 카드의 확정치. **리포트 대조용 참고값**이며 회귀 게이트는
# T24(`tests/test_budget.py`) 소관이다 — 여기서 assert 하지 않는다.
_BUDGET_REFERENCE: Dict[str, Dict[str, float]] = {
    # 06 §6.2 모드별 실행 예산 (params / GMAC@512² / GMAC@1024²)
    "s0_rgb": {"params": 8_954_390, "num_classes": 12, "gmac_512": 10.198, "gmac_1024": 40.792},
    "s1_rgb_ms4": {"params": 14_660_522, "num_classes": 2, "gmac_512": 14.291, "gmac_1024": 57.164},
    "s2_full": {"params": 15_969_023, "num_classes": 2, "gmac_512": 15.670, "gmac_1024": 62.678},
}
_MAC_MEASURE_HW = (256, 256)  # ★ 정정 A-24


def measure_latency(
    model: Any,
    inputs: Dict[str, Any],
    *,
    device: Any,
    iters: int = 100,
    warmup: int = 20,
) -> Dict[str, float]:
    """이전 구현 프로토콜 복제: untimed 1 → warmup → sync → timed × iters → sync."""
    import torch

    is_cuda = getattr(device, "type", str(device)) == "cuda"

    def _sync() -> None:
        if is_cuda:
            torch.cuda.synchronize()

    with torch.no_grad():
        model(**inputs)  # 1회 untimed (lazy init / autotune 유발)
        for _ in range(warmup):
            model(**inputs)
        _sync()
        samples: List[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(**inputs)
            _sync()
            samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    n = len(samples)
    batch = int(next(iter(inputs.values())).shape[0])
    mean_ms = sum(samples) / n
    return {
        "iters": n,
        "warmup": warmup,
        "batch": batch,
        "mean_ms": mean_ms,
        "median_ms": samples[n // 2],
        "p90_ms": samples[min(n - 1, int(0.90 * n))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "fps": (batch * 1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
    }


def _measure_macs(model: Any, cfg: Any, device: Any, mac_hw: Sequence[int]) -> Dict[str, Any]:
    from bloomnet.utils.flops import count_macs, scale_macs

    hw = (int(mac_hw[0]), int(mac_hw[1]))
    inputs = make_dummy_inputs(cfg, batch=1, hw=hw, device=device)
    report = count_macs(model, inputs, method="flop_counter", input_hw=hw)
    base = float(report.total)
    return {
        "measured_at": list(hw),
        "gmac_measured": base / 1e9,
        "gmac_512": scale_macs(base, from_hw=hw, to_hw=(512, 512)) / 1e9,
        "gmac_1024": scale_macs(base, from_hw=hw, to_hw=(1024, 1024)) / 1e9,
        "note": "정정 A-24 — 256² 실측 후 면적비 스케일 (비선형 항 = PAPPM global 30,720 MAC)",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_tool_parser("BloomNet latency / throughput / MAC 측정")
    parser.add_argument("--ckpt", default=None, help="체크포인트(선택). 미지정이면 랜덤 초기화")
    parser.add_argument("--input_hw", type=int, nargs=2, default=None, help="기본 data.eval_size")
    parser.add_argument(
        "--mac_hw",
        type=int,
        nargs=2,
        default=list(_MAC_MEASURE_HW),
        help="MAC 측정 해상도 (기본 256 256 — 정정 A-24. 결과는 512²/1024² 로 면적비 스케일)",
    )
    parser.add_argument("--bench_batch", type=int, default=1, help="측정 배치 (기본 1, 04 §10 규약)")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--macs_only", action="store_true", help="latency 생략, 파라미터·MAC 만")
    parser.add_argument("--deploy", action="store_true", help="model.deploy() 후 측정 (추론 구성)")
    parser.add_argument("--out", default=None, help="리포트 JSON 경로")
    cfg, args = parse_tool_cli(parser, argv)

    rc = handle_common_flags(cfg, args)
    if rc is not None:
        return rc

    from bloomnet.utils.seed import seed_everything_strict

    seed_everything_strict(int(cfg.seed))

    import torch

    device = resolve_device(cfg.device)  # ★ CUDA 부재 시 CPU 폴백 (경고 후 계속)
    model = build_model(cfg)
    if args.build_only:
        print(banner("build_only — 모델 빌드 성공 (측정은 하지 않는다)"))
        for k, v in model_summary(model, cfg).items():
            print(f"  {k:20s} {v}")
        return 0

    if args.ckpt:
        from bloomnet.utils.checkpoint import load_state_dict_shape_tolerant

        ck_path = resolve_out_path(args.ckpt)
        if not ck_path.is_file():
            raise SystemExit(f"체크포인트가 없다: {ck_path}")
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
        load_state_dict_shape_tolerant(model, ck.get("model", ck))

    if args.deploy:
        model.deploy()
    model.to(device).eval()

    summary = model_summary(model, cfg)
    macs = _measure_macs(model, cfg, device, args.mac_hw)
    payload: Dict[str, Any] = {
        "mode": cfg.mode,
        "device": str(device),
        "deploy_pruned": bool(args.deploy),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
        "summary": summary,
        "macs": macs,
        "reference_06_s6": _BUDGET_REFERENCE.get(cfg.mode),
    }

    print(banner(f"benchmark — mode={cfg.mode} device={device}"))
    print(f"  params        : {summary['params_total']:,}")
    print(
        f"  GMAC @{macs['measured_at'][0]}²(실측)/512²/1024² : "
        f"{macs['gmac_measured']:.4f} / {macs['gmac_512']:.3f} / {macs['gmac_1024']:.3f}"
    )
    ref = payload["reference_06_s6"]
    if ref:
        print(f"  06 §6 참고값  : {ref}")

    if not args.macs_only:
        hw = tuple(args.input_hw or cfg.data.eval_size)
        inputs = make_dummy_inputs(
            cfg, batch=int(args.bench_batch), hw=(int(hw[0]), int(hw[1])), device=device
        )
        iters, warmup = int(args.iters), int(args.warmup)
        if getattr(device, "type", str(device)) == "cpu":
            iters, warmup = min(iters, 10), min(warmup, 2)
            print(f"  CPU 폴백 — iters={iters} warmup={warmup} 로 축소 (절대값 인용 금지)")
        lat = measure_latency(model, inputs, device=device, iters=iters, warmup=warmup)
        lat["input_hw"] = list(hw)
        payload["latency"] = lat
        print(
            f"  latency @{hw[0]}×{hw[1]} B={lat['batch']} : "
            f"mean {lat['mean_ms']:.2f} ms / median {lat['median_ms']:.2f} / "
            f"p90 {lat['p90_ms']:.2f} → {lat['fps']:.1f} FPS"
        )

    out = resolve_out_path(args.out) if args.out else (
        resolve_out_path(cfg.output_dir) / f"benchmark_{cfg.mode}.json"
    )
    write_json(out, payload)
    print(f"\n리포트: {out}")
    if getattr(device, "type", str(device)) == "cpu" and not args.macs_only:
        print("  ※ CPU latency 는 Jetson Orin FP16 실시간성 판단(06 §6.4)의 근거가 될 수 없다.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
