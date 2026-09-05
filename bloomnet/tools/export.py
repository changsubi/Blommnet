"""ONNX export 진입점 — `deploy()` → `ExportWrapper` → ONNX → 검증 (04 §9.4).

사용 예 (cwd = <repo_root>)
    python -m bloomnet.tools.export --config bloomnet/configs/s1_rgb_ms4.yaml \\
        --ckpt outputs/<run>/best.pt --out outputs/<run>/bloomnet.onnx
    python -m bloomnet.tools.export --config s0_rgb_aihub092 --check_only --device cpu

04 §9.4 체크리스트 대응
    1. `model.deploy()` 후 state_dict 에 `edge_head|aux_|siam` 키 0개              → 항상 수행
    2. `torch.onnx.export(..., dynamo=False, dynamic_axes=…)`                      → onnx 필요
    3. onnxsim 단순화 후 `Shape`/`Gather` 0개 (**ExportWrapper 산출물에만**, 정정 A-21) → onnx 필요
    4. polygraphy ORT vs TRT 대조                                                  → 사용자 수동
    5. FP16 엔진에서 `conf ∈ [0.029312, 0.970688]`, `chl` finite                   → 실기 수동
    6. Jetson latency                                                    → tools/benchmark.py

onnx 계열(requirements-deploy.txt)이 없으면 **조용히 통과시키지 않고** 명시적으로 실패한다
(06 §5.0 격리 규칙 2).
"""

from __future__ import annotations

import sys
from typing import Any, Optional, Sequence

from bloomnet.tools._cli import (
    DEPLOY_PACKAGES,
    banner,
    build_model,
    build_tool_parser,
    dependency_report,
    handle_common_flags,
    missing_packages,
    parse_tool_cli,
    require_module,
    resolve_device,
    resolve_out_path,
    write_json,
)

__all__ = ["main", "assert_deploy_pruned"]

_TRAIN_ONLY_PATTERNS = ("edge_head", "aux_", "siam")


def assert_deploy_pruned(model: Any) -> int:
    """04 §9.4 step 1 / T23 — `deploy()` 후 학습 전용 키가 0개여야 한다."""
    leftover = [
        k for k in model.state_dict() if any(p in k for p in _TRAIN_ONLY_PATTERNS)
    ]
    if leftover:
        raise SystemExit(
            "deploy() 후에도 학습 전용 파라미터가 남아 있다 (04 §9.4 step 1 위반):\n  "
            + "\n  ".join(leftover[:20])
            + ("\n  ..." if len(leftover) > 20 else "")
        )
    return len(model.state_dict())


def _require_onnx_stack() -> None:
    missing = missing_packages(DEPLOY_PACKAGES)
    if not missing:
        return
    raise SystemExit(
        "ONNX export 를 실행할 수 없다 — requirements-deploy.txt 가 설치되어 있지 않다.\n"
        f"  미설치: {missing}\n"
        "  설치:   pip install -r bloomnet/requirements-deploy.txt\n"
        "          (인터넷이 없는 환경이면 오프라인 wheel 을 먼저 확보한다 — 06 §7.3 B5)\n"
        "  참고:   --check_only 는 onnx 없이도 deploy()/ExportWrapper 계약을 검사한다.\n"
        f"{dependency_report()}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_tool_parser("BloomNet ONNX export (deploy → ExportWrapper → ONNX → 검증)")
    parser.add_argument("--ckpt", default=None, help="체크포인트 (.pt). 없으면 랜덤 초기화")
    parser.add_argument(
        "--out", default=None, help="출력 .onnx (기본 <ckpt 디렉터리>/bloomnet.onnx)"
    )
    parser.add_argument("--input_hw", type=int, nargs=2, default=None, help="기본 deploy.input_hw")
    parser.add_argument("--opset", type=int, default=None, help="기본 deploy.opset")
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="onnx 없이 deploy()/ExportWrapper 계약만 검사 (04 §9.4 step 1)",
    )
    parser.add_argument(
        "--no_verify", action="store_true", help="verify_onnx / Shape 노드 검사 생략"
    )
    cfg, args = parse_tool_cli(parser, argv)

    rc = handle_common_flags(cfg, args)
    if rc is not None:
        return rc

    from bloomnet.utils.seed import seed_everything_strict

    seed_everything_strict(int(cfg.seed))

    model = build_model(cfg)
    if args.build_only:
        print(banner("build_only — 모델 빌드 성공 (export 는 하지 않는다)"))
        print(f"  active_paths : {list(cfg.active_paths)}")
        print(f"  input_hw     : {list(cfg.deploy.input_hw)}  opset={cfg.deploy.opset}")
        return 0

    import torch

    device = resolve_device("cpu" if args.check_only else cfg.device)
    if args.ckpt:
        from bloomnet.utils.checkpoint import load_state_dict_shape_tolerant

        ck_path = resolve_out_path(args.ckpt)
        if not ck_path.is_file():
            raise SystemExit(f"체크포인트가 없다: {ck_path}")
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
        load_state_dict_shape_tolerant(model, ck.get("model", ck))
    else:
        print("  경고: --ckpt 미지정 — 랜덤 초기화 가중치를 export 한다 (그래프 계약 검사용).")

    model.to(device)
    model.deploy()
    n_keys = assert_deploy_pruned(model)
    print(banner(f"deploy() 완료 — state_dict 키 {n_keys}개, 학습 전용 키 0개"))

    input_hw = tuple(args.input_hw or cfg.deploy.input_hw)
    opset = int(args.opset or cfg.deploy.opset)

    if args.check_only:
        models_mod = require_module(
            "bloomnet.models.bloomnet",
            group="구현 대기 (06 §2.2 L5)",
            hint="models/bloomnet.py 의 `ExportWrapper` 가 필요하다 (06 §3.4.3).",
        )
        wrapper = models_mod.ExportWrapper(model)
        dummy = tuple(
            t.to(device)
            for t in _dummy_export_inputs(cfg, input_hw)
        )
        with torch.no_grad():
            out = wrapper(*dummy)
        if not (isinstance(out, tuple) and len(out) == 3):
            raise SystemExit(f"ExportWrapper 반환은 3-tuple 이어야 한다 (04 §9.3-K), got {type(out)}")
        seg, chl, conf = out
        print(f"  ExportWrapper 입력 {len(dummy)}개 (active_paths={list(cfg.active_paths)})")
        print(f"  seg {tuple(seg.shape)} / chl {tuple(chl.shape)} / conf {tuple(conf.shape)}")
        print(f"  chl  ∈ [{chl.min():.4f}, {chl.max():.4f}]  (계약 [0, {cfg.deploy.chl_max_mgm3}])")
        print(
            f"  conf ∈ [{conf.min():.6f}, {conf.max():.6f}]"
            "  (계약 [0.029312, 0.970688], X-25/X-26)"
        )
        print("\n--check_only 완료. 실제 ONNX 생성은 requirements-deploy.txt 설치 후 재실행한다.")
        return 0

    _require_onnx_stack()
    ex = require_module(
        "bloomnet.deploy.export_onnx",
        group="구현 대기 (06 §2.2 L6)",
        hint="deploy/export_onnx.py 의 `export`/`verify_onnx`/`assert_no_shape_nodes` 가 필요하다.",
    )
    out_path = resolve_out_path(args.out or _default_onnx_path(args.ckpt))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = ex.export(
        model,
        out_path,
        input_hw=input_hw,
        opset=opset,
        dynamic_batch=bool(cfg.deploy.dynamic_batch),
        use_dynamo=bool(cfg.deploy.use_dynamo),  # ★ 정정 A-29 — torch 2.13 기본은 dynamo=True
    )
    print(f"  ONNX: {onnx_path}")

    payload = {
        "onnx": str(onnx_path),
        "input_hw": list(input_hw),
        "opset": opset,
        "precision": cfg.deploy.precision,
        "fp32_forced": list(cfg.deploy.fp32_forced),
        "fp16_locked": list(cfg.deploy.fp16_locked),
        "active_paths": list(cfg.active_paths),
    }
    if not args.no_verify:
        ex.assert_no_shape_nodes(onnx_path)  # 정정 A-21 — ExportWrapper 산출물에만
        payload["verify"] = ex.verify_onnx(onnx_path, model)
        print(f"  Shape/Gather 노드 0개 확인, verify={payload['verify']}")
    write_json(out_path.with_suffix(".export.json"), payload)
    print(f"\n리포트: {out_path.with_suffix('.export.json')}")
    print(
        "  다음 단계(수동): polygraphy run "
        f"{onnx_path} --trt --fp16 --onnxrt --atol 1e-2 --rtol 1e-2   (04 §9.4 step 4)"
    )
    return 0


def _default_onnx_path(ckpt: Optional[str]) -> str:
    """`--out` 미지정 시 기본 경로 — 체크포인트 옆, 없으면 `outputs/`."""
    if ckpt:
        return str(resolve_out_path(ckpt).parent / "bloomnet.onnx")
    return "outputs/bloomnet.onnx"


def _dummy_export_inputs(cfg: Any, input_hw: Sequence[int]) -> Sequence[Any]:
    """ExportWrapper 입력 순서 = `model.active_paths` 유도 (정정 A-35)."""
    from bloomnet.tools._cli import make_dummy_inputs

    kw = make_dummy_inputs(cfg, batch=1, hw=(int(input_hw[0]), int(input_hw[1])))
    order = ["rgb", "msi", "bio", "ir", "pol"]
    return [kw[k] for k in order if k in kw]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
