"""실행 스크립트 5종(`train/train_spec/eval/export/benchmark`)이 공유하는 CLI 배선.

06 §2.1 의 매니페스트는 이 5개를 `scripts/`(레벨 L7) 에 두지만 본 저장소는
`bloomnet/tools/` 에 배치한다. **레벨은 L7 (최상위 진입점)** 이며, 여기서만
`models`/`engine`/`deploy`/`pretrain` 을 import 한다.

설계 원칙
    - 무거운 import(`models`, `engine`, `deploy`)는 **함수 안에서만** 한다.
      그래야 아직 구현되지 않은 하위 모듈이 있어도 ``--help`` 와 ``--print_config`` 가 동작한다.
    - config 로드 의미론은 `bloomnet.config.parse_cli` 와 **동일**하다(06 §4.1).
      달라지는 것은 (a) 도구별 추가 인자를 붙일 수 있다는 점과
      (b) `--base` 기본값이 cwd 상대 경로가 아니라 **패키지 동봉 `configs/_base.yaml`** 이라는 점뿐이다.
    - 쓰기는 전부 저장소 루트(`<repo_root>`) 기준으로 해석한다.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

import bloomnet
from bloomnet.config import (
    FIRST_CLASS_FLAGS,
    BloomNetConfig,
    build_parser,
    dump_config,
    load_config,
    make_run_name,
    to_yaml,
)
from bloomnet.constants import MODALITY_CHANNELS, MSI_SLOTS

__all__ = [
    "PKG_DIR",
    "REPO_ROOT",
    "CONFIG_DIR",
    "DEFAULT_BASE",
    "PRESET_NAMES",
    "RUNTIME_PACKAGES",
    "DEV_PACKAGES",
    "DEPLOY_PACKAGES",
    "build_tool_parser",
    "parse_tool_cli",
    "resolve_base_path",
    "resolve_config_path",
    "resolve_out_path",
    "prepare_run_dir",
    "resolve_device",
    "require_cuda_or_explicit_cpu",
    "missing_packages",
    "dependency_report",
    "require_module",
    "model_summary",
    "make_dummy_inputs",
    "build_model",
    "write_json",
    "banner",
    "handle_common_flags",
]

PKG_DIR: Path = Path(bloomnet.__file__).resolve().parent
REPO_ROOT: Path = PKG_DIR.parent
CONFIG_DIR: Path = PKG_DIR / "configs"
DEFAULT_BASE: Path = CONFIG_DIR / "_base.yaml"

PRESET_NAMES: Tuple[str, ...] = (
    "s0_rgb_aihub092",
    "s0_spec_k235",
    "s1_rgb_ms4",
    "s2_full",
)

# 06 §5.0 (정정 A-13) 의존성 3분할. 여기 이름은 **import 이름**이다.
RUNTIME_PACKAGES: Tuple[str, ...] = (
    "torch",
    "torchvision",
    "numpy",
    "PIL",
    "yaml",
    "tqdm",
    "scipy",
)
DEV_PACKAGES: Tuple[str, ...] = ("pytest", "pytest_cov", "ruff")
DEPLOY_PACKAGES: Tuple[str, ...] = (
    "onnx",
    "onnxscript",
    "onnxruntime",
    "onnxsim",
    "polygraphy",
)


# ═══════════════════════════════════════════════════════════════════════
#  경로 해석
# ═══════════════════════════════════════════════════════════════════════
def resolve_base_path(base: Optional[str]) -> Optional[Path]:
    """`--base` 를 실제 파일로 푼다.

    기본값(패키지 동봉 `configs/_base.yaml`)일 때만 폴백/생략을 허용한다.
    사용자가 **명시한** 경로가 없으면 `bloomnet.config.parse_cli` 와 동일하게 `FileNotFoundError`.
    """
    if base is None:
        return None
    given = Path(base).expanduser()
    if given.is_file():
        return given
    if str(given) not in {str(DEFAULT_BASE), "configs/_base.yaml"}:
        raise FileNotFoundError(f"--base file not found: {base}")
    # 기본 경로가 없을 때만 폴백한다 (저장소 루트 → 없으면 dataclass 기본값이 기저)
    alt = REPO_ROOT / "configs" / "_base.yaml"
    return alt if alt.is_file() else None


def _preset_candidates() -> List[Path]:
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(CONFIG_DIR.rglob("*.yaml"))


def resolve_config_path(name: Optional[str]) -> Optional[Path]:
    """`--config` 를 푼다. 경로가 아니면 동봉 프리셋 이름으로 해석한다.

    `--config s1_rgb_ms4` / `--config a4_bio_gndvi` / `--config <절대경로>` 가 모두 동작한다.
    """
    if name is None:
        return None
    given = Path(name).expanduser()
    if given.is_file():
        return given
    stem = given.name
    for cand in (
        CONFIG_DIR / stem,
        CONFIG_DIR / f"{stem}.yaml",
        CONFIG_DIR / "ablation" / stem,
        CONFIG_DIR / "ablation" / f"{stem}.yaml",
    ):
        if cand.is_file():
            return cand
    available = ", ".join(p.stem for p in _preset_candidates() if not p.stem.startswith("_"))
    raise FileNotFoundError(
        f"config 파일을 찾을 수 없다: {name!r}\n"
        f"  경로를 직접 주거나 동봉 프리셋 이름을 쓴다 — {available or '(configs/ 비어 있음)'}"
    )


def resolve_out_path(path: str) -> Path:
    """상대 경로를 저장소 루트(`<repo_root>`) 기준으로 푼다."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


# ═══════════════════════════════════════════════════════════════════════
#  파서
# ═══════════════════════════════════════════════════════════════════════
def build_tool_parser(description: str) -> argparse.ArgumentParser:
    """1급 플래그 15종 + `--set`(06 §4.1) 에 도구 공통 플래그 3종을 얹는다."""
    p = build_parser(description)
    p.set_defaults(base=str(DEFAULT_BASE))
    g = p.add_argument_group("도구 공통")
    g.add_argument(
        "--print_config",
        action="store_true",
        help="병합·검증된 config 를 stdout 에 덤프하고 종료 (모델·데이터 미생성)",
    )
    g.add_argument(
        "--build_only",
        action="store_true",
        help="config 로드 + 모델 빌드까지만 하고 종료 (CPU 스모크). 학습·평가는 하지 않는다",
    )
    g.add_argument(
        "--deps",
        action="store_true",
        help="requirements 3분할(runtime/dev/deploy) 설치 현황을 출력하고 종료 (06 §5.0)",
    )
    g.add_argument(
        "--progress",
        choices=["auto", "bar", "plain", "off"],
        default=None,
        help=(
            "진행 표시. auto(기본)=TTY 면 tqdm 막대, 리다이렉트면 60초마다 한 줄. "
            "bar=항상 막대, plain=항상 한 줄, off=없음. 환경변수 BLOOMNET_PROGRESS 로도 지정 가능"
        ),
    )
    return p


def parse_tool_cli(
    parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None
) -> Tuple[BloomNetConfig, argparse.Namespace]:
    """`bloomnet.config.parse_cli` 와 동일 의미론 + 도구별 인자 지원.

    1급 플래그는 내부적으로 `--set` 으로 변환된다 (06 §4.1 step 4).
    """
    args = parser.parse_args(argv)
    # 진행 표시는 config 스키마가 아니라 환경변수로 전달한다(동결된 API 를 건드리지 않음).
    if getattr(args, "progress", None):
        os.environ["BLOOMNET_PROGRESS"] = str(args.progress)
    overrides = list(args.overrides)
    for dest, dotted in FIRST_CLASS_FLAGS.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        overrides.append(f"{dotted}={yaml.safe_dump(value).strip()}")

    cfg = load_config(
        resolve_config_path(args.config),
        base=resolve_base_path(args.base),
        overrides=overrides,
        allow_contract_break=bool(getattr(args, "allow_contract_break", False)),
    )
    return cfg, args


# ═══════════════════════════════════════════════════════════════════════
#  실행 디렉터리 · 디바이스
# ═══════════════════════════════════════════════════════════════════════
def prepare_run_dir(cfg: BloomNetConfig, *, create: bool = True) -> Path:
    """`run_name` 을 확정하고 `run_dir/config.resolved.yaml` 을 덤프한다 (06 §4.1 step 6).

    `run_name` 을 **스크립트에서** 확정해야 trainer/evaluator 가 같은 디렉터리를 본다.
    """
    if cfg.run_name is None:
        cfg.run_name = make_run_name(cfg)
    run_dir = resolve_out_path(cfg.output_dir) / cfg.run_name
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_config(cfg, run_dir / "config.resolved.yaml")
    return run_dir


def resolve_device(requested: str, *, allow_cpu_fallback: bool = True) -> "Any":
    """요청 디바이스를 torch.device 로 푼다. CUDA 부재 시 CPU 폴백(경고)."""
    import torch

    dev = str(requested)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu_fallback:
            raise SystemExit(
                f"device='{dev}' 인데 CUDA 를 쓸 수 없다 "
                "(CUDA_VISIBLE_DEVICES 확인). CPU 로 진행하려면 --device cpu 를 명시하라."
            )
        warnings.warn(
            f"device='{dev}' 를 쓸 수 없어 CPU 로 폴백한다 "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r})",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return torch.device(dev)


def require_cuda_or_explicit_cpu(cfg: BloomNetConfig) -> "Any":
    """학습 경로 전용 가드. 실수로 CPU 에서 며칠짜리 학습이 시작되는 것을 막는다."""
    import torch

    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "학습을 시작할 수 없다: device='cuda' 인데 CUDA 가 보이지 않는다.\n"
                "  의도한 CPU 실행이라면 --device cpu 를 명시하라 (매우 느리다)."
            )
        return torch.device(cfg.device)
    warnings.warn(
        "CPU 학습은 지원 목적이 아니다 — 스모크 용도로만 쓸 것 (헌법 C-5).",
        RuntimeWarning,
        stacklevel=2,
    )
    return torch.device(cfg.device)


# ═══════════════════════════════════════════════════════════════════════
#  의존성 · 지연 import
# ═══════════════════════════════════════════════════════════════════════
def missing_packages(names: Sequence[str]) -> List[str]:
    out: List[str] = []
    for n in names:
        try:
            found = importlib.util.find_spec(n) is not None
        except (ImportError, ValueError):  # pragma: no cover - 손상된 설치
            found = False
        if not found:
            out.append(n)
    return out


def dependency_report() -> str:
    """06 §5.0 (정정 A-13) — 미설치 그룹을 **명시 출력**한다. 조용한 skip 금지."""
    lines = ["의존성 현황 (06 §5.0, 정정 A-13)"]
    for label, names, req in (
        ("runtime", RUNTIME_PACKAGES, "requirements.txt"),
        ("dev", DEV_PACKAGES, "requirements-dev.txt"),
        ("deploy", DEPLOY_PACKAGES, "requirements-deploy.txt"),
    ):
        miss = missing_packages(names)
        status = "OK" if not miss else f"미설치 {miss}"
        lines.append(f"  [{label:8s}] {req:24s} {status}")
    if missing_packages(DEPLOY_PACKAGES):
        lines.append("  → onnx 미설치 — export 게이트(T25 slow, 04 §9.4) 미검증")
    return "\n".join(lines)


def require_module(module: str, *, group: str, hint: str) -> Any:
    """지연 import. 실패하면 **무엇을 설치/구현해야 하는지** 를 담은 SystemExit."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise SystemExit(
            f"'{module}' 를 import 할 수 없다 ({exc}).\n"
            f"  분류: {group}\n"
            f"  조치: {hint}\n"
            f"{dependency_report()}"
        ) from exc


# ═══════════════════════════════════════════════════════════════════════
#  출력 헬퍼
# ═══════════════════════════════════════════════════════════════════════
def banner(title: str) -> str:
    return f"\n{'═' * 72}\n  {title}\n{'═' * 72}"


def model_summary(model: Any, cfg: BloomNetConfig) -> Dict[str, Any]:
    """파라미터 수 + 활성 path 요약 (T24 예산표와 대조 가능한 형태)."""
    from bloomnet.utils.flops import count_parameters

    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    per_top: Dict[str, int] = {}
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        if n:
            per_top[name] = n
    return {
        "mode": cfg.mode,
        "num_classes": cfg.data.num_classes,
        "active_paths": list(cfg.active_paths),
        "modalities": list(cfg.data.modalities),
        "params_total": total,
        "params_trainable": trainable,
        "params_by_module": per_top,
    }


def build_model(cfg: BloomNetConfig) -> Any:
    """`build_bloomnet(cfg)` 지연 호출 (06 §3.4.3). L5 가 아직 없으면 친절한 SystemExit."""
    mod = require_module(
        "bloomnet.models.bloomnet",
        group="구현 대기 (06 §2.2 L5)",
        hint="models/encoder.py(L3) → backbone.py(L4) → bloomnet.py(L5) 순서로 구현되어야 한다.",
    )
    return mod.build_bloomnet(cfg)


def make_dummy_inputs(
    cfg: BloomNetConfig,
    *,
    batch: int = 1,
    hw: Optional[Tuple[int, int]] = None,
    device: Any = "cpu",
    dtype: Any = None,
) -> Dict[str, Any]:
    """활성 모달만 담은 더미 입력 dict (06 §3.4.3 forward 키워드와 동일 이름).

    채널 수는 config 에서 유도한다 — msi 는 `len(band_ids)`(센서 의존 4/5/6), bio 2, ir 1,
    pol 3(=[DoLP, sin2θ, cos2θ], 정정 B-4).
    """
    import torch

    h, w = hw or tuple(int(v) for v in cfg.data.train_size)
    kw: Dict[str, Any] = {}
    opts = {"device": device, "dtype": dtype or torch.float32}

    def _t(c: int) -> Any:
        return torch.randn(batch, c, h, w, **opts)

    mods = list(cfg.data.modalities)
    # 채널 수 리터럴은 전부 constants(L−1) 단일 출처에서 온다 (06 §2.1.1).
    kw["rgb"] = _t(MODALITY_CHANNELS["rgb"])
    if "msi" in mods:
        # msi 는 센서 의존이라 MODALITY_CHANNELS 가 -1 이다 → band_ids 로 유도, 없으면 canonical 6
        n_band = len(cfg.data.band_ids) if cfg.data.band_ids else len(MSI_SLOTS)
        kw["msi"] = _t(int(n_band))
    if "bio" in mods:
        kw["bio"] = _t(MODALITY_CHANNELS["bio"])
    if "ir" in mods:
        kw["ir"] = _t(MODALITY_CHANNELS["ir"])
    if "pol" in mods:
        kw["pol"] = _t(MODALITY_CHANNELS["pol"] if cfg.data.pol.encoding == "sincos" else 2)
    return kw


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), "utf-8")
    return path


def handle_common_flags(cfg: BloomNetConfig, args: argparse.Namespace) -> Optional[int]:
    """`--deps` / `--print_config` 처리. 처리했으면 종료 코드를, 아니면 None 을 돌려준다."""
    if getattr(args, "deps", False):
        print(dependency_report())
        return 0
    if getattr(args, "print_config", False):
        sys.stdout.write(to_yaml(cfg))
        return 0
    return None
