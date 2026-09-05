"""T01 — 의존 레벨 위반 0건 / import 순환 0건 (06 §2.2).

각 모듈을 ``ast`` 로 파싱해 ``bloomnet.*`` import 를 수집하고 §2.2 레벨표와 대조한다.
**동급·상위 레벨 import 를 발견하면 실패**한다. 순환 import 방지의 근본 장치다.

정정 A-23: ``constants.py``/``version.py`` 는 **L−1 예외**로 어느 레벨에서도 import
할 수 있고, 그 둘은 ``bloomnet.*`` 를 import 하지 않는다.

``__init__.py`` 는 레벨표 대조에서 제외한다 — 06 §2.1 원칙 3 이 "re-export 만" 을
요구하므로 그 파일의 import 는 정의상 자기 패키지 하위를 가리키고, 레벨이 정의되어
있지 않다. 대신 "re-export 만 하는가"(함수/클래스 정의 0개)를 별도로 검사한다.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:  # cwd 무관 부트스트랩
    sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 레벨표 (06 §2.2 위상 정렬 결과)
# ─────────────────────────────────────────────────────────────────────────────
#: ``-1`` = 레벨 예외(어디서나 import 가능, 정정 A-23).
LEVELS: Dict[str, int] = {
    # L−1 (정정 A-23)
    "constants.py": -1,
    "version.py": -1,
    # L0
    "utils/seed.py": 0,
    "utils/metrics_seg.py": 0,
    "utils/metrics_reg.py": 0,
    "data/boundary.py": 0,
    # L1
    "config.py": 1,
    "modules/common.py": 1,
    "data/bundle.py": 1,
    "data/indices.py": 1,
    "data/transforms.py": 1,
    "data/samplers.py": 1,
    "losses/seg.py": 1,
    "losses/regression.py": 1,
    "losses/distill.py": 1,
    "engine/sched.py": 1,
    "engine/ema.py": 1,
    "utils/logging_csv.py": 1,
    "utils/checkpoint.py": 1,
    "utils/flops.py": 1,
    "utils/metrics_boundary.py": 1,
    "utils/distributed.py": 1,
    "deploy/trt_policy.py": 1,
    "deploy/postprocess.py": 1,
    # L2
    "modules/stems.py": 2,
    "modules/blocks_ela.py": 2,
    "modules/blocks_biospec.py": 2,
    "modules/blocks_physlite.py": 2,
    "modules/bmef.py": 2,
    "modules/decoder_blocks.py": 2,
    "modules/heads.py": 2,
    "data/aihub092.py": 2,
    "data/k235.py": 2,
    "data/drone_m3m.py": 2,
    "losses/boundary_loss.py": 2,
    "engine/optim.py": 2,
    "pretrain/spec_mlp.py": 2,
    # L3
    "models/encoder.py": 3,
    "modules/pid_decoder.py": 3,
    "losses/criterion.py": 3,
    "data/build.py": 3,
    "engine/evaluator.py": 3,
    "pretrain/loso.py": 3,
    "tools/link_aihub092.py": 3,
    "tools/make_group_split.py": 3,
    "tools/build_k235_cache.py": 3,
    "tools/compute_class_stats.py": 3,
    "tools/colorize_labels.py": 3,
    "tools/calibrate_k_sensor.py": 3,
    # L4 / L5
    "models/backbone.py": 4,
    "models/bloomnet.py": 5,
    # L6
    "engine/trainer.py": 6,
    "pretrain/transplant.py": 6,
    "deploy/export_onnx.py": 6,
    # ★ 매니페스트 밖 신설 파일. 5개 진입점이 공유하는 CLI 배선이며 정적 import 는
    #   config/constants/utils.flops(≤L1) 뿐이다. 진입점(L7)보다 아래여야 하므로 L6.
    "tools/_cli.py": 6,
    # L7 — 06 §2.1 은 이들을 `scripts/` 에 두지만 본 구현은 `tools/` 에 있다.
    #      **위치는 tools/ 지만 레벨은 L7** 이다 (tools 담당자 인계사항).
    "tools/train.py": 7,
    "tools/train_spec.py": 7,
    "tools/eval.py": 7,
    "tools/export.py": 7,
    "tools/predict_media.py": 7,
    "tools/benchmark.py": 7,
}

#: L−1 예외 모듈 (어느 레벨에서도 import 허용).
LEVEL_EXEMPT: Tuple[str, ...] = ("bloomnet.constants", "bloomnet.version")


def _module_name(rel: str) -> str:
    return "bloomnet." + rel[:-3].replace("/", ".")


MODULE_LEVEL: Dict[str, int] = {_module_name(k): v for k, v in LEVELS.items()}


def _iter_source_files() -> List[Path]:
    return sorted(
        p
        for p in PKG_ROOT.rglob("*.py")
        if "tests" not in p.relative_to(PKG_ROOT).parts and p.name != "__init__.py"
    )


def _collect_bloomnet_imports(path: Path) -> Set[str]:
    """``bloomnet.*`` 절대/상대 import 를 전부 수집한다 (``from X import Y`` 의 Y 포함)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    self_mod = _module_name(str(path.relative_to(PKG_ROOT)))
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "bloomnet" or a.name.startswith("bloomnet."):
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 상대 import → 절대 경로로 환원
                pkg = ".".join(self_mod.split(".")[: -node.level])
                base = f"{pkg}.{node.module}" if node.module else pkg
            elif node.module and (node.module == "bloomnet" or node.module.startswith("bloomnet.")):
                base = node.module
            else:
                continue
            out.add(base)
            for a in node.names:
                out.add(f"{base}.{a.name}")
    return out


def _resolve(imported: str) -> str | None:
    """import 문자열 → 레벨표에 있는 **가장 긴** 모듈 접두."""
    best = None
    for mod in MODULE_LEVEL:
        if imported == mod or imported.startswith(mod + "."):
            if best is None or len(mod) > len(best):
                best = mod
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 테스트
# ─────────────────────────────────────────────────────────────────────────────
def test_every_source_file_has_a_level() -> None:
    """레벨표에 없는 새 파일이 생기면 즉시 실패한다 (레벨 미정의 = 계약 미정의)."""
    unmapped = [
        str(p.relative_to(PKG_ROOT)) for p in _iter_source_files()
        if str(p.relative_to(PKG_ROOT)) not in LEVELS
    ]
    assert unmapped == [], (
        f"레벨표에 없는 소스 파일 {unmapped}. 06 §2.2 레벨표와 이 테스트의 LEVELS 를 함께 갱신하라."
    )


def test_no_upward_or_sideways_imports() -> None:
    """★ 근본 장치 — 레벨 i 파일은 레벨 < i 만 import 한다 (L−1 예외 제외)."""
    violations: List[str] = []
    for path in _iter_source_files():
        rel = str(path.relative_to(PKG_ROOT))
        level = LEVELS.get(rel)
        if level is None:
            continue
        for imported in sorted(_collect_bloomnet_imports(path)):
            target = _resolve(imported)
            if target is None or target in LEVEL_EXEMPT:
                continue
            tlevel = MODULE_LEVEL[target]
            if tlevel == -1:
                continue
            if tlevel >= level:
                kind = "동급" if tlevel == level else "상위"
                violations.append(f"{rel} (L{level}) → {target} (L{tlevel}) [{kind} 레벨 import]")
    assert violations == [], "06 §2.2 레벨 위반:\n  " + "\n  ".join(violations)


def test_level_minus_one_imports_nothing_from_bloomnet() -> None:
    """정정 A-23 — ``constants.py``/``version.py`` 는 ``bloomnet.*`` 를 import 하지 않는다."""
    for rel, level in LEVELS.items():
        if level != -1:
            continue
        imports = _collect_bloomnet_imports(PKG_ROOT / rel)
        assert imports == set(), f"{rel} (L−1) 이 {sorted(imports)} 를 import 한다"


def test_init_files_only_reexport() -> None:
    """06 §2.1 원칙 3 — ``__init__.py`` 는 re-export 만 한다 (함수/클래스 정의 0개)."""
    offenders: List[str] = []
    for path in sorted(PKG_ROOT.rglob("__init__.py")):
        if "tests" in path.relative_to(PKG_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                offenders.append(f"{path.relative_to(PKG_ROOT)}::{node.name}")
    assert offenders == [], f"__init__.py 는 re-export 전용이다: {offenders}"


@pytest.mark.parametrize("rel", sorted(LEVELS))
def test_module_imports_without_cycle(rel: str) -> None:
    """★ 실제 import 순환 0건 — 각 모듈을 **단독으로** import 해 본다.

    ast 검사는 정적 구조만 본다. 지연 import(``importlib.import_module``)로 생기는
    순환은 잡지 못하므로 실행으로 확인한다. 순환이 있으면 ``ImportError``
    (partially initialized module) 로 즉시 터진다.
    """
    if rel == "deploy/export_onnx.py":
        pytest.importorskip("torch")  # onnx 는 지연 import 라 없어도 모듈 import 는 된다
    importlib.import_module(_module_name(rel))


def test_full_package_import_graph_is_acyclic() -> None:
    """전 모듈을 한 프로세스에서 import 한 뒤 부분 초기화 모듈이 없는지 확인한다."""
    for rel in sorted(LEVELS):
        importlib.import_module(_module_name(rel))
    partial = [
        name
        for name, mod in list(sys.modules.items())
        if name.startswith("bloomnet") and mod is not None and not hasattr(mod, "__file__")
    ]
    assert partial == [], f"부분 초기화된 모듈(순환 징후): {partial}"
