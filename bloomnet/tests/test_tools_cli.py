"""도구 CLI · 동봉 config 계약 (06 §2.1 scripts/*, §4.2 configs/*).

실행:
    cd <repo_root> && CUDA_VISIBLE_DEVICES="" \\
        python -m pytest bloomnet/tests/test_tools_cli.py -q

검사 범위
    1. 5개 도구가 `--help` / `--deps` / `--print_config` 로 **정상 종료**한다.
    2. 동봉 config 11개가 전부 로드되고, `_base.yaml` 이 dataclass 기본값과 정확히 같다.
    3. `config 로드 → build_bloomnet` dry-run 이 CPU 에서 통과한다.
    4. 잘못된 config(미지 키 / V2 / V16 / V19 / 없는 파일)를 **거부**한다.
    5. 학습·평가·export 가 시작되지 않는다 — 각 도구는 계약 검사 지점에서 멈춘다.

주의: 헌법 C-5.2(GPU 금지) / C-5.6(쓰기 범위) 준수.
      임시 파일은 `/tmp` 가 아니라 `<repo_root>/.pytest_tmp` 아래에만 만든다.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
import torch
import yaml

import bloomnet
from bloomnet.config import default_config, to_dict
from bloomnet.tools import _cli
from bloomnet.tools import benchmark as t_benchmark
from bloomnet.tools import eval as t_eval  # noqa: A004 — 스크립트 이름이 계약이다
from bloomnet.tools import export as t_export
from bloomnet.tools import train as t_train
from bloomnet.tools import train_spec as t_train_spec

REPO_ROOT = Path(bloomnet.__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "bloomnet" / "configs"

TOOLS = {
    "train": t_train,
    "train_spec": t_train_spec,
    "eval": t_eval,
    "export": t_export,
    "benchmark": t_benchmark,
}

# 실제 CPU forward 를 도는 테스트의 입력 크기 (헌법 C-5.1 / §5 공통 규약: B<=2, 64²)
SMALL_HW = ["64", "64"]


@pytest.fixture(autouse=True)
def _no_gpu() -> None:
    """헌법 C-5.2 — 어떤 테스트도 GPU 를 건드리지 않는다."""
    assert not torch.cuda.is_available(), "GPU 가 보인다 — CUDA_VISIBLE_DEVICES='' 로 실행하라"


@pytest.fixture()
def kw_tmp() -> Iterator[Path]:
    base = REPO_ROOT / ".pytest_tmp"
    base.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            base.rmdir()
        except OSError:
            pass


def write_yaml(path: Path, data: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def config_files() -> List[Path]:
    return sorted(CONFIG_DIR.rglob("*.yaml"))


# ═══════════════════════════════════════════════════════════════════════
#  1. --help / --deps / --print_config
# ═══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("name", sorted(TOOLS))
def test_help_exits_zero(name: str, capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc:
        TOOLS[name].main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--config", "--set", "--print_config", "--build_only", "--deps"):
        assert flag in out, f"{name} --help 에 {flag} 가 없다"


def test_help_works_as_module_subprocess() -> None:
    """`python -m bloomnet.tools.train --help` 가 실제로 실행 가능해야 한다 (README 명령)."""
    r = subprocess.run(
        [sys.executable, "-m", "bloomnet.tools.train", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
        timeout=180,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "--config" in r.stdout


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_deps_report_is_explicit(name: str, capsys: pytest.CaptureFixture) -> None:
    """06 §5.0 격리 규칙 2 — 미설치를 조용히 넘기지 않고 명시 출력한다."""
    assert TOOLS[name].main(["--deps"]) == 0
    out = capsys.readouterr().out
    assert "requirements.txt" in out and "requirements-deploy.txt" in out


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_print_config_dumps_resolved_yaml(name: str, capsys: pytest.CaptureFixture) -> None:
    assert TOOLS[name].main(["--config", "s0_rgb_aihub092", "--print_config"]) == 0
    data = yaml.safe_load(capsys.readouterr().out)
    assert data["mode"] == "s0_rgb"
    assert data["data"]["num_classes"] == 12
    # 병합 완료본이므로 스키마 전 키가 들어 있어야 한다 (06 §4.1 step 6)
    assert set(data) == set(to_dict(default_config()))


# ═══════════════════════════════════════════════════════════════════════
#  2. 동봉 config 계약
# ═══════════════════════════════════════════════════════════════════════
def test_expected_config_files_exist() -> None:
    names = {p.relative_to(CONFIG_DIR).as_posix() for p in config_files()}
    expected = {
        "_base.yaml",
        "s0_rgb_aihub092.yaml",
        "s0_spec_k235.yaml",
        "s1_rgb_ms4.yaml",
        "s2_full.yaml",
        "kwater_ft_s0.yaml",
        "ablation/a1_dice.yaml",
        "ablation/a2_rfs.yaml",
        "ablation/a3_layerscale.yaml",
        "ablation/a4_bio_gndvi.yaml",
        "ablation/a5_aux_p8.yaml",
        "ablation/a6_attn_all_stages.yaml",
    }
    assert names == expected, f"누락 {expected - names} / 잉여 {names - expected}"


def test_repo_root_configs_alias_points_at_package_configs() -> None:
    """저장소 루트 `configs/` 는 `bloomnet/configs` 의 별칭이어야 한다 (실체 1개).

    06 §2.1 매니페스트는 `configs/` 를 저장소 루트에 두고 `config.DEFAULT_BASE_PATH` 도
    `"configs/_base.yaml"`(cwd 상대)이다. 본 저장소는 실체를 패키지 안에 두므로 링크로 잇는다.
    링크가 사라지면 `parse_cli()` 의 기본 base 가 조용히 dataclass 기본값으로 폴백한다.
    """
    alias = REPO_ROOT / "configs"
    assert alias.is_dir(), f"{alias} 가 없다 — `ln -sfn bloomnet/configs configs` 로 만든다"
    assert (alias / "_base.yaml").resolve() == (CONFIG_DIR / "_base.yaml").resolve()


def test_base_yaml_is_exact_dump_of_dataclass_defaults() -> None:
    """`_base.yaml` 은 스키마 정본의 덤프본이다 — 손으로 고치면 여기서 깨진다."""
    data = yaml.safe_load((CONFIG_DIR / "_base.yaml").read_text(encoding="utf-8"))
    assert data == to_dict(default_config())


@pytest.mark.parametrize("path", config_files(), ids=lambda p: p.stem)
def test_every_shipped_config_loads(path: Path) -> None:
    from bloomnet.config import BloomNetConfig, from_dict, load_config

    cfg = load_config(path)
    assert isinstance(cfg, BloomNetConfig)
    # round-trip 동일성 (config.resolved.yaml 재로드 계약)
    assert to_dict(from_dict(to_dict(cfg))) == to_dict(cfg)


def test_base_yaml_is_a_noop_base() -> None:
    """`--base configs/_base.yaml` 를 기저로 써도 dataclass 기본값과 결과가 같아야 한다."""
    from bloomnet.config import load_config

    a = load_config(CONFIG_DIR / "s0_rgb_aihub092.yaml", base=CONFIG_DIR / "_base.yaml")
    b = load_config(CONFIG_DIR / "s0_rgb_aihub092.yaml")
    assert to_dict(a) == to_dict(b)


def test_preset_resolves_by_bare_name() -> None:
    assert _cli.resolve_config_path("s1_rgb_ms4") == CONFIG_DIR / "s1_rgb_ms4.yaml"
    assert _cli.resolve_config_path("a4_bio_gndvi") == CONFIG_DIR / "ablation" / "a4_bio_gndvi.yaml"
    assert _cli.resolve_config_path(str(CONFIG_DIR / "s2_full.yaml")).name == "s2_full.yaml"
    with pytest.raises(FileNotFoundError, match="찾을 수 없다"):
        _cli.resolve_config_path("no_such_preset")


def test_explicit_missing_base_is_an_error_but_default_falls_back(kw_tmp: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--base"):
        _cli.resolve_base_path(str(kw_tmp / "nope.yaml"))
    assert _cli.resolve_base_path(str(_cli.DEFAULT_BASE)) == _cli.DEFAULT_BASE


def test_preset_contracts() -> None:
    """각 프리셋이 스펙이 요구하는 값을 실제로 담고 있는지 (주석이 아니라 값으로)."""
    from bloomnet.config import load_config

    s0 = load_config(CONFIG_DIR / "s0_rgb_aihub092.yaml")
    assert s0.mode == "s0_rgb" and s0.data.num_classes == 12
    assert s0.data.split_variant == "group"  # 01 §7.3 [M3] 누수 실측
    assert s0.data.sampler.kind == "rfs" and s0.data.sampler.repeat_t == 0.05  # 05 §4.3 #1
    assert s0.loss.ohem_thresh == 0.7 and s0.optim.lr == 4.0e-4
    assert tuple(s0.data.eval_size) == (512, 512)  # aihub092 원본이 512²

    spec = load_config(CONFIG_DIR / "s0_spec_k235.yaml")
    assert spec.mode == "s0_spec" and spec.data.dataset == "k235"
    assert spec.spec.transplant.enabled is False  # ★ 정정 A-19
    assert spec.spec.band_order_confirmed is False  # ★ V18
    assert spec.spec.use_blue is False and spec.spec.device == "cpu"  # X-03 / CPU 실행

    s1 = load_config(CONFIG_DIR / "s1_rgb_ms4.yaml")
    assert s1.active_paths == ("rgb", "spec") and s1.data.num_classes == 2
    assert s1.data.bio.kind == "mci" and s1.data.bio.source == "msi"  # X-01/X-27
    assert s1.loss.ohem_thresh == 0.9  # 05 §4.4 저데이터 → 약한 마이닝
    assert s1.train.modality_dropout.enabled is True  # ★ 정정 B-11(c)
    assert s1.optim.lr_encoder == 6.0e-5 and s1.optim.lr == 6.0e-4  # 정정 B-28 그룹 분리
    assert s1.optim.lr * s1.optim.physics_lr_mult == pytest.approx(3.0e-3)
    assert s1.data.msi.k_sensor is not None and s1.data.msi.k_sensor > 0  # V21

    s2 = load_config(CONFIG_DIR / "s2_full.yaml")
    assert s2.active_paths == ("rgb", "spec", "phys")
    assert s2.model.ppn.use_pol is True  # V6 자동 동기화
    assert s2.data.augment.geometric.rot90 is False  # ★ V7 (AoLP 회전 규약)
    assert s2.data.pol.encoding == "sincos"  # 3ch [DoLP, sin2θ, cos2θ] (정정 B-4)
    assert s2.train.modality_dropout.enabled is True


def test_ablation_contracts() -> None:
    from bloomnet.config import load_config

    ab = CONFIG_DIR / "ablation"
    base = load_config(CONFIG_DIR / "s0_rgb_aihub092.yaml")

    a1 = load_config(ab / "a1_dice.yaml")
    assert a1.loss.w_dice == 0.0 and base.loss.w_dice == 0.4

    a2 = load_config(ab / "a2_rfs.yaml")
    assert a2.data.sampler.repeat_t == 0.0

    a3 = load_config(ab / "a3_layerscale.yaml")
    assert a3.model.layer_scale_init == [0.1, 0.1, 0.01, 0.01]

    a4 = load_config(ab / "a4_bio_gndvi.yaml")
    assert a4.data.bio.kind == "gndvi" and a4.data.bio.source == "msi"  # V16 을 만족해야 한다

    a5 = load_config(ab / "a5_aux_p8.yaml")
    assert a5.model.aux_taps == ["p8"] and "p8" in a5.loss.lambda_aux  # V9

    a6 = load_config(ab / "a6_attn_all_stages.yaml")
    assert a6.model.ela.attn_stages == [1, 2, 3, 4]


def test_all_configs_satisfy_fp16_precision_contract() -> None:
    """V13 — fp16/int8 경로에서 BMEF·LiteMLA fp32 강제가 빠지면 안 된다 (정정 A-22)."""
    from bloomnet.config import load_config

    for path in config_files():
        cfg = load_config(path)
        if cfg.train.amp == "fp16" or cfg.deploy.precision in ("fp16", "int8"):
            assert "bmef.fusion" in cfg.deploy.fp32_forced, path
            assert "encoder.*.litemla.attention" in cfg.deploy.fp32_forced, path


# ═══════════════════════════════════════════════════════════════════════
#  3. dry-run: config 로드 → build_bloomnet (CPU)
# ═══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "preset", ["s0_rgb_aihub092", "s1_rgb_ms4", "s2_full", "a5_aux_p8"]
)
def test_train_build_only_cpu(preset: str, capsys: pytest.CaptureFixture) -> None:
    """`--build_only` 는 모델을 만들고 **학습을 시작하지 않는다**."""
    assert t_train.main(["--config", preset, "--build_only", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "build_only" in out and "학습은 하지 않는다" in out
    assert "params_total" in out


def test_build_only_param_counts_match_frozen_budget(capsys: pytest.CaptureFixture) -> None:
    """06 §10 요약 카드의 모드별 파라미터(추론 구성 = 학습 전용 헤드 제외)와 일치."""
    from bloomnet.config import load_config

    from bloomnet.tools._cli import build_model

    expected = {  # (preset, num_classes) -> 추론 구성 params
        "s0_rgb_aihub092": 8_954_390,  # K=12
        "s1_rgb_ms4": 14_660_522,  # K=2  (정정 A-15)
        "s2_full": 15_969_023,  # K=2  (06 §6.2)
    }
    for preset, want in expected.items():
        cfg = load_config(CONFIG_DIR / f"{preset}.yaml")
        model = build_model(cfg)
        train_only = sum(
            p.numel()
            for n, m in model.named_children()
            if n in ("edge_head", "aux_heads", "siam_proj")
            for p in m.parameters()
        )
        total = sum(p.numel() for p in model.parameters())
        assert total - train_only == want, f"{preset}: {total - train_only} != {want}"
    capsys.readouterr()


@pytest.mark.parametrize("name", ["eval", "export", "benchmark"])
def test_other_tools_build_only(name: str, capsys: pytest.CaptureFixture) -> None:
    assert TOOLS[name].main(["--config", "s0_rgb_aihub092", "--build_only", "--device", "cpu"]) == 0
    assert "build_only" in capsys.readouterr().out


def test_eval_builds_criterion_and_evaluators_from_config() -> None:
    """config → `BloomNetCriterion` / Seg·Reg·Boundary Evaluator 배선 (데이터 없이)."""
    from bloomnet.config import load_config

    cfg = load_config(CONFIG_DIR / "s0_rgb_aihub092.yaml")
    crit = t_eval.build_criterion(cfg)
    assert crit.lambda_seg == 1.0 and crit.lambda_reg == 0.5  # V2 계약이 실제로 전달됐다
    assert tuple(crit.unc_clamp) == (-7.0, 7.0)  # X-26 / 정정 A-20
    evs = t_eval._build_evaluators(cfg)
    assert len(evs) == 3
    assert evs[0].num_classes == 12
    assert tuple(evs[2].tolerances) == (1, 3, 5)


def test_train_spec_loso_kwargs_are_accepted_by_run_loso() -> None:
    """`_LOSO_KEYS` ⊆ `loso.RUN_DEFAULTS` — 미지 키는 `run_loso` 가 TypeError 를 낸다.

    실제 LOSO 실행에는 235 npz 캐시가 필요하므로, 여기서는 **키 계약만** 고정한다.
    이 테스트가 없으면 스키마 키 이름이 바뀌었을 때 몇 시간짜리 LOSO 가 시작 직후 죽는다.
    """
    from bloomnet.config import load_config
    from bloomnet.pretrain.loso import RUN_DEFAULTS, run_holdout, run_loso

    cfg = load_config(CONFIG_DIR / "s0_spec_k235.yaml")
    kwargs = t_train_spec._loso_kwargs(cfg)
    unknown = sorted(set(kwargs) - set(RUN_DEFAULTS))
    assert not unknown, f"run_loso 가 모르는 키: {unknown}"
    assert callable(run_loso) and callable(run_holdout)
    assert cfg.spec.device == "cpu"  # loso 는 cuda 요청을 ValueError 로 거부한다


def test_train_spec_build_only_gives_1089_params(capsys: pytest.CaptureFixture) -> None:
    assert t_train_spec.main(["--config", "s0_spec_k235", "--build_only"]) == 0
    out = capsys.readouterr().out
    assert "params     : 1089" in out, out


def test_export_check_only_contract_cpu(capsys: pytest.CaptureFixture) -> None:
    """04 §9.4 step 1 + ExportWrapper 3-tuple + conf/chl 범위 (onnx 불필요)."""
    rc = t_export.main(
        ["--config", "s0_rgb_aihub092", "--check_only", "--device", "cpu", "--input_hw", *SMALL_HW]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "학습 전용 키 0개" in out
    assert "ExportWrapper 입력 1개" in out  # active_paths == ('rgb',) (정정 A-35)
    assert "seg (1, 12, 64, 64)" in out


def test_benchmark_macs_only_cpu(kw_tmp: Path, capsys: pytest.CaptureFixture) -> None:
    out_json = kw_tmp / "bench.json"
    rc = t_benchmark.main(
        [
            "--config", "s0_rgb_aihub092", "--device", "cpu", "--macs_only",
            "--mac_hw", *SMALL_HW, "--out", str(out_json),
        ]
    )
    assert rc == 0
    capsys.readouterr()
    payload = yaml.safe_load(out_json.read_text(encoding="utf-8"))
    assert payload["macs"]["measured_at"] == [64, 64]
    # 면적비 스케일 (정정 A-24): 512² = ×64, 1024² = ×256
    assert payload["macs"]["gmac_512"] == pytest.approx(payload["macs"]["gmac_measured"] * 64)
    assert payload["macs"]["gmac_1024"] == pytest.approx(payload["macs"]["gmac_measured"] * 256)
    assert "latency" not in payload  # --macs_only 는 forward 반복을 돌지 않는다
    assert payload["summary"]["params_total"] == 9_447_195  # 학습 구성(edge/aux 포함)


# ═══════════════════════════════════════════════════════════════════════
#  4. 잘못된 config 거부
# ═══════════════════════════════════════════════════════════════════════
def test_unknown_key_is_rejected_with_full_dotted_path(kw_tmp: Path) -> None:
    bad = write_yaml(kw_tmp / "bad.yaml", {"model": {"ela": {"head_dimm": [16]}}})
    with pytest.raises(ValueError) as exc:
        t_train.main(["--config", str(bad), "--print_config"])
    msg = str(exc.value)
    assert "model.ela.head_dimm" in msg and "bad.yaml" in msg


def test_unknown_key_via_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="loss.lambda_typo"):
        t_train.main(["--set", "loss.lambda_typo=1.0", "--print_config"])


@pytest.mark.parametrize(
    "overrides,rule",
    [
        (["loss.lambda_seg=2.0"], "V2"),
        (["data.ignore_index=0"], "V3"),
        (["data.modalities=[msi]"], "V4"),
        (["data.bio.kind=mci", "data.bio.source=rgb_proxy"], "V16"),
        (["data.bio.source=rgb_proxy", "data.bio.kind=mci"], "V16"),
        (["model.bmef.stages=[2,3,4]"], "V19"),
        (["loss.aux_loss_stride=2"], "V22"),
        (["model.aux_taps=[enc_s99]"], "V9"),
    ],
)
def test_validation_rules_reject_bad_configs(overrides: List[str], rule: str) -> None:
    argv: List[str] = []
    for o in overrides:
        argv += ["--set", o]
    with pytest.raises(ValueError, match=rule):
        t_train.main(argv + ["--print_config"])


def test_contract_break_flag_unlocks_v2_only(capsys: pytest.CaptureFixture) -> None:
    rc = t_train.main(
        ["--set", "loss.lambda_seg=2.0", "--allow_contract_break", "--print_config"]
    )
    assert rc == 0
    assert yaml.safe_load(capsys.readouterr().out)["loss"]["lambda_seg"] == 2.0


def test_missing_config_file_is_rejected(kw_tmp: Path) -> None:
    with pytest.raises(FileNotFoundError):
        t_train.main(["--config", str(kw_tmp / "nope.yaml"), "--print_config"])


def test_transplant_requires_msi_mci_and_confirmed_band_order() -> None:
    """V12/V18 — S0-Spec 프리셋에서 이식만 켜면 반드시 거부돼야 한다 (정정 A-11/A-19)."""
    with pytest.raises(ValueError, match="V12|V18"):
        t_train_spec.main(
            ["--config", "s0_spec_k235", "--set", "spec.transplant.enabled=true", "--print_config"]
        )


# ═══════════════════════════════════════════════════════════════════════
#  5. 학습이 실수로 시작되지 않는다
# ═══════════════════════════════════════════════════════════════════════
def test_train_refuses_s0_spec_mode() -> None:
    with pytest.raises(SystemExit, match="train_spec"):
        t_train.main(["--config", "s0_spec_k235"])


def test_train_refuses_cuda_when_unavailable() -> None:
    """CUDA 부재 시 **학습을 시작하기 전에** 멈춘다 (며칠짜리 CPU 학습 사고 방지)."""
    with pytest.raises(SystemExit, match="CUDA"):
        t_train.main(["--config", "s0_rgb_aihub092", "--device", "cuda"])


def test_export_without_onnx_fails_loudly() -> None:
    """06 §5.0 격리 규칙 2 — 조용한 skip 금지. 미설치면 명시적으로 실패한다."""
    if not _cli.missing_packages(_cli.DEPLOY_PACKAGES):
        pytest.skip("requirements-deploy.txt 가 설치되어 있다 — 실패 경로를 재현할 수 없다")
    with pytest.raises(SystemExit, match="requirements-deploy.txt"):
        t_export.main(["--config", "s0_rgb_aihub092", "--device", "cpu", "--input_hw", *SMALL_HW])


def test_eval_rejects_unknown_split() -> None:
    with pytest.raises((SystemExit, RuntimeError, FileNotFoundError, ValueError)):
        t_eval.main(["--config", "s0_rgb_aihub092", "--device", "cpu", "--split", "train"])


# ═══════════════════════════════════════════════════════════════════════
#  6. 패키징 파일
# ═══════════════════════════════════════════════════════════════════════
def _requirement_names(path: Path) -> List[str]:
    """주석(#)과 `-r` 포함 줄을 걷어내고 **실제 요구 패키지 이름**만 남긴다."""
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        for sep in (">=", "==", "<=", "~=", ">", "<", "["):
            line = line.split(sep, 1)[0]
        out.append(line.strip())
    return out


def test_requirements_split_matches_spec() -> None:
    pkg = REPO_ROOT / "bloomnet"
    runtime = _requirement_names(pkg / "requirements.txt")
    dev = _requirement_names(pkg / "requirements-dev.txt")
    deploy = _requirement_names(pkg / "requirements-deploy.txt")
    assert {"torch", "torchvision", "numpy", "pillow", "pyyaml", "tqdm", "scipy"} <= set(runtime)
    # 06 §5.0 — cv2/opencv 는 **어느 목록에도** 넣지 않는다 (경계 연산자가 torch max_pool 로 대체)
    for group in (runtime, dev, deploy):
        assert not [n for n in group if "opencv" in n or n == "cv2"]
    assert {"pytest", "pytest-cov", "ruff"} <= set(dev)
    assert {"onnx", "onnxscript", "onnxruntime", "onnxsim", "polygraphy"} <= set(deploy)
    assert "tensorrt" not in deploy  # Jetson 이미지에 포함되므로 제외
    # dev/deploy 는 runtime 을 상속한다 (`-r requirements.txt`)
    for f in ("requirements-dev.txt", "requirements-deploy.txt"):
        assert "-r requirements.txt" in (pkg / f).read_text(encoding="utf-8")


def test_pyproject_declares_line_length_and_markers() -> None:
    text = (REPO_ROOT / "bloomnet" / "pyproject.toml").read_text(encoding="utf-8")
    assert "line-length = 100" in text
    assert 'requires-python = ">=3.9"' in text
    assert "data:" in text and "slow:" in text  # 06 §5.3 CI 게이트 마커


def test_readme_contains_runnable_entrypoints() -> None:
    text = (REPO_ROOT / "bloomnet" / "README.md").read_text(encoding="utf-8")
    for tool in TOOLS:
        assert f"python -m bloomnet.tools.{tool}" in text, tool
    assert "cd <repo_root>" in text
    assert 'CUDA_VISIBLE_DEVICES=""' in text  # GPU 금지 규약이 문서에 있어야 한다
