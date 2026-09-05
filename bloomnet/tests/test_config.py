"""T03 — `bloomnet/config.py` 스키마·로드·검증 계약 (06 §5.1 T03).

실행:
    cd <repo_root> && CUDA_VISIBLE_DEVICES="" \
        python -m pytest bloomnet/tests/test_config.py -q

주의: 헌법 C-5 제약상 임시 파일도 `<repo_root>` **안에서만** 만든다
      (pytest 기본 tmp_path 는 /tmp 라 쓰지 않는다 — `kw_tmp` fixture 참조).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
import yaml

import bloomnet
from bloomnet import config as cfgmod
from bloomnet.config import (
    BloomNetConfig,
    apply_overrides,
    default_config,
    dump_config,
    from_dict,
    load_config,
    merge_dicts,
    parse_cli,
    resolve_schedule,
    to_dict,
    to_yaml,
)
from bloomnet.constants import CHANNELS, IGNORE_INDEX, K_SENSOR, SENSOR_BAND_IDS

REPO_ROOT = Path(bloomnet.__file__).resolve().parent.parent


@pytest.fixture()
def kw_tmp() -> Iterator[Path]:
    """k_water 내부에만 만드는 임시 디렉터리 (쓰기 범위 제약)."""
    base = REPO_ROOT / ".pytest_tmp"
    base.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            base.rmdir()  # 비어 있을 때만
        except OSError:
            pass


def write_yaml(path: Path, data: Dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════════
#  스키마 · 기본값
# ═══════════════════════════════════════════════════════════════════════
def test_default_config_is_valid_and_frozen_values() -> None:
    c = default_config()
    assert c.mode == "s0_rgb"
    assert c.data.modalities == ["rgb"]
    assert c.data.ignore_index == IGNORE_INDEX
    assert tuple(c.model.channels) == CHANNELS
    assert c.model.decoder.dec_ch == 128
    assert (c.loss.lambda_seg, c.loss.lambda_reg) == (1.0, 0.5)  # 사업문서 확정값
    assert c.loss.unc_clamp == [-7.0, 7.0]  # X-26 / 정정 A-20
    assert c.loss.aux_loss_stride == 1  # 정정 A-32
    assert c.schedule.warmup_epochs == 2 and c.schedule.warmup_iters is None  # 정정 A-18
    assert c.spec.transplant.enabled is False  # 정정 A-19
    assert c.spec.band_order_confirmed is False
    assert c.model.bmef.stages == [1, 2, 3, 4] and c.model.bmef.stage1_identity is False
    assert c.data.boundary.radius == {512: 1, 1024: 2}  # 정정 A-28
    assert c.deploy.use_dynamo is False  # 정정 A-29
    assert "encoder.*.litemla.attention" in c.deploy.fp32_forced  # 정정 A-22
    assert "bmef.fusion" in c.deploy.fp32_forced
    assert c.data.augment.geometric.scale_range == [1.0, 2.0]  # 정정 B-27
    assert c.active_paths == ("rgb",)


def test_schema_roundtrip_dict() -> None:
    c = default_config()
    d = to_dict(c)
    assert "allow_contract_break" not in d  # 런타임 플래그는 스키마 밖
    assert to_dict(from_dict(d)) == d


def test_resolved_yaml_roundtrip(kw_tmp: Path) -> None:
    """`config.resolved.yaml` round-trip 동일성 (06 §4.1 step 6)."""
    c = load_config(overrides=["data.sensor=m3m", "data.modalities=[rgb, msi]"])
    path = dump_config(c, kw_tmp / "run" / "config.resolved.yaml")
    assert path.is_file()
    reloaded = load_config(path)
    assert to_dict(reloaded) == to_dict(c)
    assert to_yaml(reloaded) == to_yaml(c)


def test_base_yaml_matches_schema_if_present() -> None:
    """`configs/_base.yaml` 은 dataclass 트리의 덤프본이어야 한다 (누락·잉여 0)."""
    base = REPO_ROOT / "configs" / "_base.yaml"
    if not base.is_file():
        pytest.skip(
            f"{base} 미생성 — `python -m bloomnet.config --dump-base --out configs/_base.yaml` "
            "로 생성하면 이 테스트가 스키마 완전 일치를 검사한다"
        )
    data = yaml.safe_load(base.read_text(encoding="utf-8"))
    assert _key_tree(data) == _key_tree(to_dict(default_config()))
    from_dict(data, source=str(base))  # 미지 키 0건


def _key_tree(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _key_tree(v) for k, v in sorted(d.items())}
    return None


# ═══════════════════════════════════════════════════════════════════════
#  미지 키 거부 · 타입
# ═══════════════════════════════════════════════════════════════════════
def test_unknown_key_rejected_with_full_dotted_path(kw_tmp: Path) -> None:
    path = write_yaml(kw_tmp / "bad.yaml", {"model": {"ela": {"head_dimm": [16, 16, 32, 32]}}})
    with pytest.raises(ValueError) as exc:
        load_config(path)
    msg = str(exc.value)
    assert "model.ela.head_dimm" in msg  # 전체 dotted path
    assert str(path) in msg  # 어느 파일인지


def test_unknown_top_level_key_rejected(kw_tmp: Path) -> None:
    path = write_yaml(kw_tmp / "bad2.yaml", {"modell": {}})
    with pytest.raises(ValueError, match="Unknown config key 'modell'"):
        load_config(path)


def test_type_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="schedule.batch_size"):
        load_config(overrides=["schedule.batch_size=abc"])
    with pytest.raises(ValueError, match="must be a bool"):
        load_config(overrides=["train.ema.enabled=1"])
    with pytest.raises(ValueError, match="must be a list"):
        load_config(overrides=["data.modalities=rgb"])


# ═══════════════════════════════════════════════════════════════════════
#  병합 · 오버라이드
# ═══════════════════════════════════════════════════════════════════════
def test_merge_dicts_replaces_lists() -> None:
    merged = merge_dicts({"a": {"b": [1, 2, 3], "c": 1}}, {"a": {"b": [9]}})
    assert merged == {"a": {"b": [9], "c": 1}}  # 리스트는 교체, 병합 아님


def test_base_chain_merge(kw_tmp: Path) -> None:
    parent = write_yaml(kw_tmp / "parent.yaml", {"schedule": {"epochs": 10, "batch_size": 8}})
    child = write_yaml(
        kw_tmp / "child.yaml", {"_base": parent.name, "schedule": {"epochs": 3}, "seed": 7}
    )
    c = load_config(child)
    assert (c.schedule.epochs, c.schedule.batch_size, c.seed) == (3, 8, 7)


def test_set_override_flips_bool_true_to_false() -> None:
    """이전 구현의 `store_true` 결함 제거 — `--set x=false` 로 양방향 (06 §4.1)."""
    assert default_config().data.augment.photometric.enabled is True
    c = load_config(overrides=["data.augment.photometric.enabled=false"])
    assert c.data.augment.photometric.enabled is False
    assert load_config(overrides=["data.pin_memory=false"]).data.pin_memory is False


def test_set_override_parses_yaml_values() -> None:
    c = load_config(overrides=["model.ela.head_dim=[16, 16, 32, 32]", "optim.lr=1.0e-5"])
    assert c.model.ela.head_dim == [16, 16, 32, 32]
    assert c.optim.lr == pytest.approx(1e-5)
    c2 = load_config(overrides=["schedule.scheduler_kwargs={power: 0.5}"])
    assert c2.schedule.scheduler_kwargs == {"power": 0.5}  # 중첩 dict 를 직접 표현


def test_cli_overrides_yaml(kw_tmp: Path) -> None:
    path = write_yaml(kw_tmp / "c.yaml", {"schedule": {"batch_size": 4}, "seed": 1})
    c = load_config(path, overrides=["schedule.batch_size=16"])
    assert c.schedule.batch_size == 16  # CLI > YAML > 기본값


def test_apply_overrides_requires_equals() -> None:
    with pytest.raises(ValueError, match="Invalid override"):
        apply_overrides({}, ["schedule.batch_size"])


def test_first_class_flags_map_to_dotted_paths() -> None:
    argv = [
        "--batch_size", "8",
        "--lr", "1e-3",
        "--epochs", "5",
        "--seed", "99",
        "--device", "cpu",
        "--num_workers", "2",
        "--num_classes", "2",
        "--mode", "s0_rgb",
        "--output_dir", "outputs/x",
        "--run_name", "smoke",
        "--data_root", "data/tiny",
        "--dry_run",
        "--no_amp",
        "--set", "loss.w_dice=0.0",
    ]
    c, args = parse_cli(argv)
    assert c.schedule.batch_size == 8
    assert c.optim.lr == pytest.approx(1e-3)
    assert c.schedule.epochs == 5
    assert c.seed == 99 and c.device == "cpu" and c.data.num_workers == 2
    assert c.data.num_classes == 2 and c.mode == "s0_rgb"
    assert c.output_dir == "outputs/x" and c.run_name == "smoke" and c.data.root == "data/tiny"
    assert c.dry_run is True
    assert c.train.amp == "off"  # --no_amp
    assert c.loss.w_dice == 0.0
    assert args.config is None
    assert len(cfgmod.FIRST_CLASS_FLAGS) == 15


def test_cli_missing_explicit_base_raises(kw_tmp: Path) -> None:
    """기본 경로(configs/_base.yaml) 부재는 dataclass 기본값으로 폴백하지만, 명시 지정은 에러다."""
    with pytest.raises(FileNotFoundError):
        parse_cli(["--base", str(kw_tmp / "nope.yaml")])
    c, _ = parse_cli([])  # configs/_base.yaml 이 없어도 기본값으로 동작
    assert c.mode == "s0_rgb"


def test_cli_amp_flag() -> None:
    c, _ = parse_cli(["--amp"])
    assert c.train.amp == "bf16"
    c2, _ = parse_cli(["--amp", "fp16", "--set", "deploy.precision=fp16"])
    assert c2.train.amp == "fp16"


# ═══════════════════════════════════════════════════════════════════════
#  유도값 해결
# ═══════════════════════════════════════════════════════════════════════
def test_sensor_derived_values() -> None:
    c = load_config(overrides=["data.sensor=m3m", "data.modalities=[rgb, msi]"])
    assert c.data.band_ids == list(SENSOR_BAND_IDS["m3m"])
    assert c.data.msi.k_sensor == K_SENSOR["m3m"]  # 정정 A-40 / V21
    assert c.spec.mci_c == pytest.approx((730 - 650) / (860 - 650), abs=1e-9)  # 정정 A-27
    assert default_config().data.band_ids is None  # sensor=none 이면 유도하지 않는다


def test_boundary_radius_lookup() -> None:
    c = default_config()
    assert (c.boundary_radius(512), c.boundary_radius(1024)) == (1, 2)  # 01 C-10


def test_active_paths_by_modalities() -> None:
    assert load_config(overrides=["data.modalities=[rgb]"]).active_paths == ("rgb",)
    c = load_config(
        overrides=["data.modalities=[rgb, msi, bio]", "data.sensor=m3m", "data.bio.source=msi"]
    )
    assert c.active_paths == ("rgb", "spec")
    c2 = load_config(
        overrides=[
            "data.modalities=[rgb, msi, bio, ir, pol]",
            "data.sensor=m3m",
            "data.bio.source=msi",
            "data.augment.geometric.rot90=false",
        ]
    )
    assert c2.active_paths == ("rgb", "spec", "phys")
    assert c2.model.ppn.use_pol is True  # V6 자동 동기화


# ═══════════════════════════════════════════════════════════════════════
#  V1 ~ V22  (각 1개 negative)
# ═══════════════════════════════════════════════════════════════════════
S1_BASE: List[str] = [
    "data.modalities=[rgb, msi, bio]",
    "data.sensor=m3m",
    "data.bio.source=msi",
    "data.bio.kind=mci",
    "mode=s1_rgb_ms4",
]


def test_v1_size_divisor() -> None:
    with pytest.raises(ValueError, match="V1"):
        load_config(overrides=["data.train_size=[500, 512]"])


def test_v2_lambda_contract_and_bypass() -> None:
    with pytest.raises(ValueError, match="V2"):
        load_config(overrides=["loss.lambda_reg=0.4"])
    with pytest.raises(ValueError, match="V2"):
        load_config(overrides=["loss.lambda_seg=2.0"])
    c = load_config(overrides=["loss.lambda_reg=0.4"], allow_contract_break=True)
    assert c.loss.lambda_reg == 0.4  # --allow_contract_break 로만 우회


@pytest.mark.parametrize(
    "override",
    ["data.ignore_index=0", "model.channels=[32, 64, 160, 321]", "model.decoder.dec_ch=64"],
)
def test_v3_constitution_fixed_values(override: str) -> None:
    with pytest.raises(ValueError, match="V3"):
        load_config(overrides=[override])


def test_v4_rgb_required() -> None:
    with pytest.raises(ValueError, match="V4"):
        load_config(overrides=["data.modalities=[msi]"])


def test_v5_bio_requires_msi_or_proxy() -> None:
    with pytest.raises(ValueError, match="V5"):
        load_config(overrides=["data.modalities=[rgb, bio]", "data.bio.source=none"])
    # 완화 경로(정정 A-11): rgb_proxy 는 msi 없이도 허용된다
    c = load_config(
        overrides=[
            "data.modalities=[rgb, bio]",
            "data.bio.source=rgb_proxy",
            "data.bio.kind=rgb_proxy",
        ]
    )
    assert c.data.bio.source == "rgb_proxy"


def test_v6_use_pol_sync() -> None:
    with pytest.raises(ValueError, match="V6"):
        load_config(
            overrides=S1_BASE
            + [
                "data.modalities=[rgb, msi, bio, ir, pol]",
                "model.ppn.use_pol=false",
                "data.augment.geometric.rot90=false",
            ]
        )
    with pytest.raises(ValueError, match="V6"):
        load_config(overrides=["model.ppn.use_pol=true"])  # pol 없는데 켜면 에러


def test_v7_rot90_disabled_with_pol() -> None:
    with pytest.warns(RuntimeWarning, match="V7"):
        c = load_config(
            overrides=S1_BASE
            + ["data.modalities=[rgb, msi, bio, ir, pol]", "mode=s2_full"]
        )
    assert c.data.augment.geometric.rot90 is False  # 강제 비활성 (01 §7.4.1)
    c2 = load_config(
        overrides=S1_BASE
        + [
            "data.modalities=[rgb, msi, bio, ir, pol]",
            "mode=s2_full",
            "data.augment.geometric.allow_rot90_with_pol=true",
        ]
    )
    assert c2.data.augment.geometric.rot90 is True  # 명시 opt-in 은 유지


def test_v8_bio_source_msi_requires_sensor() -> None:
    with pytest.raises(ValueError, match="V8"):
        load_config(overrides=["data.bio.source=msi"])  # modalities 에 msi 없음 + sensor none


def test_v9_aux_taps() -> None:
    with pytest.raises(ValueError, match="V9"):
        load_config(overrides=["model.aux_taps=[enc_s8, enc_s99]"])
    with pytest.raises(ValueError, match="V9"):
        load_config(overrides=["loss.lambda_aux={enc_s8: 0.2}"])  # enc_s16/s32 누락


def test_v10_boundary_radius_mapping() -> None:
    with pytest.raises(ValueError, match="V10"):
        load_config(overrides=["data.boundary.radius={512: 1}"])  # eval_size 1024 키 없음
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(overrides=["data.boundary.radius=1"])  # 스칼라 금지 (정정 A-28)


def test_v11_siam_teacher_must_exist(kw_tmp: Path) -> None:
    with pytest.raises(ValueError, match="V11"):
        load_config(overrides=["model.siam.enabled=true"])
    teacher = kw_tmp / "teacher.pt"
    teacher.write_text("dummy", encoding="utf-8")
    c = load_config(overrides=["model.siam.enabled=true", f"model.siam.teacher={teacher}"])
    assert c.model.siam.enabled is True


def test_v12_transplant_requires_msi_mci() -> None:
    """(정정 A-11) 초판은 kind 만 봐서 source=rgb_proxy + kind=mci 를 통과시켰다."""
    with pytest.raises(ValueError, match="V12"):
        load_config(
            overrides=[
                "spec.transplant.enabled=true",
                "spec.band_order_confirmed=true",
                "data.modalities=[rgb, bio]",
                "data.bio.source=rgb_proxy",
                "data.bio.kind=rgb_proxy",
            ]
        )


def test_v13_fp32_forced_requires_bmef_and_litemla() -> None:
    with pytest.raises(ValueError, match="V13"):
        load_config(overrides=["deploy.fp32_forced=[bmef.fusion]"])  # litemla 누락 (정정 A-22)
    with pytest.raises(ValueError, match="V13"):
        load_config(
            overrides=[
                "train.amp=fp16",
                "deploy.precision=fp32",
                "deploy.fp32_forced=[encoder.*.litemla.attention]",
            ]
        )


def test_v14_mode_modalities_warning() -> None:
    with pytest.warns(RuntimeWarning, match="V14"):
        load_config(overrides=["mode=s1_rgb_ms4"])  # modalities 는 [rgb] 그대로


def test_v15_schedule_derivation_and_negative() -> None:
    c = load_config(overrides=["schedule.epochs=60", "schedule.warmup_epochs=2"])
    resolve_schedule(c, 2108)
    assert c.schedule.warmup_iters == 4216  # round(2 × 2108)
    assert c.schedule.scheduler_kwargs["total_iters"] == 60 * 2108 - 4216
    assert c.schedule.iters_per_epoch == 2108
    assert c.schedule.warmup_iters < c.schedule.scheduler_kwargs["total_iters"]

    bad = load_config(overrides=["schedule.warmup_iters=5604"])  # asis 기준 하드코딩 (정정 A-18)
    with pytest.raises(ValueError, match="V15"):
        resolve_schedule(bad, 2108)

    bad2 = load_config(overrides=["schedule.scheduler_kwargs={power: 0.9, total_iters: 123}"])
    with pytest.raises(ValueError, match="V15"):
        resolve_schedule(bad2, 2108)


def test_v15_derived_values_are_dumped(kw_tmp: Path) -> None:
    c = default_config()
    resolve_schedule(c, 100)
    path = dump_config(c, kw_tmp / "config.resolved.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["schedule"]["warmup_iters"] == 200
    assert data["schedule"]["iters_per_epoch"] == 100
    assert data["schedule"]["scheduler_kwargs"]["total_iters"] == 60 * 100 - 200
    assert to_dict(load_config(path)) == to_dict(c)  # 유도값 포함 round-trip


def test_v16_re_nir_index_needs_msi_source() -> None:
    with pytest.raises(ValueError, match="V16"):
        load_config(
            overrides=[
                "data.modalities=[rgb, bio]",
                "data.bio.source=rgb_proxy",
                "data.bio.kind=mci",
            ]
        )


def test_v17_rgb_proxy_kind_must_match() -> None:
    with pytest.raises(ValueError, match="V17"):
        load_config(
            overrides=[
                "data.modalities=[rgb, bio]",
                "data.bio.source=rgb_proxy",
                "data.bio.kind=none",
            ]
        )


def test_v18_transplant_requires_band_order_confirmed() -> None:
    with pytest.raises(ValueError, match="V18"):
        load_config(overrides=S1_BASE + ["spec.transplant.enabled=true"])
    c = load_config(
        overrides=S1_BASE + ["spec.transplant.enabled=true", "spec.band_order_confirmed=true"]
    )
    assert c.spec.transplant.enabled is True


def test_v19_bmef_stage1_required() -> None:
    with pytest.raises(ValueError, match="V19"):
        load_config(overrides=["model.bmef.stages=[2, 3, 4]"])
    c = load_config(overrides=["model.bmef.stage1_identity=true"])  # ablation 은 플래그로
    assert c.model.bmef.stage1_identity is True


def test_v20_gn8_widths() -> None:
    c = load_config(overrides=["model.norm=gn8"])
    assert c.model.norm == "gn8"
    widths = cfgmod._norm_group_widths(c)
    assert widths["biogate_c_r.stage3"] == 20  # 유일한 비8배수 (정정 A-37 / B-1)
    assert cfgmod._gn_num_groups(20) == 4 and 20 % 4 == 0
    assert cfgmod._gn_num_groups(160) == 8
    with pytest.raises(ValueError, match="V20"):
        load_config(overrides=["model.norm=gn8", "model.decoder.bd_ch=0"])


def test_v21_k_sensor_positive() -> None:
    with pytest.raises(ValueError, match="V21"):
        load_config(
            overrides=["data.modalities=[rgb, msi]", "data.sensor=m3m", "data.msi.k_sensor=0.0"]
        )


def test_v22_aux_loss_stride() -> None:
    with pytest.raises(ValueError, match="V22"):
        load_config(overrides=["loss.aux_loss_stride=2"])
    assert load_config(overrides=["loss.aux_loss_stride=4"]).loss.aux_loss_stride == 4


# ═══════════════════════════════════════════════════════════════════════
#  프리셋 형태의 config 로드 (실제 configs/*.yaml 이 생기기 전 계약 검증)
# ═══════════════════════════════════════════════════════════════════════
PRESETS: Dict[str, Dict[str, Any]] = {
    "s0_rgb_aihub092": {"mode": "s0_rgb", "data": {"dataset": "aihub092", "num_classes": 12}},
    "s0_spec_k235": {
        "mode": "s0_spec",
        "data": {"dataset": "k235", "num_classes": 2},
        "spec": {"loso": True},
    },
    "s1_rgb_ms4": {
        "mode": "s1_rgb_ms4",
        "data": {
            "modalities": ["rgb", "msi", "bio"],
            "sensor": "m3m",
            "num_classes": 2,
            "bio": {"source": "msi", "kind": "mci"},
        },
        "loss": {"ohem_thresh": 0.9},
        "train": {"modality_dropout": {"enabled": True}},
    },
    "s2_full": {
        "mode": "s2_full",
        "data": {
            "modalities": ["rgb", "msi", "bio", "ir", "pol"],
            "sensor": "m3m",
            "num_classes": 2,
            "bio": {"source": "msi", "kind": "mci"},
            "augment": {"geometric": {"rot90": False}},
        },
    },
    "a4_bio_gndvi": {
        "mode": "s1_rgb_ms4",
        "data": {
            "modalities": ["rgb", "msi", "bio"],
            "sensor": "rededge_p",
            "num_classes": 2,
            "bio": {"source": "msi", "kind": "gndvi"},
        },
    },
}


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_load(name: str, kw_tmp: Path) -> None:
    path = write_yaml(kw_tmp / f"{name}.yaml", PRESETS[name])
    c = load_config(path)
    assert isinstance(c, BloomNetConfig)
    assert to_dict(from_dict(to_dict(c))) == to_dict(c)


def test_preset_with_base_file(kw_tmp: Path) -> None:
    base = kw_tmp / "_base.yaml"
    write_yaml(base, to_dict(default_config()))
    path = write_yaml(kw_tmp / "s0.yaml", PRESETS["s0_rgb_aihub092"])
    c = load_config(path, base=base)
    assert c.mode == "s0_rgb"
    assert to_dict(c) == to_dict(load_config(path))  # 기저 파일 == dataclass 기본값


def test_dump_base_cli_output(kw_tmp: Path) -> None:
    out = kw_tmp / "_base.yaml"
    assert cfgmod._main(["--dump-base", "--out", str(out)]) == 0
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert _key_tree(data) == _key_tree(to_dict(default_config()))
    from_dict(data, source=str(out))
