"""``bloomnet/deploy/*`` 와 새로 채운 ``bloomnet/tools/*`` 의 계약 테스트.

T25(export 계약)의 **onnx 무관 부분**과 데이터 준비 도구 4종을 다룬다.
onnx 계열이 필요한 항목만 ``skipif`` 로 격리하며(06 §5.0 규칙 1),
미설치 사실은 ``conftest.pytest_terminal_summary`` 가 명시 출력한다(규칙 2).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bloomnet import constants as C  # noqa: E402
from bloomnet.deploy import export_onnx as eo  # noqa: E402
from bloomnet.deploy.postprocess import (  # noqa: E402
    CONF_MAX,
    CONF_MIN,
    chl_to_alert_level,
    confidence_map,
    sigma_chl,
)
from bloomnet.deploy.trt_policy import (  # noqa: E402
    FP16_LOCKED_MODULES,
    FP32_FORCED_MODULES,
    RISK_TABLE,
    assert_precision_policy,
    matches_module_pattern,
)

HAS_ONNX = importlib.util.find_spec("onnx") is not None


# ─────────────────────────────────────────────────────────────────────────────
# deploy/postprocess.py
# ─────────────────────────────────────────────────────────────────────────────
def test_alert_level_boundaries_are_inclusive() -> None:
    """임계값 **이상**이 상위 단계다. ``right=False`` 를 쓰면 x==15 를 놓친다."""
    x = torch.tensor([0.0, 14.999, 15.0, 24.999, 25.0, 99.999, 100.0, 1e9])
    assert chl_to_alert_level(x).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert chl_to_alert_level(x).dtype == torch.int64


def test_alert_level_seg_masking() -> None:
    """녹조 클래스가 아닌 픽셀은 경보가 0 으로 마스킹된다."""
    chl = torch.full((1, 1, 2, 2), 200.0)
    seg = torch.tensor([[[[1, 0], [1, 2]]]])
    lv = chl_to_alert_level(chl, seg_id=seg, algae_id=1)
    assert lv.tolist() == [[[[3, 0], [3, 0]]]]
    # seg_id 없으면 전부 3.
    assert int(chl_to_alert_level(chl).min()) == 3


def test_alert_level_accepts_squeezed_seg() -> None:
    chl = torch.full((2, 1, 4, 4), 30.0)
    seg = torch.ones(2, 4, 4, dtype=torch.long)  # 채널 축 없음
    assert int(chl_to_alert_level(chl, seg_id=seg).max()) == 2
    with pytest.raises(ValueError):
        chl_to_alert_level(chl, seg_id=torch.ones(2, 1, 3, 3, dtype=torch.long))


def test_confidence_map_is_bounded_and_monotone() -> None:
    """★X-25 — ``conf = 1/(1+exp(0.5 s))`` 는 ``[0.0293, 0.9705]`` 안이고 단조 감소."""
    s = torch.linspace(-7.0, 7.0, 141)
    conf = confidence_map(s)
    assert torch.all(conf[1:] < conf[:-1]), "s 에 대해 단조 감소가 아니다"
    assert float(conf.min()) >= CONF_MIN - 1e-6
    assert float(conf.max()) <= CONF_MAX + 1e-6
    assert float(confidence_map(torch.tensor(0.0))) == pytest.approx(0.5)
    # 폐기된 05 §2.5.4 정의(exp(-0.5 s))는 s=-7 에서 33.1 → 유계 계약 위반.
    assert float(torch.exp(-0.5 * torch.tensor(-7.0))) > 1.0


def test_confidence_map_upcasts_fp16() -> None:
    conf = confidence_map(torch.tensor([-7.0, 7.0], dtype=torch.float16))
    assert conf.dtype == torch.float32 and torch.isfinite(conf).all()


def test_sigma_chl_delta_method() -> None:
    """``σ_chl = σ_u·(1+chl)``. s=0 → σ_u=1 → σ_chl = 1+chl."""
    chl = torch.tensor([0.0, 4.0, 99.0])
    assert torch.allclose(sigma_chl(torch.zeros(3), chl), torch.tensor([1.0, 5.0, 100.0]))
    s = torch.full((3,), -7.0)
    assert torch.allclose(sigma_chl(s, chl), torch.exp(torch.tensor(-3.5)) * (1.0 + chl))


# ─────────────────────────────────────────────────────────────────────────────
# deploy/trt_policy.py
# ─────────────────────────────────────────────────────────────────────────────
def test_fp32_forced_contains_both_required_modules() -> None:
    """정정 A-22 / B-15 — ``bmef.fusion`` 과 ``litemla.attention`` **둘 다**."""
    assert "bmef.fusion" in FP32_FORCED_MODULES
    assert any("litemla.attention" in m for m in FP32_FORCED_MODULES)
    assert any("litemla.attention" in m for m in FP16_LOCKED_MODULES)


def test_risk_table_covers_a_to_o() -> None:
    ids = [r["id"] for r in RISK_TABLE]
    assert ids == list("ABCDEFGHIJKLMNO"), ids
    for row in RISK_TABLE:
        assert set(row) == {"id", "op", "why", "fix", "where"}
        assert all(row[k] for k in row)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("encoder.rgb.stage3.0.litemla.attention", True),
        ("model.encoder.rgb.stage4.2.litemla.attention", True),
        ("backbone.bmef.0.fusion", True),  # 중간의 stage 인덱스를 건너뛴다
        ("/backbone/bmef/2/fusion/Exp", True),  # ONNX/TRT layer 이름 형태
        ("decoder.pagfm_4", False),
        ("seg_head.cls", False),
        ("fusion.bmef.0", False),  # 순서가 뒤바뀌면 매칭되지 않는다
    ],
)
def test_matches_module_pattern(name: str, expected: bool) -> None:
    assert matches_module_pattern(name, FP32_FORCED_MODULES) is expected


def test_fp16_locked_tokens_match_real_module_paths() -> None:
    """``chl_head``/``unc_head``/``pagfm`` 는 실제 모듈 경로에 걸린다."""
    assert matches_module_pattern("chl_head.out", FP16_LOCKED_MODULES)
    assert matches_module_pattern("unc_head.out", FP16_LOCKED_MODULES)
    assert matches_module_pattern("decoder.pagfm_4.f_x", FP16_LOCKED_MODULES)
    assert not matches_module_pattern("seg_head.cls", FP16_LOCKED_MODULES)


def test_module_token_notes_cover_every_policy_token() -> None:
    """★ 정책 토큰이 PyTorch 모듈 경로가 아닌 경우가 있으므로 소재를 명문화한다."""
    from bloomnet.deploy.trt_policy import MODULE_TOKEN_NOTES

    for token in set(FP32_FORCED_MODULES) | set(FP16_LOCKED_MODULES):
        assert token in MODULE_TOKEN_NOTES, f"{token} 의 소재가 기록되어 있지 않다"


def test_precision_policy_v13() -> None:
    ok = list(FP32_FORCED_MODULES)
    assert_precision_policy(ok, list(FP16_LOCKED_MODULES), precision="fp32")  # 무관
    assert_precision_policy(ok, list(FP16_LOCKED_MODULES), precision="fp16")
    with pytest.raises(ValueError, match="V13"):
        assert_precision_policy(["bmef.fusion"], [], precision="fp16")
    with pytest.raises(ValueError, match="V13"):
        assert_precision_policy([], [], precision="fp32", amp="fp16")
    with pytest.raises(ValueError, match="int8"):
        assert_precision_policy(ok, ["chl_head"], precision="int8")


# ─────────────────────────────────────────────────────────────────────────────
# deploy/export_onnx.py
# ─────────────────────────────────────────────────────────────────────────────
def test_export_onnx_imports_without_onnx() -> None:
    """모듈 import 자체는 onnx 없이 성공해야 한다 (지연 import)."""
    assert eo.SHAPE_DEPENDENT_OPS == ("Shape", "Gather")
    assert "onnx" in eo.DEPLOY_PACKAGES


@pytest.mark.skipif(HAS_ONNX, reason="onnx 가 설치되어 있으면 이 negative 는 의미가 없다")
def test_export_onnx_fails_loudly_when_missing() -> None:
    """★ 조용한 skip 금지 — 미설치 시 무엇을 설치해야 하는지 적힌 예외를 낸다."""
    with pytest.raises(ModuleNotFoundError) as ei:
        eo.assert_no_shape_nodes(Path("nonexistent.onnx"))
    msg = str(ei.value)
    assert "requirements-deploy.txt" in msg and "04 §9.4" in msg


@pytest.mark.slow
@pytest.mark.skipif(not HAS_ONNX, reason="requirements-deploy.txt 미설치")
def test_export_and_verify_roundtrip(kw_tmp: Path) -> None:  # pragma: no cover - onnx 미설치
    from bloomnet.config import load_config
    from bloomnet.models.bloomnet import build_bloomnet

    cfg = load_config(str(REPO_ROOT / "configs" / "s0_rgb_aihub092.yaml"))
    model = build_bloomnet(cfg)
    model.deploy()
    out = eo.export(model, kw_tmp / "m.onnx", input_hw=(64, 64), opset=17)
    eo.assert_no_shape_nodes(out)  # 04 §9.4 step 3
    stats = eo.verify_onnx(out, model)
    assert stats["max_abs_diff_seg"] < 1e-2


# ─────────────────────────────────────────────────────────────────────────────
# tools/ — 합성 트리로 검증
# ─────────────────────────────────────────────────────────────────────────────
def _make_fake_aihub092_tree(root: Path, n_per_scene: int = 3) -> None:
    from PIL import Image

    for split in ("train", "val"):
        for scene in C.SCENES[:2]:
            (root / "images" / split / scene).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split / scene).mkdir(parents=True, exist_ok=True)
            for i in range(n_per_scene):
                stem = f"L01_11000_100_20240101_N01_{i:03d}"
                Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(
                    root / "images" / split / scene / f"{stem}.png"
                )
                lab = np.zeros((16, 16), np.uint8)
                lab[:8] = 1
                lab[8:12, :4] = 10
                lab[12:, 12:] = C.IGNORE_INDEX
                Image.fromarray(lab, "L").save(
                    root / "labels" / split / scene / f"{stem}_labelids.png"
                )


def test_link_aihub092_creates_links_and_manifest(kw_tmp: Path) -> None:
    from bloomnet.tools.link_aihub092 import link_aihub092

    src = kw_tmp / "src"
    _make_fake_aihub092_tree(src)
    out = kw_tmp / "asis"
    manifest_path = link_aihub092(src, out, kinds=("images", "labels"))

    assert (out / "images").is_symlink() and (out / "labels").is_symlink()
    assert (out / "images").resolve() == (src / "images").resolve()
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert man["source_root"] == str(src.resolve())
    assert man["n_files_total"]["images"] == 12  # 2 split × 2 scene × 3
    assert man["kinds"] == ["images", "labels"]

    # 덮어쓰기 금지 (01 §7.1 규칙 1).
    with pytest.raises(FileExistsError):
        link_aihub092(src, out)


def test_link_aihub092_rolls_back_on_failure(kw_tmp: Path) -> None:
    """실패하면 자기가 만든 것만 되돌린다 (규칙 2) — 부분 트리를 남기지 않는다."""
    from bloomnet.tools.link_aihub092 import link_aihub092

    src = kw_tmp / "src"
    (src / "images").mkdir(parents=True)  # labels 없음 → 필수 kind 부재
    out = kw_tmp / "asis"
    with pytest.raises(FileNotFoundError):
        link_aihub092(src, out)
    assert not out.exists()


def test_compute_class_stats_matches_sampler_contract(kw_tmp: Path) -> None:
    from bloomnet.data.samplers import RepeatFactorSampler
    from bloomnet.tools.compute_class_stats import compute_class_stats

    root = kw_tmp / "tree"
    _make_fake_aihub092_tree(root)
    out = compute_class_stats(root, "train", kw_tmp / "stats.json")
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["n_images"] == 6
    assert len(data["per_class"]) == 12
    by_id = {r["class_id"]: r for r in data["per_class"]}
    assert by_id[1]["image_rate"] == 1.0 and by_id[10]["image_rate"] == 1.0
    assert by_id[2]["n_images"] == 0
    # ignore 픽셀은 라벨 픽셀 총계에서 빠진다.
    assert data["n_pixels_ignore"] == 6 * 16  # 4행 × 4열 = 16 px/장
    assert data["n_pixels_labeled"] == 6 * (256 - 16)

    presence = np.asarray(data["class_presence"], dtype=bool)
    assert presence.shape == (6, 12)
    assert np.array_equal(presence, np.load(out.with_suffix(".npy")))
    # 실제 소비자가 받아들이는가.
    sampler = RepeatFactorSampler(presence, t=0.05, seed=0)
    assert len(list(iter(sampler))) > 0

    # data/build.py 의 로더도 같은 파일을 읽는다 (포맷 계약 단일화).
    from bloomnet.data.build import load_class_presence

    assert load_class_presence(str(out), 6).shape == (6, 12)
    assert load_class_presence(str(out.with_suffix(".npy")), 6).shape == (6, 12)


def test_colorize_labels_does_not_touch_originals(kw_tmp: Path) -> None:
    from PIL import Image

    from bloomnet.tools.colorize_labels import IGNORE_COLOR, PALETTE, colorize_labels

    root = kw_tmp / "tree"
    _make_fake_aihub092_tree(root)
    before = {p: p.read_bytes() for p in sorted((root / "labels").rglob("*.png"))}

    n = colorize_labels(root, workers=1)
    assert n == 12  # train + val
    after = {p: p.read_bytes() for p in sorted((root / "labels").rglob("*.png"))}
    assert before == after, "★ 원본 라벨이 변경됐다 — id 가 깨져 학습이 망가진다"

    color = sorted((root / "labels_color").rglob("*_color.png"))
    assert len(color) == 12
    arr = np.array(Image.open(color[0]).convert("RGB"))
    assert tuple(arr[0, 0]) == tuple(PALETTE[1])  # 상단 = class 1
    assert tuple(arr[15, 15]) == IGNORE_COLOR  # ★ 255 가 class 11 로 둔갑하지 않는다
    assert tuple(arr[15, 15]) != tuple(PALETTE[11])


def test_calibrate_k_sensor_auto_water_mask() -> None:
    """자동 수면 마스크가 NIR 저반사 픽셀을 고른다 (실데이터 없이 검증 가능한 부분)."""
    from bloomnet.tools.calibrate_k_sensor import auto_water_mask

    rho = np.ones((4, 8, 8), dtype=np.float32) * 0.2
    rho[3, :4, :] = 0.01  # 상단 절반 = 물 (NIR 흡수)
    mask = auto_water_mask(rho, nir_index=3, pctl=50.0)
    assert mask[:4].all() and not mask[4:].any()

    rho_bad = np.zeros((4, 4, 4), dtype=np.float32)
    assert not auto_water_mask(rho_bad).any()  # 전부 0 → 물 픽셀 없음


def test_calibrate_k_sensor_reports_current_constant_conflict() -> None:
    """★ M-13 — 잠정 상수 ``K_SENSOR['m3m']=3.5e3`` 은 [H1] 상한을 깬다.

    실데이터 없이도 산술만으로 확인할 수 있다: 전수 실측 ``rho_rel`` median 0.0122 에
    3.5e3 을 곱하면 43 이 되어 ``median < 1.0`` 을 위반한다.
    """
    rho_rel_median = 0.0122  # data/drone_m3m.py 담당자 XMP 전수 실측
    lo, hi = C.MSI_MEDIAN_RANGE
    assert not (lo < rho_rel_median * C.K_SENSOR["m3m"] < hi), (
        "K_SENSOR['m3m'] 가 갱신됐다면 이 회귀도 함께 갱신하라 (tools/calibrate_k_sensor.py)"
    )
    assert lo < rho_rel_median * 1.0 < hi, "k≈1 이면 [H1] 을 만족한다"
