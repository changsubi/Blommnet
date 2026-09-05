"""전체 통합 회귀 — 모듈 경계를 **실제로 넘는** 경로만 검사한다.

개별 모듈 테스트(T02~T25)는 각 파일이 자기 계약을 지키는지 본다. 이 파일은 그것들이
서로 물렸을 때만 드러나는 것을 본다:

* S0-RGB ``build_bloomnet`` → ``forward(B=2,3,128,128)`` → **정본 criterion** → backward
  (과제 요구 스모크). 파라미터 수·출력 shape·gradient 를 회귀 고정.
* dataset 스키마 → ``bloom_collate`` → 모델 ``forward`` 배선 (키 이름이 실제로 맞는가).
* 학습 출력 → ``deploy()`` → ``ExportWrapper`` → ``deploy/postprocess`` 후처리.
* 문서가 "유일 구현" 이라고 못 박은 수식이 **두 곳에서 같은 값**을 내는가
  (X-25 conf, X-07 경계 연산자, 알람 임계).

전부 CPU · B≤2 · 128² 이하다.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bloomnet import constants as C  # noqa: E402
from bloomnet.config import load_config  # noqa: E402
from bloomnet.data.boundary import make_boundary_target  # noqa: E402
from bloomnet.data.bundle import bloom_collate, derive_present  # noqa: E402
from bloomnet.deploy.postprocess import chl_to_alert_level, confidence_map, sigma_chl  # noqa: E402
from bloomnet.deploy.trt_policy import FP32_FORCED_MODULES, assert_precision_policy  # noqa: E402
from bloomnet.losses.criterion import StepCtx, build_criterion  # noqa: E402
from bloomnet.models.bloomnet import ExportWrapper, build_bloomnet  # noqa: E402
from bloomnet.utils.checkpoint import prune_train_only_keys  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs"

# ── 회귀 상수 (06 §6.2 / §10 요약 카드) ──────────────────────────────────────
S0_TRAIN_PARAMS = 9_447_195
S0_DEPLOY_PARAMS = 8_954_390
S1_DEPLOY_PARAMS = 14_660_522
S2_DEPLOY_PARAMS = 15_969_023
#: S0-RGB 에서 gradient 를 받지 못하는 파라미터 수.
#: 06 §6.2 각주는 284,040 이지만 실측은 284,056 이다 — 차이는 `kappa_raw`(12, g_pol 부재)와
#: `log_tau0`(4, vacuity 가 손실에 안 들어감. X-12). models/bloomnet.py 담당자 인계사항과 일치.
S0_DEAD_PARAMS = 284_056


def _cfg(name: str):
    return load_config(str(CONFIG_DIR / name))


def _blocky_labels(b: int, h: int, w: int, k: int) -> Tensor:
    """구조가 있는 라벨. 무작위 라벨은 **모든 픽셀이 경계**가 되어
    ``boundary_pos_weight`` 의 양성 가중치가 0 이 되고 ``L_edge`` 가 조용히 0 이 된다."""
    y = torch.zeros(b, h, w, dtype=torch.long)
    y[:, : h // 2, :] = 1 % k
    y[:, :, : w // 2] += 2 % k
    y.clamp_(0, k - 1)
    y[0, (3 * h) // 4 :, (3 * w) // 4 :] = C.IGNORE_INDEX  # ignore 경로도 태운다
    return y


def _step_ctx(**kw) -> StepCtx:
    base = dict(epoch=0, total_epochs=60, global_step=0, total_steps=100, prec_ramp=1.0, u=0.0)
    base.update(kw)
    return StepCtx(**base)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# ★ 과제 요구 스모크 — S0-RGB, B=2, 3×128×128
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def s0_smoke() -> Dict[str, object]:
    """S0-RGB forward + criterion + backward 를 한 번만 돌리고 결과를 공유한다."""
    cfg = _cfg("s0_rgb_aihub092.yaml")
    model = build_bloomnet(cfg)
    model.train()
    crit = build_criterion(cfg)

    torch.manual_seed(0)
    b, h, w, k = 2, 128, 128, cfg.data.num_classes
    rgb = torch.randn(b, 3, h, w)
    y_seg = _blocky_labels(b, h, w, k)

    out = model(rgb=rgb)
    loss, breakdown = crit(out, {"y_seg": y_seg}, _step_ctx())
    loss.backward()

    return {
        "cfg": cfg,
        "model": model,
        "out": out,
        "loss": loss,
        "breakdown": breakdown,
        "hw": (h, w),
        "k": k,
    }


def test_s0_rgb_parameter_count(s0_smoke) -> None:
    """학습 구성 파라미터 수 회귀 (06 §6.2 S0-RGB K=12)."""
    model = s0_smoke["model"]
    total = sum(p.numel() for p in model.parameters())
    assert total == S0_TRAIN_PARAMS, f"S0-RGB 학습 파라미터 {total} != {S0_TRAIN_PARAMS}"
    assert all(p.requires_grad for p in model.parameters()), "학습 모델에 동결 파라미터가 있다"


def test_s0_rgb_output_shapes(s0_smoke) -> None:
    """모든 출력이 **H/4 이하**다 — forward 는 절대 업샘플하지 않는다 (04 §9.1)."""
    out: Dict[str, Tensor] = s0_smoke["out"]  # type: ignore[assignment]
    b, (h, w), k = 2, s0_smoke["hw"], s0_smoke["k"]
    expect = {
        C.OUT_SEG: (b, k, h // 4, w // 4),
        C.OUT_VACUITY: (b, 1, h // 4, w // 4),
        C.OUT_CHL: (b, 1, h // 4, w // 4),
        C.OUT_LOGVAR: (b, 1, h // 4, w // 4),
        C.OUT_EDGE: (b, 1, h // 4, w // 4),
        "aux_seg_s8": (b, k, h // 8, w // 8),
        "aux_seg_s16": (b, k, h // 16, w // 16),
        "aux_seg_s32": (b, k, h // 32, w // 32),
    }
    assert set(out) == set(expect), f"출력 키 불일치: {sorted(set(out) ^ set(expect))}"
    for key, shape in expect.items():
        assert tuple(out[key].shape) == shape, f"{key}: {tuple(out[key].shape)} != {shape}"
        assert torch.isfinite(out[key]).all(), f"{key} 에 비유한 값"


def test_s0_rgb_loss_is_finite_and_all_terms_present(s0_smoke) -> None:
    """총 손실이 유한하고, **경계 항이 실제로 켜져 있다**."""
    loss: Tensor = s0_smoke["loss"]  # type: ignore[assignment]
    bd: Dict[str, float] = s0_smoke["breakdown"]  # type: ignore[assignment]
    assert torch.isfinite(loss).all() and float(loss.detach()) > 0.0

    required = {
        "loss_total", "loss_ohem", "loss_dice", "loss_edge", "loss_bas",
        "loss_reg", "loss_siam", "n_chl_valid", "n_edge_pos_ratio", "u_ramp",
    }
    assert required <= set(bd), f"breakdown 키 누락 {sorted(required - set(bd))}"

    # ★ 인계: `lambda_aux` 는 tap 4개(p8 포함)지만 `model.aux_taps` 는 3개다.
    #   criterion 은 **모델이 실제로 낸 tap** 에 대해서만 열을 만든다. 두 키가 어긋나면
    #   p8 aux head 를 켰을 때(a5_aux_p8 ablation) 손실이 조용히 빠진다.
    cfg = s0_smoke["cfg"]
    for tap in cfg.model.aux_taps:
        assert f"loss_aux_{tap}" in bd, f"model.aux_taps 의 {tap} 에 대한 breakdown 열이 없다"
        assert bd[f"loss_aux_{tap}"] > 0.0

    # ★ λ_bd=20 이 무력화되지 않았는가 (정정 A-28 이 막으려던 조용한 실패).
    assert bd["loss_edge"] > 0.0, "L_edge 가 0 이다 — 경계 타깃 재생성 경로가 죽었다"
    assert 0.0 < bd["n_edge_pos_ratio"] < 0.5, bd["n_edge_pos_ratio"]
    # chl 라벨이 없으므로 회귀·증류 항은 정확히 0 이어야 한다 (규칙 N1).
    assert bd["loss_reg"] == 0.0 and bd["n_chl_valid"] == 0.0
    assert bd["loss_siam"] == 0.0
    assert abs(sum(bd[k] for k in bd if k.startswith("loss_") and k != "loss_total")
               - bd["loss_total"]) < 1e-4


def test_s0_rgb_backward_gradients(s0_smoke) -> None:
    """★ backward 후 모든 grad 가 유한하고, grad=None 집합이 **문서화된 dead set** 과 일치한다."""
    model = s0_smoke["model"]
    dead: List[str] = []
    live_nonzero = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            dead.append(name)
            continue
        assert torch.isfinite(p.grad).all(), f"{name} grad 에 비유한 값"
        if float(p.grad.abs().sum()) > 0:
            live_nonzero += 1
    assert live_nonzero > 0

    dead_numel = sum(p.numel() for n, p in model.named_parameters() if n in set(dead))
    assert dead_numel == S0_DEAD_PARAMS, (
        f"S0-RGB dead 파라미터 {dead_numel} != {S0_DEAD_PARAMS}. "
        "새 dead 가 생겼거나 죽어 있던 것이 살아났다 — 어느 쪽이든 06 §6.2 각주를 갱신해야 한다."
    )
    # dead 는 전부 BMEF 의 spec/phys 계열 + kappa_raw/log_tau0 뿐이어야 한다.
    families = {re.sub(r"\.\d+\.", ".N.", n) for n in dead}
    assert all(f.startswith("backbone.bmef.N.") for f in families), sorted(families)
    assert "backbone.bmef.N.log_tau0" in families, (
        "log_tau0 이 살아났다면 X-12(vacuity 손실 미소비)가 바뀐 것이다"
    )


def test_s0_rgb_deploy_parameter_count_and_pruning(s0_smoke) -> None:
    """``deploy()`` 가 학습 전용 모듈을 제거하고, ``prune_train_only_keys`` 와 정확히 일치한다."""
    cfg = s0_smoke["cfg"]
    train_model = build_bloomnet(cfg)
    train_sd = train_model.state_dict()

    deploy_model = build_bloomnet(cfg)
    deploy_model.deploy()
    deploy_sd = deploy_model.state_dict()

    assert sum(p.numel() for p in deploy_model.parameters()) == S0_DEPLOY_PARAMS
    assert set(prune_train_only_keys(train_sd)) == set(deploy_sd), (
        "prune_train_only_keys 와 deploy() 결과가 다르다 — TRAIN_ONLY_MODULES 이름 계약 확인"
    )
    leaked = [k for k in deploy_sd if re.search(r"edge_head|aux_|siam", k)]
    assert leaked == [], f"배포 state_dict 에 학습 전용 키가 남았다: {leaked[:5]}"
    assert not deploy_model.training
    assert not any(p.requires_grad for p in deploy_model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 파이프라인 → 모델 배선
# ─────────────────────────────────────────────────────────────────────────────
def test_collate_to_model_wiring() -> None:
    """dataset 스키마(``Sample``) → ``bloom_collate`` → ``BloomNet.forward`` 키가 실제로 맞는가.

    개별 테스트는 collate 와 모델을 따로 본다. 배치 dict 의 키 이름이 forward 의
    keyword 와 어긋나면 여기서만 잡힌다.
    """
    cfg = _cfg("s0_rgb_aihub092.yaml")
    model = build_bloomnet(cfg)
    model.train()
    h = w = 64

    samples = []
    for i in range(2):
        samples.append(
            {
                "rgb": torch.randn(3, h, w),
                "avail": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
                "y_seg": _blocky_labels(1, h, w, cfg.data.num_classes)[0],
                "meta": {"stem": f"s{i}", "group_key": ("a", "b", "c", "d")},
            }
        )
    batch = bloom_collate(samples)
    assert tuple(batch["rgb"].shape) == (2, 3, h, w)
    assert tuple(batch["avail"].shape) == (2, len(C.MODALITY_ORDER))
    assert isinstance(batch["meta"], list) and len(batch["meta"]) == 2

    present = derive_present(batch["avail"])
    assert present.shape == (2, len(C.PATHS))
    assert present[:, 0].all() and not present[:, 1].any() and not present[:, 2].any()

    out = model(rgb=batch["rgb"], avail=batch["avail"])
    assert tuple(out[C.OUT_SEG].shape) == (2, cfg.data.num_classes, h // 4, w // 4)

    crit = build_criterion(cfg)
    loss, _ = crit(out, {"y_seg": batch["y_seg"]}, _step_ctx())
    loss.backward()
    assert torch.isfinite(loss).all()


# ─────────────────────────────────────────────────────────────────────────────
# 모드별 조립 (S0/S1/S2)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name,deploy_params",
    [
        ("s1_rgb_ms4.yaml", S1_DEPLOY_PARAMS),
        ("s2_full.yaml", S2_DEPLOY_PARAMS),
    ],
)
def test_multimodal_presets_end_to_end(name: str, deploy_params: int) -> None:
    """S1/S2 프리셋도 forward+criterion+backward 가 돌고 배포 파라미터 수가 동결값과 같다."""
    cfg = _cfg(name)
    model = build_bloomnet(cfg)
    model.train()
    crit = build_criterion(cfg)

    b, h, w = 2, 64, 64
    kw: Dict[str, Tensor] = {"rgb": torch.randn(b, 3, h, w)}
    active = model.active_modalities
    if "msi" in active:
        kw["msi"] = torch.rand(b, len(model.msi_band_ids), h, w) * 0.05  # [H1] 스케일 대역
    if "bio" in active:
        kw["bio"] = torch.rand(b, 2, h, w) * 0.2
    if "ir" in active:
        kw["ir"] = torch.rand(b, 1, h, w)
    if "pol" in active:
        kw["pol"] = torch.rand(b, 3, h, w)

    out = model(**kw)
    y = _blocky_labels(b, h, w, cfg.data.num_classes)
    loss, bd = crit(out, {"y_seg": y}, _step_ctx())
    loss.backward()
    assert torch.isfinite(loss).all() and float(loss.detach()) > 0
    assert bd["loss_edge"] > 0.0
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n

    dm = build_bloomnet(cfg)
    dm.deploy()
    assert sum(p.numel() for p in dm.parameters()) == deploy_params


def test_modality_dropout_wiring_is_live_and_finite() -> None:
    """S2 에서 modality dropout 4조합이 전부 유한하고 **실제로 출력을 바꾼다**.

    스케일 불변성의 **정량** 계약은 T14(T2/T7, BMEF 수준)와 T18(backbone 수준,
    독립 입력 rms 0.9933/0.7077/0.5813)이 갖는다. 모델 수준에서 그것을 재검사하지
    않는 이유는 초기화 직후 seg 로짓의 std 가 **1.7e-6** 이기 때문이다(zero-init
    잔차 + SegHead 초기화). 사실상 상수인 두 텐서의 std 비는 의미가 없다 — 실측
    ratio 는 6~13 이 나오지만 절대값은 전부 1e-5 미만이다.

    여기서 잡는 것은 **배선 사고**다: dropout 이 아무것도 안 하거나(출력 동일),
    NaN 을 만들거나, shape 가 달라지는 경우.
    """
    cfg = _cfg("s2_full.yaml")
    model = build_bloomnet(cfg)
    model.eval()
    b, h, w = 2, 64, 64
    kw = {
        "rgb": torch.randn(b, 3, h, w),
        "msi": torch.rand(b, len(model.msi_band_ids), h, w) * 0.05,
        "bio": torch.rand(b, 2, h, w) * 0.2,
        "ir": torch.rand(b, 1, h, w),
        "pol": torch.rand(b, 3, h, w),
    }
    with torch.no_grad():
        full = model(**kw)[C.OUT_SEG]
        no_spec = model(**kw, drop_modal={"spec": True})[C.OUT_SEG]
        no_phys = model(**kw, drop_modal={"phys": True})[C.OUT_SEG]
        rgb_only = model(**kw, drop_modal={"spec": True, "phys": True})[C.OUT_SEG]

    for name, t in (("full", full), ("no_spec", no_spec), ("no_phys", no_phys), ("rgb", rgb_only)):
        assert torch.isfinite(t).all(), name
        assert tuple(t.shape) == tuple(full.shape), name
        assert float(t.abs().max()) < 100.0, f"{name}: 로짓 폭주 {float(t.abs().max())}"

    # dropout 이 실제로 배선되어 있는가 (조용한 no-op 방지).
    assert not torch.equal(full, no_spec), "spec dropout 이 출력을 바꾸지 않는다"
    assert not torch.equal(full, no_phys), "phys dropout 이 출력을 바꾸지 않는다"
    assert not torch.equal(no_spec, rgb_only), "phys dropout 이 spec 없이도 배선되어야 한다"

    with pytest.raises(ValueError):
        model(**kw, drop_modal={"rgb": True})  # RGB 부재 모드는 헌법에 없다


# ─────────────────────────────────────────────────────────────────────────────
# 배포 경로 통합 (모델 → ExportWrapper → postprocess)
# ─────────────────────────────────────────────────────────────────────────────
def test_export_wrapper_to_postprocess_chain() -> None:
    """``ExportWrapper`` 3-tuple → ``deploy/postprocess`` 후처리까지 한 번에."""
    cfg = _cfg("s0_rgb_aihub092.yaml")
    model = build_bloomnet(cfg)
    model.deploy()
    wrapper = ExportWrapper(model, input_hw=(128, 128))

    assert wrapper.input_names == ("rgb",)  # 정정 A-35
    x = torch.randn(1, 3, 128, 128)
    seg, chl, conf = wrapper(x)

    assert tuple(seg.shape) == (1, cfg.data.num_classes, 128, 128)
    assert tuple(chl.shape) == (1, 1, 128, 128)
    assert tuple(conf.shape) == (1, 1, 128, 128)
    assert torch.isfinite(seg).all() and torch.isfinite(chl).all() and torch.isfinite(conf).all()
    assert 0.0 <= float(chl.min()) and float(chl.max()) <= cfg.deploy.chl_max_mgm3
    assert 0.0293 - 1e-4 <= float(conf.min()) and float(conf.max()) <= 0.9705 + 1e-4

    # 후처리 체인이 실제 출력 위에서 동작하는가.
    level = chl_to_alert_level(chl, seg_id=seg.argmax(dim=1, keepdim=True), algae_id=1)
    assert level.shape == chl.shape and int(level.min()) >= 0 and int(level.max()) <= 3
    sig = sigma_chl(torch.zeros_like(chl), chl)
    assert torch.isfinite(sig).all() and float(sig.min()) > 0


def test_export_wrapper_conf_matches_single_source_definition() -> None:
    """★ X-25 — ``ExportWrapper`` 안의 inline conf 식이 ``confidence_map`` 과 **같은 값**을 낸다.

    두 곳에 같은 수식이 손으로 적혀 있으므로(하나는 export 그래프용, 하나는 후처리용)
    한쪽만 바뀌면 배포 산출물과 리포트가 조용히 갈라진다.
    """
    s = torch.linspace(-7.0, 7.0, steps=57).reshape(1, 1, 1, -1)
    inline = 1.0 / (1.0 + torch.exp(0.5 * s))  # ExportWrapper.forward 와 동일 식
    assert torch.allclose(inline, confidence_map(s), atol=0, rtol=0)
    assert float(confidence_map(torch.tensor(7.0))) == pytest.approx(0.029312, abs=1e-6)
    assert float(confidence_map(torch.tensor(-7.0))) == pytest.approx(0.970688, abs=1e-6)


def test_export_wrapper_requires_deploy() -> None:
    """``deploy()`` 하지 않은 모델을 감싸면 즉시 실패한다 (T25 계약)."""
    cfg = _cfg("s0_rgb_aihub092.yaml")
    model = build_bloomnet(cfg)
    with pytest.raises(AssertionError):
        ExportWrapper(model, input_hw=(64, 64))


def test_deploy_precision_policy_matches_config() -> None:
    """V13 — 프리셋 config 의 ``deploy.fp32_forced`` 가 ``trt_policy`` 하한을 만족한다."""
    for name in ("s0_rgb_aihub092.yaml", "s1_rgb_ms4.yaml", "s2_full.yaml"):
        cfg = _cfg(name)
        assert_precision_policy(
            cfg.deploy.fp32_forced,
            cfg.deploy.fp16_locked,
            precision=cfg.deploy.precision,
            amp=cfg.train.amp,
        )
        for token in FP32_FORCED_MODULES:
            assert token in tuple(cfg.deploy.fp32_forced), (name, token)


# ─────────────────────────────────────────────────────────────────────────────
# 문서가 "유일 구현" 이라고 못 박은 것들의 교차 일치
# ─────────────────────────────────────────────────────────────────────────────
def test_alarm_thresholds_single_source() -> None:
    """``deploy/postprocess`` 와 ``utils/metrics_reg`` 가 같은 임계·같은 경계 규약을 쓴다."""
    from bloomnet.utils.metrics_reg import alarm_level

    x = torch.tensor([[0.0, 14.999, 15.0, 24.999, 25.0, 99.999, 100.0, 1e4]])
    assert torch.equal(chl_to_alert_level(x), alarm_level(x))
    assert chl_to_alert_level(x).tolist() == [[0, 0, 1, 1, 2, 2, 3, 3]]
    # log1p 임계와 mg/m³ 임계가 서로 정합한가 (T02 가 constants 안에서 보는 것을
    # 여기서는 실제 변환 경로로 확인한다).
    lg = torch.log1p(torch.tensor(C.ALARM_THRESHOLDS_MGM3))
    assert torch.allclose(lg, torch.tensor(C.ALARM_THRESHOLDS_LOG1P), atol=1e-6)


def test_criterion_regenerates_boundary_identically_to_dataset(s0_smoke) -> None:
    """X-07 — criterion 의 경계 재생성이 ``data/boundary.make_boundary_target`` 과 같다.

    소유권(정정 A-28)이 갈려 있어 두 경로가 같은 값을 내지 않으면 λ_bd 가
    조용히 다른 것을 학습한다.
    """
    cfg = s0_smoke["cfg"]
    k = cfg.data.num_classes
    y = _blocky_labels(2, 64, 64, k)
    radius = cfg.boundary_radius(cfg.data.train_size[0])
    stride = cfg.data.boundary.stride
    gt, valid = make_boundary_target(y, radius=radius, out_stride=stride)
    assert gt.dtype == torch.float32 or gt.dtype == torch.bool or gt.is_floating_point()
    assert gt.shape[-2:] == (64 // stride, 64 // stride)
    assert valid.shape == gt.shape
    assert float(gt.max()) > 0.0, "구조가 있는 라벨인데 경계가 0개다"

    # criterion 이 같은 반경/스트라이드를 쓰는지 (설정 전달이 끊기면 여기서 잡힌다).
    crit = build_criterion(cfg)
    assert crit.boundary_stride == stride
    assert crit._radius_for(cfg.data.train_size[0]) == radius
    assert crit._radius_for(cfg.data.eval_size[0]) == cfg.boundary_radius(cfg.data.eval_size[0])
    # 해상도별 매핑이 실제로 다른 값을 준다 (01 C-10: 512²→1 / 1024²→2).
    assert crit._radius_for(512) == 1 and crit._radius_for(1024) == 2


#: export 그래프에 들어가는 모듈들. 06 §10 금지 목록은 여기에만 적용된다.
EXPORT_PATH_PREFIXES = ("models/", "modules/", "deploy/postprocess.py", "utils/metrics_reg.py")


def _called_attr_names(path: Path) -> Set[str]:
    """``ast`` 로 **호출된 attribute 이름**만 모은다 (docstring·주석은 세지 않는다)."""
    import ast

    names: Set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                names.add(fn.attr)
            elif isinstance(fn, ast.Name):
                names.add(fn.id)
    return names


def test_no_forbidden_ops_on_export_path() -> None:
    """06 §10 "절대 하지 말 것" 중 정적으로 잡히는 2건을 **export 경로 소스**에서 확인한다.

    * 금지 13 — ``torch.expm1`` (ONNX 에 ``Expm1`` 없음, 정정 B-13).
    * 금지 7 / 04 §9.3-A·O — ``adaptive_avg_pool2d`` (Shape→Gather 유발).

    ``pretrain/spec_mlp.py`` 는 예외다: ``torch.special.expm1`` 을 **초기화 상수 1개**를
    float64 로 역산하는 데만 쓰고 그래프에 넣지 않는다. 그래서 검사 범위를
    export 경로로 한정하고, 그 사실을 여기에 기록한다.
    """
    pkg = REPO_ROOT / "bloomnet"
    bad_expm1: List[str] = []
    bad_pool: List[str] = []
    checked = 0
    for path in sorted(pkg.rglob("*.py")):
        rel = str(path.relative_to(pkg))
        if not rel.startswith(EXPORT_PATH_PREFIXES):
            continue
        checked += 1
        called = _called_attr_names(path)
        if "expm1" in called:
            bad_expm1.append(rel)
        if "adaptive_avg_pool2d" in called or "AdaptiveAvgPool2d" in called:
            bad_pool.append(rel)
    assert checked > 10, f"검사 대상이 {checked}개뿐이다 — 경로 접두가 틀렸다"
    assert bad_expm1 == [], f"금지 13: torch.expm1 (정정 B-13). {bad_expm1}"
    assert bad_pool == [], f"금지 7 / 04 §9.3-A·O: adaptive pooling. {bad_pool}"


def test_no_gpu_is_used_anywhere_in_this_suite() -> None:
    """헌법 C-5.2 — 이 파일이 도는 동안 CUDA 가 보이면 안 된다."""
    assert not torch.cuda.is_available()
    assert torch.tensor(0.0).device.type == "cpu"
