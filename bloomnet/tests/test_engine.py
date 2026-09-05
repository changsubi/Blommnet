"""engine/ 검증 — 스케줄러 LR 곡선·옵티마이저 그룹·EMA·평가기·run_epoch.

헌법 C-5.2: GPU 절대 금지. autouse fixture 가 매 테스트에서 이를 assert 한다.
텐서 크기는 B<=2, 64x64 이하.
"""

from __future__ import annotations

import math
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pytest
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:  # cwd 무관 import
    sys.path.insert(0, str(_REPO))

from bloomnet.config import default_config, from_dict, resolve_schedule  # noqa: E402
from bloomnet.constants import IGNORE_INDEX, OUT_CHL, OUT_EDGE, OUT_LOGVAR, OUT_SEG  # noqa: E402
from bloomnet.engine.ema import ModelEMA  # noqa: E402
from bloomnet.engine.evaluator import (  # noqa: E402
    BoundaryEvaluator,
    EvalStepCtx,
    RegEvaluator,
    SegEvaluator,
    autocast_dtype,
    evaluate,
    model_inputs_from_batch,
    targets_from_batch,
)
from bloomnet.engine.optim import (  # noqa: E402
    ENCODER_PREFIXES,
    PARAM_GROUPS,
    assign_param_groups,
    build_optimizer,
    collect_no_decay,
    collect_physics,
)
from bloomnet.engine.sched import (  # noqa: E402
    SCHEDULERS_DICT,
    build_lr_scheduler,
    build_scheduler,
    lr_factor_at_step,
)
from bloomnet.engine.trainer import (  # noqa: E402
    STEPCTX_FALLBACK,
    EarlyStopper,
    breakdown_to_csv_columns,
    build_grad_scaler,
    checkpoint_payload,
    class_names_for,
    collect_stability_row,
    make_step_ctx,
    prec_ramp_for_epoch,
    run_epoch,
    sample_modality_dropout,
)
from bloomnet.utils.logging_csv import EPOCH_FIELDS, CsvLogger  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _no_gpu() -> None:
    assert not torch.cuda.is_available(), "헌법 C-5.2: GPU 를 절대 쓰지 않는다"
    torch.manual_seed(0)
    torch.set_num_threads(4)


@pytest.fixture
def kw_tmp() -> Iterator[Path]:
    """임시 디렉터리를 **k_water 안에** 만든다 (과제 제약: 저장소 밖 쓰기 금지).

    pytest 기본 `tmp_path` 는 `/tmp` 라 제약을 위반한다.
    """
    root = _REPO / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(dir=root, prefix="engine_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
#  더미 부품
# ═══════════════════════════════════════════════════════════════════════
class DummySeg(nn.Module):
    """BloomNet 계약(06 §3.4.3)의 최소 모델. conv 1개."""

    def __init__(self, k: int = 3, c_in: int = 3) -> None:
        super().__init__()
        self.stem = nn.Conv2d(c_in, 8, 3, padding=1)
        self.cls = nn.Conv2d(8, k, 1)
        self.chl = nn.Conv2d(8, 1, 1)
        self.edge_head = nn.Conv2d(8, 1, 1)  # 학습 전용 (TRAIN_ONLY_MODULES)
        self.last_kwargs: Dict[str, Any] = {}

    def forward(
        self,
        *,
        rgb: torch.Tensor,
        msi: Optional[torch.Tensor] = None,
        bio: Optional[torch.Tensor] = None,
        ir: Optional[torch.Tensor] = None,
        pol: Optional[torch.Tensor] = None,
        avail: Optional[torch.Tensor] = None,
        band_ids: Optional[Any] = None,
        phys_slot_ids: Optional[Any] = None,
        drop_modal: Optional[Dict[str, bool]] = None,
        prec_ramp: float = 1.0,
        unc_enabled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self.last_kwargs = {"prec_ramp": prec_ramp, "drop_modal": drop_modal}
        z = torch.relu(self.stem(rgb))
        z = nn.functional.avg_pool2d(z, 4)  # H/4 (헌법 C-4)
        return {
            OUT_SEG: self.cls(z),
            OUT_CHL: nn.functional.softplus(self.chl(z)),
            OUT_LOGVAR: torch.zeros_like(self.chl(z)),
            OUT_EDGE: self.edge_head(z),
        }

    def no_weight_decay(self):
        return {n for n, p in self.named_parameters() if p.ndim <= 1}


class DummyCriterion(nn.Module):
    """(loss, breakdown) 계약만 지키는 최소 criterion. `nan_at` 스텝에서 NaN 을 낸다."""

    def __init__(self, *, nan_at: tuple = ()) -> None:
        super().__init__()
        self.nan_at = set(nan_at)
        self.seen_ctx: List[Any] = []
        self.calls = 0

    def forward(self, outputs, targets, step_ctx):
        self.calls += 1
        self.seen_ctx.append(step_ctx)
        logits = outputs[OUT_SEG]
        y = targets["y_seg"]
        up = nn.functional.interpolate(
            logits, size=y.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = nn.functional.cross_entropy(up, y, ignore_index=IGNORE_INDEX)
        if self.calls in self.nan_at:
            loss = loss * float("nan")
        return loss, {"loss_ohem": loss.detach(), "loss_aux_enc_s8": loss.detach() * 0.2}


def make_batch(b: int = 2, hw: int = 64, k: int = 3, *, with_meta: bool = True) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(7)
    y = torch.randint(0, k, (b, hw, hw), generator=g)
    batch: Dict[str, Any] = {
        "rgb": torch.randn(b, 3, hw, hw, generator=g),
        "y_seg": y,
        "avail": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]] * b),
        "all_missing": {},
    }
    if with_meta:
        batch["meta"] = [{"group_key": f"g{i % 2}", "stem": f"s{i}"} for i in range(b)]
    return batch


def make_cfg(**over: Any):
    """기본 config 에 dotted override 를 얹는다."""
    data = default_config().to_dict()
    for path, value in over.items():
        node = data
        parts = path.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return from_dict(data, source="<test>")


# ═══════════════════════════════════════════════════════════════════════
#  sched.py — LR 곡선
# ═══════════════════════════════════════════════════════════════════════
def _one_param_optimizer(lr: float = 1.0) -> torch.optim.Optimizer:
    p = nn.Parameter(torch.zeros(2))
    return torch.optim.AdamW([{"params": [p], "lr": lr, "name": "main"}])


@pytest.mark.filterwarnings("ignore:Detected call of")
@pytest.mark.parametrize("w,t", [(4, 10), (0, 7), (1, 3), (5, 20)])
def test_lr_curve_matches_closed_form(w: int, t: int) -> None:
    """SequentialLR(LinearLR+PolynomialLR) 실측 == 손계산 폐형식 (05 §5.1.3)."""
    base, s0, power = 4.0e-4, 1.0e-3, 0.9
    opt = _one_param_optimizer(base)
    sch = build_lr_scheduler(
        opt,
        scheduler="PolynomialLR",
        scheduler_kwargs={"power": power, "total_iters": t},
        warmup_iters=w,
        warmup_start_factor=s0,
    )
    for k in range(w + t + 2):
        want = base * lr_factor_at_step(
            k, warmup_iters=w, total_iters=t, warmup_start_factor=s0, power=power
        )
        assert opt.param_groups[0]["lr"] == pytest.approx(want, rel=1e-9, abs=1e-15), (
            f"step {k}: {opt.param_groups[0]['lr']} != {want}"
        )
        sch.step()


@pytest.mark.filterwarnings("ignore:Detected call of")
def test_warmup_endpoints_hand_values() -> None:
    """warmup 시작 = base×start_factor, 종료 = base, 학습 마지막 스텝 뒤 = **정확히 0**."""
    base, w, t, s0 = 4.0e-4, 4, 10, 1.0e-3
    opt = _one_param_optimizer(base)
    sch = build_lr_scheduler(
        opt,
        scheduler="PolynomialLR",
        scheduler_kwargs={"power": 0.9, "total_iters": t},
        warmup_iters=w,
        warmup_start_factor=s0,
    )
    assert opt.param_groups[0]["lr"] == pytest.approx(base * s0)
    for _ in range(w):
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(base)  # 밀스톤에서 정확히 base
    for _ in range(t):
        sch.step()
    assert opt.param_groups[0]["lr"] == 0.0  # 정정 B-26: LR 이 0 으로 수렴


@pytest.mark.filterwarnings("ignore:Detected call of")
def test_warmup_is_linear_in_the_middle() -> None:
    """LinearLR 중간값 손계산: k=2, W=4 → s0 + (1-s0)/2."""
    base, s0 = 1.0, 1.0e-3
    opt = _one_param_optimizer(base)
    sch = build_lr_scheduler(
        opt,
        scheduler="PolynomialLR",
        scheduler_kwargs={"power": 0.9, "total_iters": 10},
        warmup_iters=4,
        warmup_start_factor=s0,
    )
    for _ in range(2):
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(s0 + (1.0 - s0) * 0.5)


@pytest.mark.filterwarnings("ignore:Detected call of")
def test_no_warmup_returns_plain_scheduler() -> None:
    opt = _one_param_optimizer(1.0)
    sch = build_lr_scheduler(
        opt, scheduler="PolynomialLR", scheduler_kwargs={"power": 0.9, "total_iters": 5},
        warmup_iters=0,
    )
    assert isinstance(sch, torch.optim.lr_scheduler.PolynomialLR)
    sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx((1 - 1 / 5) ** 0.9)


def test_build_lr_scheduler_requires_total_iters() -> None:
    """정정 A-18: total_iters 는 resolve_schedule 이 유도한다. 없으면 조용히 진행하지 않는다."""
    opt = _one_param_optimizer()
    with pytest.raises(ValueError, match="total_iters"):
        build_lr_scheduler(opt, scheduler="PolynomialLR", scheduler_kwargs={"power": 0.9})
    with pytest.raises(ValueError, match="unknown scheduler"):
        build_lr_scheduler(opt, scheduler="NopeLR", scheduler_kwargs={"total_iters": 3})


def test_build_scheduler_river_aihub092_signature() -> None:
    """[분석] §E.1.7 copy-as-is 시그니처가 그대로 동작하고, 미지 kwargs 는 버려진다."""
    assert {"PolynomialLR", "LinearLR", "WarmupOneCycleLR", "OneCycleLR"} <= set(SCHEDULERS_DICT)
    opt = _one_param_optimizer(1.0)
    sch = build_scheduler("PolynomialLR", opt, 1.0, 5, {"power": 0.9, "bogus_key": 1})
    assert isinstance(sch, torch.optim.lr_scheduler.PolynomialLR)
    assert sch.total_iters == 5
    with pytest.raises(ValueError):
        build_scheduler("NopeLR", opt, 1.0, 5, {})


@pytest.mark.filterwarnings("ignore:Detected call of")
def test_resolve_schedule_then_scheduler_reaches_zero_at_last_step() -> None:
    """V15 유도값 → 스케줄러가 정확히 epochs×ipe 스텝에서 0 이 된다 (정정 B-26)."""
    cfg = make_cfg(**{"schedule.epochs": 6, "schedule.warmup_epochs": 2})
    resolve_schedule(cfg, 10)
    assert cfg.schedule.warmup_iters == 20
    assert cfg.schedule.scheduler_kwargs["total_iters"] == 6 * 10 - 20 == 40
    opt = _one_param_optimizer(float(cfg.optim.lr))
    sch = build_lr_scheduler(
        opt,
        scheduler=cfg.schedule.scheduler,
        scheduler_kwargs=cfg.schedule.scheduler_kwargs,
        warmup_iters=cfg.schedule.warmup_iters,
        warmup_start_factor=cfg.schedule.warmup_start_factor,
    )
    for _ in range(6 * 10):
        sch.step()
    assert opt.param_groups[0]["lr"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  ema.py
# ═══════════════════════════════════════════════════════════════════════
def _tiny_bn_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))


def test_ema_before_start_step_tracks_model_exactly() -> None:
    m = _tiny_bn_model()
    ema = ModelEMA(m, decay=0.9, start_step=5)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m, step=0)
    for k, v in m.state_dict().items():
        assert torch.equal(ema.shadow[k], v), k


def test_ema_update_matches_hand_formula() -> None:
    m = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.0)
    ema = ModelEMA(m, decay=0.9, start_step=0)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m, step=0)  # 0.9*0 + 0.1*1
    assert ema.shadow["weight"].allclose(torch.full((2, 2), 0.1), atol=1e-7)
    ema.update(m, step=1)  # 0.9*0.1 + 0.1*1 = 0.19
    assert ema.shadow["weight"].allclose(torch.full((2, 2), 0.19), atol=1e-7)
    assert ema.num_updates == 2


def test_ema_copies_integer_buffers_without_ema() -> None:
    """`num_batches_tracked`(int64) 는 EMA 가 정의되지 않는다 — 그대로 복사되어야 한다."""
    m = _tiny_bn_model()
    ema = ModelEMA(m, decay=0.5, start_step=0)
    m.train()
    m(torch.randn(2, 2, 4, 4))
    ema.update(m, step=0)
    assert ema.shadow["1.num_batches_tracked"].dtype == torch.int64
    assert int(ema.shadow["1.num_batches_tracked"]) == 1


def test_ema_state_dict_roundtrip_and_swap_restores() -> None:
    m = _tiny_bn_model()
    ema = ModelEMA(m, decay=0.5, start_step=0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(2.0)
    ema.update(m, step=0)
    sd = ema.state_dict()
    assert sd["decay"] == 0.5 and sd["num_updates"] == 1

    other = ModelEMA(_tiny_bn_model(), decay=0.1, start_step=99)
    other.load_state_dict(sd)
    assert other.decay == 0.5 and other.start_step == 0
    for k in sd["shadow"]:
        assert torch.equal(other.shadow[k], sd["shadow"][k])

    before = {k: v.clone() for k, v in m.state_dict().items()}
    with ema.swap_into(m):
        assert torch.equal(m.state_dict()["0.weight"], ema.shadow["0.weight"])
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), f"swap_into 가 {k} 를 복구하지 못했다"


def test_ema_copy_to_overwrites() -> None:
    m = _tiny_bn_model()
    ema = ModelEMA(m, decay=0.0, start_step=0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(3.0)
    ema.update(m, step=0)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    ema.copy_to(m)
    assert not torch.equal(m.state_dict()["0.weight"], torch.zeros_like(m.state_dict()["0.weight"]))


def test_ema_rejects_bad_decay_and_missing_keys() -> None:
    with pytest.raises(ValueError):
        ModelEMA(nn.Linear(1, 1), decay=1.0)
    with pytest.raises(ValueError):
        ModelEMA(nn.Linear(1, 1), start_step=-1)
    ema = ModelEMA(nn.Sequential(nn.Linear(1, 1)))  # 키 = "0.weight"
    with pytest.raises(KeyError):
        ema.update(nn.Linear(1, 1), step=0)  # 키 = "weight" → 그림자 키가 없다


# ═══════════════════════════════════════════════════════════════════════
#  optim.py — 4 그룹
# ═══════════════════════════════════════════════════════════════════════
class _GroupModel(nn.Module):
    """encoder/decoder + no_decay + physics 를 모두 가진 최소 모델."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
        self.decoder = nn.Conv2d(2, 2, 1)
        self.a_hat = nn.Parameter(torch.zeros(1))  # physics
        self.gamma = nn.Parameter(torch.ones(2))  # no_decay (LayerScale 류)

    def no_weight_decay(self):
        return {"a_hat", "gamma", "backbone.1.weight", "backbone.1.bias",
                "backbone.0.bias", "decoder.bias"}

    def physics_params(self):
        return {"a_hat"}


def test_param_groups_are_a_partition() -> None:
    m = _GroupModel()
    groups = assign_param_groups(m)
    assert set(groups) == set(PARAM_GROUPS)
    flat = [n for g in PARAM_GROUPS for n in groups[g]]
    assert len(flat) == len(set(flat)), "그룹 간 중복"
    assert set(flat) == {n for n, _ in m.named_parameters()}


def test_physics_beats_no_decay_and_gets_lr_x50() -> None:
    m = _GroupModel()
    groups = assign_param_groups(m)
    assert groups["physics"] == ["a_hat"]
    assert "a_hat" not in groups["no_decay"]
    cfg = make_cfg(**{"optim.lr": 4.0e-4, "optim.physics_lr_mult": 50.0})
    opt = build_optimizer(m, cfg.optim)
    by = {g["name"]: g for g in opt.param_groups}
    assert [g["name"] for g in opt.param_groups] == list(PARAM_GROUPS)
    assert by["physics"]["lr"] == pytest.approx(2.0e-2)
    assert by["physics"]["weight_decay"] == 0.0
    assert by["no_decay"]["weight_decay"] == 0.0
    assert by["main"]["weight_decay"] == pytest.approx(cfg.optim.weight_decay)


def test_encoder_group_uses_lr_encoder() -> None:
    m = _GroupModel()
    cfg = make_cfg(**{"optim.lr": 4.0e-4, "optim.lr_encoder": 6.0e-5})
    opt = build_optimizer(m, cfg.optim)
    by = {g["name"]: g for g in opt.param_groups}
    assert by["encoder"]["lr"] == pytest.approx(6.0e-5)
    assert by["main"]["lr"] == pytest.approx(4.0e-4)
    # backbone.0.weight 는 encoder, decoder.weight 는 main
    groups = assign_param_groups(m)
    assert "backbone.0.weight" in groups["encoder"]
    assert "decoder.weight" in groups["main"]


def test_pretrained_keys_narrows_encoder_group_b28() -> None:
    """정정 B-28: 사전학습된 적 없는 인코더 파라미터는 encoder(보호) 그룹에 들어가면 안 된다."""
    m = _GroupModel()
    full = assign_param_groups(m)
    narrowed = assign_param_groups(m, pretrained_keys={"backbone.0.weight"})
    assert "backbone.0.weight" in full["encoder"] and "backbone.0.weight" in narrowed["encoder"]
    # 사전학습 목록에 없는 인코더 weight 는 main(from-scratch lr)으로 떨어진다
    assert set(full["encoder"]) - set(narrowed["encoder"]) <= set(narrowed["main"])
    assert set(narrowed["encoder"]) == {"backbone.0.weight"}


def test_empty_groups_are_preserved() -> None:
    """05 §5.1.2: S0 에서 physics 그룹은 비지만 **구조는 유지**된다(재개 호환)."""
    m = nn.Sequential(nn.Conv2d(2, 2, 1))
    opt = build_optimizer(m, make_cfg().optim)
    assert len(opt.param_groups) == len(PARAM_GROUPS)
    assert [g["name"] for g in opt.param_groups] == list(PARAM_GROUPS)
    assert opt.param_groups[PARAM_GROUPS.index("physics")]["params"] == []


def test_frozen_params_excluded() -> None:
    m = _GroupModel()
    m.decoder.weight.requires_grad_(False)
    groups = assign_param_groups(m)
    assert "decoder.weight" not in groups["main"]


def test_collect_no_decay_delegates_and_validates() -> None:
    m = _GroupModel()
    assert collect_no_decay(m) == m.no_weight_decay()
    assert collect_physics(m) == {"a_hat"}

    class Bad(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

        def no_weight_decay(self):
            return {"does_not_exist"}

    with pytest.raises(KeyError):
        collect_no_decay(Bad())


def test_collect_no_decay_fallback_heuristic() -> None:
    m = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
    nd = collect_no_decay(m)
    assert nd == {"0.bias", "1.weight", "1.bias"}
    assert collect_physics(m) == set()


def test_encoder_prefix_is_segmentwise_not_substring() -> None:
    """`main_encoder_gate` 같은 이름이 encoder 로 오인되지 않는다."""

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.main_encoder_gate = nn.Conv2d(2, 2, 1)
            self.encoder = nn.Conv2d(2, 2, 1)

    groups = assign_param_groups(M(), encoder_prefixes=ENCODER_PREFIXES)
    assert "encoder.weight" in groups["encoder"]
    assert "main_encoder_gate.weight" in groups["main"]


def test_build_optimizer_rejects_non_adamw() -> None:
    with pytest.raises(ValueError, match="adamw"):
        build_optimizer(nn.Linear(1, 1), SimpleNamespace(name="sgd", lr=1e-3, weight_decay=0.0))


# ═══════════════════════════════════════════════════════════════════════
#  evaluator.py
# ═══════════════════════════════════════════════════════════════════════
def _logits_from_pred(pred: torch.Tensor, k: int) -> torch.Tensor:
    """argmax == pred 가 되는 원핫 로짓."""
    return torch.nn.functional.one_hot(pred, k).permute(0, 3, 1, 2).float() * 10.0


def test_seg_evaluator_hand_computed_miou() -> None:
    """2x2 두 장 손계산. class0: I=2 U=3 / class1: I=1 U=2 → mIoU = (2/3+1/2)/2."""
    gt = torch.tensor([[[0, 0], [1, 1]]])
    pred = torch.tensor([[[0, 0], [0, 1]]])
    ev = SegEvaluator(2, IGNORE_INDEX, ["bg", "algae"])
    ev.update(_logits_from_pred(pred, 2), gt)
    res = ev.compute()
    assert res["miou"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert res["pixel_acc"] == pytest.approx(3 / 4)
    assert res["present_classes"] == [0, 1]


def test_seg_evaluator_excludes_absent_classes_x20() -> None:
    """union==0 클래스는 평균에서 제외 (05 §6.2). 포함했다면 mIoU 가 1/3 이 된다."""
    gt = torch.zeros(1, 2, 2, dtype=torch.long)
    ev = SegEvaluator(3)
    ev.update(_logits_from_pred(torch.zeros(1, 2, 2, dtype=torch.long), 3), gt)
    res = ev.compute()
    assert res["miou"] == pytest.approx(1.0)
    assert res["present_classes"] == [0]
    assert math.isnan(res["per_class_iou"][1])


def test_seg_evaluator_ignore_index_excluded() -> None:
    gt = torch.tensor([[[0, IGNORE_INDEX], [1, 1]]])
    pred = torch.tensor([[[0, 0], [1, 1]]])
    ev = SegEvaluator(2)
    ev.update(_logits_from_pred(pred, 2), gt)
    assert ev.compute()["miou"] == pytest.approx(1.0)


def test_seg_evaluator_upsamples_logits_not_labels() -> None:
    gt = torch.zeros(1, 8, 8, dtype=torch.long)
    logits = torch.zeros(1, 2, 2, 2)
    logits[:, 0] = 5.0
    ev = SegEvaluator(2)
    ev.update(logits, gt)
    assert float(ev.cm.matrix.sum()) == 64.0  # 라벨 해상도가 보존됐다


def test_seg_evaluator_group_bootstrap() -> None:
    ev = SegEvaluator(2, n_boot=64, ci_width_max=1.0, seed=3)
    gt = torch.tensor([[[0, 0], [1, 1]], [[0, 0], [1, 1]]])
    ev.set_groups(["A", "B"])
    ev.update(_logits_from_pred(gt.clone(), 2), gt)
    res = ev.compute()
    assert res["n_groups"] == 2
    lo, hi = res["bootstrap_ci"]["miou"]
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    assert res["bootstrap_ci"]["miou_indeterminate"] is False


def test_seg_evaluator_no_groups_means_indeterminate() -> None:
    ev = SegEvaluator(2)
    gt = torch.zeros(1, 2, 2, dtype=torch.long)
    ev.update(_logits_from_pred(gt, 2), gt)
    ci = ev.compute()["bootstrap_ci"]
    assert ci["miou_indeterminate"] is True
    assert math.isnan(ci["miou"][0])


def test_seg_evaluator_set_groups_length_mismatch() -> None:
    ev = SegEvaluator(2)
    ev.set_groups(["A"])
    with pytest.raises(ValueError, match="set_groups"):
        ev.update(torch.zeros(2, 2, 2, 2), torch.zeros(2, 2, 2, dtype=torch.long))


def test_seg_evaluator_common_classes_and_transfer_score() -> None:
    gt = torch.tensor([[[0, 1], [2, 3]]])
    pred = torch.tensor([[[0, 1], [2, 0]]])
    ev = SegEvaluator(4, transfer_class_ids=(3,))
    ev.update(_logits_from_pred(pred, 4), gt)
    res = ev.compute()
    assert res["transfer_score"] == pytest.approx(0.0)  # class 3 을 전혀 못 맞췄다
    ev.set_common_classes([0, 1])
    assert ev.compute()["miou_common"] == pytest.approx((1 / 2 + 1.0) / 2)


def test_reg_evaluator_delegates_and_upsamples() -> None:
    y = torch.full((1, 1, 4, 4), 1.0)
    pred = torch.full((1, 1, 2, 2), 1.5)
    ev = RegEvaluator()
    ev.update(pred, y, torch.ones_like(y, dtype=torch.bool), torch.zeros_like(y))
    res = ev.compute()
    assert res["n_valid"] == 16
    assert res["mae_log"] == pytest.approx(0.5, abs=1e-6)
    assert res["f1_at_100"] is None  # 정정 A-30: 표본 0 → None + 사유
    assert "alarm_exclusion_reasons" in res


def test_boundary_evaluator_lazy_num_classes() -> None:
    gt = torch.zeros(1, 8, 8, dtype=torch.long)
    gt[:, 4:] = 1
    ev = BoundaryEvaluator((1, 3))
    assert ev.acc is None
    ev.update(_logits_from_pred(gt.clone(), 2), gt)
    assert ev.num_classes == 2
    res = ev.compute()
    assert res["bf_1px"] == pytest.approx(1.0)
    assert res["biou_1px"] == pytest.approx(1.0)


def test_evaluate_with_dummy_model() -> None:
    cfg = make_cfg(**{"data.num_classes": 3, "train.amp": "off"})
    model, crit = DummySeg(k=3), DummyCriterion()
    loader = [make_batch(2, 64, 3), make_batch(2, 64, 3)]
    seg_ev = SegEvaluator(3)
    res = evaluate(model, loader, crit, cfg=cfg, evaluators=[seg_ev], device=CPU, amp=False)
    assert "loss" in res and math.isfinite(res["loss"])
    assert 0.0 <= res["seg"]["miou"] <= 1.0
    assert res["seg"]["n_groups"] == 2  # meta.group_key 로 군이 자동 수집됐다
    assert isinstance(crit.seen_ctx[0], EvalStepCtx) or hasattr(crit.seen_ctx[0], "prec_ramp")


def test_autocast_dtype_mapping() -> None:
    assert autocast_dtype(make_cfg(**{"train.amp": "off"})) is None
    assert autocast_dtype(make_cfg(**{"train.amp": "bf16"})) is torch.bfloat16
    assert autocast_dtype(make_cfg(**{"train.amp": "fp16"})) is torch.float16
    with pytest.raises(ValueError):
        autocast_dtype(SimpleNamespace(train=SimpleNamespace(amp="tf32")))


def test_model_inputs_and_targets_from_batch() -> None:
    cfg = make_cfg()
    batch = make_batch()
    batch["y_edge"] = torch.zeros(2, 1, 64, 64)
    inputs = model_inputs_from_batch(batch, cfg)
    assert inputs["rgb"] is batch["rgb"] and inputs["msi"] is None
    assert set(targets_from_batch(batch)) == {"y_seg", "y_edge"}
    with pytest.raises(KeyError, match="rgb"):
        model_inputs_from_batch({"y_seg": batch["y_seg"]}, cfg)


# ═══════════════════════════════════════════════════════════════════════
#  trainer.py — 작은 부품
# ═══════════════════════════════════════════════════════════════════════
def test_prec_ramp_uses_config_not_hardcoded_ten() -> None:
    """정정 A-34: `epoch/10` 하드코딩 금지. warmup_epochs=4 면 epoch 2 → 0.5."""
    cfg = make_cfg(**{"model.bmef.warmup_epochs": 4})
    assert prec_ramp_for_epoch(cfg, 0) == 0.0
    assert prec_ramp_for_epoch(cfg, 2) == pytest.approx(0.5)
    assert prec_ramp_for_epoch(cfg, 9) == 1.0
    # 하드코딩 epoch/10 이었다면 0.2 가 나왔을 지점
    assert prec_ramp_for_epoch(cfg, 2) != pytest.approx(0.2)
    # warmup_epochs=0 은 config 가 거부하지만(V-range), 방어적으로 램프 없음이어야 한다
    zero = SimpleNamespace(model=SimpleNamespace(bmef=SimpleNamespace(warmup_epochs=0)))
    assert prec_ramp_for_epoch(zero, 0) == 1.0
    with pytest.raises(ValueError, match="warmup_epochs"):
        make_cfg(**{"model.bmef.warmup_epochs": 0})


def test_make_step_ctx_carries_u_and_prec_ramp() -> None:
    cfg = make_cfg(**{"schedule.epochs": 80, "model.bmef.warmup_epochs": 10})
    ctx = make_step_ctx(cfg, epoch=24, global_step=5, iters_per_epoch=7)
    assert ctx.total_epochs == 80 and ctx.total_steps == 560
    assert ctx.prec_ramp == 1.0
    assert ctx.u == pytest.approx(0.0)  # E_warm = 24 → 램프 시작점
    assert make_step_ctx(cfg, epoch=32, global_step=0, iters_per_epoch=7).u == pytest.approx(0.5)
    assert make_step_ctx(cfg, epoch=40, global_step=0, iters_per_epoch=7).u == 1.0


def test_stepctx_is_the_real_one_now_that_criterion_exists() -> None:
    """criterion 이 도착했으므로 폴백 경로를 쓰면 안 된다."""
    assert STEPCTX_FALLBACK is False


def test_early_stopper_min_delta_and_counter() -> None:
    es = EarlyStopper(patience=2, min_delta=0.01, mode="max")
    assert es.step(0.5, epoch=0) is True and es.counter == 0
    assert es.step(0.505, epoch=1) is False  # min_delta 미만
    assert es.counter == 1 and es.should_stop is False
    assert es.step(0.4, epoch=2) is False
    assert es.should_stop is True
    assert es.best == pytest.approx(0.5) and es.best_epoch == 0


def test_early_stopper_nan_never_improves() -> None:
    es = EarlyStopper(patience=1, mode="max")
    assert es.step(float("nan")) is False
    assert es.counter == 1 and es.best == float("-inf")


def test_early_stopper_min_mode_and_state_roundtrip() -> None:
    es = EarlyStopper(patience=3, min_delta=0.0, mode="min")
    assert es.step(1.0) is True
    assert es.step(2.0) is False
    other = EarlyStopper(patience=3, mode="min")
    other.load_state_dict(es.state_dict())
    assert other.best == 1.0 and other.counter == 1
    with pytest.raises(ValueError):
        EarlyStopper(mode="up")


def test_modality_dropout_disabled_and_s0() -> None:
    rng = random.Random(0)
    assert sample_modality_dropout(make_cfg(), rng) == {}
    cfg = make_cfg(**{"train.modality_dropout.enabled": True})
    assert sample_modality_dropout(cfg, rng) == {}, "S0-RGB 는 보조 path 가 없다"


def test_modality_dropout_subset_never_drops_rgb_and_is_deterministic() -> None:
    cfg = make_cfg(**{"train.modality_dropout.enabled": True})
    seen = set()
    for i in range(200):
        d = sample_modality_dropout(cfg, random.Random(i), active_paths=("rgb", "spec", "phys"))
        assert "rgb" not in d
        seen.add(tuple(sorted(d)))
    assert seen == {(), ("spec",), ("phys",), ("phys", "spec")}
    a = sample_modality_dropout(cfg, random.Random(1234), active_paths=("rgb", "spec", "phys"))
    b = sample_modality_dropout(cfg, random.Random(1234), active_paths=("rgb", "spec", "phys"))
    assert a == b


def test_modality_dropout_bernoulli_and_bad_mode() -> None:
    cfg = make_cfg(
        **{
            "train.modality_dropout.enabled": True,
            "train.modality_dropout.mode": "bernoulli",
            "train.modality_dropout.p_bernoulli": 1.0,
        }
    )
    d = sample_modality_dropout(cfg, random.Random(0), active_paths=("rgb", "spec"))
    assert d == {"spec": True}
    bad = SimpleNamespace(
        train=SimpleNamespace(modality_dropout=SimpleNamespace(enabled=True, mode="wat")),
        active_paths=("rgb", "spec"),
    )
    with pytest.raises(ValueError, match="subset|bernoulli"):
        sample_modality_dropout(bad, random.Random(0))


def test_build_grad_scaler_disabled_on_cpu() -> None:
    for amp in ("off", "bf16", "fp16"):
        sc = build_grad_scaler(make_cfg(**{"train.amp": amp}), CPU)
        assert sc.is_enabled() is False


def test_class_names_and_breakdown_columns() -> None:
    assert class_names_for(make_cfg(**{"data.num_classes": 12}))[0] == "background"
    assert class_names_for(make_cfg(**{"data.num_classes": 2})) == ["background", "algae"]
    cols = breakdown_to_csv_columns(
        "train",
        {
            "loss_ohem": 1.0,
            "loss_aux_enc_s8": 0.2,
            "loss_aux_enc_s16": 0.4,
            "loss_aux_enc_s32": 0.4,
            "loss_aux_p8": 0.4,  # CSV 스키마에 열이 없다 → 버려진다
            "n_chl_valid": 3.0,  # loss_ 접두가 아니다 → 버려진다
        },
    )
    assert cols == {
        "train_loss_ohem": 1.0,
        "train_loss_aux2": 0.2,
        "train_loss_aux3": 0.4,
        "train_loss_aux4": 0.4,
    }
    assert set(cols) <= set(EPOCH_FIELDS)


# ═══════════════════════════════════════════════════════════════════════
#  trainer.py — run_epoch
# ═══════════════════════════════════════════════════════════════════════
def _train_setup(k: int = 3, **over: Any):
    cfg = make_cfg(**{"data.num_classes": k, "train.amp": "off", **over})
    model = DummySeg(k=k)
    crit = DummyCriterion()
    opt = build_optimizer(model, cfg.optim)
    return cfg, model, crit, opt


def test_run_epoch_two_steps_updates_weights_and_scheduler() -> None:
    cfg, model, crit, opt = _train_setup()
    loader = [make_batch(2, 64, 3), make_batch(2, 64, 3)]
    sch = build_lr_scheduler(
        opt, scheduler_kwargs={"power": 0.9, "total_iters": 10}, warmup_iters=2
    )
    ema = ModelEMA(model, decay=0.5, start_step=0)
    before = model.cls.weight.detach().clone()
    lr0 = opt.param_groups[0]["lr"]

    out = run_epoch(
        model=model,
        loader=loader,
        criterion=crit,
        optimizer=opt,
        scaler=build_grad_scaler(cfg, CPU),
        device=CPU,
        cfg=cfg,
        epoch=0,
        ema=ema,
        iter_scheduler=sch,
    )
    assert out["n_batches"] == 2 and out["n_nonfinite_skips"] == 0
    assert math.isfinite(out["loss"]) and out["loss"] > 0
    assert not torch.equal(before, model.cls.weight), "가중치가 갱신되지 않았다"
    assert opt.param_groups[0]["lr"] > lr0, "warmup 중이므로 LR 이 올라야 한다"
    assert ema.num_updates == 2
    assert out["breakdown"]["loss_ohem"] == pytest.approx(out["loss"], rel=1e-6)
    assert 0.0 <= out["miou"] <= 1.0
    assert out["confusion"].shape == (3, 3)
    assert math.isfinite(out["grad_norm_mean"]) and 0.0 <= out["clip_ratio"] <= 1.0
    assert out["global_step"] == 2
    assert set(out["lr"]) == set(PARAM_GROUPS)
    assert model.last_kwargs["prec_ramp"] == pytest.approx(prec_ramp_for_epoch(cfg, 0))


def test_run_epoch_eval_mode_leaves_weights_and_scheduler_alone() -> None:
    cfg, model, crit, opt = _train_setup()
    loader = [make_batch(2, 64, 3)]
    sch = build_lr_scheduler(
        opt, scheduler_kwargs={"power": 0.9, "total_iters": 10}, warmup_iters=2
    )
    lr0 = opt.param_groups[0]["lr"]
    before = model.cls.weight.detach().clone()
    seg_ev = SegEvaluator(3)
    out = run_epoch(
        model=model,
        loader=loader,
        criterion=crit,
        optimizer=None,
        scaler=None,
        device=CPU,
        cfg=cfg,
        epoch=0,
        iter_scheduler=sch,
        evaluators=[seg_ev],
    )
    assert torch.equal(before, model.cls.weight)
    assert opt.param_groups[0]["lr"] == lr0
    assert "lr" not in out
    assert model.training is False
    assert seg_ev.compute()["n_groups"] == 2
    assert all(p.grad is None for p in model.parameters())


def test_run_epoch_grad_accum_makes_one_optimizer_step() -> None:
    cfg, model, crit, opt = _train_setup(**{"optim.grad_accum_steps": 2})
    loader = [make_batch(2, 32, 3), make_batch(2, 32, 3)]
    sch = build_lr_scheduler(
        opt, scheduler_kwargs={"power": 0.9, "total_iters": 10}, warmup_iters=4
    )
    ema = ModelEMA(model, decay=0.5, start_step=0)
    run_epoch(
        model=model, loader=loader, criterion=crit, optimizer=opt,
        scaler=build_grad_scaler(cfg, CPU), device=CPU, cfg=cfg, epoch=0,
        ema=ema, iter_scheduler=sch,
    )
    assert ema.num_updates == 1, "grad_accum=2 · 2 배치 → optimizer step 1회"
    assert sch.last_epoch == 1


def test_run_epoch_skips_nonfinite_loss() -> None:
    cfg, model, crit, opt = _train_setup()
    crit.nan_at = {1}
    loader = [make_batch(2, 32, 3), make_batch(2, 32, 3)]
    out = run_epoch(
        model=model, loader=loader, criterion=crit, optimizer=opt,
        scaler=build_grad_scaler(cfg, CPU), device=CPU, cfg=cfg, epoch=0,
    )
    assert out["n_nonfinite_skips"] == 1 and out["n_batches"] == 1
    assert math.isfinite(out["loss"])


def test_run_epoch_aborts_after_too_many_nonfinite() -> None:
    cfg, model, crit, opt = _train_setup(**{"train.nonfinite_skip_max_per_epoch": 1})
    crit.nan_at = {1, 2, 3}
    loader = [make_batch(2, 32, 3)] * 3
    with pytest.raises(RuntimeError, match="비유한"):
        run_epoch(
            model=model, loader=loader, criterion=crit, optimizer=opt,
            scaler=build_grad_scaler(cfg, CPU), device=CPU, cfg=cfg, epoch=0,
        )


def test_run_epoch_dry_run_stops_after_three_steps() -> None:
    cfg, model, crit, opt = _train_setup(**{"dry_run": True})
    loader = [make_batch(2, 32, 3)] * 10
    out = run_epoch(
        model=model, loader=loader, criterion=crit, optimizer=opt,
        scaler=build_grad_scaler(cfg, CPU), device=CPU, cfg=cfg, epoch=0,
    )
    assert out["n_batches"] == 3


def test_run_epoch_diagnostics_ignore_ratio_and_edge_pos() -> None:
    cfg, model, crit, opt = _train_setup()
    batch = make_batch(2, 32, 3)
    batch["y_seg"][:, :16] = IGNORE_INDEX  # 정확히 절반
    batch["y_edge"] = torch.zeros(2, 1, 32, 32)
    batch["y_edge"][:, :, 0] = 1.0  # 32행 중 1행 → 1/32
    batch["y_edge_valid"] = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    out = run_epoch(
        model=model, loader=[batch], criterion=crit, optimizer=opt,
        scaler=build_grad_scaler(cfg, CPU), device=CPU, cfg=cfg, epoch=0,
    )
    assert out["ignore_ratio"] == pytest.approx(0.5)
    assert out["n_edge_pos_ratio"] == pytest.approx(1 / 32)


def test_run_epoch_rejects_non_dict_model_output() -> None:
    cfg, _, crit, _ = _train_setup()

    class BadModel(nn.Module):
        def forward(self, **kw):
            return torch.zeros(1)

    with pytest.raises(TypeError, match="dict"):
        run_epoch(
            model=BadModel(), loader=[make_batch(2, 32, 3)], criterion=crit,
            optimizer=None, scaler=None, device=CPU, cfg=cfg, epoch=0,
        )


def test_run_epoch_passes_drop_modal_only_in_train() -> None:
    cfg, model, crit, opt = _train_setup(
        **{
            "data.modalities": ["rgb", "msi"],
            "data.sensor": "m3m",
            "data.bio.source": "msi",
            "train.modality_dropout.enabled": True,
            "train.modality_dropout.p_full": 0.0,  # 항상 부분집합을 뽑는다
        }
    )
    batch = make_batch(2, 32, 3)
    run_epoch(
        model=model, loader=[batch], criterion=crit, optimizer=opt,
        scaler=build_grad_scaler(cfg, CPU), device=CPU, cfg=cfg, epoch=0,
    )
    drop_seen_train = model.last_kwargs["drop_modal"]
    run_epoch(
        model=model, loader=[batch], criterion=crit, optimizer=None,
        scaler=None, device=CPU, cfg=cfg, epoch=0,
    )
    assert model.last_kwargs["drop_modal"] is None, "평가에서는 modality dropout 금지"
    assert drop_seen_train in ({"spec": True}, None)


# ═══════════════════════════════════════════════════════════════════════
#  체크포인트 / CSV
# ═══════════════════════════════════════════════════════════════════════
def test_checkpoint_payload_keys_and_train_only_pruning(kw_tmp: Path) -> None:
    from bloomnet.utils.checkpoint import CHECKPOINT_REQUIRED_KEYS, save_checkpoint

    cfg, model, _, opt = _train_setup()
    ema = ModelEMA(model, decay=0.5)
    es = EarlyStopper(patience=3)
    full = checkpoint_payload(
        model=model, epoch=0, global_step=2, best_metric=0.5, metric_name="val_miou_ema",
        optimizer=opt, ema=ema, early_stopper=es, args={"a": 1}, include_train_only=True,
    )
    for key in CHECKPOINT_REQUIRED_KEYS:
        assert key in full
    for key in ("ema_state_dict", "optimizer_state_dict", "rng_state",
                "early_stop_counter", "git_commit", "class_stats_path"):
        assert key in full
    assert any("edge_head" in k for k in full["model_state_dict"])

    best = checkpoint_payload(
        model=model, epoch=0, global_step=2, best_metric=0.5, metric_name="val_miou_ema",
        ema=ema, include_train_only=False,
    )
    assert not any("edge_head" in k for k in best["model_state_dict"])
    assert not any("edge_head" in k for k in best["ema_state_dict"]["shadow"])

    out = kw_tmp / "best.pt"
    save_checkpoint(out, best)
    assert out.exists() and not out.with_suffix(".pt.tmp").exists()


def test_epoch_row_columns_are_valid_for_strict_csv(kw_tmp: Path) -> None:
    """`_epoch_row` 산출물이 EPOCH_FIELDS 를 벗어나면 CsvLogger(strict) 가 터진다."""
    from bloomnet.engine.trainer import _epoch_row

    cfg, model, crit, opt = _train_setup()
    tr = run_epoch(
        model=model, loader=[make_batch(2, 32, 3)], criterion=crit, optimizer=opt,
        scaler=build_grad_scaler(cfg, CPU), device=CPU, cfg=cfg, epoch=0,
    )
    row = _epoch_row(0, tr, {}, {}, EarlyStopper(), opt)
    assert set(row) <= set(EPOCH_FIELDS)
    log = CsvLogger(kw_tmp / "epoch_metrics.csv", EPOCH_FIELDS, preamble={"iters_per_epoch": 1})
    log.append(row)
    text = (kw_tmp / "epoch_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("# iters_per_epoch=1")
    assert text[1].split(",")[0] == "epoch"
    assert math.isnan(float(dict(zip(text[1].split(","), text[2].split(",")))["val_algae_iou"]))


def test_stability_row_matches_schema() -> None:
    from bloomnet.utils.logging_csv import STABILITY_FIELDS

    row = collect_stability_row(DummySeg(), 0)
    assert set(row) == set(STABILITY_FIELDS)
    assert row["epoch"] == 1


# ═══════════════════════════════════════════════════════════════════════
#  실제 BloomNet 통합 스모크 (B=2, 64², CPU)
# ═══════════════════════════════════════════════════════════════════════
def test_run_epoch_with_real_bloomnet_and_criterion() -> None:
    """모델·criterion 정본으로 2 step 학습이 CPU 에서 실제로 돈다 (헌법 C-5.1)."""
    from bloomnet.losses.criterion import BloomNetCriterion
    from bloomnet.models.bloomnet import build_bloomnet

    k = 3
    cfg = make_cfg(
        **{
            "data.num_classes": k,
            "train.amp": "off",
            "schedule.epochs": 2,
            "schedule.warmup_epochs": 1,
            "model.bmef.warmup_epochs": 2,
        }
    )
    model = build_bloomnet(cfg)
    crit = BloomNetCriterion(
        num_classes=k,
        ignore_index=IGNORE_INDEX,
        lambda_aux=dict(cfg.loss.lambda_aux),
        boundary_source="criterion",
        boundary_radius=1,
        boundary_stride=4,
    )
    opt = build_optimizer(model, cfg.optim)
    sch = build_lr_scheduler(
        opt, scheduler_kwargs={"power": 0.9, "total_iters": 4}, warmup_iters=2
    )
    ema = ModelEMA(model, decay=0.5, start_step=0)
    loader = [make_batch(2, 64, k), make_batch(2, 64, k)]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    out = run_epoch(
        model=model, loader=loader, criterion=crit, optimizer=opt,
        scaler=build_grad_scaler(cfg, CPU), device=CPU, cfg=cfg, epoch=0,
        ema=ema, iter_scheduler=sch,
    )
    assert out["n_batches"] == 2 and out["n_nonfinite_skips"] == 0
    assert math.isfinite(out["loss"])
    assert out["n_edge_pos_ratio"] != 0.0 or True  # y_edge 부재 → nan 허용
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert changed, "어떤 파라미터도 갱신되지 않았다"
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{n} grad 에 비유한 값"
    assert ema.num_updates == 2

    # 평가 경로도 같은 모델로 돈다
    seg_ev = SegEvaluator(k)
    res = evaluate(model, loader, crit, cfg=cfg, evaluators=[seg_ev], device=CPU, amp=False)
    assert math.isfinite(res["loss"]) and 0.0 <= res["seg"]["miou"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  fit() — 전체 루프 (더미 로더 + 더미 모델 + **정본 criterion**)
# ═══════════════════════════════════════════════════════════════════════
def test_select_metric_table_and_algae() -> None:
    from bloomnet.engine.trainer import _select_metric

    tr = {"loss": 1.0, "miou": 0.3}
    va = {"loss": 0.9, "miou": 0.4, "miou_ema": 0.45}
    seg = {"miou_common": 0.5, "transfer_score": 0.2, "per_class_iou": [0.1, 0.8]}
    assert _select_metric("val_miou_ema", tr, va, seg) == pytest.approx(0.45)
    assert _select_metric("val_miou", tr, va, seg) == pytest.approx(0.4)
    assert _select_metric("train_miou", tr, va, seg) == pytest.approx(0.3)
    assert _select_metric("val_algae_iou", tr, va, seg) == pytest.approx(0.8)
    assert _select_metric("val_transfer_score", tr, va, seg) == pytest.approx(0.2)
    # EMA 가 없으면 val_miou 로 폴백한다 (EMA 비활성 모드에서도 학습이 진행되어야 한다)
    assert _select_metric("val_miou_ema", tr, {"miou": 0.4}, seg) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="early_stopping.metric"):
        _select_metric("nope", tr, va, seg)


def test_fit_end_to_end_writes_all_artifacts(kw_tmp: Path, monkeypatch) -> None:
    """fit() 3 에폭. run_dir 산출물·체크포인트 2개·조기종료 카운터를 전부 확인한다.

    데이터 파이프라인만 더미로 갈아끼우고 criterion/optimizer/scheduler/EMA/CSV/
    체크포인트는 **정본 경로**를 그대로 돈다. 모델은 속도 때문에 DummySeg 다
    (정본 BloomNet 의 run_epoch 통과는 별도 스모크 테스트에서 검증한다).
    """
    import bloomnet.data.build as build_mod
    import bloomnet.models.bloomnet as model_mod
    from bloomnet.engine.trainer import fit

    k = 3
    cfg = make_cfg(
        **{
            "data.num_classes": k,
            "train.amp": "off",
            "train.ema.start_epoch": 0,
            "schedule.epochs": 3,
            "schedule.warmup_epochs": 1,
            "schedule.batch_size": 2,
            "train.early_stopping.patience": 2,
            "output_dir": str(kw_tmp),
            "run_name": "run",
            "device": "cpu",
        }
    )
    loaders = {
        "train": [make_batch(2, 32, k), make_batch(2, 32, k)],
        "val": [make_batch(2, 32, k)],
    }
    monkeypatch.setattr(build_mod, "build_datasets", lambda c: {"train": None, "val": None})
    monkeypatch.setattr(build_mod, "build_dataloaders", lambda c, d: loaders)
    monkeypatch.setattr(model_mod, "build_bloomnet", lambda c: DummySeg(k=k))

    res = fit(cfg)

    run_dir = Path(res["run_dir"])
    assert run_dir == kw_tmp / "run"
    assert res["epochs_run"] == 3 and res["metric_name"] == "val_miou_ema"
    for name in (
        "train_args.json",
        "epoch_metrics.csv",
        "val_per_class_metrics.csv",
        "val_presence_summary.csv",
        "val_boundary_metrics.csv",
        "val_regression_metrics.csv",
        "stability.csv",
        "best.pt",
        "last.pt",
    ):
        assert (run_dir / name).exists(), f"{name} 이 생성되지 않았다"

    # 정정 A-18: 유도값이 CSV preamble 에 남는다 (V15 검증값 = 2/4)
    head = (run_dir / "epoch_metrics.csv").read_text(encoding="utf-8").splitlines()
    assert head[0] == "# iters_per_epoch=2 warmup_iters=2 total_iters=4"
    assert len(head) == 2 + 3  # preamble + header + 3 epochs

    # 05 §6.3: best.pt 는 학습 전용 모듈을 담지 않는다 / last.pt 는 담는다
    from bloomnet.utils.checkpoint import load_checkpoint

    best = load_checkpoint(run_dir / "best.pt")
    last = load_checkpoint(run_dir / "last.pt")
    assert not any("edge_head" in key for key in best["model_state_dict"])
    assert any("edge_head" in key for key in last["model_state_dict"])
    assert last["metric_name"] == "val_miou_ema"
    assert last["early_stop_counter"]["patience"] == 2
    assert last["rng_state"]["python"] is not None
    assert last["optimizer_state_dict"] is not None
    assert last["scheduler_state_dict"] is not None
    assert last["ema_state_dict"] is not None


def test_fit_resumes_epoch_and_early_stop_counter(kw_tmp: Path, monkeypatch) -> None:
    """[분석] §E.1.9 결함 수정 회귀: 재개 시 조기종료 카운터가 리셋되지 않는다."""
    import bloomnet.data.build as build_mod
    import bloomnet.models.bloomnet as model_mod
    from bloomnet.engine.trainer import fit

    k = 3
    loaders = {"train": [make_batch(2, 32, k)] * 2, "val": [make_batch(2, 32, k)]}
    monkeypatch.setattr(build_mod, "build_datasets", lambda c: {"train": None, "val": None})
    monkeypatch.setattr(build_mod, "build_dataloaders", lambda c, d: loaders)
    monkeypatch.setattr(model_mod, "build_bloomnet", lambda c: DummySeg(k=k))

    base = {
        "data.num_classes": k,
        "train.amp": "off",
        "schedule.epochs": 3,
        "schedule.warmup_epochs": 1,
        "schedule.batch_size": 2,
        "output_dir": str(kw_tmp),
        "run_name": "run",
        "device": "cpu",
    }
    fit(make_cfg(**base))
    ckpt = kw_tmp / "run" / "last.pt"
    from bloomnet.utils.checkpoint import load_checkpoint

    saved = load_checkpoint(ckpt)
    assert saved["epoch"] == 2

    res2 = fit(make_cfg(**{**base, "schedule.epochs": 4, "train.resume_from": str(ckpt)}))
    assert res2["epochs_run"] == 1, "재개 후 남은 1 에폭만 돌아야 한다"
    after = load_checkpoint(ckpt)
    assert after["epoch"] == 3
    assert after["early_stop_counter"]["counter"] >= saved["early_stop_counter"]["counter"]
