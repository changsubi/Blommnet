"""L2/L3 손실 조립 테스트 — 05 §0.2 / §1 / §2.4 / §3, 06 T20 (+ T19 경계분).

헌법 C-5.1/C-5.2: **CPU 전용**, B<=2, 64x64 이하.
중점:
  (a) 오늘의 S0 상황(seg 라벨만) — 유한 + backward NaN 없음 + 전 파라미터 grad 유한
  (b) 항별 on/off 와 λ 배분이 총합과 정확히 일치
  (c) 경계 동적 가중치(PIDNet weighted_bce)가 05 §1.3 크기 검산을 재현
  (d) deep supervision 가중치 적용
  (e) 라벨 전무 → 정확히 0.0 (텐서, requires_grad 유지)
  (f) 정정 A-28: boundary_source="criterion" 이면 y_edge(전부 0)를 무시하고 재생성
"""

from __future__ import annotations

import inspect
import math
import pathlib
import sys
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # cwd 와 무관하게 bloomnet 패키지를 찾게 한다
    sys.path.insert(0, str(_ROOT))

from bloomnet.constants import (  # noqa: E402
    IGNORE_INDEX,
    OUT_AUX,
    OUT_CHL,
    OUT_EDGE,
    OUT_LOGVAR,
    OUT_SEG,
    OUT_SIAM,
)
from bloomnet.data.boundary import boundary_pos_weight, make_boundary_target  # noqa: E402
from bloomnet.losses.boundary_loss import bas_loss, boundary_bce  # noqa: E402
from bloomnet.losses.criterion import (  # noqa: E402
    SIAM_PAIR_WEIGHTS,
    TEACHER_KEY,
    BloomNetCriterion,
    StepCtx,
)
from bloomnet.losses.seg import ohem_ce, plain_ce  # noqa: E402

IGNORE = IGNORE_INDEX
K12 = 12
H = 64


@pytest.fixture(autouse=True)
def _no_gpu() -> None:
    """헌법 C-5.2 — GPU 를 절대 쓰지 않는다."""
    assert not torch.cuda.is_available(), "GPU 가 보인다 — CUDA_VISIBLE_DEVICES='' 로 실행하라"


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def _structured_mask(b: int = 2, h: int = H, w: int = H, k: int = K12) -> torch.Tensor:
    """경계가 실제로 존재하는 결정론적 라벨 (사분면 + 띠)."""
    y = torch.zeros(b, h, w, dtype=torch.long)
    y[:, : h // 2, :] = 1
    y[:, :, w // 2 :] += 2
    y[:, h // 4 : h // 4 + 4, :] = 7 % k
    return y.clamp_max(k - 1)


def _outputs(
    b: int = 2,
    k: int = K12,
    h: int = H,
    *,
    stride: int = 4,
    edge: bool = True,
    chl: bool = True,
    aux: Tuple[str, ...] = ("enc_s8", "enc_s16", "enc_s32"),
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    hs = h // stride
    out: Dict[str, torch.Tensor] = {
        OUT_SEG: torch.randn(b, k, hs, hs, generator=g, requires_grad=True),
    }
    if edge:
        out[OUT_EDGE] = torch.randn(b, 1, hs, hs, generator=g, requires_grad=True)
    if chl:
        out[OUT_CHL] = F.softplus(torch.randn(b, 1, hs, hs, generator=g)).requires_grad_(True)
        out[OUT_LOGVAR] = torch.randn(b, 1, hs, hs, generator=g, requires_grad=True)
    for tap in aux:
        s = {"enc_s8": 8, "enc_s16": 16, "enc_s32": 32, "p8": 8}[tap]
        out[OUT_AUX[tap]] = torch.randn(b, k, h // s, h // s, generator=g, requires_grad=True)
    return out


def _crit(**kw: object) -> BloomNetCriterion:
    kw.setdefault("num_classes", K12)
    return BloomNetCriterion(**kw)  # type: ignore[arg-type]


def _ctx(epoch: int = 0, total: int = 10, u: float = 0.0) -> StepCtx:
    return StepCtx(epoch=epoch, total_epochs=total, global_step=epoch, total_steps=total, u=u)


class TinyNet(nn.Module):
    """전 헤드를 가진 최소 모델 — 전 파라미터 grad 유한성 검사용 (B<=2, 64²)."""

    def __init__(self, k: int = K12, c: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, stride=4, padding=1)  # H/4
        self.seg = nn.Conv2d(c, k, 1)
        self.edge = nn.Conv2d(c, 1, 1)
        self.chl = nn.Conv2d(c, 1, 1)
        self.unc = nn.Conv2d(c, 1, 1)
        self.aux8 = nn.Conv2d(c, k, 1)
        self.aux16 = nn.Conv2d(c, k, 1)
        self.aux32 = nn.Conv2d(c, k, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.stem(x)
        return {
            OUT_SEG: self.seg(f),
            OUT_EDGE: self.edge(f),
            OUT_CHL: F.softplus(self.chl(f)),
            OUT_LOGVAR: self.unc(f).clamp(-7.0, 7.0),
            OUT_AUX["enc_s8"]: self.aux8(F.avg_pool2d(f, 2)),
            OUT_AUX["enc_s16"]: self.aux16(F.avg_pool2d(f, 4)),
            OUT_AUX["enc_s32"]: self.aux32(F.avg_pool2d(f, 8)),
        }


def _component_sum(bd: Dict[str, float]) -> float:
    keys = ("loss_ohem", "loss_dice", "loss_edge", "loss_bas", "loss_reg", "loss_siam")
    s = sum(bd[k] for k in keys)
    s += sum(v for k, v in bd.items() if k.startswith("loss_aux_"))
    return float(s)


# ═════════════════════════════════════════════════════════════════════════════
# 1. 생성자 계약 (T20: λ 계약)
# ═════════════════════════════════════════════════════════════════════════════
def test_lambda_contract_creation_fails() -> None:
    """헌법 C-1 / config V2 — 사업문서 확정값을 바꾸면 **생성 자체가 실패**한다."""
    with pytest.raises(ValueError, match="C-1"):
        _crit(lambda_seg=0.5)
    with pytest.raises(ValueError, match="C-1"):
        _crit(lambda_reg=1.0)
    # 명시적 우회는 허용 (ablation 전용)
    c = _crit(lambda_seg=0.5, lambda_reg=1.0, allow_contract_break=True)
    assert c.lambda_seg == 0.5 and c.lambda_reg == 1.0


def test_ctor_validation() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        _crit(num_classes=1)
    with pytest.raises(ValueError, match="boundary_source"):
        _crit(boundary_source="dataloader")
    with pytest.raises(ValueError, match="aux_loss_stride"):
        _crit(aux_loss_stride=2)
    with pytest.raises(ValueError, match="unc_clamp"):
        _crit(unc_clamp=(7.0, -7.0))
    with pytest.raises(ValueError, match="tap"):
        _crit(lambda_aux={"enc_s2": 0.4})
    with pytest.raises(ValueError, match="class_weight"):
        _crit(class_weight=torch.ones(3))
    with pytest.raises(ValueError, match="u_source"):
        _crit(u_source="epoch")


def test_frozen_defaults_match_spec() -> None:
    """06 §3.5 동결표의 기본값 — 시그니처를 직접 읽어 고정한다."""
    p = inspect.signature(BloomNetCriterion.__init__).parameters
    assert p["ignore_index"].default == 255
    assert p["lambda_seg"].default == 1.0
    assert p["lambda_reg"].default == 0.5
    assert p["lambda_bd"].default == 20.0
    assert p["lambda_bas"].default == 1.0
    assert p["lambda_siam"].default == 0.0
    assert p["w_dice"].default == 0.4
    assert p["ohem_thresh"].default == 0.7
    assert p["ohem_keep_frac"].default == 0.0625
    assert p["bas_tau"].default == 0.8
    assert p["huber_beta"].default == 1.0
    # ★X-26 / 정정 A-20 — UncHead S_MIN/S_MAX 와 동일해야 이중 clamp 가 항등이 된다
    assert p["unc_clamp"].default == (-7.0, 7.0)
    assert p["u_warm_frac"].default == 0.30 and p["u_ramp_frac"].default == 0.20
    assert p["boundary_source"].default == "dataset"
    assert p["aux_loss_stride"].default == 1
    assert _crit().lambda_aux == {"enc_s8": 0.2, "enc_s16": 0.4, "enc_s32": 0.4, "p8": 0.4}


def test_step_ctx_fields() -> None:
    """정정 A-34 — prec_ramp(모델용) / u(criterion용) 가 StepCtx 에 있다."""
    ctx = StepCtx(epoch=1, total_epochs=10, global_step=5, total_steps=100)
    assert ctx.prec_ramp == 1.0 and ctx.u == 0.0
    ctx2 = StepCtx(1, 10, 5, 100, 0.5, 0.25)
    assert ctx2.prec_ramp == 0.5 and ctx2.u == 0.25


# ═════════════════════════════════════════════════════════════════════════════
# 2. (a) 오늘의 S0 상황 — seg 라벨만 존재
# ═════════════════════════════════════════════════════════════════════════════
def test_s0_today_finite_and_all_param_grads_finite() -> None:
    """chl/unc 라벨이 전무한 오늘의 정상 경로. 05 §3.1 / §1.4."""
    torch.manual_seed(0)
    net = TinyNet()
    crit = _crit()
    x = torch.randn(2, 3, H, H)
    out = net(x)
    tgt = {"y_seg": _structured_mask()}

    loss, bd = crit(out, tgt, _ctx())
    assert torch.isfinite(loss) and loss.requires_grad
    assert loss.item() > 0.0
    # 라벨 부재 항은 정확히 0
    assert bd["loss_reg"] == 0.0
    assert bd["n_chl_valid"] == 0.0
    assert bd["loss_siam"] == 0.0
    # 라벨이 있는 항은 실제로 살아 있다
    for k in ("loss_ohem", "loss_dice", "loss_edge"):
        assert bd[k] > 0.0, k
    # 05 §1.3 — 초기에는 sigmoid(edge) ≈ 0.5 < τ=0.8 이라 L_bas ≈ 0 이 **정상**이다
    # (PIDNet 원본은 이 지점에서 NaN 을 낸다 — 규칙 N2).
    assert bd["loss_bas"] == 0.0

    loss.backward()
    missing = [n for n, p in net.named_parameters() if p.grad is None]
    assert not missing, f"grad 가 없는 파라미터: {missing}"
    for n, p in net.named_parameters():
        assert torch.isfinite(p.grad).all(), f"{n} grad 비유한"
    # UncHead 는 손실에 들어가지 않지만 graph 는 연결돼 있어야 한다 (DDP 안전)
    assert torch.equal(net.unc.weight.grad, torch.zeros_like(net.unc.weight))


def test_lambda_reg_stays_05_when_labels_absent() -> None:
    """05 §1.4 — S0 에서 λ_reg 를 0 으로 '바꾸지' 않는다. 마스킹으로 항이 0 이 된다."""
    crit = _crit()
    assert crit.lambda_reg == 0.5
    out = _outputs()
    loss, bd = crit(out, {"y_seg": _structured_mask()}, _ctx())
    assert bd["loss_reg"] == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 3. (b) 항별 on/off · 총합 일치 · λ 배분
# ═════════════════════════════════════════════════════════════════════════════
def test_total_equals_sum_of_breakdown_components() -> None:
    crit = _crit()
    out = _outputs()
    loss, bd = crit(out, {"y_seg": _structured_mask()}, _ctx())
    assert bd["loss_total"] == pytest.approx(float(loss.detach()), rel=1e-6)
    assert _component_sum(bd) == pytest.approx(bd["loss_total"], rel=1e-5, abs=1e-6)


def test_breakdown_keys_complete() -> None:
    """06 §3.5 가 요구하는 breakdown 키 전부 (T20)."""
    crit = _crit()
    _, bd = crit(_outputs(), {"y_seg": _structured_mask()}, _ctx())
    required = {
        "loss_ohem", "loss_dice", "loss_edge", "loss_bas", "loss_reg", "loss_siam",
        "n_chl_valid", "n_edge_pos_ratio", "u_ramp",
    }
    required |= {f"loss_aux_{t}" for t in ("enc_s8", "enc_s16", "enc_s32", "p8")}
    assert required <= set(bd)
    assert all(isinstance(v, float) for v in bd.values())


def test_breakdown_columns_stable_across_modes() -> None:
    """05 §3.2 — 라벨 유무와 무관하게 CSV 열 스키마가 동일해야 한다."""
    crit = _crit()
    y = _structured_mask()
    _, bd_full = crit(_outputs(edge=True, chl=True), {"y_seg": y}, _ctx())
    _, bd_min = crit(_outputs(edge=False, chl=False, aux=()), {"y_seg": y}, _ctx())
    assert set(bd_full) == set(bd_min)


def test_terms_switch_off_when_outputs_absent() -> None:
    """edge_logits 미제공 → L_edge == L_bas == 0 이고 총합이 정확히 그만큼 줄어든다."""
    crit = _crit()
    y = _structured_mask()
    full = _outputs(edge=True, chl=True, seed=3)
    loss_f, bd_f = crit(full, {"y_seg": y}, _ctx())

    noedge = {k: v for k, v in full.items() if k != OUT_EDGE}
    loss_n, bd_n = crit(noedge, {"y_seg": y}, _ctx())
    assert bd_n["loss_edge"] == 0.0 and bd_n["loss_bas"] == 0.0
    assert bd_n["n_edge_pos_ratio"] == 0.0
    expect = bd_f["loss_total"] - bd_f["loss_edge"] - bd_f["loss_bas"]
    assert bd_n["loss_total"] == pytest.approx(expect, rel=1e-5)


def test_lambda_scaling_is_exact() -> None:
    """λ 를 2배로 하면 해당 항의 breakdown 도 정확히 2배."""
    y = _structured_mask()
    out = _outputs(seed=5)
    _, bd1 = _crit(lambda_bd=20.0)(out, {"y_seg": y}, _ctx())
    _, bd2 = _crit(lambda_bd=40.0)(out, {"y_seg": y}, _ctx())
    assert bd2["loss_edge"] == pytest.approx(2.0 * bd1["loss_edge"], rel=1e-6)
    assert bd2["loss_ohem"] == pytest.approx(bd1["loss_ohem"], rel=1e-6)


def test_w_dice_is_inside_lambda_seg() -> None:
    """05 §1.1 — Dice 는 top-level λ 를 건드리지 않고 λ_seg 안에 w_dice 로 들어간다."""
    y = _structured_mask()
    out = _outputs(seed=7)
    crit = _crit(w_dice=0.4)
    _, bd = crit(out, {"y_seg": y}, _ctx())
    from bloomnet.losses.seg import batch_soft_dice

    up = F.interpolate(out[OUT_SEG].float(), size=(H, H), mode="bilinear", align_corners=False)
    raw = float(batch_soft_dice(up, y, num_classes=K12, ignore_index=IGNORE).detach())
    assert bd["loss_dice"] == pytest.approx(1.0 * 0.4 * raw, rel=1e-5)


def test_ohem_term_matches_primitive() -> None:
    y = _structured_mask()
    out = _outputs(seed=11)
    crit = _crit(ohem_thresh=0.7, ohem_keep_frac=0.0625)
    _, bd = crit(out, {"y_seg": y}, _ctx())
    up = F.interpolate(out[OUT_SEG].float(), size=(H, H), mode="bilinear", align_corners=False)
    raw = float(ohem_ce(up, y, ignore_index=IGNORE, thresh=0.7, keep_frac=0.0625).detach())
    assert bd["loss_ohem"] == pytest.approx(raw, rel=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# 4. (c) 경계 — PIDNet weighted_bce 의미론과 동적 가중치
# ═════════════════════════════════════════════════════════════════════════════
def _pidnet_weighted_bce_reference(
    edge_logits: torch.Tensor, bd_gt: torch.Tensor, bd_valid: torch.Tensor
) -> torch.Tensor:
    """05 §2.4.2 의사코드를 **문자 그대로** 옮긴 참조 구현."""
    if edge_logits.shape[-2:] != bd_gt.shape[-2:]:
        edge_logits = F.interpolate(
            edge_logits, size=bd_gt.shape[-2:], mode="bilinear", align_corners=False
        )
    v = bd_valid.reshape(-1).bool()
    if int(v.sum()) == 0:
        return edge_logits.sum() * 0.0
    logit = edge_logits.reshape(-1)[v].float()
    tgt = bd_gt.reshape(-1)[v].float()
    n_pos = tgt.sum()
    n_all = float(tgt.numel())
    n_neg = n_all - n_pos
    w = torch.empty_like(tgt)
    w[tgt > 0.5] = (n_neg / n_all)
    w[tgt <= 0.5] = (n_pos / n_all)
    return F.binary_cross_entropy_with_logits(logit, tgt, weight=w, reduction="mean")


def test_boundary_bce_matches_pidnet_reference() -> None:
    """동기화 없는 등가식이 05 의사코드와 수치적으로 같아야 한다."""
    torch.manual_seed(1)
    for pos_frac in (0.03, 0.2, 0.5):
        gt = (torch.rand(2, 1, 16, 16) < pos_frac).float()
        valid = torch.rand(2, 1, 16, 16) > 0.1
        logits = torch.randn(2, 1, 16, 16) * 2.0
        got = boundary_bce(logits, gt, valid)
        ref = _pidnet_weighted_bce_reference(logits, gt, valid)
        assert float(got) == pytest.approx(float(ref), rel=1e-6, abs=1e-7)


def test_boundary_bce_weights_come_from_single_source() -> None:
    """가중치 정의가 data/boundary.py::boundary_pos_weight 와 일치 (리터럴 복제 방지)."""
    torch.manual_seed(2)
    gt = (torch.rand(2, 1, 16, 16) < 0.1).float()
    valid = torch.rand(2, 1, 16, 16) > 0.2
    w_pos, w_neg = boundary_pos_weight(gt, valid)
    # 같은 (w_pos, w_neg) 로 손계산한 값과 boundary_bce 가 일치해야 한다.
    e = torch.randn(2, 1, 16, 16)
    bce = F.binary_cross_entropy_with_logits(e, gt, reduction="none")
    w = torch.where(gt > 0.5, torch.full_like(gt, w_pos), torch.full_like(gt, w_neg))
    v = valid.float()
    manual = (w * v * bce).sum() / v.sum()
    assert float(boundary_bce(e, gt, valid).detach()) == pytest.approx(float(manual), rel=1e-6)
    assert w_pos + w_neg == pytest.approx(1.0, rel=1e-6)


def test_boundary_bce_magnitude_reproduces_spec_check() -> None:
    """05 §1.3 (정정 B-20) 크기 검산 재현.

    균일 로짓(0) → BCE = ln2. 가중치 평균 = 2·p·(1−p) (p = 양성비).
    p = 0.03112 → L_edge = 0.041799, ×λ_bd(20) = 0.836 → seg 항(ln12 = 2.4849)의 ≈ 1/3.
    """
    n = 10_000
    n_pos = 311  # p = 0.0311
    gt = torch.zeros(1, 1, 1, n)
    gt[..., :n_pos] = 1.0
    valid = torch.ones_like(gt, dtype=torch.bool)
    logits = torch.zeros_like(gt)

    l_edge = float(boundary_bce(logits, gt, valid).detach())
    p = n_pos / n
    assert l_edge == pytest.approx(math.log(2.0) * 2 * p * (1 - p), rel=1e-6)
    assert l_edge == pytest.approx(0.0418, abs=5e-4)  # 05 §1.3 정정본
    scaled = 20.0 * l_edge
    assert scaled == pytest.approx(0.836, abs=0.01)
    # pos_weight 방식이었다면 ≈ 1,000 배 과대해진다는 05 §1.3 경고를 반대로 고정한다.
    assert scaled < math.log(K12)  # seg 항을 압도하지 않는다
    assert 0.2 < scaled / math.log(K12) < 0.5  # ≈ 1/3


def test_boundary_bce_zero_positive_is_exactly_zero() -> None:
    """규칙 N4 — 경계가 하나도 없으면 손실이 정확히 0 (0-나눗셈 없음)."""
    gt = torch.zeros(2, 1, 8, 8)
    valid = torch.ones_like(gt, dtype=torch.bool)
    e = torch.randn(2, 1, 8, 8, requires_grad=True)
    loss = boundary_bce(e, gt, valid)
    assert float(loss.detach()) == 0.0
    assert loss.requires_grad  # graph 유지 (규칙 N1)
    loss.backward()
    assert torch.isfinite(e.grad).all() and float(e.grad.abs().sum()) == 0.0


def test_boundary_bce_all_invalid_is_zero_and_finite() -> None:
    gt = (torch.rand(2, 1, 8, 8) < 0.5).float()
    valid = torch.zeros_like(gt, dtype=torch.bool)
    e = (torch.randn(2, 1, 8, 8) * 1e4).requires_grad_(True)
    loss = boundary_bce(e, gt, valid)
    assert float(loss.detach()) == 0.0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(e.grad).all()


def test_boundary_bce_upsamples_logits() -> None:
    gt = (torch.rand(1, 1, 16, 16) < 0.2).float()
    valid = torch.ones_like(gt, dtype=torch.bool)
    e = torch.randn(1, 1, 4, 4)
    assert torch.isfinite(boundary_bce(e, gt, valid))


def test_dynamic_pos_weight_is_in_sane_range_on_synthetic_masks() -> None:
    """(c) 동적 pos_weight 가 합리적 범위 — 01 [M4] anchor (p ≈ 3%) 주변."""
    y = _structured_mask()
    gt, valid = make_boundary_target(y, ignore_index=IGNORE, radius=1, out_stride=4)
    w_pos, w_neg = boundary_pos_weight(gt, valid)
    assert 0.5 < w_pos < 1.0, w_pos  # 양성이 희소하므로 양성 가중치가 크다
    assert 0.0 < w_neg < 0.5, w_neg
    assert w_pos > w_neg
    mean_w = 2.0 * w_neg * (1.0 - w_neg)
    assert 0.0 < mean_w < 0.5  # 05 §1.3: 가중치 평균이 크게 내려간다


# ═════════════════════════════════════════════════════════════════════════════
# 5. L_bas — 초기 NaN 경로 (규칙 N2)
# ═════════════════════════════════════════════════════════════════════════════
def test_bas_loss_zero_at_init_and_no_nan() -> None:
    """학습 초기 sigmoid(edge) ≈ 0.5 < τ=0.8 → 전 픽셀 ignore.

    **PIDNet 원본은 여기서 NaN 을 낸다** (05 §2.4.3). 우리 ohem_ce 가 0.0 을 낸다.
    """
    y = _structured_mask()
    seg = torch.randn(2, K12, 16, 16, requires_grad=True)
    edge = torch.zeros(2, 1, 16, 16, requires_grad=True)  # sigmoid = 0.5
    loss = bas_loss(seg, y, edge, tau=0.8, ignore_index=IGNORE)
    assert float(loss.detach()) == 0.0 and torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert torch.isfinite(seg.grad).all()


def test_bas_loss_active_when_edge_confident() -> None:
    y = _structured_mask()
    seg = torch.randn(2, K12, 16, 16)
    edge = torch.full((2, 1, 16, 16), 5.0)  # sigmoid ≈ 0.993 > 0.8 → 전 픽셀 감독
    loss = bas_loss(seg, y, edge, tau=0.8, ignore_index=IGNORE)
    ref = ohem_ce(seg, y, ignore_index=IGNORE)
    assert float(loss.detach()) == pytest.approx(float(ref.detach()), rel=1e-6)


def test_bas_loss_selects_only_confident_pixels() -> None:
    """경계 확률이 절반만 τ 를 넘으면 그 절반만 감독된다 (전체 CE 와 달라야 한다)."""
    y = _structured_mask(b=1)
    seg = torch.randn(1, K12, 16, 16)
    edge = torch.full((1, 1, 16, 16), -5.0)
    edge[..., :8] = 5.0
    part = float(bas_loss(seg, y, edge, tau=0.8, ignore_index=IGNORE).detach())
    whole = float(
        bas_loss(seg, y, torch.full_like(edge, 5.0), tau=0.8, ignore_index=IGNORE).detach()
    )
    assert part != pytest.approx(whole, rel=1e-4)
    assert part > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 6. (f) 경계 타깃 소유권 — 정정 A-28 (T20 필수 케이스)
# ═════════════════════════════════════════════════════════════════════════════
def test_boundary_source_criterion_regenerates_and_ignores_zero_y_edge() -> None:
    """source="criterion" 이면 (전부 0, valid=False) 자리표시자를 무시하고 재생성한다.

    그렇게 하지 않으면 L_edge 가 학습 내내 조용히 0 이 되어 λ_bd=20 이 무력화된다.
    """
    y = _structured_mask()
    out = _outputs(seed=13)
    zeros_edge = torch.zeros(2, 1, H // 4, H // 4)
    tgt = {
        "y_seg": y,
        "y_edge": zeros_edge,
        "y_edge_valid": torch.zeros_like(zeros_edge, dtype=torch.bool),
    }
    _, bd = _crit(boundary_source="criterion")(out, tgt, _ctx())
    assert bd["loss_edge"] > 0.0, "재생성 경로가 돌지 않았다 (A-28)"
    assert bd["n_edge_pos_ratio"] > 0.0


def test_boundary_source_dataset_uses_targets_verbatim() -> None:
    """source="dataset" 이면 주어진 타깃을 그대로 쓴다 (전부 0/invalid → L_edge = 0)."""
    y = _structured_mask()
    out = _outputs(seed=13)
    zeros_edge = torch.zeros(2, 1, H // 4, H // 4)
    tgt = {
        "y_seg": y,
        "y_edge": zeros_edge,
        "y_edge_valid": torch.zeros_like(zeros_edge, dtype=torch.bool),
    }
    _, bd = _crit(boundary_source="dataset")(out, tgt, _ctx())
    assert bd["loss_edge"] == 0.0
    assert bd["n_edge_pos_ratio"] == 0.0


def test_boundary_source_dataset_falls_back_when_target_missing() -> None:
    """05 §0.2 — y_edge 가 아예 없으면 같은 함수로 생성한다."""
    y = _structured_mask()
    out = _outputs(seed=13)
    _, bd_ds = _crit(boundary_source="dataset")(out, {"y_seg": y}, _ctx())
    _, bd_cr = _crit(boundary_source="criterion")(out, {"y_seg": y}, _ctx())
    assert bd_ds["loss_edge"] == pytest.approx(bd_cr["loss_edge"], rel=1e-6)
    assert bd_ds["loss_edge"] > 0.0


def test_dataset_target_matches_criterion_regeneration() -> None:
    """같은 연산자(X-07 단일 구현)를 쓰므로 dataset 경로와 재생성 경로가 동일해야 한다."""
    y = _structured_mask()
    gt, valid = make_boundary_target(y, ignore_index=IGNORE, radius=1, out_stride=4)
    out = _outputs(seed=17)
    _, bd_ds = _crit(boundary_source="dataset")(
        out, {"y_seg": y, "y_edge": gt, "y_edge_valid": valid}, _ctx()
    )
    _, bd_cr = _crit(boundary_source="criterion", boundary_radius=1)(out, {"y_seg": y}, _ctx())
    assert bd_ds["loss_edge"] == pytest.approx(bd_cr["loss_edge"], rel=1e-6)
    assert bd_ds["loss_bas"] == pytest.approx(bd_cr["loss_bas"], rel=1e-6)


def test_boundary_radius_map_overrides_scalar() -> None:
    """01 C-10 — 해상도별 반경(512²→1 / 1024²→2). 한 criterion 으로 여러 해상도를
    평가할 때 반경이 조용히 틀리는 것을 막는다."""
    y = _structured_mask()
    out = _outputs(seed=19)
    # out_stride=1 로 봐야 max-pool 포화 없이 반경 차이가 드러난다.
    c1 = _crit(boundary_source="criterion", boundary_radius=1, boundary_stride=1)
    c2 = _crit(
        boundary_source="criterion",
        boundary_radius=1,
        boundary_stride=1,
        boundary_radius_map={H: 2, 1024: 2},
    )
    assert c1._radius_for(H) == 1 and c2._radius_for(H) == 2
    assert c2._radius_for(512) == 1  # 매핑에 없는 해상도는 스칼라 기본값
    _, bd1 = c1(out, {"y_seg": y}, _ctx())
    _, bd2 = c2(out, {"y_seg": y}, _ctx())
    assert bd2["n_edge_pos_ratio"] > bd1["n_edge_pos_ratio"]  # 반경이 크면 경계가 두꺼워진다


# ═════════════════════════════════════════════════════════════════════════════
# 7. (d) Deep supervision
# ═════════════════════════════════════════════════════════════════════════════
def test_aux_weights_applied_exactly() -> None:
    """λ_aux = (0.2, 0.4, 0.4) 가 tap 별로 정확히 곱해진다 (05 §2.3.3)."""
    y = _structured_mask()
    out = _outputs(seed=23)
    crit = _crit()
    _, bd = crit(out, {"y_seg": y}, _ctx())
    for tap, lam in (("enc_s8", 0.2), ("enc_s16", 0.4), ("enc_s32", 0.4)):
        raw = float(plain_ce(out[OUT_AUX[tap]].float(), y, ignore_index=IGNORE).detach())
        assert bd[f"loss_aux_{tap}"] == pytest.approx(lam * raw, rel=1e-6), tap
    assert bd["loss_aux_p8"] == 0.0  # 모델이 내지 않은 tap


def test_aux_uses_plain_ce_not_ohem() -> None:
    """PIDNet 공식과 동일하게 보조 헤드에는 마이닝을 걸지 않는다 (05 §2.3.3).

    판별력을 위해 ohem 을 **강한 마이닝**(thresh=0.0)으로 설정한다 — 기본 0.7 에서는
    랜덤 로짓의 p_gt 가 거의 전부 임계 미만이라 OHEM 이 평균 CE 로 환원되어
    두 구현을 구분하지 못한다.
    """
    y = _structured_mask()
    out = _outputs(aux=("enc_s8",), seed=29)
    _, bd = _crit(lambda_aux={"enc_s8": 1.0}, ohem_thresh=0.0, ohem_keep_frac=0.0625)(
        out, {"y_seg": y}, _ctx()
    )
    logits = out[OUT_AUX["enc_s8"]].float()
    ce = float(plain_ce(logits, y, ignore_index=IGNORE).detach())
    oh = float(ohem_ce(logits, y, ignore_index=IGNORE, thresh=0.0, keep_frac=0.0625).detach())
    assert ce != pytest.approx(oh, rel=1e-2), "마이닝이 실제로 다른 값을 내는 설정이어야 한다"
    assert bd["loss_aux_enc_s8"] == pytest.approx(ce, rel=1e-6)


def test_aux_decay_last_20_percent() -> None:
    crit = _crit(aux_decay=True)
    assert crit.aux_scale(0, 100) == 1.0
    assert crit.aux_scale(80, 100) == 1.0
    assert crit.aux_scale(90, 100) == pytest.approx(0.5)
    assert crit.aux_scale(100, 100) == pytest.approx(0.0)
    assert _crit(aux_decay=False).aux_scale(95, 100) == 1.0

    y = _structured_mask()
    out = _outputs(seed=31)
    _, bd0 = crit(out, {"y_seg": y}, _ctx(epoch=0, total=100))
    _, bd9 = crit(out, {"y_seg": y}, _ctx(epoch=90, total=100))
    assert bd9["loss_aux_enc_s8"] == pytest.approx(0.5 * bd0["loss_aux_enc_s8"], rel=1e-6)


def test_aux_loss_stride_4_lever(caplog: pytest.LogCaptureFixture) -> None:
    """정정 A-32(c) — aux 항을 1/4 해상도에서 계산하는 OOM 레버."""
    y = _structured_mask()
    out = _outputs(aux=("enc_s8",), seed=37)
    _, bd1 = _crit(aux_loss_stride=1, lambda_aux={"enc_s8": 1.0})(out, {"y_seg": y}, _ctx())
    _, bd4 = _crit(aux_loss_stride=4, lambda_aux={"enc_s8": 1.0})(out, {"y_seg": y}, _ctx())
    ref4 = float(
        plain_ce(out[OUT_AUX["enc_s8"]].float(), y[:, ::4, ::4], ignore_index=IGNORE).detach()
    )
    assert bd4["loss_aux_enc_s8"] == pytest.approx(ref4, rel=1e-6)
    assert bd4["loss_aux_enc_s8"] != pytest.approx(bd1["loss_aux_enc_s8"], rel=1e-4)
    # 주 손실(ohem/dice)은 stride 레버의 영향을 받지 않는다
    assert bd4["loss_ohem"] == pytest.approx(bd1["loss_ohem"], rel=1e-6)


def test_aux_loss_stride_rejects_indivisible_shape() -> None:
    y = torch.zeros(1, 30, 30, dtype=torch.long)
    out = {OUT_SEG: torch.randn(1, K12, 30, 30)}
    with pytest.raises(ValueError, match="aux_loss_stride"):
        _crit(aux_loss_stride=4)(out, {"y_seg": y}, _ctx())


def test_unconfigured_aux_tap_still_gets_grad() -> None:
    """lambda_aux 에 없는 tap 을 모델이 내면 손실 기여는 0 이되 graph 는 연결된다."""
    y = _structured_mask()
    out = _outputs(aux=("enc_s8", "enc_s16"), seed=41)
    crit = _crit(lambda_aux={"enc_s8": 0.2})
    loss, bd = crit(out, {"y_seg": y}, _ctx())
    assert "loss_aux_enc_s16" not in bd
    loss.backward()
    g = out[OUT_AUX["enc_s16"]].grad
    assert g is not None and float(g.abs().sum()) == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 8. (e) 라벨 전무 → 0.0  /  T20 NaN 방지 5경우
# ═════════════════════════════════════════════════════════════════════════════
def test_all_labels_absent_returns_exact_zero() -> None:
    """(e) y_seg 전부 ignore + 다른 라벨 전무 → L_total 이 정확히 0.0, 유한, graph 유지."""
    torch.manual_seed(0)
    net = TinyNet()
    crit = _crit()
    out = net(torch.randn(2, 3, H, H))
    tgt = {"y_seg": torch.full((2, H, H), IGNORE, dtype=torch.long)}

    loss, bd = crit(out, tgt, _ctx())
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    assert loss.requires_grad, "torch.tensor(0.0) 이 아니라 graph 에 연결된 0 이어야 한다 (N1)"
    for k, v in bd.items():
        assert v == 0.0, f"{k} = {v}"

    loss.backward()
    for n, p in net.named_parameters():
        assert p.grad is not None, n
        assert torch.isfinite(p.grad).all(), n
        assert float(p.grad.abs().sum()) == 0.0, n


def test_t20_case_a_all_ignore() -> None:
    net, crit = TinyNet(), _crit()
    out = net(torch.randn(2, 3, H, H))
    loss, _ = crit(out, {"y_seg": torch.full((2, H, H), IGNORE, dtype=torch.long)}, _ctx())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_t20_case_b_no_boundary() -> None:
    """단일 클래스 마스크 → 경계 0개."""
    net, crit = TinyNet(), _crit()
    out = net(torch.randn(2, 3, H, H))
    y = torch.zeros(2, H, H, dtype=torch.long)
    loss, bd = crit(out, {"y_seg": y}, _ctx())
    assert bd["loss_edge"] == 0.0 and bd["n_edge_pos_ratio"] == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_t20_case_c_chl_valid_zero() -> None:
    """m_chl 전부 False → L_reg == 0, backward 통과 (05 §3.3 필수 목록 2)."""
    net, crit = TinyNet(), _crit()
    out = net(torch.randn(2, 3, H, H))
    tgt = {
        "y_seg": _structured_mask(),
        "y_chl": torch.rand(2, 1, H, H) * 4.0,  # log1p 공간 (★X-14)
        "y_chl_valid": torch.zeros(2, 1, H, H, dtype=torch.bool),
    }
    loss, bd = crit(out, tgt, _ctx(u=1.0))
    assert bd["loss_reg"] == 0.0 and bd["n_chl_valid"] == 0.0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_t20_case_d_extreme_logits() -> None:
    """로짓 ±1e4 + log_var ±1e4 → clamp 후 유한 (규칙 N5)."""
    crit = _crit()
    b, hs = 2, H // 4
    out = {
        OUT_SEG: (torch.randn(b, K12, hs, hs) * 1e4).requires_grad_(True),
        OUT_EDGE: (torch.randn(b, 1, hs, hs) * 1e4).requires_grad_(True),
        OUT_CHL: F.softplus(torch.randn(b, 1, hs, hs)).requires_grad_(True),
        OUT_LOGVAR: torch.full((b, 1, hs, hs), 1e4).requires_grad_(True),
    }
    tgt = {
        "y_seg": _structured_mask(),
        "y_chl": torch.rand(b, 1, H, H) * 4.0,
        "y_chl_valid": torch.ones(b, 1, H, H, dtype=torch.bool),
    }
    loss, bd = crit(out, tgt, _ctx(u=1.0))
    assert torch.isfinite(loss), bd
    loss.backward()
    for k, v in out.items():
        assert torch.isfinite(v.grad).all(), k


def test_t20_case_e_missing_everything_but_seg() -> None:
    """전부-결측 배치: seg 로짓만 존재."""
    crit = _crit()
    out = {OUT_SEG: torch.randn(2, K12, H // 4, H // 4, requires_grad=True)}
    loss, bd = crit(out, {"y_seg": _structured_mask()}, _ctx())
    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    assert bd["loss_edge"] == 0.0 and bd["loss_bas"] == 0.0 and bd["loss_reg"] == 0.0
    loss.backward()
    assert torch.isfinite(out[OUT_SEG].grad).all()


def test_single_class_batch_dice_finite() -> None:
    """05 §3.3 필수 목록 4 — 배치에 클래스가 1개뿐이어도 Dice 유한 (규칙 N6)."""
    crit = _crit()
    out = {OUT_SEG: torch.randn(2, K12, H // 4, H // 4, requires_grad=True)}
    y = torch.full((2, H, H), 5, dtype=torch.long)
    loss, bd = crit(out, {"y_seg": y}, _ctx())
    assert torch.isfinite(loss) and math.isfinite(bd["loss_dice"])
    loss.backward()
    assert torch.isfinite(out[OUT_SEG].grad).all()


def test_debug_assert_catches_nonfinite_logits() -> None:
    """규칙 N8 — nan_to_num 으로 덮지 않고 드러낸다."""
    crit = _crit(debug_assert=True)
    out = {OUT_SEG: torch.full((1, K12, 16, 16), float("nan"))}
    with pytest.raises(AssertionError, match="seg_logits"):
        crit(out, {"y_seg": _structured_mask(b=1)}, _ctx())


# ═════════════════════════════════════════════════════════════════════════════
# 9. 해상도 관용 (T20) · 필수 키
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("stride", [1, 4, 8])
def test_any_logit_resolution(stride: int) -> None:
    """05 §0.2 — 로짓 해상도가 H/4·H/8·H 어느 것이든 동작한다."""
    crit = _crit()
    out = _outputs(stride=stride, aux=("enc_s8",), seed=43)
    loss, bd = crit(out, {"y_seg": _structured_mask()}, _ctx())
    assert torch.isfinite(loss) and float(loss.detach()) > 0.0
    assert bd["loss_edge"] > 0.0


def test_missing_required_keys_raise() -> None:
    crit = _crit()
    with pytest.raises(KeyError, match="y_seg"):
        crit({OUT_SEG: torch.randn(1, K12, 8, 8)}, {}, _ctx())
    with pytest.raises(KeyError, match=OUT_SEG):
        crit({}, {"y_seg": _structured_mask(b=1)}, _ctx())


def test_num_classes_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        _crit(num_classes=2)(
            {OUT_SEG: torch.randn(1, K12, 8, 8)}, {"y_seg": _structured_mask(b=1)}, _ctx()
        )


def test_accepts_b1hw_seg_target() -> None:
    crit = _crit()
    y = _structured_mask(b=1).unsqueeze(1)  # (B,1,H,W)
    out = _outputs(b=1, seed=47)
    loss, _ = crit(out, {"y_seg": y}, _ctx())
    assert torch.isfinite(loss)


# ═════════════════════════════════════════════════════════════════════════════
# 10. 회귀 항 · u ramp (정정 A-34)
# ═════════════════════════════════════════════════════════════════════════════
def _chl_targets(b: int = 2) -> Dict[str, torch.Tensor]:
    return {
        "y_seg": _structured_mask(b=b),
        "y_chl": torch.full((b, 1, H, H), 1.5),  # log1p 공간 (★X-14)
        "y_chl_valid": torch.ones(b, 1, H, H, dtype=torch.bool),
    }


def test_reg_term_active_when_labels_present() -> None:
    crit = _crit()
    out = _outputs(seed=53)
    loss, bd = crit(out, _chl_targets(), _ctx())
    assert bd["loss_reg"] > 0.0
    assert bd["n_chl_valid"] == float(2 * H * H)
    from bloomnet.losses.regression import chl_reg_loss

    raw, _ = chl_reg_loss(
        F.interpolate(out[OUT_CHL].float(), size=(H, H), mode="bilinear", align_corners=False),
        None,
        torch.full((2, 1, H, H), 1.5),
        torch.ones(2, 1, H, H, dtype=torch.bool),
        beta=1.0,
        u=0.0,
    )
    assert bd["loss_reg"] == pytest.approx(0.5 * float(raw.detach()), rel=1e-5)


def test_u_ramp_plumbed_from_step_ctx() -> None:
    """정정 A-34 — criterion 이 u 의 유일 소비자. StepCtx.u 가 그대로 쓰인다."""
    crit = _crit()
    out = _outputs(seed=59)
    tgt = _chl_targets()
    _, bd0 = crit(out, tgt, _ctx(u=0.0))
    _, bd1 = crit(out, tgt, _ctx(u=1.0))
    assert bd0["u_ramp"] == 0.0 and bd1["u_ramp"] == 1.0
    assert bd0["loss_reg"] != pytest.approx(bd1["loss_reg"], rel=1e-4)


def test_u_source_auto_derives_from_epoch() -> None:
    crit = _crit(u_source="auto")
    out = _outputs(seed=61)
    tgt = _chl_targets()
    _, bd_warm = crit(out, tgt, _ctx(epoch=0, total=100))
    _, bd_mid = crit(out, tgt, _ctx(epoch=40, total=100))
    _, bd_late = crit(out, tgt, _ctx(epoch=90, total=100))
    assert bd_warm["u_ramp"] == 0.0
    assert bd_mid["u_ramp"] == pytest.approx(0.5)
    assert bd_late["u_ramp"] == 1.0


def test_u_at_matches_u_ramp_primitive() -> None:
    from bloomnet.losses.regression import u_ramp as prim

    crit = _crit(u_warm_frac=0.3, u_ramp_frac=0.2)
    for e in (0, 24, 32, 40, 79):
        assert crit.u_at(e, 80) == prim(e, 80, warm_frac=0.3, ramp_frac=0.2)


def test_u_zero_reduces_to_smooth_l1_exactly() -> None:
    """u=0 에서 사업문서 형태(SmoothL1 δ=1)로 정확히 환원된다 (05 §2.5.2 근거 1)."""
    crit = _crit()
    b, hs = 1, H
    pred = torch.full((b, 1, hs, hs), 2.0)
    out = {
        OUT_SEG: torch.randn(b, K12, hs, hs),
        OUT_CHL: pred,
        OUT_LOGVAR: torch.full((b, 1, hs, hs), 3.0),
    }
    tgt = {
        "y_seg": _structured_mask(b=b),
        "y_chl": torch.full((b, 1, hs, hs), 1.5),
        "y_chl_valid": torch.ones(b, 1, hs, hs, dtype=torch.bool),
    }
    _, bd = crit(out, tgt, _ctx(u=0.0))
    expect = 0.5 * (0.5 * 0.5 ** 2)  # λ_reg · SmoothL1(0.5)
    assert bd["loss_reg"] == pytest.approx(expect, abs=1e-6)


def test_log_var_gets_zero_grad_during_warmup() -> None:
    """u=0 이면 log_var 는 손실에 안 들어가지만 graph 는 연결돼 있어야 한다(DDP)."""
    crit = _crit()
    out = _outputs(seed=67)
    loss, _ = crit(out, _chl_targets(), _ctx(u=0.0))
    loss.backward()
    g = out[OUT_LOGVAR].grad
    assert g is not None and float(g.abs().sum()) == 0.0


def test_x14_double_log_guard_via_debug_assert() -> None:
    """★X-14 — 원단위 mg/m³ 를 넣으면 debug_assert 경로에서 잡힌다."""
    crit = _crit(debug_assert=True)
    b = 1
    out = _outputs(b=b, seed=71)
    tgt = {
        "y_seg": _structured_mask(b=b),
        "y_chl": torch.full((b, 1, H, H), 90.0),  # 원단위 → log1p 공간이 아니다
        "y_chl_valid": torch.ones(b, 1, H, H, dtype=torch.bool),
    }
    with pytest.raises(AssertionError, match="log1p"):
        crit(out, tgt, _ctx())


# ═════════════════════════════════════════════════════════════════════════════
# 11. SIAM (기본 OFF, 코드 경로만)
# ═════════════════════════════════════════════════════════════════════════════
def test_siam_off_by_default() -> None:
    crit = _crit()
    assert crit.lambda_siam == 0.0
    out = _outputs(seed=73)
    out[OUT_SIAM["s3"]] = torch.randn(2, 4, 8, 8, requires_grad=True)
    tgt = {"y_seg": _structured_mask(), TEACHER_KEY: {"s3": torch.randn(2, 4, 8, 8)}}
    _, bd = crit(out, tgt, _ctx())
    assert bd["loss_siam"] == 0.0


def test_siam_path_runs_when_enabled() -> None:
    from bloomnet.losses.distill import cwd_loss

    crit = _crit(lambda_siam=1.0)
    out = _outputs(seed=79)
    s3 = torch.randn(2, 4, 8, 8, requires_grad=True)
    out[OUT_SIAM["s3"]] = s3
    t3 = torch.randn(2, 4, 8, 8)
    tgt = {"y_seg": _structured_mask(), TEACHER_KEY: {"s3": t3}}
    loss, bd = crit(out, tgt, _ctx())
    expect = SIAM_PAIR_WEIGHTS["s3"] * float(cwd_loss(s3, t3, T=4.0).detach())
    assert bd["loss_siam"] == pytest.approx(expect, rel=1e-5)
    loss.backward()
    assert torch.isfinite(s3.grad).all() and float(s3.grad.abs().sum()) > 0.0


def test_siam_zero_without_teacher() -> None:
    crit = _crit(lambda_siam=1.0)
    out = _outputs(seed=83)
    out[OUT_SIAM["s3"]] = torch.randn(2, 4, 8, 8)
    _, bd = crit(out, {"y_seg": _structured_mask()}, _ctx())
    assert bd["loss_siam"] == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 12. 기타 계약
# ═════════════════════════════════════════════════════════════════════════════
def test_class_weight_is_buffer_and_applied() -> None:
    cw = torch.linspace(0.5, 2.0, K12)
    crit = _crit(class_weight=cw)
    assert "class_weight" in dict(crit.named_buffers())
    y = _structured_mask()
    out = _outputs(seed=89)
    _, bd_w = crit(out, {"y_seg": y}, _ctx())
    _, bd_0 = _crit()(out, {"y_seg": y}, _ctx())
    assert bd_w["loss_ohem"] != pytest.approx(bd_0["loss_ohem"], rel=1e-4)
    up = F.interpolate(out[OUT_SEG].float(), size=(H, H), mode="bilinear", align_corners=False)
    ref = float(
        ohem_ce(up, y, ignore_index=IGNORE, thresh=0.7, keep_frac=0.0625, class_weight=cw).detach()
    )
    assert bd_w["loss_ohem"] == pytest.approx(ref, rel=1e-6)


def test_fp32_forced_under_autocast() -> None:
    """규칙 N7 — autocast(bf16) 안에서도 criterion 은 fp32 로 계산한다."""
    torch.manual_seed(0)
    net = TinyNet()
    crit = _crit()
    x = torch.randn(2, 3, H, H)
    tgt = {"y_seg": _structured_mask()}
    loss_ref, _ = crit(net(x), tgt, _ctx())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = {k: v for k, v in net(x).items()}
        loss_ac, _ = crit(out, tgt, _ctx())
    assert loss_ac.dtype == torch.float32
    # 모델 출력이 bf16 이라 값 자체는 달라지지만, 손실 계산은 fp32 로 수행된다.
    assert torch.isfinite(loss_ac)


def test_upsampled_logits_shared_across_terms() -> None:
    """정정 A-32(a) — 로짓 업샘플이 항마다가 아니라 1회만 일어난다.

    ``F.interpolate`` 호출 횟수를 세어 고정한다: seg 로짓 1회 + edge 로짓 1~2회
    (boundary_bce, bas_loss). ohem/dice/bas 가 각자 seg 를 올리면 4회가 된다.
    """
    calls = {"n": 0}
    real = F.interpolate

    def counting(inp, *a, **kw):  # type: ignore[no-untyped-def]
        if inp.shape[1] == K12:  # seg 로짓만 센다
            calls["n"] += 1
        return real(inp, *a, **kw)

    crit = _crit(lambda_aux={})
    out = _outputs(aux=(), seed=97)
    try:
        F.interpolate = counting  # type: ignore[assignment]
        crit(out, {"y_seg": _structured_mask()}, _ctx())
    finally:
        F.interpolate = real  # type: ignore[assignment]
    assert calls["n"] == 1, f"seg 로짓 업샘플 {calls['n']}회 (기대 1회)"


def test_deterministic() -> None:
    crit = _crit()
    out = _outputs(seed=101)
    tgt = _chl_targets()
    a = crit(out, tgt, _ctx(u=0.5))[1]
    b = crit(out, tgt, _ctx(u=0.5))[1]
    assert a == b


def test_reproduces_spec_magnitude_check_at_init() -> None:
    """05 §1.3 (정정 B-20) 초기 항별 크기 검산을 end-to-end 로 재현한다.

    균일 로짓(전부 0) → CE = ln12 = 2.4849, OHEM 은 p_gt = 1/12 < thresh 라 전 픽셀 유지
    → 평균 CE 로 환원. 경계는 p = 0.03112 를 갖는 타깃을 주입해 ×20 = 0.836 을 확인.
    합계가 05 §1.3 의 6.19 근방(dice 는 데이터 의존)이어야 한다.
    """
    b, hs = 1, 64
    y = _structured_mask(b=b, h=hs, w=hs)
    gt = torch.zeros(b, 1, hs // 4, hs // 4)
    n_edge = gt[0, 0].numel()
    gt.view(-1)[: max(1, round(0.03112 * n_edge))] = 1.0
    valid = torch.ones_like(gt, dtype=torch.bool)

    out = {
        OUT_SEG: torch.zeros(b, K12, hs // 4, hs // 4),
        OUT_EDGE: torch.zeros(b, 1, hs // 4, hs // 4),
        OUT_AUX["enc_s8"]: torch.zeros(b, K12, hs // 8, hs // 8),
        OUT_AUX["enc_s16"]: torch.zeros(b, K12, hs // 16, hs // 16),
        OUT_AUX["enc_s32"]: torch.zeros(b, K12, hs // 32, hs // 32),
    }
    crit = _crit()
    _, bd = crit(out, {"y_seg": y, "y_edge": gt, "y_edge_valid": valid}, _ctx())

    ln12 = math.log(K12)
    assert bd["loss_ohem"] == pytest.approx(ln12, rel=1e-5)  # 2.4849
    aux_sum = sum(v for k, v in bd.items() if k.startswith("loss_aux_"))
    assert aux_sum == pytest.approx((0.2 + 0.4 + 0.4) * ln12, rel=1e-5)  # 2.4849
    p = float((gt * valid).sum() / valid.sum())
    assert bd["loss_edge"] == pytest.approx(20.0 * math.log(2.0) * 2 * p * (1 - p), rel=1e-5)
    assert bd["loss_edge"] == pytest.approx(0.836, abs=0.06)
    assert bd["loss_bas"] == 0.0  # sigmoid(0) = 0.5 < τ = 0.8
    assert bd["loss_reg"] == 0.0  # 라벨 부재
    assert 5.3 < bd["loss_total"] < 7.0, bd["loss_total"]  # 05 §1.3 합계 ≈ 6.19


def test_owned_files_respect_import_levels() -> None:
    """T01 선행 자가검사 — boundary_loss(L2) / criterion(L3) 의 bloomnet import 레벨."""
    import ast

    level = {
        "bloomnet.constants": -1,
        "bloomnet.version": -1,
        "bloomnet.data.boundary": 0,
        "bloomnet.config": 1,
        "bloomnet.losses.seg": 1,
        "bloomnet.losses.regression": 1,
        "bloomnet.losses.distill": 1,
        "bloomnet.losses.boundary_loss": 2,
    }
    for rel, own in (("losses/boundary_loss.py", 2), ("losses/criterion.py", 3)):
        src = (_ROOT / "bloomnet" / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module)
            elif isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
        for m in sorted(x for x in mods if x.startswith("bloomnet")):
            assert m in level, f"{rel}: 레벨 미상 import {m}"
            assert level[m] < own, f"{rel}(L{own}) 가 {m}(L{level[m]}) 를 import 한다"


def test_build_criterion_from_config() -> None:
    from bloomnet.config import default_config
    from bloomnet.losses.criterion import build_criterion

    cfg = default_config()
    crit = build_criterion(cfg)
    assert crit.num_classes == cfg.data.num_classes
    assert crit.lambda_seg == 1.0 and crit.lambda_reg == 0.5
    assert crit.unc_clamp == (-7.0, 7.0)
    assert crit.boundary_stride == 4
    assert set(crit.lambda_aux) == set(cfg.model.aux_taps)
    loss, bd = crit(_outputs(), {"y_seg": _structured_mask()}, _ctx())
    assert torch.isfinite(loss)
