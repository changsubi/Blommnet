"""BMEF 테스트 — 03 §10 의 T1~T17 전부 + 메모리 회귀 (06 §5.2 T14).

헌법 C-5.1/C-5.2: **CPU 전용**, B<=2, 64x64 이하 소형 텐서, GPU 금지.
공통 fixture: B=2, C=32, H=W=64, stage=1, seed 0, `model.eval()` (03 §10).
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # cwd 와 무관하게 bloomnet 패키지를 찾게 한다
    sys.path.insert(0, str(_ROOT))

from bloomnet.constants import PATHS as CONST_PATHS  # noqa: E402
from bloomnet.modules.bmef import (  # noqa: E402
    EPS,
    FEEDBACK_PATHS,
    NEG,
    PATHS,
    BMEF,
    BMEFOutput,
    identity_fuse,
)

# ---------------------------------------------------------------- 공통 설정
B, C, H, W = 2, 32, 64, 64
STAGE_CHANNELS = {1: 32, 2: 64, 3: 160, 4: 320}
EXPECTED_PARAMS = {1: 5135, 2: 16399, 3: 90907, 4: 347567}

# g_pol 은 **full resolution** 계약이라 stage1 에서 feature 의 4배다.
# 헌법 C-5.1 의 "64x64 이하" 를 g_pol 쪽에서도 지키기 위해 g_pol 테스트는 feature 를 GH=16 으로 둔다.
GH = 16

COMBOS = {
    "rgb": (1, 0, 0),
    "rgb+spec": (1, 1, 0),
    "spec": (0, 1, 0),
    "all": (1, 1, 1),
    "none": (0, 0, 0),
}
NONEMPTY = {k: v for k, v in COMBOS.items() if any(v)}


@pytest.fixture(autouse=True)
def _no_cuda() -> None:
    """헌법 C-5.2 — 어떤 테스트도 GPU 를 건드리지 않는다."""
    assert torch.cuda.is_available() is False, "GPU 금지 (헌법 C-5.2)"


def make_bmef(**kw) -> BMEF:
    torch.manual_seed(0)
    m = BMEF(kw.pop("channels", C), kw.pop("stage_idx", 1), **kw)
    m.eval()
    return m


def pres_of(flags, b: int = B) -> torch.Tensor:
    return torch.tensor([list(flags)] * b, dtype=torch.bool)


def same_feats(x: torch.Tensor):
    return {p: x.clone() for p in PATHS}


def indep_feats(b=B, c=C, h=H, w=W, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return {p: torch.randn(b, c, h, w, generator=g) for p in PATHS}


# ===================================================================== T17
def test_t17_param_count_regression():
    """03 §7.1 / 06 §6.1. 값이 바뀌면 설계 문서 갱신을 강제한다."""
    total = 0
    for stage, ch in STAGE_CHANNELS.items():
        m = BMEF(ch, stage)
        n = sum(p.numel() for p in m.parameters())
        assert n == EXPECTED_PARAMS[stage], f"stage {stage}: {n} != {EXPECTED_PARAMS[stage]}"
        total += n
    assert total == 460_008


def test_param_count_invariant_under_gn():
    """gn8 스위치(정정 A-37/B-1)가 파라미터 수를 보존해야 한다 (T12 파라메트릭)."""
    for stage, ch in STAGE_CHANNELS.items():
        n_bn = sum(p.numel() for p in BMEF(ch, stage, norm_layer="bn").parameters())
        n_gn = sum(p.numel() for p in BMEF(ch, stage, norm_layer="gn").parameters())
        assert n_bn == n_gn == EXPECTED_PARAMS[stage]
    # gcd(8, C) 규약: BMEF 의 norm 폭은 전부 8의 배수이므로 num_groups == 8
    m = BMEF(160, 3, norm_layer="gn")
    assert m.mean_norm["rgb"].num_groups == 8
    assert m.prec_norm["phys"].num_groups == 8


def test_norm_layer_delegates_to_common_build_norm():
    """gn8 정의(정정 B-1)를 복제하지 않고 modules/common.py 단일 빌더에 위임한다."""
    from bloomnet.modules.common import build_norm

    ref = build_norm("gn8", 160)
    got = BMEF(160, 3, norm_layer="gn").mean_norm["spec"]
    assert type(got) is type(ref)
    assert (got.num_groups, got.num_channels) == (ref.num_groups, ref.num_channels)
    # 06 §3.3.6 어휘 {"bn","gn"} 와 02 §1.2 어휘 {"bn","syncbn","gn8"} 를 모두 받는다
    assert type(BMEF(C, 1, norm_layer="gn8").prec_norm["rgb"]) is torch.nn.GroupNorm
    assert type(BMEF(C, 1, norm_layer="bn").prec_norm["rgb"]) is torch.nn.BatchNorm2d


def test_param_count_analytic_formula():
    """path 당 C^2 + 15C + 1, SE 2·C·Cr + Cr + C, fb 2C(stage<4), 기타 4."""
    for stage, c in STAGE_CHANNELS.items():
        cr = max(8, c // 8)
        expect = (3 * c * c + 45 * c + 3) + (2 * c * cr + cr + c) + (2 * c if stage < 4 else 0) + 4
        assert expect == EXPECTED_PARAMS[stage]


# ===================================================================== T1
def test_t1_shape_and_finite_all_combos():
    m = make_bmef()
    x = torch.randn(B, C, H, W)
    for name, flags in COMBOS.items():
        out = m(same_feats(x), pres_of(flags), return_weights=True)
        assert isinstance(out, BMEFOutput)
        assert out.fused.shape == (B, C, H, W), name
        assert out.vacuity.shape == (B, 1, H, W), name
        assert out.log_tau.shape == (B, C, H, W), name
        assert out.weights is not None and out.weights.shape == (B, 3, H, W), name
        for t in (out.fused, out.vacuity, out.log_tau, out.weights):
            assert torch.isfinite(t).all(), name
        # stage 1 이므로 spec/phys 피드백이 생성된다
        assert set(out.feedback) == set(FEEDBACK_PATHS), name
        for v in out.feedback.values():
            assert v.shape == (B, C, H, W) and torch.isfinite(v).all()


def test_vacuity_strictly_in_unit_interval():
    m = make_bmef()
    out_all = m(same_feats(torch.randn(B, C, H, W)), pres_of((1, 1, 1)))
    assert (out_all.vacuity > 0).all() and (out_all.vacuity < 1).all()


# ===================================================================== T2
def test_t2_same_signal_subset_invariance():
    """핵심. 헌법 C-3 '모달 부재가 융합 로짓의 스케일을 바꾸면 안 된다'의 형식화."""
    m = make_bmef()
    x = torch.randn(B, C, H, W)
    ref = m(same_feats(x), pres_of((1, 1, 1))).fused
    for name, flags in NONEMPTY.items():
        out = m(same_feats(x), pres_of(flags))
        diff = (out.fused - ref).abs().max().item()
        assert diff < 1e-5, f"{name}: {diff:.3e}"  # 03 부록 B 실측 2.384e-07


def test_l2_mode_trades_t2_for_t7_exactly_as_documented():
    """`norm_mode='l2'` 는 T2(1차 모멘트 불변)를 **버리고** 2차 모멘트를 보존한다.

    03 §3.3: `mu_f <- mu_f / (||w||_2 + eps)`. 동일신호에서 `||w||_2 = 1/sqrt(|S|)` 이므로
    `fused_l2(S) == sqrt(|S|) · fused_l1(S)` 이고, sup-norm 유계가 `sqrt(3)` 배 느슨해진다.
    03 §12-U-2 의 "T2/T7 이 두 모드 모두 통과" 는 두 성질이 상호배타이므로 성립할 수 없다 —
    **필수 게이트인 T2 는 기본값 l1 에 대한 것**이고, l2 는 M-12 fallback 이다.
    """
    m1 = make_bmef(norm_mode="l1")
    m2 = make_bmef(norm_mode="l2")
    x = torch.randn(B, C, H, W)
    for flags, n in (((1, 0, 0), 1), ((1, 1, 0), 2), ((1, 1, 1), 3)):
        f1 = m1(same_feats(x), pres_of(flags)).fused
        f2 = m2(same_feats(x), pres_of(flags)).fused
        assert torch.allclose(f2, math.sqrt(n) * f1, atol=1e-4), n
    # sup-norm 유계 완화의 크기: sqrt(3) 배 이내
    feats = indep_feats()
    out = m2(feats, pres_of((1, 1, 1)))
    with torch.no_grad():
        mus = torch.stack([m2.mean_norm[p](m2.mean_conv[p](feats[p])) for p in PATHS], 0)
    assert out.fused.abs().max() <= math.sqrt(3.0) * mus.abs().max() + 1e-4


def test_l2_single_modality_is_still_passthrough():
    """|S| = 1 이면 ||w||_2 = 1 이라 l2 도 정확한 passthrough."""
    m = make_bmef(norm_mode="l2")
    feats = indep_feats()
    out = m(feats, pres_of((0, 1, 0)))
    with torch.no_grad():
        mu = m.mean_norm["spec"](m.mean_conv["spec"](feats["spec"]))
        gap = mu.mean(dim=(2, 3), keepdim=True)
        expect = mu * (2.0 * torch.sigmoid(m.se2(F.relu(m.se1(gap)))))
    assert torch.allclose(out.fused, expect, atol=1e-5)


# ===================================================================== T3
def test_t3_single_modality_passthrough():
    m = make_bmef()
    x = torch.randn(B, C, H, W)
    feats = {p: torch.randn(B, C, H, W) for p in PATHS}
    feats["spec"] = x
    out = m(feats, pres_of((0, 1, 0)), return_weights=True)
    with torch.no_grad():
        mu = m.mean_norm["spec"](m.mean_conv["spec"](x))
        gap = mu.mean(dim=(2, 3), keepdim=True)
        gain = 2.0 * torch.sigmoid(m.se2(F.relu(m.se1(gap))))
        expect = mu * gain
    assert torch.allclose(out.fused, expect, atol=1e-6)
    w = out.weights
    assert torch.allclose(w[:, 1], torch.ones_like(w[:, 1]), atol=1e-6)
    assert (w[:, 0] == 0.0).all() and (w[:, 2] == 0.0).all()


def test_t3_passthrough_is_independent_of_other_path_content():
    """absent path 의 텐서 내용이 결과에 절대 새지 않는다."""
    m = make_bmef()
    x = torch.randn(B, C, H, W)
    a = {"rgb": torch.zeros(B, C, H, W), "spec": x, "phys": torch.zeros(B, C, H, W)}
    b = {"rgb": 1e3 * torch.randn(B, C, H, W), "spec": x, "phys": 1e3 * torch.randn(B, C, H, W)}
    o1 = m(a, pres_of((0, 1, 0)))
    o2 = m(b, pres_of((0, 1, 0)))
    assert torch.allclose(o1.fused, o2.fused, atol=1e-6)


# ===================================================================== T4
def test_t4_weights_sum_to_one_and_absent_exactly_zero():
    m = make_bmef()
    feats = indep_feats()
    for name, flags in NONEMPTY.items():
        out = m(feats, pres_of(flags), return_weights=True)
        s = out.weights.sum(dim=1)
        assert s.min().item() == pytest.approx(1.0, abs=1e-5), name
        assert s.max().item() == pytest.approx(1.0, abs=1e-5), name
        for k, f in enumerate(flags):
            if not f:
                assert (out.weights[:, k] == 0.0).all(), f"{name}/{PATHS[k]}"


def test_t4_weights_nonnegative():
    m = make_bmef()
    out = m(indep_feats(), pres_of((1, 1, 1)), return_weights=True)
    assert (out.weights >= 0.0).all()


# ===================================================================== T5
def test_t5_all_absent_is_safe():
    m = make_bmef()
    out = m(indep_feats(), pres_of((0, 0, 0)), return_weights=True)
    assert out.fused.abs().max().item() == 0.0
    assert out.vacuity.mean().item() > 0.99  # 03 부록 B 실측 0.99909
    assert not torch.isnan(out.fused).any()
    assert torch.isfinite(out.log_tau).all()
    # log_tau 는 하한으로 포화 (-(R+3) = -7.0)
    assert out.log_tau.min().item() == pytest.approx(-7.0, abs=1e-5)
    assert out.log_tau.max().item() == pytest.approx(-7.0, abs=1e-5)
    assert (out.weights == 0.0).all()


def test_t5_negative_clamp_trap_reproduces_nan():
    """`exp(A - amax)` 에서 `clamp(max=0)` 를 빼면 전부-결측이 NaN 이 된다 (06 T14 negative)."""
    a = torch.zeros(B, 1, 8, 8)
    d = torch.zeros(B, 1, 1, 1)  # 전부 결측
    amax = a.masked_fill(torch.ones_like(d, dtype=torch.bool), NEG)
    bad = torch.exp(a - amax) * d  # clamp 없음 -> exp(1e4) = inf, inf*0 = NaN
    assert torch.isinf(torch.exp(a - amax)).all()
    assert torch.isnan(bad).all()
    good = torch.exp((a - amax).clamp(max=0.0)) * d
    assert (good == 0.0).all() and torch.isfinite(good).all()


def test_t5_backward_from_all_absent_has_no_nan():
    m = make_bmef()
    feats = {p: torch.randn(B, C, H, W, requires_grad=True) for p in PATHS}
    out = m(feats, pres_of((0, 0, 0)))
    (out.fused.sum() + out.vacuity.sum() + out.log_tau.sum()).backward()
    for p in PATHS:
        g = feats[p].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum().item() == 0.0
    for name, par in m.named_parameters():
        if par.grad is not None:
            assert torch.isfinite(par.grad).all(), name


# ===================================================================== T6
def test_t6_convex_hull_bound_l1():
    m = make_bmef()
    feats = indep_feats()
    out = m(feats, pres_of((1, 1, 1)))
    with torch.no_grad():
        mus = torch.stack([m.mean_norm[p](m.mean_conv[p](feats[p])) for p in PATHS], 0)
        lo, hi = mus.min(dim=0).values, mus.max(dim=0).values
        # SE gain 이 init 시 정확히 1 이므로 fused == mu_f. 여유는 gain ∈ (0,2) 만큼만.
        assert (out.fused >= lo - 1e-5).all()
        assert (out.fused <= hi + 1e-5).all()
        assert out.fused.abs().max() <= 2.0 * mus.abs().max() + 1e-5


# ===================================================================== T7
def test_t7_noise_attenuation_regression_lock():
    """정정 B-11 / 06 M-12: **문서화된 동작**의 회귀 고정 (버그가 아니다)."""
    m = make_bmef()
    feats = indep_feats(seed=7)
    for flags, n in (((1, 0, 0), 1), ((1, 1, 0), 2), ((1, 1, 1), 3)):
        out = m(feats, pres_of(flags))
        rms = out.fused.pow(2).mean().sqrt().item()
        target = 1.0 / math.sqrt(n)
        assert abs(rms - target) / target < 0.03, f"|S|={n}: rms={rms:.4f} vs {target:.4f}"


def test_t7_l2_mode_preserves_second_moment():
    """U-2 fallback: norm_mode='l2' 는 |S| 에 무관하게 RMS 를 보존한다."""
    m = make_bmef(norm_mode="l2")
    feats = indep_feats(seed=7)
    rms = []
    for flags in ((1, 0, 0), (1, 1, 0), (1, 1, 1)):
        rms.append(m(feats, pres_of(flags)).fused.pow(2).mean().sqrt().item())
    for r in rms:
        assert abs(r - rms[0]) / rms[0] < 0.03, rms


# ===================================================================== T8
def test_t8_backward_all_combos():
    m = make_bmef()
    for name, flags in COMBOS.items():
        feats = {p: torch.randn(B, C, H, W, requires_grad=True) for p in PATHS}
        m.zero_grad(set_to_none=True)
        out = m(feats, pres_of(flags))
        out.fused.sum().backward()
        for k, p in enumerate(PATHS):
            g = feats[p].grad
            assert g is not None and torch.isfinite(g).all(), f"{name}/{p}"
            if flags[k]:
                assert g.abs().sum().item() > 0, f"{name}/{p} present 인데 grad 0"
            else:
                assert g.abs().sum().item() == 0.0, f"{name}/{p} absent 인데 grad != 0"
        for pname, par in m.named_parameters():
            if par.grad is not None:
                assert torch.isfinite(par.grad).all(), f"{name}/{pname}"


def test_backward_reaches_precision_and_kappa_params():
    """prec_pw zero-init 이어도 gradient 는 흐른다 (dead-parameter 아님)."""
    m = make_bmef()
    m.train()
    feats = indep_feats(h=GH, w=GH)
    g_pol = torch.rand(B, 1, GH * 4, GH * 4)
    out = m(feats, pres_of((1, 1, 1)), g_pol, prec_ramp=1.0)
    out.fused.pow(2).sum().backward()
    assert m.prec_pw["rgb"].weight.grad.abs().sum().item() > 0
    assert m.p_raw["spec"].grad.abs().sum().item() > 0
    assert m.kappa_raw.grad.abs().sum().item() > 0
    assert m.se2.weight.grad.abs().sum().item() > 0


# ===================================================================== T9
def test_t9_gradient_reach_is_full_unlike_max_reduce():
    m = make_bmef()
    feats = {p: torch.randn(B, C, H, W, requires_grad=True) for p in PATHS}
    m(feats, pres_of((1, 1, 1))).fused.sum().backward()
    for p in PATHS:
        frac = (feats[p].grad != 0).float().mean().item()
        assert frac == 1.0, f"{p}: {frac}"
    # 대조군: 원소별 max-reduce 는 대략 1/3 만 gradient 를 받는다 (CMNeXt Self-Query Hub)
    ctrl = {p: torch.randn(B, C, H, W, requires_grad=True) for p in PATHS}
    torch.maximum(torch.maximum(ctrl["rgb"], ctrl["spec"]), ctrl["phys"]).sum().backward()
    fr = [(ctrl[p].grad != 0).float().mean().item() for p in PATHS]
    assert all(0.30 < f < 0.37 for f in fr), fr


# ===================================================================== T10
def test_t10_gpol_monotonicity():
    m = make_bmef()
    feats = indep_feats(h=GH, w=GH)
    ws = []
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        gp = torch.full((B, 1, GH * 4, GH * 4), float(g))
        out = m(feats, pres_of((1, 1, 1)), gp, return_weights=True)
        ws.append(out.weights.mean(dim=(0, 2, 3)).tolist())
    w_rgb = [row[0] for row in ws]
    assert all(w_rgb[i] > w_rgb[i + 1] for i in range(len(w_rgb) - 1)), w_rgb
    assert w_rgb[0] == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert w_rgb[-1] < 0.06  # 03 §9.2 손검산 0.040960
    # phys 는 kappa≈0 이라 오히려 증가한다 (glint 는 물리 경로를 억제하지 않는다)
    w_phys = [row[2] for row in ws]
    assert all(w_phys[i] < w_phys[i + 1] for i in range(len(w_phys) - 1)), w_phys


def test_t10_gpol_hand_computation():
    """03 §9.2 5번 손검산: g=0.5, kappa=(1.0, 0.5, 0.018) 권고 init."""
    m = make_bmef()
    feats = indep_feats(h=GH, w=GH)
    gp = torch.full((B, 1, GH * 4, GH * 4), 0.5)
    out = m(feats, pres_of((1, 1, 1)), gp, return_weights=True)
    r = 0.05 + 0.95 * 0.5
    log_r = math.log(r)
    kappa = m.kappa.tolist()
    a = [math.exp(k * log_r) for k in kappa]
    tot = sum(a)
    expect = [v / tot for v in a]
    got = out.weights.mean(dim=(0, 2, 3)).tolist()
    for e, g in zip(expect, got):
        assert g == pytest.approx(e, abs=1e-5), (expect, got)
    # vacuity = sigmoid(-log Σ) at init
    v_expect = 1.0 / (1.0 + tot)
    assert out.vacuity.mean().item() == pytest.approx(v_expect, abs=1e-5)


def test_gpol_pool_max_is_conservative():
    """gpol_pool='max' 는 셀 안의 최악값을 쓰므로 area 보다 w_rgb 가 작거나 같다."""
    feats = indep_feats(h=GH, w=GH)
    g = torch.zeros(B, 1, GH * 4, GH * 4)
    g[..., ::4, ::4] = 1.0  # 산발 speckle: 셀당 1/16 만 glint
    w = {}
    for pool in ("area", "max"):
        m = make_bmef(gpol_pool=pool)
        w[pool] = m(feats, pres_of((1, 1, 1)), g, return_weights=True).weights[:, 0].mean().item()
    assert w["max"] < w["area"]


def test_gpol_wrong_resolution_raises():
    m = make_bmef()
    feats = indep_feats()
    with pytest.raises(ValueError, match="full resolution"):
        m(feats, pres_of((1, 1, 1)), torch.rand(B, 1, H, W))
    with pytest.raises(ValueError):
        m(feats, pres_of((1, 1, 1)), torch.rand(B, 3, GH * 4, GH * 4))


def test_gpol_stride_per_stage():
    for stage, ch in STAGE_CHANNELS.items():
        m = BMEF(ch, stage).eval()
        assert m.gpol_stride == 2 ** (stage + 1)
        h = 8
        feats = {p: torch.randn(1, ch, h, h) for p in PATHS}
        gp = torch.rand(1, 1, h * m.gpol_stride, h * m.gpol_stride)
        out = m(feats, torch.ones(1, 3, dtype=torch.bool), gp)
        assert out.fused.shape == (1, ch, h, h)


# ===================================================================== T11
def test_t11_gpol_none_equals_zeros():
    m = make_bmef()
    feats = indep_feats(h=GH, w=GH)
    o_none = m(feats, pres_of((1, 1, 1)), None, return_weights=True)
    o_zero = m(feats, pres_of((1, 1, 1)), torch.zeros(B, 1, GH * 4, GH * 4), return_weights=True)
    assert torch.allclose(o_none.fused, o_zero.fused, atol=1e-6)
    assert torch.allclose(o_none.weights, o_zero.weights, atol=1e-6)
    assert torch.allclose(o_none.vacuity, o_zero.vacuity, atol=1e-6)


# ===================================================================== T12
@pytest.mark.parametrize("b,h", [(1, 64), (2, 64), (2, 128), (3, 32)])
def test_t12_shape_agnostic(b, h):
    m = make_bmef()
    x = torch.randn(b, C, h, h)
    ref = m(same_feats(x), pres_of((1, 1, 1), b)).fused
    for flags in NONEMPTY.values():
        out = m(same_feats(x), pres_of(flags, b), return_weights=True)
        assert (out.fused - ref).abs().max().item() < 1e-5  # T2
        s = out.weights.sum(dim=1)                            # T4
        assert s.min().item() == pytest.approx(1.0, abs=1e-5)
        assert s.max().item() == pytest.approx(1.0, abs=1e-5)


# ===================================================================== T13
def test_t13_prec_ramp_zero_gives_uniform_weights():
    m = make_bmef()
    # 학습된 정밀도가 실제로 편차를 만들도록 prec_pw 를 랜덤화
    with torch.no_grad():
        for p in PATHS:
            m.prec_pw[p].weight.normal_(0, 0.5)
            m.prec_pw[p].bias.normal_(0, 0.5)
    feats = indep_feats()
    w0 = m(feats, pres_of((1, 1, 1)), prec_ramp=0.0, return_weights=True).weights
    assert torch.allclose(w0, torch.full_like(w0, 1.0 / 3.0), atol=1e-6)
    w1 = m(feats, pres_of((1, 1, 1)), prec_ramp=1.0, return_weights=True).weights
    assert (w1 - w0).abs().max().item() > 1e-3  # ramp=1 과는 확실히 다르다


def test_t13_physical_reliability_is_not_ramped():
    """03 §8.3: g_pol 은 학습되지 않은 물리량이라 ramp=0 에서도 활성이다."""
    m = make_bmef()
    feats = indep_feats(h=GH, w=GH)
    gp = torch.full((B, 1, GH * 4, GH * 4), 1.0)
    w = m(feats, pres_of((1, 1, 1)), gp, prec_ramp=0.0, return_weights=True).weights
    assert w[:, 0].mean().item() < 0.06


# ===================================================================== T14
def test_t14_init_is_uniform_mean_of_normalized_paths():
    m = make_bmef()
    x = torch.randn(B, C, H, W)
    feats = {p: torch.randn(B, C, H, W) for p in PATHS}
    feats["rgb"] = x
    out = m(feats, pres_of((1, 0, 0)))
    with torch.no_grad():
        expect = m.mean_norm["rgb"](m.mean_conv["rgb"](x))
    assert torch.allclose(out.fused, expect, atol=1e-5)
    # mean conv 는 항등 초기화 -> eval BN(running 0/1) 하에서 거의 입력 그대로
    assert torch.allclose(out.fused, x, atol=1e-3)


def test_init_table_matches_spec():
    """03 §8.2 초기화표."""
    m = BMEF(C, 1)
    eye = torch.eye(C).view(C, C, 1, 1)
    for p in PATHS:
        assert torch.equal(m.mean_conv[p].weight, eye)
        assert m.mean_conv[p].bias is None
        assert torch.count_nonzero(m.prec_pw[p].weight).item() == 0
        assert torch.count_nonzero(m.prec_pw[p].bias).item() == 0
        assert torch.count_nonzero(m.p_raw[p]).item() == 0
        assert m.prec_dw[p].groups == C and m.prec_dw[p].bias is None
        assert torch.count_nonzero(m.prec_dw[p].weight).item() > 0  # kaiming
    assert torch.count_nonzero(m.se2.weight).item() == 0
    assert torch.count_nonzero(m.se2.bias).item() == 0
    assert torch.count_nonzero(m.se1.weight).item() > 0
    assert torch.count_nonzero(m.se1.bias).item() == 0
    assert m.log_tau0.item() == 0.0
    for p in FEEDBACK_PATHS:
        assert torch.count_nonzero(m.fb_gamma[p]).item() == 0


def test_kappa_init_values():
    m = BMEF(C, 1)
    assert torch.allclose(m.kappa, torch.tensor([1.0, 0.5, 0.018]), atol=1e-6)
    # 06 §3.3.6 의 리터럴 [+0.5413, -0.4328, -4.0084]
    assert torch.allclose(
        m.kappa_raw, torch.tensor([0.5413, -0.4328, -4.0084]), atol=1e-3
    )
    assert (m.kappa >= 0).all()  # softplus -> 부호 반전 불가


def test_init_vacuity_equals_one_over_one_plus_n():
    """03 §9.2 6번: init 에서 v = 1/(1+n)."""
    m = make_bmef()
    feats = indep_feats()
    for flags, n in (((1, 0, 0), 1), ((1, 1, 0), 2), ((1, 1, 1), 3)):
        v = m(feats, pres_of(flags)).vacuity.mean().item()
        assert v == pytest.approx(1.0 / (1.0 + n), abs=1e-5)


# ===================================================================== T15
def test_t15_bfloat16_autocast_stays_finite():
    m = make_bmef()
    feats = indep_feats()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m(feats, pres_of((1, 1, 1)), return_weights=True)
        out_none = m(feats, pres_of((0, 0, 0)))
    for t in (out.fused, out.vacuity, out.log_tau, out.weights):
        assert torch.isfinite(t).all()
    assert not torch.isnan(out_none.fused).any()
    assert out_none.fused.abs().max().item() == 0.0
    # dtype 계약: 융합부 fp32 강제 -> 출력은 항상 float32
    assert out.fused.dtype == torch.float32
    assert out.vacuity.dtype == torch.float32
    assert out.log_tau.dtype == torch.float32
    assert out.weights.dtype == torch.float32
    assert all(v.dtype == torch.float32 for v in out.feedback.values())


def test_worst_case_weight_survives_in_fp32():
    """03 §8.1 최악 `w_min = 1/(1+2·e^{2R+2Rp}) = 3.07e-6` 이 0 으로 소멸하지 않는다."""
    m = make_bmef()
    feats = indep_feats(h=16, w=16)
    with torch.no_grad():
        # 최악 배치: **두 path 가 상한(+R+Rp), 한 path 가 하한(-R-Rp)** -> 1/(1+2e^{2R+2Rp})
        for p, sign in (("rgb", 1.0), ("spec", 1.0), ("phys", -1.0)):
            m.prec_pw[p].bias.fill_(sign * 1e3)
            m.p_raw[p].fill_(sign * 1e3)
    out = m(feats, pres_of((1, 1, 1)), return_weights=True)
    assert torch.isfinite(out.fused).all()
    assert out.fused.dtype == torch.float32
    w_min = out.weights[:, 2].min().item()
    expect = 1.0 / (1.0 + 2.0 * math.exp(2 * 4.0 + 2 * 2.0))
    assert w_min == pytest.approx(expect, rel=1e-3), (w_min, expect)
    assert w_min > 0.0  # gradient 가 완전히 끊기지 않는다
    s = out.weights.sum(dim=1)
    assert s.min().item() == pytest.approx(1.0, abs=1e-5)


def test_negative_control_fp16_loses_the_worst_case_weight():
    """왜 fp32 강제인가: `w_min = 3.07e-6` 은 fp16 subnormal 영역이라 유효숫자가 무너진다."""
    w_min = 1.0 / (1.0 + 2.0 * math.exp(12.0))
    got16 = torch.tensor(w_min, dtype=torch.float16).float().item()
    rel = abs(got16 - w_min) / w_min
    assert rel > 5e-3, rel  # 실측 8.9e-3
    assert torch.tensor(w_min, dtype=torch.float32).item() == pytest.approx(w_min, rel=1e-7)


# ===================================================================== T16
def test_t16_per_sample_presence_no_leak():
    m = make_bmef()
    feats = indep_feats(b=2)
    present = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    out = m(feats, present, return_weights=True)

    f0 = {p: feats[p][:1] for p in PATHS}
    f1 = {p: feats[p][1:] for p in PATHS}
    o0 = m(f0, torch.tensor([[1, 1, 1]], dtype=torch.bool))
    o1 = m(f1, torch.tensor([[1, 0, 0]], dtype=torch.bool))
    assert torch.allclose(out.fused[0], o0.fused[0], atol=1e-6)
    assert torch.allclose(out.fused[1], o1.fused[0], atol=1e-6)
    assert out.weights[1, 1].abs().max().item() == 0.0
    assert out.weights[1, 2].abs().max().item() == 0.0
    assert out.weights[0].sum(dim=0).mean().item() == pytest.approx(1.0, abs=1e-5)


def test_per_sample_all_absent_row_is_zero():
    m = make_bmef()
    feats = indep_feats(b=2)
    present = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.bool)
    out = m(feats, present)
    assert out.fused[1].abs().max().item() == 0.0
    assert out.fused[0].abs().max().item() > 0.0
    assert torch.isfinite(out.fused).all()


# ================================================================= feedback
def test_feedback_zero_at_init_and_shapes():
    m = make_bmef()
    out = m(indep_feats(), pres_of((1, 1, 1)))
    assert set(out.feedback) == {"spec", "phys"}
    for v in out.feedback.values():
        assert v.shape == (B, C, H, W)
        assert v.abs().max().item() == 0.0  # gamma zero-init -> 초기엔 완전 private stream


def test_feedback_absent_at_stage4_and_when_disabled():
    m4 = BMEF(320, 4).eval()
    feats = {p: torch.randn(1, 320, 4, 4) for p in PATHS}
    out = m4(feats, torch.ones(1, 3, dtype=torch.bool))
    assert out.feedback == {}
    assert m4.fb_gamma is None

    m1 = BMEF(C, 1, enable_feedback=False).eval()
    n = sum(p.numel() for p in m1.parameters())
    assert n == EXPECTED_PARAMS[1] - 2 * C
    o = m1(indep_feats(), pres_of((1, 1, 1)))
    assert o.feedback == {}


def test_feedback_follows_gamma():
    m = make_bmef()
    with torch.no_grad():
        m.fb_gamma["spec"].fill_(0.5)
    out = m(indep_feats(), pres_of((1, 1, 1)))
    assert torch.allclose(out.feedback["spec"], 0.5 * out.fused, atol=1e-6)
    assert out.feedback["phys"].abs().max().item() == 0.0


# =============================================================== log_tau 범위
def test_log_tau_clamp_range():
    m = make_bmef()
    hi = 4.0 + 3.0 + math.log(3.0)
    with torch.no_grad():
        for p in PATHS:
            m.prec_pw[p].bias.fill_(1e3)  # tanh 로 이미 유계지만 상한 포화를 유도
            m.p_raw[p].fill_(1e3)
    out = m(indep_feats(), pres_of((1, 1, 1)))
    assert out.log_tau.max().item() <= hi + 1e-5
    assert out.log_tau.min().item() >= -7.0 - 1e-5
    assert torch.isfinite(out.log_tau).all()


def test_bounded_ranges_of_internal_quantities():
    """03 §8.1 유계화표: s_m ∈ (-R,R), p_m ∈ (-Rp,Rp), r_base ∈ [eps_r, 1]."""
    m = make_bmef()
    with torch.no_grad():
        x = torch.randn(B, C, H, W)
        # (a) 중간 크기에서는 tanh 가 매끄럽게 유계화한다 (하드 clamp 아님)
        for p in PATHS:
            m.prec_pw[p].bias.fill_(8.0)  # tanh(2) = 0.96403 -> s = 3.856
        _, s = m._evidence("rgb", x, 1.0)
        assert s.max().item() == pytest.approx(4.0 * math.tanh(2.0), abs=1e-4)
        assert s.max().item() < 4.0
        # (b) 극단값에서도 발산하지 않고 R 에서 포화 (fp32 tanh 는 정확히 1.0 으로 포화)
        for p in PATHS:
            m.prec_pw[p].bias.fill_(1e4)
            m.p_raw[p].fill_(1e4)
        _, s = m._evidence("rgb", x, 1.0)
        assert torch.isfinite(s).all() and s.max().item() <= 4.0
        pv = m.chan_logtau_range * torch.tanh(m.p_raw["rgb"] / m.chan_logtau_range)
        assert torch.isfinite(pv).all() and pv.max().item() <= 2.0
        log_r = m._reduce_gpol(torch.ones(B, 1, GH * 4, GH * 4), GH, GH)
        assert log_r.max().item() == pytest.approx(math.log(0.05), abs=1e-6)
        log_r0 = m._reduce_gpol(torch.zeros(B, 1, GH * 4, GH * 4), GH, GH)
        assert log_r0.abs().max().item() == pytest.approx(0.0, abs=1e-7)


# =================================================================== 메모리
class _TensorSizeProbe(TorchDispatchMode):
    """forward 중 생성되는 모든 aten 출력 텐서의 numel 을 관측한다."""

    def __init__(self) -> None:
        self.max_numel = 0
        self.total_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        items = out if isinstance(out, (tuple, list)) else (out,)
        for t in items:
            if isinstance(t, torch.Tensor):
                self.max_numel = max(self.max_numel, t.numel())
                self.total_numel += t.numel()
        return out


def test_no_b3chw_materialization_and_linear_scaling():
    """03 §7.3 / 06 §3.3.6 함정 (2): (B,3,C,H,W) 버퍼를 만들면 1024² 에서 47.2 MiB 낭비."""
    m = make_bmef()
    stats = {}
    for h in (32, 64):
        feats = {p: torch.randn(B, C, h, h) for p in PATHS}
        probe = _TensorSizeProbe()
        with probe, torch.no_grad():
            m(feats, pres_of((1, 1, 1)), return_weights=True)
        base = B * C * h * h
        stats[h] = (probe.max_numel / base, probe.total_numel / base)
        # 단일 최대 텐서가 (B,C,H,W) 를 넘지 않는다 -> 3배 버퍼 없음
        assert probe.max_numel <= base, (h, probe.max_numel, base)
    # 해상도가 4배가 되어도 base 당 총 할당량이 일정 -> 폭발 항 없음(전부 O(B·C·H·W))
    assert abs(stats[64][1] - stats[32][1]) / stats[32][1] < 0.05, stats
    # 1024² 외삽 상한 (stage1: B=1, C=32, 256²): 계수 × base 가 03 §7.3 의 규모와 정합
    assert stats[64][1] < 60.0, stats


# ================================================================ 에러 계약
def test_input_validation():
    m = make_bmef()
    feats = indep_feats()
    with pytest.raises(ValueError, match="전부 None"):
        m({p: None for p in PATHS}, pres_of((1, 1, 1)))
    with pytest.raises(ValueError, match="present"):
        m(feats, torch.ones(B, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="present"):
        m(feats, torch.ones(B + 1, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        m({"rgb": feats["rgb"], "ir": feats["spec"]}, pres_of((1, 0, 0)))
    with pytest.raises(ValueError, match="C="):
        m({"rgb": torch.randn(B, C + 8, H, W)}, pres_of((1, 0, 0)))
    bad = dict(feats)
    bad["spec"] = torch.randn(B, C, H // 2, W)
    with pytest.raises(ValueError, match="shape 불일치"):
        m(bad, pres_of((1, 1, 0)))


def test_constructor_validation():
    with pytest.raises(ValueError):
        BMEF(C, 5)
    with pytest.raises(ValueError):
        BMEF(C, 1, gpol_pool="bilinear")
    with pytest.raises(ValueError):
        BMEF(C, 1, norm_mode="linf")
    with pytest.raises(ValueError):
        BMEF(C, 1, norm_layer="ln")
    with pytest.raises(ValueError):
        BMEF(C, 1, kappa_init=(1.0, 0.5))
    with pytest.raises(ValueError):
        BMEF(C, 1, kappa_init=(1.0, 0.0, 0.018))  # softplus 치역은 (0, ∞)


def test_present_accepts_non_bool_dtypes():
    m = make_bmef()
    feats = indep_feats()
    ref = m(feats, pres_of((1, 1, 0))).fused
    for dt in (torch.int64, torch.float32, torch.uint8):
        got = m(feats, torch.tensor([[1, 1, 0]] * B, dtype=dt)).fused
        assert torch.allclose(got, ref, atol=1e-6)


# ================================================== 리터럴 단일 출처 / 상수
def test_paths_come_from_constants():
    assert PATHS is CONST_PATHS
    assert PATHS == ("rgb", "spec", "phys")
    assert (NEG, EPS) == (-1e4, 1e-12)
    assert FEEDBACK_PATHS == ("spec", "phys")


def test_channels_are_not_changed():
    for stage, ch in STAGE_CHANNELS.items():
        m = BMEF(ch, stage).eval()
        feats = {p: torch.randn(1, ch, 8, 8) for p in PATHS}
        out = m(feats, torch.ones(1, 3, dtype=torch.bool))
        assert out.fused.shape == (1, ch, 8, 8)
        assert m.se_channels == max(8, ch // 8)


# ============================================ identity(mean-only) 모드 (A-26)
def test_identity_fuse_contract():
    feats = indep_feats()
    out = identity_fuse(feats, pres_of((1, 1, 0)), stage_idx=1, return_weights=True)
    expect = (feats["rgb"] + feats["spec"]) / 2.0
    assert torch.allclose(out.fused, expect, atol=1e-6)
    assert (out.vacuity == 1.0).all()
    assert (out.log_tau == 0.0).all()
    assert set(out.feedback) == {"spec", "phys"}
    assert all(v.abs().max().item() == 0.0 for v in out.feedback.values())
    assert torch.allclose(out.weights.sum(dim=1), torch.ones(B, H, W), atol=1e-6)
    assert (out.weights[:, 2] == 0.0).all()


def test_identity_fuse_edge_cases():
    feats = indep_feats()
    assert identity_fuse(feats, pres_of((0, 0, 0))).fused.abs().max().item() == 0.0
    assert identity_fuse(feats, pres_of((1, 1, 1)), stage_idx=4).feedback == {}
    single = identity_fuse(feats, pres_of((0, 1, 0)))
    assert torch.allclose(single.fused, feats["spec"], atol=1e-6)
    with pytest.raises(ValueError):
        identity_fuse({p: None for p in PATHS}, pres_of((1, 1, 1)))


# ============================================================ 통합 backward
def test_end_to_end_backward_through_conv_stack():
    """헌법 C-5.1: CPU forward/backward. g_pol 이 full-res 라 feature 는 GH² 로 둔다."""
    torch.manual_seed(0)
    enc = {p: torch.nn.Conv2d(3, C, 3, padding=1) for p in PATHS}
    mods = torch.nn.ModuleDict(enc)
    m = BMEF(C, 1)
    m.train()
    x = torch.randn(B, 3, GH, GH)
    feats = {p: mods[p](x) for p in PATHS}
    g_pol = torch.rand(B, 1, GH * 4, GH * 4)
    out = m(feats, pres_of((1, 1, 1)), g_pol, prec_ramp=0.3, return_weights=True)
    loss = out.fused.pow(2).mean() + out.vacuity.mean() + sum(
        v.pow(2).mean() for v in out.feedback.values()
    )
    loss.backward()
    for name, par in list(mods.named_parameters()) + list(m.named_parameters()):
        if par.grad is not None:
            assert torch.isfinite(par.grad).all(), name
    assert mods["phys"].weight.grad.abs().sum().item() > 0
