"""T16 — `modules/heads.py` (04 §8, 05 §2.3.2, 06 §3.3.8).

헌법 C-5.1/C-5.2: **CPU 전용**, B<=2, 소형 텐서.
중점:
  (a) 파라미터 회귀 (06 §10 핵심 수치) + `num_classes` 의존 텐서가 cls 2개뿐 (헌법 R7),
  (b) **test_chl_fp16_safe** — z ∈ [-100,100] 2,001점에서 chl finite ∈ [0,500], fp16 도 finite,
  (c) **ChlHead 초기 출력 == 1.7559** (★X-10 weight=0, bias=softplus⁻¹),
  (d) **test_no_expm1** (정정 B-13) — 소스에 그 함수 호출이 없고 exp(u)-1 이 참조와 일치,
  (e) UncHead s ∈ [-7,7] 가 **유일한 clamp**, enabled=False → zeros (정정 B-17/A-20),
  (f) EdgeHead 초기 sigmoid ≈ prior_pos,
  (g) 헤드 activation 이 GELU(approximate='tanh') (04 §9.3-G),
  (h) forward/backward finite.
"""

from __future__ import annotations

import ast
import math
import pathlib
import sys

import pytest
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # cwd 와 무관하게 bloomnet 패키지를 찾게 한다
    sys.path.insert(0, str(_ROOT))

from bloomnet.constants import AUX_TAP_CH, AUX_TAP_MID, K235_LOG1P_MEAN  # noqa: E402
from bloomnet.modules.heads import (  # noqa: E402
    AuxSegHead,
    ChlHead,
    EdgeHead,
    RegTrunk,
    SegHead,
    SiamProjection,
    UncHead,
)

_SRC_HEADS = (_ROOT / "bloomnet" / "modules" / "heads.py").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _cpu_only():
    """헌법 C-5.2 — 어떤 테스트도 GPU 를 건드리지 않는다."""
    assert not torch.cuda.is_available(), "GPU 가 보이면 안 된다 (CUDA_VISIBLE_DEVICES='')"
    torch.manual_seed(0)
    torch.set_num_threads(4)
    yield


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# 파라미터 회귀 (06 §10)
# ─────────────────────────────────────────────────────────────────────────────
def test_seghead_params():
    assert n_params(SegHead(num_classes=2)) == 147_970  # 9·128² + 2·128 + 128·2 + 2
    assert n_params(SegHead(num_classes=12)) == 149_260


def test_regtrunk_chl_unc_params():
    assert n_params(RegTrunk()) == 9_728  # 9·128 + 2·128 + 128·64 + 2·64
    assert n_params(ChlHead()) == 65  # 64·1 + 1
    assert n_params(UncHead()) == 65
    assert n_params(UncHead(use_vacuity=True)) == 66  # ★X-12


def test_edgehead_params():
    assert n_params(EdgeHead()) == 9_313  # 9·32² + 2·32 + 32·1 + 1


@pytest.mark.parametrize(
    "tap,k12,k2",
    [
        ("enc_s8", 19_020, 18_690),
        ("enc_s16", 93_388, 92_738),
        ("enc_s32", 371_084, 369_794),
        ("p8", 74_892, 74_242),
    ],
)
def test_auxseghead_params(tap, k12, k2):
    """★(정정 B-21) c_mid 는 `C_in//2` 규칙이 아니라 AUX_TAP_MID 리터럴이다."""
    c_in, c_mid = AUX_TAP_CH[tap], AUX_TAP_MID[tap]
    assert n_params(AuxSegHead(c_in, c_mid, 12)) == k12
    assert n_params(AuxSegHead(c_in, c_mid, 2)) == k2
    # 손검산: 2·c_in + 9·c_in·c_mid + 2·c_mid + c_mid·K + K
    assert 2 * c_in + 9 * c_in * c_mid + 2 * c_mid + c_mid * 12 + 12 == k12


def test_auxseghead_encoder_taps_sum():
    total = sum(
        n_params(AuxSegHead(AUX_TAP_CH[t], AUX_TAP_MID[t], 12))
        for t in ("enc_s8", "enc_s16", "enc_s32")
    )
    assert total == 483_492


def test_auxseghead_cmid_halving_rule_is_wrong():
    """규칙 `C_in//2` 로 구현했을 때 실제로 값이 달라짐을 고정한다 (정정 B-21 의 근거)."""
    assert n_params(AuxSegHead(160, 80, 12)) == 116_652 != 93_388
    assert n_params(AuxSegHead(320, 160, 12)) == 463_692 != 371_084


def test_siamprojection_params():
    assert n_params(SiamProjection(160, 320)) == 51_840
    assert n_params(SiamProjection(320, 512)) == 164_864
    assert n_params(SiamProjection(128, 256)) == 33_280


# ─────────────────────────────────────────────────────────────────────────────
# SegHead
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("K", [2, 12])
def test_seghead_shape_and_backward(K):
    m = SegHead(num_classes=K).train()
    x = torch.randn(2, 128, 16, 16, requires_grad=True)
    y = m(x)
    assert y.shape == (2, K, 16, 16)  # ★ 헤드는 업샘플하지 않는다 (04 §9.1)
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_seghead_structure_matches_business_doc():
    """사업문서 s42/43: Conv3×3 → BatchNorm → GELU → 1×1 (헌법 C-4 가 채널/클래스만 정정)."""
    m = SegHead()
    assert isinstance(m.conv, nn.Conv2d) and m.conv.kernel_size == (3, 3)
    assert m.conv.bias is None  # BN 이 따르므로 bias=False
    assert isinstance(m.bn, nn.BatchNorm2d)
    assert isinstance(m.act, nn.GELU) and m.act.approximate == "tanh"  # 04 §9.3-G
    assert m.cls.kernel_size == (1, 1) and m.cls.bias is not None


def test_seghead_num_classes_tensors_are_exactly_two():
    """헌법 C-6 R7 — S0(12) → S1(2) 전이 시 갈아끼울 텐서는 cls.weight/bias 둘뿐."""
    a, b = SegHead(num_classes=12).state_dict(), SegHead(num_classes=2).state_dict()
    assert set(a) == set(b)
    differ = {k for k in a if a[k].shape != b[k].shape}
    assert differ == {"cls.weight", "cls.bias"}


def test_seghead_shape_filtered_load():
    """04 §8.1 의 shape-filtered load 가 정확히 2개만 누락시킨다."""
    src = SegHead(num_classes=12).state_dict()
    dst = SegHead(num_classes=2)
    own = dst.state_dict()
    keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    missing = sorted(k for k in own if k not in keep)
    assert missing == ["cls.bias", "cls.weight"]
    dst.load_state_dict(keep, strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# RegTrunk
# ─────────────────────────────────────────────────────────────────────────────
def test_regtrunk_shape_structure_backward():
    m = RegTrunk().train()
    x = torch.randn(2, 128, 16, 16, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 64, 16, 16)
    assert m.dw.groups == 128 and m.dw.kernel_size == (3, 3)  # depthwise-separable
    assert m.pw.kernel_size == (1, 1)
    assert isinstance(m.act, nn.GELU) and m.act.approximate == "tanh"
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_regtrunk_mac_ratio_vs_full_conv():
    """04 §8.2 의 '8배 싸다' 근거 (full Conv3x3(128→64) 대비)."""
    full = 9 * 128 * 64
    sep = 9 * 128 + 128 * 64
    assert full == 73_728 and sep == 9_344
    assert 7.5 < full / sep < 8.5


# ─────────────────────────────────────────────────────────────────────────────
# ChlHead
# ─────────────────────────────────────────────────────────────────────────────
def test_chlhead_constants():
    assert ChlHead.U_MAX == pytest.approx(math.log1p(500.0), abs=1e-6)
    assert (ChlHead.Z_MIN, ChlHead.Z_MAX) == (-20.0, 11.0)


def test_chlhead_initial_prediction_is_prior_mean():
    """★X-10 — weight=0, bias=softplus⁻¹(1.7559)=1.5662 → 초기 출력이 정확히 1.7559."""
    m = ChlHead().eval()
    assert m.prior_log1p_mean == K235_LOG1P_MEAN == 1.7559
    assert torch.equal(m.out.weight, torch.zeros_like(m.out.weight))
    assert m.out.bias.item() == pytest.approx(1.5662, abs=1e-4)
    with torch.no_grad():
        u = m(torch.randn(2, 64, 8, 8) * 100.0)  # 입력과 무관해야 한다
    assert torch.allclose(u, torch.full_like(u, 1.7559), atol=1e-4)


def test_chlhead_prior_is_configurable():
    m = ChlHead(prior_log1p_mean=2.13180).eval()  # 04 초판 값(median 7.43) — 설정으로만 복원
    with torch.no_grad():
        u = m(torch.zeros(1, 64, 4, 4))
    assert torch.allclose(u, torch.full_like(u, 2.13180), atol=1e-5)


def test_chlhead_rejects_degenerate_prior():
    with pytest.raises(ValueError):
        ChlHead(prior_log1p_mean=0.0)


def test_chl_fp16_safe():
    """T16 필수 — z ∈ [-100,100] 2,001점 sweep 에서 chl finite ∈ [0,500], fp16 도 finite."""
    m = ChlHead(in_ch=1).eval()
    with torch.no_grad():  # z = 입력값이 그대로 되도록 항등 사영
        m.out.weight.fill_(1.0)
        m.out.bias.zero_()
        z = torch.linspace(-100.0, 100.0, 2001).view(1, 1, 1, 2001)
        u = m(z)
        chl = m(z, return_mgm3=True)
    assert torch.isfinite(u).all() and (u >= 0).all()
    assert torch.isfinite(chl).all()
    assert chl.min().item() >= 0.0
    assert chl.max().item() <= 500.0 + 1e-3  # exp(U_MAX)-1 = 500.00003
    half = chl.half()
    assert torch.isfinite(half).all() and half.max().item() <= 500.0


def test_chlhead_no_expm1_call():
    """(정정 B-13) ONNX 에 해당 연산자가 없다 — 소스에 그 호출이 있으면 export 가 실패한다.

    ast 호출 검사 + 원문 문자열 검사 둘 다 한다(주석·docstring 에도 남기지 않는 것이 계약).
    """
    banned = "exp" + "m1"  # 이 테스트 파일 자신의 참조 구현과 구분하기 위해 분리 표기
    called = set()
    for node in ast.walk(ast.parse(_SRC_HEADS)):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                called.add(fn.attr)
            elif isinstance(fn, ast.Name):
                called.add(fn.id)
    assert banned not in called, f"{banned} 호출 발견 — 06 §10 규칙 13 위반"
    assert banned not in _SRC_HEADS, f"{banned} 문자열이 heads.py 에 남아 있다"
    assert "exp" in called  # torch.exp(u) - 1.0 로 구현


def test_chlhead_exp_minus_one_matches_reference():
    """교체가 무해함을 참조(expm1)와 atol=1e-4 로 대조 (04 §13.7)."""
    u = torch.linspace(0.0, ChlHead.U_MAX, 5000, dtype=torch.float32)
    ours = torch.exp(u) - 1.0
    ref = torch.expm1(u)
    assert torch.allclose(ours, ref, atol=1e-4, rtol=1e-4)
    assert (ours - ref).abs().max().item() < 1e-4


def test_chlhead_backward_and_nonneg():
    m = ChlHead().train()
    f = torch.randn(2, 64, 8, 8, requires_grad=True)
    u = m(f)
    assert u.shape == (2, 1, 8, 8) and (u >= 0).all()
    u.sum().backward()
    assert torch.isfinite(f.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_chlhead_mgm3_roundtrip_of_prior():
    """log1p 평균 1.7559 → 약 4.79 mg/m³ (8-site 필터후, 01 부록)."""
    m = ChlHead().eval()
    with torch.no_grad():
        chl = m(torch.zeros(1, 64, 2, 2), return_mgm3=True)
    assert chl.mean().item() == pytest.approx(math.exp(1.7559) - 1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# UncHead
# ─────────────────────────────────────────────────────────────────────────────
def test_unchead_clamp_is_the_only_clamp():
    """(정정 B-17/A-20, X-26) 헤드 [-7,7] 이 정본. σ ∈ [0.0302, 33.115]."""
    assert (UncHead.S_MIN, UncHead.S_MAX) == (-7.0, 7.0)
    assert math.exp(UncHead.S_MIN / 2) == pytest.approx(0.030197, abs=1e-5)
    assert math.exp(UncHead.S_MAX / 2) == pytest.approx(33.1155, abs=1e-3)
    # 파생 계약: conf = 1/(1+σ) 는 1 에 도달하지 못한다 (X-25, T25 가 이 구간을 검사한다)
    assert 1.0 / (1.0 + math.exp(0.5 * UncHead.S_MIN)) == pytest.approx(0.970688, abs=1e-5)
    assert 1.0 / (1.0 + math.exp(0.5 * UncHead.S_MAX)) == pytest.approx(0.029312, abs=1e-5)


def test_unchead_initial_s_is_zero():
    """weight=0, bias=0 → σ²=1 → 초기 손실이 정확히 ½·Huber."""
    m = UncHead().eval()
    with torch.no_grad():
        s = m(torch.randn(2, 64, 8, 8) * 50.0)
    assert torch.equal(s, torch.zeros_like(s))


def test_unchead_clamp_range_enforced():
    m = UncHead().eval()
    with torch.no_grad():
        m.out.weight.fill_(1.0)
        m.out.bias.zero_()
        f = torch.randn(2, 64, 8, 8) * 1e4
        s = m(f)
    assert torch.isfinite(s).all()
    assert s.min().item() >= UncHead.S_MIN and s.max().item() <= UncHead.S_MAX
    assert s.min().item() == pytest.approx(UncHead.S_MIN)  # 실제로 clamp 가 발동했는지
    assert s.max().item() == pytest.approx(UncHead.S_MAX)


def test_unchead_disabled_returns_zeros():
    m = UncHead().eval()
    with torch.no_grad():
        m.out.weight.fill_(0.5)
        m.out.bias.fill_(3.0)
        f = torch.randn(1, 64, 4, 4)
        assert not torch.equal(m(f, enabled=True), torch.zeros(1, 1, 4, 4))
        assert torch.equal(m(f, enabled=False), torch.zeros(1, 1, 4, 4))


def test_unchead_vacuity_variant():
    m = UncHead(use_vacuity=True).train()
    f = torch.randn(2, 64, 8, 8)
    v = torch.rand(2, 1, 8, 8)
    assert m.out.in_channels == 65
    s = m(f, vacuity=v)
    assert s.shape == (2, 1, 8, 8)
    with pytest.raises(ValueError):
        m(f)


def test_unchead_ignores_vacuity_when_disabled_variant():
    """기본 경로(use_vacuity=False)는 vacuity 를 받아도 조용히 무시한다 (진단 전용, X-12)."""
    m = UncHead().eval()
    f = torch.randn(1, 64, 4, 4)
    with torch.no_grad():
        assert torch.equal(m(f), m(f, vacuity=torch.rand(1, 1, 4, 4)))


def test_unchead_backward():
    m = UncHead().train()
    f = torch.randn(2, 64, 8, 8, requires_grad=True)
    # weight=0 초기화라 grad 가 흐르는지 보려면 한 스텝 흔들어 준다
    with torch.no_grad():
        m.out.weight.normal_(0.0, 0.1)
    m(f).sum().backward()
    assert torch.isfinite(f.grad).all() and f.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# EdgeHead
# ─────────────────────────────────────────────────────────────────────────────
def test_edgehead_initial_sigmoid_matches_prior():
    """prior init — 초기 sigmoid 평균 ≈ 0.03 (01 [M4] 실측 H/4 양성비 3.112 %)."""
    m = EdgeHead().train()
    logits = m(torch.randn(2, 32, 16, 16))
    assert m.out.bias.item() == pytest.approx(math.log(0.03 / 0.97), abs=1e-6)
    assert torch.sigmoid(logits).mean().item() == pytest.approx(0.03, abs=0.005)


def test_edgehead_prior_configurable():
    m = EdgeHead(prior_pos=0.03112)  # 01 [M4] 실측치
    assert m.out.bias.item() == pytest.approx(math.log(0.03112 / (1 - 0.03112)), abs=1e-6)
    with pytest.raises(ValueError):
        EdgeHead(prior_pos=0.0)


def test_edgehead_shape_and_backward():
    m = EdgeHead().train()
    d32 = torch.randn(2, 32, 16, 16, requires_grad=True)
    y = m(d32)
    assert y.shape == (2, 1, 16, 16)  # H/4 감독 (04 §7.4)
    y.sum().backward()
    assert torch.isfinite(d32.grad).all() and d32.grad.abs().sum() > 0
    assert isinstance(m.act, nn.GELU) and m.act.approximate == "tanh"


# ─────────────────────────────────────────────────────────────────────────────
# AuxSegHead
# ─────────────────────────────────────────────────────────────────────────────
def test_auxseghead_is_preactivation():
    """★X-06 — 05 §2.3.2 의 pre-activation 형태(BN→ReLU→Conv3→BN→ReLU→1×1)."""
    m = AuxSegHead(64, 32, 12)
    assert isinstance(m.bn1, nn.BatchNorm2d) and m.bn1.num_features == 64
    assert isinstance(m.relu, nn.ReLU)
    assert m.conv.kernel_size == (3, 3) and m.conv.bias is None
    assert isinstance(m.bn2, nn.BatchNorm2d) and m.bn2.num_features == 32
    assert m.cls.kernel_size == (1, 1) and m.cls.bias is not None
    # 04 §8.4 의 post-activation(GELU) 형태가 아니다
    assert not any(isinstance(mod, nn.GELU) for mod in m.modules())


@pytest.mark.parametrize("tap", ["enc_s8", "enc_s16", "enc_s32", "p8"])
def test_auxseghead_shape_and_backward(tap):
    c_in, c_mid = AUX_TAP_CH[tap], AUX_TAP_MID[tap]
    m = AuxSegHead(c_in, c_mid, 12).train()
    x = torch.randn(2, c_in, 4, 4, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 12, 4, 4)  # 해상도 보존 — 업샘플은 criterion 소관
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# SiamProjection
# ─────────────────────────────────────────────────────────────────────────────
def test_siamprojection_shape_and_structure():
    m = SiamProjection(128, 256).train()
    x = torch.randn(2, 128, 8, 8, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 256, 8, 8)
    assert isinstance(m[0], nn.Conv2d) and m[0].kernel_size == (1, 1) and m[0].bias is None
    assert isinstance(m[1], nn.BatchNorm2d)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


# ─────────────────────────────────────────────────────────────────────────────
# MAC 회귀 — 06 §6.1 / 정정 A-14·B-21 (T24 가 소비할 선행 고정)
# ─────────────────────────────────────────────────────────────────────────────
def _gmac_1024(module: nn.Module, inputs, *, base: int = 64) -> float:
    """base² 에서 1회 측정하고 (1024/base)² 로 스케일한다 (정정 A-24). 헤드는 전부 conv 라 정확."""
    from bloomnet.utils.flops import count_macs_hooks, scale_macs

    total = count_macs_hooks(module, inputs).total
    return scale_macs(total, from_hw=(base, base), to_hw=(1024, 1024)) / 1e9


def test_head_macs_match_budget_table():
    S = 64
    f_dec = torch.randn(1, 128, S // 4, S // 4)
    r = torch.randn(1, 64, S // 4, S // 4)
    assert _gmac_1024(SegHead(num_classes=2).eval(), f_dec) == pytest.approx(9.6805, abs=5e-4)
    assert _gmac_1024(SegHead(num_classes=12).eval(), f_dec) == pytest.approx(9.7643, abs=5e-4)
    assert _gmac_1024(RegTrunk().eval(), f_dec) == pytest.approx(0.6124, abs=5e-4)
    chl = _gmac_1024(ChlHead().eval(), r)
    unc = _gmac_1024(UncHead().eval(), r)
    assert chl == pytest.approx(0.0042, abs=5e-4) and unc == pytest.approx(0.0042, abs=5e-4)
    assert _gmac_1024(EdgeHead().eval(), torch.randn(1, 32, S // 4, S // 4)) == pytest.approx(
        0.6061, abs=5e-4
    )
    # 추론 헤드 소계 10.3013 (04 §11.2 가 디코더 13.235 와 더해 23.536 을 만드는 값)
    seg2 = _gmac_1024(SegHead(num_classes=2).eval(), f_dec)
    assert seg2 + _gmac_1024(RegTrunk().eval(), f_dec) + chl + unc == pytest.approx(
        10.3013, abs=1e-3
    )


def test_auxseghead_macs_1p068():
    """★(정정 B-21) AuxSegHead×3 = 0.267 @512² / 1.068 @1024². 06 초판 0.409/1.635 는 폐기."""
    S = 64
    total = sum(
        _gmac_1024(
            AuxSegHead(AUX_TAP_CH[t], AUX_TAP_MID[t], 12).eval(),
            torch.randn(1, AUX_TAP_CH[t], S // s, S // s),
        )
        for t, s in (("enc_s8", 8), ("enc_s16", 16), ("enc_s32", 32))
    )
    assert total == pytest.approx(1.068, abs=2e-3)
    assert total / 4.0 == pytest.approx(0.267, abs=1e-3)  # @512²


# ─────────────────────────────────────────────────────────────────────────────
# 공통 규약
# ─────────────────────────────────────────────────────────────────────────────
def test_all_heads_use_tanh_gelu_or_relu_only():
    """04 §1.2/§9.3-G — GELU 는 반드시 approximate='tanh' (erf 형과 수치가 다르다)."""
    heads = [SegHead(), RegTrunk(), ChlHead(), UncHead(), EdgeHead(), AuxSegHead(64, 32, 2)]
    for h in heads:
        for mod in h.modules():
            if isinstance(mod, nn.GELU):
                assert mod.approximate == "tanh", f"{type(h).__name__} 의 GELU 가 erf 형이다"


def test_heads_use_batchnorm_only():
    """04 §1.2 — GN/LN 금지 (TRT conv+BN fusion)."""
    heads = [SegHead(), RegTrunk(), EdgeHead(), AuxSegHead(64, 32, 2), SiamProjection(128, 256)]
    for h in heads:
        assert not any(isinstance(m, (nn.GroupNorm, nn.LayerNorm)) for m in h.modules())


def test_reg_chain_end_to_end():
    """F_dec → RegTrunk → (ChlHead, UncHead) 배선을 소형으로 (04 §8.0 부착 지점)."""
    trunk, chl, unc = RegTrunk().train(), ChlHead().train(), UncHead().train()
    f_dec = torch.randn(2, 128, 16, 16, requires_grad=True)
    r = trunk(f_dec)
    u = chl(r)
    s = unc(r)
    assert r.shape == (2, 64, 16, 16) and u.shape == s.shape == (2, 1, 16, 16)
    (u.sum() + s.sum()).backward()
    assert torch.isfinite(f_dec.grad).all()
    for mod in (trunk, chl, unc):
        for p in mod.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()


def test_heads_finite_on_extreme_inputs():
    """헌법 C-5.4/C-5.5 — 결측 모달로 F_dec 이 튀어도 NaN/Inf 가 나오면 안 된다."""
    f_dec = torch.full((1, 128, 8, 8), 1e4)
    trunk = RegTrunk().eval()
    with torch.no_grad():
        r = trunk(f_dec)
        assert torch.isfinite(r).all()
        assert torch.isfinite(ChlHead().eval()(r, return_mgm3=True)).all()
        assert torch.isfinite(UncHead().eval()(r)).all()
        assert torch.isfinite(SegHead().eval()(f_dec)).all()


def test_heads_zero_input_is_finite():
    zero = torch.zeros(2, 128, 8, 8)
    with torch.no_grad():
        assert torch.isfinite(SegHead().eval()(zero)).all()
        r = RegTrunk().eval()(zero)
        assert torch.isfinite(ChlHead().eval()(r)).all()
