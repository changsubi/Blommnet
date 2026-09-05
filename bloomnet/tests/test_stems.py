"""L1 공통 부품 + L2 stem 테스트 — 02 §1.2~1.5·§2·§9, 06 §3.3.1/§3.3.2, T10.

헌법 C-5.1/C-5.2: **CPU 전용**, B<=2, 64x64 이하 소형 텐서. GPU 사용 금지.

중점
    (a) 06 §6.1/§6.2 회귀 상수(파라미터 수·MAC)와 **정확히** 일치
    (b) 헌법 R3 glint gate 초기값의 수치 검증
    (c) A7 결측 전파 — avail_k 가 m·C_abs·stem 출력까지 도달하는지
    (d) 정정 B-3 폴백 사다리 F0~F3 와 bio_valid
    (e) 정정 B-1 gn8(gcd) norm 스위치가 C_r=20 에서 살아남고 파라미터 수를 보존하는지
    (f) 정정 B-7 bio pyramid 가 adaptive pooling 과 수치적으로 동일한지
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # cwd 와 무관하게 bloomnet 패키지를 찾게 한다
    sys.path.insert(0, str(_ROOT))

from bloomnet.data.indices import normalize_imagenet  # noqa: E402
from bloomnet.modules.common import (  # noqa: E402
    ConvBN,
    Downsample,
    DropPath,
    LayerScale,
    SEGate,
    act_layer,
    build_act,
    build_bio_pyramid,
    build_norm,
    init_encoder,
)
from bloomnet.modules.stems import PPN, SPS, TPS, CanonicalScatter, SPSOut  # noqa: E402
from bloomnet.utils.flops import count_macs_hooks, scale_macs  # noqa: E402

DEV = torch.device("cpu")


@pytest.fixture(autouse=True)
def _no_gpu() -> None:
    """헌법 C-5.2: 다른 학습이 GPU 를 쓰고 있다. 테스트가 GPU 를 잡으면 안 된다."""
    assert not torch.cuda.is_available(), "CUDA_VISIBLE_DEVICES='' 로 실행해야 한다"


def fnum(t: object) -> float:
    """requires_grad 텐서에서도 경고 없이 스칼라를 뽑는다 (테스트 편의)."""
    if isinstance(t, torch.Tensor):  # type: ignore[union-attr]
        return float(t.detach())
    return float(t)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _finite_backward(y: torch.Tensor, m: nn.Module) -> None:
    y.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "gradient 가 하나도 흐르지 않았다"
    assert all(torch.isfinite(g).all() for g in grads)


# ═════════════════════════════════════════════════════════════════════════════
# common.py — norm / act 빌더 (02 §1.2, 정정 B-1 / A-37 / V20)
# ═════════════════════════════════════════════════════════════════════════════
# 02 §1.2 전수 점검: 모델이 만드는 norm 폭 중 8의 배수가 아닌 것은 C_r=20 뿐이다.
NORM_WIDTHS = [8, 16, 20, 32, 40, 64, 80, 128, 160, 320, 384, 480]


@pytest.mark.parametrize("c", NORM_WIDTHS)
def test_gn8_uses_gcd_not_literal_8(c: int) -> None:
    """`gn8` = GroupNorm(gcd(8,C), C). 문자 그대로 8 이면 C=20 에서 모델 생성이 실패한다."""
    g = build_norm("gn8", c)
    assert isinstance(g, nn.GroupNorm)
    assert g.num_groups == math.gcd(8, c)
    assert c % g.num_groups == 0


def test_gn8_biogate_cr20_is_the_only_offender() -> None:
    """BioGate stage3 의 C_r=20 이 실제 충돌 지점임을 회귀 고정한다 (정정 B-1)."""
    with pytest.raises(ValueError):
        nn.GroupNorm(8, 20)  # 초판 구현이 냈던 바로 그 에러
    assert build_norm("gn8", 20).num_groups == 4  # 그룹당 5채널
    # C_r = max(8, C//8) 을 전 stage 에 적용했을 때 20 만 8의 배수가 아니다.
    crs = [max(8, c // 8) for c in (32, 64, 160, 320)]
    assert crs == [8, 8, 20, 40]
    assert [c for c in crs if c % 8 != 0] == [20]


@pytest.mark.parametrize("c", NORM_WIDTHS)
def test_norm_param_count_is_invariant_to_kind(c: int) -> None:
    """BN(C) 와 GN(g,C) 모두 2C. → §2~§7 파라미터 표가 `norm` 값과 무관하다."""
    assert n_params(build_norm("bn", c)) == 2 * c
    assert n_params(build_norm("gn8", c)) == 2 * c
    assert n_params(build_norm("bn", 20)) == n_params(build_norm("gn8", 20)) == 40


def test_build_norm_syncbn_constructs() -> None:
    """DDP 경로용. forward 는 process group 이 필요하므로 생성만 검증한다."""
    sbn = build_norm("syncbn", 32)
    assert isinstance(sbn, nn.SyncBatchNorm) and n_params(sbn) == 64


def test_build_norm_rejects_unknown_and_nonpositive() -> None:
    with pytest.raises(ValueError):
        build_norm("layernorm", 32)
    with pytest.raises(ValueError):
        build_norm("bn", 0)


def test_bn_hyperparams_frozen() -> None:
    bn = build_norm("bn", 32)
    assert (bn.eps, bn.momentum, bn.affine, bn.track_running_stats) == (1e-5, 0.1, True, True)


def test_build_act_kinds_and_alias() -> None:
    assert act_layer is build_act  # 06 매니페스트 이름 == 02 §1.2 동결 코드의 이름
    assert isinstance(build_act("hswish"), nn.Hardswish)
    assert isinstance(build_act("relu"), nn.ReLU)
    assert isinstance(build_act("sigmoid"), nn.Sigmoid)
    assert isinstance(build_act("hsigmoid"), nn.Hardsigmoid)
    g = build_act("gelu")
    assert isinstance(g, nn.GELU) and g.approximate == "tanh"
    with pytest.raises(ValueError):
        build_act("swish")


# ═════════════════════════════════════════════════════════════════════════════
# common.py — ConvBN / Downsample / LayerScale / DropPath / SEGate
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "k,cin,cout,groups",
    [(1, 8, 16, 1), (3, 16, 16, 16), (5, 16, 32, 1), (7, 3, 32, 1), (3, 64, 160, 1)],
)
def test_convbn_param_formula(k: int, cin: int, cout: int, groups: int) -> None:
    m = ConvBN(k, cin, cout, groups=groups)
    assert n_params(m) == k * k * cin * cout // groups + 2 * cout
    assert m[0].bias is None  # bias=False (BN 이 흡수)


@pytest.mark.parametrize("k,dil", [(3, 1), (3, 2), (3, 4), (5, 1), (7, 1)])
def test_convbn_padding_preserves_shape(k: int, dil: int) -> None:
    m = ConvBN(k, 4, 4, dilation=dil)
    assert m(torch.randn(2, 4, 32, 32)).shape == (2, 4, 32, 32)


@pytest.mark.parametrize(
    "cin,cout,expected", [(32, 64, 18_560), (64, 160, 92_480), (160, 320, 461_440)]
)
def test_downsample_params_and_shape(cin: int, cout: int, expected: int) -> None:
    """02 §6.2 회귀 상수."""
    d = Downsample(cin, cout)
    assert n_params(d) == expected
    y = d(torch.randn(2, cin, 8, 8))
    assert y.shape == (2, cout, 4, 4)
    _finite_backward(y, d)


def test_downsample_no_activation() -> None:
    d = Downsample(32, 64)
    kinds = {type(m) for m in d.modules() if not isinstance(m, (Downsample, nn.Conv2d))}
    assert kinds == {nn.BatchNorm2d}  # activation 없음 (02 §6.2)


def test_layer_scale_is_per_channel() -> None:
    ls = LayerScale(8, 0.01)
    assert ls.gamma.shape == (8,)  # 스칼라가 아니다 (CMNeXt 근거)
    assert torch.allclose(ls.gamma, torch.full((8,), 0.01))
    x = torch.randn(2, 8, 4, 4)
    assert torch.allclose(ls(x), x * 0.01)
    # 채널별로 다른 γ 가 채널 축에 걸리는지
    with torch.no_grad():
        ls.gamma.copy_(torch.arange(8, dtype=torch.float32))
    assert torch.allclose(ls(x), x * torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1))


def test_drop_path_identity_in_eval_and_p0() -> None:
    x = torch.randn(2, 4, 8, 8)
    dp = DropPath(0.5).eval()
    assert torch.equal(dp(x), x)
    assert torch.equal(DropPath(0.0).train()(x), x)
    with pytest.raises(ValueError):
        DropPath(1.0)


def test_drop_path_is_per_sample_and_unbiased() -> None:
    torch.manual_seed(0)
    dp = DropPath(0.5).train()
    x = torch.ones(512, 4, 2, 2)
    y = dp(x)
    per_sample = y.reshape(512, -1)
    # 샘플 단위로 통째 0 이거나 통째 1/keep
    assert torch.all((per_sample == 0).all(1) | (per_sample == 2.0).all(1))
    assert abs(fnum(y.mean()) - 1.0) < 0.1  # 기댓값 보존


def test_segate_initial_gate_is_half() -> None:
    """2nd conv zero-init → 초기 gate 정확히 0.5 ([분석] D-PHY-01 발산 차단)."""
    se = SEGate(32).eval()
    assert torch.equal(se.fc2.weight, torch.zeros_like(se.fc2.weight))
    x = torch.randn(2, 32, 8, 8)
    assert torch.allclose(se(x), x * 0.5, atol=1e-6)


@pytest.mark.parametrize("c,expected_cr", [(32, 8), (64, 8), (160, 20), (320, 40)])
def test_segate_reduction_floor_and_params(c: int, expected_cr: int) -> None:
    """C_se = max(8, C//8) — floor 8 (02 §5.2). params = 2·C·C_se + C_se + C."""
    se = SEGate(c, reduction=8, se_min=8)
    assert se.fc1.out_channels == expected_cr
    assert n_params(se) == 2 * c * expected_cr + expected_cr + c


def test_segate_gate_is_bounded() -> None:
    se = SEGate(16)
    with torch.no_grad():  # gate 를 강제로 흔들어도 [0,1] 밖으로 못 나간다
        se.fc2.weight.normal_(0, 5.0)
        se.fc2.bias.normal_(0, 5.0)
    x = torch.ones(2, 16, 4, 4)
    y = se(x).detach()
    assert fnum(y.min()) >= 0.0 and fnum(y.max()) <= 1.0


def test_segate_uses_mean_not_adaptive_pool() -> None:
    """04 §9.3-A: AdaptiveAvgPool2d 는 ONNX 에 shape 의존 서브그래프를 만든다."""
    se = SEGate(8)
    assert not any(isinstance(m, nn.AdaptiveAvgPool2d) for m in se.modules())


# ═════════════════════════════════════════════════════════════════════════════
# common.py — build_bio_pyramid (02 §4.3, 정정 B-7)
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("h", [64, 128])
def test_bio_pyramid_matches_adaptive_pool_exactly(h: int) -> None:
    """정정 B-7: 고정 커널 avg_pool2d == adaptive_avg_pool2d (축소비가 정수이므로)."""
    x = torch.randn(2, 2, h, h)
    sizes = [(h // s, h // s) for s in (4, 8, 16, 32)]
    got = build_bio_pyramid(x, sizes)
    for t, (hi, wi) in zip(got, sizes):
        ref = F.adaptive_avg_pool2d(x, (hi, wi))
        assert t is not None and t.shape == (2, 2, hi, wi)
        assert fnum((t - ref).abs().max()) == 0.0


def test_bio_pyramid_none_and_identity_and_error() -> None:
    sizes = [(16, 16), (8, 8), (4, 4), (2, 2)]
    assert build_bio_pyramid(None, sizes) == [None] * 4
    x = torch.randn(2, 2, 16, 16)
    assert build_bio_pyramid(x, sizes)[0] is x  # 동일 해상도는 무연산 통과
    with pytest.raises(ValueError):  # 32 의 배수가 아니면 조기 실패 (조용한 반올림 금지)
        build_bio_pyramid(torch.randn(2, 2, 30, 30), [(4, 4)])


def test_bio_pyramid_is_parameter_free() -> None:
    x = torch.randn(1, 2, 64, 64, requires_grad=True)
    out = build_bio_pyramid(x, [(16, 16)])
    out[0].sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ═════════════════════════════════════════════════════════════════════════════
# common.py — init_encoder (02 §1.5)
# ═════════════════════════════════════════════════════════════════════════════
def test_init_encoder_rules() -> None:
    m = nn.Sequential(
        nn.Conv2d(8, 16, 1, bias=True),
        nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=True),
        nn.BatchNorm2d(16),
        nn.GroupNorm(4, 16),
    )
    with torch.no_grad():  # 규칙이 실제로 덮어쓰는지 보이도록 오염시켜 둔다
        for p in m.parameters():
            p.fill_(7.0)
    m.apply(init_encoder)
    assert float(m[0].weight.detach().std()) < 0.05  # trunc_normal_(std=0.02)
    assert torch.equal(m[0].bias, torch.zeros(16))
    assert float(m[1].weight.detach().std()) > 0.05  # kaiming fan_out (DW 3x3)
    assert torch.equal(m[1].bias, torch.zeros(16))
    for norm in (m[2], m[3]):
        assert torch.equal(norm.weight, torch.ones(16))
        assert torch.equal(norm.bias, torch.zeros(16))


# ═════════════════════════════════════════════════════════════════════════════
# CanonicalScatter (02 §2.1, T10, 정정 A-25/B-2)
# ═════════════════════════════════════════════════════════════════════════════
M3M_IDS = [1, 2, 3, 5]  # G560 R650 RE730 NIR860 → 결측 slot 0, 4


def test_canonical_scatter_basic() -> None:
    cs = CanonicalScatter(6)
    x = torch.randn(2, 4, 64, 64)
    s, m = cs(x, M3M_IDS)
    assert s.shape == (2, 6, 64, 64) and m.shape == (2, 6)
    assert fnum(s[:, 0].abs().max()) == 0.0  # BLUE 결측
    assert fnum(s[:, 4].abs().max()) == 0.0  # REDEDGE2 결측
    assert m.tolist() == [[0, 1, 1, 1, 0, 1]] * 2
    for k, slot in enumerate(M3M_IDS):
        assert torch.equal(s[:, slot], x[:, k])  # 값이 올바른 slot 에 갔는가
    assert n_params(cs) == 0


def test_canonical_scatter_per_sample_avail() -> None:
    """★ A7(i): band_ids 는 배치 상수다. 샘플 결측은 avail_k 로만 표현된다."""
    cs = CanonicalScatter(6)
    x = torch.randn(2, 4, 8, 8)
    s, m = cs(x, M3M_IDS, torch.tensor([1.0, 0.0]))
    assert fnum(m[1].sum()) == 0.0  # 샘플 1 은 전 slot absent
    assert fnum(s[1].abs().max()) == 0.0
    assert m[0].tolist() == [0, 1, 1, 1, 0, 1]  # 샘플 0 은 그대로
    assert torch.equal(s[0], cs(x, M3M_IDS)[0][0])


def test_canonical_scatter_avail_none_is_bit_identical() -> None:
    cs = CanonicalScatter(6)
    x = torch.randn(2, 4, 8, 8)
    a, b = cs(x, M3M_IDS)
    c, d = cs(x, M3M_IDS, torch.ones(2))
    assert torch.equal(a, c) and torch.equal(b, d)


def test_c_abs_fires_only_on_the_missing_sample() -> None:
    """``(1-m) @ C_abs`` 가 결측 샘플에서만 전 slot 보상을 낸다 (02 §9 표)."""
    cs = CanonicalScatter(6)
    x = torch.randn(2, 4, 8, 8)
    _, m = cs(x, M3M_IDS, torch.tensor([1.0, 0.0]))
    c_abs = torch.arange(6 * 16, dtype=torch.float32).reshape(6, 16)
    comp = (1.0 - m) @ c_abs
    assert torch.equal(comp[0], c_abs[[0, 4]].sum(0))  # 결측 slot 0,4 만
    assert torch.equal(comp[1], c_abs.sum(0))  # 전 slot


def test_canonical_scatter_validates_band_ids() -> None:
    cs = CanonicalScatter(6)
    x = torch.randn(2, 4, 8, 8)
    with pytest.raises(ValueError):
        cs(x, [1, 2, 3])  # 길이 불일치
    with pytest.raises(ValueError):
        cs(x, [1, 1, 2, 3])  # 중복 slot
    with pytest.raises(ValueError):
        cs(x, [1, 2, 3, 6])  # 범위 밖
    with pytest.raises(ValueError):
        cs(torch.randn(4, 8, 8), M3M_IDS)  # 3D 입력


def test_canonical_scatter_gradient_flows_to_source_bands() -> None:
    cs = CanonicalScatter(6)
    x = torch.randn(2, 4, 4, 4, requires_grad=True)
    s, _ = cs(x, M3M_IDS)
    s.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


# ═════════════════════════════════════════════════════════════════════════════
# PPN (02 §2.2, 헌법 R3, T10)
# ═════════════════════════════════════════════════════════════════════════════
def test_ppn_nopol_params_and_identity_path() -> None:
    """정정 B-4/X-05: gate·inpainter 를 **생성하지 않는다**."""
    p = PPN(use_pol=False)
    assert n_params(p) == 4_768
    assert not hasattr(p, "a_hat") and not hasattr(p, "inp1")
    x = torch.randn(2, 3, 64, 64)
    stem, g = p(x)
    assert stem.shape == (2, 32, 16, 16)
    assert g is None  # zeros 가 아니라 None (03 §4.1 에서 log_r = 0)
    x_clean, g2 = p.precondition(x)
    assert x_clean is x and g2 is None  # 동일 객체 — 연산 0
    _finite_backward(stem, p)


def test_ppn_pol_params_and_gate_range() -> None:
    p = PPN(use_pol=True)
    assert n_params(p) == 10_101
    x = torch.randn(2, 3, 64, 64)
    x_pol = torch.rand(2, 3, 64, 64)
    stem, g = p(x, x_pol)
    assert stem.shape == (2, 32, 16, 16)
    assert g is not None and g.shape == (2, 1, 64, 64)
    gd = g.detach()
    assert fnum(gd.min()) > 0.0 and fnum(gd.max()) < 1.0
    assert torch.isfinite(stem).all()
    _finite_backward(stem, p)


def test_ppn_glint_gate_values_match_constitution_r3() -> None:
    """헌법 R3 초기값 검증 — 02 §2.2 표 / T10 (±1e-4)."""
    p = PPN(use_pol=True)
    dolp = torch.tensor([0.02, 0.15, 0.30, 0.95]).view(1, 1, 1, 4)
    g = p.glint_gate(dolp).detach().flatten()
    expected = torch.tensor([0.0100, 0.1192, 0.7311, 1.0000])
    assert torch.allclose(g, expected, atol=1e-4), g.tolist()
    # 원설계 a=10,b=-3 이면 깨끗한 수체(DoLP 0.15)를 18% 오검출한다 — 그 값이 아님을 확인
    assert abs(fnum(g[1]) - 0.1824) > 0.05


def test_ppn_gate_is_monotone_in_dolp() -> None:
    p = PPN(use_pol=True)
    x = torch.randn(1, 3, 16, 16)
    means = []
    for d in (0.05, 0.2, 0.4, 0.8):
        x_pol = torch.full((1, 3, 16, 16), d)
        means.append(float(p(x, x_pol)[1].detach().mean()))
    assert means == sorted(means)


def test_ppn_a_is_positive_by_construction() -> None:
    """a = softplus(a_hat) — Fresnel 단조성을 **구조적으로** 보장한다 (헌법 R3)."""
    p = PPN(use_pol=True)
    with torch.no_grad():
        p.a_hat.fill_(-50.0)  # 학습이 아무리 밀어도
    d = torch.tensor([0.0, 1.0]).view(1, 1, 1, 2)
    g = p.glint_gate(d).detach().flatten()
    assert fnum(g[1]) >= fnum(g[0])  # 여전히 단조 증가
    assert float(F.softplus(p.a_hat).detach()) > 0.0


def test_ppn_inpainter_head_is_zero_init() -> None:
    """P2: Sigmoid 대신 zero-init linear. 초기 x_inp = 0 = ImageNet 평균색."""
    p = PPN(use_pol=True)
    assert torch.equal(p.inp_head.weight, torch.zeros_like(p.inp_head.weight))
    assert torch.equal(p.inp_head.bias, torch.zeros_like(p.inp_head.bias))
    x = torch.randn(2, 3, 32, 32)
    x_pol = torch.rand(2, 3, 32, 32)
    x_clean, g = p.precondition(x, x_pol)
    assert torch.allclose(x_clean, (1.0 - g) * x, atol=1e-6)


def test_ppn_pol_runtime_dropout_returns_none() -> None:
    """use_pol=True 인데 배치에 pol 이 없으면 gate 를 건너뛴다 (모달 dropout)."""
    p = PPN(use_pol=True)
    x = torch.randn(2, 3, 32, 32)
    stem, g = p(x)
    assert g is None and stem.shape == (2, 32, 8, 8)


def test_ppn_uses_only_dolp_channel() -> None:
    """x_pol 은 3ch [DoLP, sin2θ, cos2θ] 지만 PPN 은 ch0 만 본다 (정정 B-4)."""
    p = PPN(use_pol=True).eval()
    x = torch.randn(2, 3, 32, 32)
    pol_a = torch.rand(2, 3, 32, 32)
    pol_b = pol_a.clone()
    pol_b[:, 1:] = torch.randn(2, 2, 32, 32)  # sin/cos 만 바꾼다
    with torch.no_grad():
        assert torch.equal(p(x, pol_a)[0], p(x, pol_b)[0])


def test_ppn_rgb_proxy_gate() -> None:
    """01 §5.2 opt-in 게이트. ★ 물리적 glint 가 아니다 — 기본 비활성."""
    p = PPN(rgb_proxy_gate=True)
    assert n_params(p) == 10_102  # 4,768 + 5,331 + (a_hat, b, c_hat)
    # 무채색(Sat=0) 이면 spec == v, ExG == 0 → g = sigmoid(40·(v − 0.55))
    for v, want in ((0.475, 0.047), (0.566, 0.655), (0.650, 0.982)):
        x01 = torch.full((1, 3, 4, 4), v)
        g = p.rgb_gate(normalize_imagenet(x01)).detach()
        assert abs(fnum(g.mean()) - want) < 5e-3, (v, fnum(g.mean()))


def test_ppn_rejects_both_gates() -> None:
    with pytest.raises(ValueError):
        PPN(use_pol=True, rgb_proxy_gate=True)


def test_ppn_always_build_keeps_inpainter() -> None:
    p = PPN(use_pol=False, always_build=True)
    assert n_params(p) == 4_768 + 5_331  # gate 파라미터는 없다
    assert p(torch.randn(2, 3, 32, 32))[1] is None


def test_ppn_patch_embed_is_7x7_stride4_no_act() -> None:
    p = PPN(use_pol=False)
    conv = p.patch_embed[0]
    assert conv.kernel_size == (7, 7) and conv.stride == (4, 4) and conv.padding == (3, 3)
    assert conv.bias is None
    assert isinstance(p.patch_embed[1], nn.BatchNorm2d) and len(p.patch_embed) == 2


# ═════════════════════════════════════════════════════════════════════════════
# SPS (02 §2.3, T10, 정정 B-3)
# ═════════════════════════════════════════════════════════════════════════════
def test_sps_params_frozen() -> None:
    assert n_params(SPS()) == 13_312


@pytest.mark.parametrize(
    "band_ids", [[1, 2, 3, 5], [0, 1, 2, 3, 5], [0, 1, 2, 3, 4, 5]]  # M3M / 235 / RedEdge-P
)
def test_sps_single_weight_handles_4_5_6_bands(band_ids: list) -> None:
    """헌법 R2 / E1: 하나의 weight 가 4·5·6밴드를 전부 처리한다."""
    sps = SPS()
    x = torch.rand(2, len(band_ids), 64, 64)
    out = sps(x, band_ids)
    assert isinstance(out, SPSOut)
    assert out.spec_stem.shape == (2, 32, 16, 16)
    assert out.x_bio_full.shape == (2, 2, 64, 64)
    assert out.bio_valid.shape == (2, 2)
    assert torch.isfinite(out.spec_stem).all()
    _finite_backward(out.spec_stem, sps)


def test_sps_bio_range_and_f0() -> None:
    sps = SPS()
    x = torch.rand(2, 4, 64, 64)
    out = sps(x, M3M_IDS)
    assert fnum(out.x_bio_full.min()) >= -1.0 and fnum(out.x_bio_full.max()) <= 1.0
    assert out.bio_valid.tolist() == [[1.0, 1.0], [1.0, 1.0]]  # F0
    assert sps.bio_kind_of(out.bio_valid) == "mci"


def test_sps_fallback_f2_ndci_only() -> None:
    """NIR·REDEDGE2 결측 → ch1 = 0, bio_valid = [1,0] (02 §9 표)."""
    sps = SPS()
    out = sps(torch.rand(2, 3, 32, 32), [1, 2, 3])
    assert out.bio_valid.tolist() == [[1.0, 0.0]] * 2
    assert fnum(out.x_bio_full[:, 1].abs().max()) == 0.0
    assert fnum(out.x_bio_full[:, 0].abs().max()) > 0.0
    assert sps.bio_kind_of(out.bio_valid) == "ndci_only"


def test_sps_fallback_f3_all_zero() -> None:
    """RED 또는 REDEDGE1 결측 → 전부 0, bio_valid = [0,0]. spec path 는 끄지 않는다."""
    sps = SPS()
    out = sps(torch.rand(2, 2, 32, 32), [1, 5])
    assert out.bio_valid.tolist() == [[0.0, 0.0]] * 2
    assert fnum(out.x_bio_full.abs().max()) == 0.0
    assert sps.bio_kind_of(out.bio_valid) == "none"
    assert torch.isfinite(out.spec_stem).all()  # F3 에서도 raw projection 은 유효


def test_sps_fallback_f1_rededge2_substitution() -> None:
    """F1: NIR 없음 + REDEDGE2 있음 → c 재유도해 MCI 를 살린다 (정정 B-3)."""
    c2 = (705.0 - 665.0) / (740.0 - 665.0)  # RedEdge-P: 0.533333 (F0 의 1.57배)
    sps = SPS(mci_c=0.338983, mci_c_re2=c2)
    x = torch.rand(2, 4, 32, 32)
    out = sps(x, [1, 2, 3, 4])  # G, R, RE1, RE2 — NIR 없음
    assert out.bio_valid.tolist() == [[1.0, 1.0]] * 2
    assert fnum(out.x_bio_full[:, 1].abs().max()) > 0.0
    # F1 을 끄면 F2 로 강등된다
    off = SPS(mci_c=0.338983, mci_c_re2=c2, allow_mci_re2=False)
    out_off = off(x, [1, 2, 3, 4])
    assert out_off.bio_valid.tolist() == [[1.0, 0.0]] * 2
    # c_re2 를 안 주면 F1 을 계산할 수 없으므로 역시 F2
    assert SPS()(x, [1, 2, 3, 4]).bio_valid.tolist() == [[1.0, 0.0]] * 2


def test_sps_bio_is_scale_invariant() -> None:
    """NDCI·MCI_norm 은 정규화형 → 절대 캘리브레이션(R5) 미확정 상태에서도 안전."""
    sps = SPS()
    x = torch.rand(2, 4, 16, 16) + 0.1
    a = sps(x, M3M_IDS).x_bio_full
    b = sps(x * 2.0, M3M_IDS).x_bio_full
    assert torch.allclose(a, b, atol=1e-4)


def test_sps_external_bio_is_used_verbatim() -> None:
    """X-02: dataset 이 기하변환 후 계산한 x_bio 가 정본. SPS 는 fallback 일 뿐."""
    sps = SPS()
    x = torch.rand(2, 4, 16, 16)
    bio = torch.full((2, 2, 16, 16), 0.25)
    out = sps(x, M3M_IDS, bio)
    assert out.x_bio_full is bio
    assert out.bio_valid.tolist() == [[1.0, 1.0]] * 2
    got = sps(x, M3M_IDS, bio, bio_valid=torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    assert got.bio_valid.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    with pytest.raises(ValueError):
        sps(x, M3M_IDS, torch.rand(2, 3, 16, 16))


def test_sps_avail_masks_stem_output_and_bio() -> None:
    """★ A7(ii): 결측 샘플은 stem 출력도 0 이어야 BN 이 오염되지 않는다."""
    sps = SPS().eval()
    x = torch.rand(2, 4, 32, 32)
    out = sps(x, M3M_IDS, avail_msi=torch.tensor([1.0, 0.0]))
    assert fnum(out.spec_stem[1].abs().max()) == 0.0
    assert fnum(out.spec_stem[0].abs().max()) > 0.0
    assert fnum(out.x_bio_full[1].abs().max()) == 0.0
    assert out.bio_valid.tolist() == [[1.0, 1.0], [0.0, 0.0]]
    # 외부 bio 를 줘도 결측 샘플에서는 0 (A1 계약)
    out2 = sps(x, M3M_IDS, torch.full((2, 2, 32, 32), 0.3), avail_msi=torch.tensor([1.0, 0.0]))
    assert fnum(out2.x_bio_full[1].abs().max()) == 0.0
    assert out2.bio_valid.tolist() == [[1.0, 1.0], [0.0, 0.0]]


def test_sps_c_abs_is_zero_init() -> None:
    """init 0 → 초기 동작이 순수 zero-fill 과 **완전히 동일** (회귀 위험 0)."""
    sps = SPS()
    assert torch.equal(sps.body.c_abs, torch.zeros(6, 16))
    assert sps.body.c_abs.shape == (6, 16)  # 96 params


def test_sps_bio_init_gain_applied_to_columns_6_7() -> None:
    """02 §2.3 Step3 초기화 특칙 — bio 열만 ×4 ([U-U5] ablation 대상)."""
    torch.manual_seed(0)
    a = SPS(bio_init_gain=1.0)
    torch.manual_seed(0)
    b = SPS(bio_init_gain=4.0)
    wa, wb = a.body.proj.weight, b.body.proj.weight
    assert torch.allclose(wa[:, :6], wb[:, :6])  # raw 밴드 열은 불변
    assert torch.allclose(wa[:, 6:] * 4.0, wb[:, 6:])


def test_sps_rejects_bad_config() -> None:
    with pytest.raises(ValueError):
        SPS(num_slots=4)
    with pytest.raises(ValueError):
        SPS(bio_kind="flh")  # 헌법 R1: FLH 는 폐기됐다 (r = −0.059)


def test_sps_gndvi_is_available_as_ablation() -> None:
    """정정 B-3: GNDVI 는 채택이 아니라 ablation 대조군이다."""
    sps = SPS(bio_kind="gndvi")
    out = sps(torch.rand(2, 4, 16, 16), M3M_IDS)
    assert out.bio_valid.tolist() == [[1.0, 1.0]] * 2  # G, NIR 존재
    assert sps.bio_kind_of(out.bio_valid) == "gndvi"
    out2 = sps(torch.rand(2, 3, 16, 16), [1, 2, 3])  # NIR 없음 → GNDVI 불가
    assert out2.bio_valid.tolist() == [[1.0, 0.0]] * 2


# ═════════════════════════════════════════════════════════════════════════════
# TPS (02 §2.4, T10, 정정 B-4)
# ═════════════════════════════════════════════════════════════════════════════
def test_tps_params_frozen_at_13216() -> None:
    """★X-04 회귀 고정: PHYS canonical slot 3 → 4 (AoLP 를 sin/cos 로 분해)."""
    t = TPS()
    assert n_params(t) == 13_216
    assert t.body.c_abs.shape == (4, 16)


@pytest.mark.parametrize("slot_ids", [[0], [1, 2, 3], [0, 1, 2, 3]])
def test_tps_variable_slots(slot_ids: list) -> None:
    t = TPS()
    y = t(torch.randn(2, len(slot_ids), 64, 64), slot_ids)
    assert y.shape == (2, 32, 16, 16)
    assert torch.isfinite(y).all()
    _finite_backward(y, t)


def test_tps_avail_masks_output() -> None:
    t = TPS().eval()
    y = t(torch.randn(2, 1, 32, 32), [0], torch.tensor([1.0, 0.0]))
    assert fnum(y[1].abs().max()) == 0.0
    assert fnum(y[0].abs().max()) > 0.0


def test_tps_rejects_wrong_num_slots() -> None:
    with pytest.raises(ValueError):
        TPS(num_slots=3)  # 정정 B-4 이전 값


def test_tps_has_no_bio_stage() -> None:
    """02 §2.4 [판단]: IR·DoLP 에는 검증된 파생 지수가 하나도 없다."""
    t = TPS()
    assert t.body.proj.in_channels == 4  # bio 2채널이 붙지 않는다


# ═════════════════════════════════════════════════════════════════════════════
# 전 stem 공통 — shape / norm 파라메트릭 / eval / MAC
# ═════════════════════════════════════════════════════════════════════════════
def _build_all(norm: str) -> dict:
    return {
        "ppn_nopol": PPN(use_pol=False, norm=norm),
        "ppn_pol": PPN(use_pol=True, norm=norm),
        "sps": SPS(norm=norm),
        "tps": TPS(norm=norm),
    }


def _run(name: str, m: nn.Module) -> torch.Tensor:
    if name == "ppn_nopol":
        return m(torch.randn(2, 3, 64, 64))[0]
    if name == "ppn_pol":
        return m(torch.randn(2, 3, 64, 64), torch.rand(2, 3, 64, 64))[0]
    if name == "sps":
        return m(torch.rand(2, 4, 64, 64), M3M_IDS).spec_stem
    return m(torch.randn(2, 1, 64, 64), [0])


@pytest.mark.parametrize("name", ["ppn_nopol", "ppn_pol", "sps", "tps"])
def test_all_stems_output_b32_hquarter(name: str) -> None:
    m = _build_all("bn")[name]
    y = _run(name, m)
    assert y.shape == (2, 32, 16, 16)  # 헌법 C-4: stem 은 H/4 × 32ch


@pytest.mark.parametrize("name", ["ppn_nopol", "ppn_pol", "sps", "tps"])
def test_norm_switch_bn_vs_gn8(name: str) -> None:
    """정정 B-1: gn8 로도 생성·forward·backward 가 되고 파라미터 수가 동일하다."""
    bn = _build_all("bn")[name]
    gn = _build_all("gn8")[name]
    assert n_params(bn) == n_params(gn)
    assert any(isinstance(mm, nn.GroupNorm) for mm in gn.modules())
    assert not any(isinstance(mm, nn.BatchNorm2d) for mm in gn.modules())
    y = _run(name, gn)
    assert torch.isfinite(y).all()
    _finite_backward(y, gn)


@pytest.mark.parametrize("name", ["ppn_nopol", "ppn_pol", "sps", "tps"])
def test_all_stems_eval_mode_finite(name: str) -> None:
    m = _build_all("bn")[name]
    _run(name, m).sum().backward()  # BN running stats 갱신
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(_run(name, m)).all()


@pytest.mark.parametrize(
    "name,expected_gmac",
    [("ppn_nopol", 0.0771), ("ppn_pol", 0.4200), ("sps", 0.2810), ("tps", 0.2642)],
)
def test_stem_mac_budget_at_512(name: str, expected_gmac: float) -> None:
    """06 §6.1 / 02 §2.5 MAC 회귀. 64² 에서 재고 512² 로 스케일한다 (정정 A-24 방식)."""
    m = _build_all("bn")[name]
    if name == "ppn_nopol":
        inputs = torch.randn(1, 3, 64, 64)
    elif name == "ppn_pol":
        inputs = (torch.randn(1, 3, 64, 64), torch.rand(1, 3, 64, 64))
    elif name == "sps":
        inputs = (torch.rand(1, 4, 64, 64), M3M_IDS)
    else:
        inputs = (torch.randn(1, 1, 64, 64), [0])
    rep = count_macs_hooks(m, inputs, input_hw=(64, 64))
    gmac512 = scale_macs(rep.total, from_hw=(64, 64), to_hw=(512, 512)) / 1e9
    assert abs(gmac512 - expected_gmac) < 5e-4, (name, gmac512)


def test_stem_channel_and_resolution_schedule_is_exact() -> None:
    """크기 검증 (02 §2.3): floor((512+4−5)/4)+1 = 128, 64² → 16."""
    sps = SPS()
    for h in (32, 64, 128):
        assert sps(torch.rand(1, 4, h, h), M3M_IDS).spec_stem.shape == (1, 32, h // 4, h // 4)
    ppn = PPN(use_pol=False)
    for h in (32, 64, 128):
        assert ppn(torch.randn(1, 3, h, h))[0].shape == (1, 32, h // 4, h // 4)
