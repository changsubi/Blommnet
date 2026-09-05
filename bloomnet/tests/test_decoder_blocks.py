"""T15 선행분 — `modules/decoder_blocks.py` (04 §3~§6, 06 §3.3.7).

헌법 C-5.1/C-5.2: **CPU 전용**, B<=2, 최대 텐서 (2,320,16,16).
중점:
  (a) 파라미터 회귀 (06 §10 핵심 수치),
  (b) **H ∈ {64,128,512} 전부에서 PAPPM/PagFM 이 동작** — 정정 A-21/B-14 의 blocker.
      H=64 (h=2) 는 pooled 가 전부 1×1 이라 고정 scale_factor 로는 반드시 터진다.
  (c) B=1 + train() 에서 PAPPM 예외 없음 (04 §6.2 편차 1 이 없으면 실패, 06 R-9),
  (d) PagFM 게이트가 포화하지 않음 (dot_scale=1/8 의 존재 이유),
  (e) out_hw / y_up 재사용이 **bit-identical** (정정 B-14/B-16),
  (f) LightBag 게이트가 128채널이고 참조식과 일치,
  (g) forward/backward finite.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # cwd 와 무관하게 bloomnet 패키지를 찾게 한다
    sys.path.insert(0, str(_ROOT))

from bloomnet.constants import CHANNELS, DEC_CH  # noqa: E402
from bloomnet.modules.decoder_blocks import (  # noqa: E402
    PAPPM,
    BasicBlock,
    LightBag,
    PagFM,
    SepConvBNAct,
)

# 04 §1.2 / 헌법 C-5.1 — 04 §6.3 표의 세 해상도. h = H/32.
TEST_H = (64, 128, 512)

_SRC_DECODER = (_ROOT / "bloomnet" / "modules" / "decoder_blocks.py").read_text(encoding="utf-8")


def _called_names(src: str) -> set:
    """소스에서 **호출된** 함수/메서드 이름 집합 (주석·docstring 제외)."""
    import ast

    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                names.add(fn.attr)
            elif isinstance(fn, ast.Name):
                names.add(fn.id)
    return names


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
# 파라미터 회귀 (06 §10 "핵심 수치")
# ─────────────────────────────────────────────────────────────────────────────
def test_param_counts_frozen():
    assert n_params(BasicBlock(DEC_CH)) == 295_424  # 2*(9*128² + 2*128)
    assert n_params(BasicBlock(32)) == 18_560  # 2*(9*32² + 2*32)
    assert n_params(SepConvBNAct(128, 128)) == 18_048  # 9*128 + 2*128 + 128² + 2*128
    assert n_params(PagFM()) == 16_640  # 2*(128*64 + 2*64)
    assert n_params(LightBag()) == 33_280  # 2*(128*128 + 2*128)
    assert n_params(PAPPM()) == 590_144


def test_pappm_param_hand_arithmetic():
    """04 §6.4 손 산술을 항목별로 재현한다 (구성이 바뀌면 여기서 먼저 깨진다)."""
    m = PAPPM(320, 96, 128)
    pre = 2 * 320
    branches = 5 * (320 * 96)
    proc = 2 * 384 + 9 * 384 * 384 // 4
    comp = 2 * 480 + 480 * 128
    short = 320 * 128
    assert pre + branches + proc + comp + short == 590_144
    assert n_params(m) == 590_144


def test_default_channels_come_from_constants():
    """리터럴 복제 금지 (06 §2.1.1) — 기본값이 constants 와 일치."""
    assert PagFM().f_x[0].in_channels == DEC_CH
    assert LightBag().conv_p[0].out_channels == DEC_CH
    assert PAPPM().s0.in_channels == CHANNELS[3] == 320


# ─────────────────────────────────────────────────────────────────────────────
# BasicBlock
# ─────────────────────────────────────────────────────────────────────────────
def test_basicblock_zero_init_residual_is_identity_relu():
    """`BN2.weight = 0` → 초기 잔차가 정확히 0 → out == ReLU(x)."""
    m = BasicBlock(32, zero_init_residual=True).eval()
    assert torch.equal(m.bn2.weight, torch.zeros(32))
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        out = m(x)
    assert torch.equal(out, F.relu(x))


def test_basicblock_zero_init_can_be_disabled():
    m = BasicBlock(32, zero_init_residual=False).eval()
    assert torch.equal(m.bn2.weight, torch.ones(32))
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        assert not torch.equal(m(x), F.relu(x))


def test_basicblock_shape_and_backward():
    m = BasicBlock(128).train()
    x = torch.randn(2, 128, 8, 8, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_basicblock_uses_batchnorm_only():
    """04 §1.2 — 디코더 본체에 GN/LN 금지 (TRT conv+BN fusion)."""
    m = BasicBlock(32)
    kinds = {type(mod) for mod in m.modules() if isinstance(mod, nn.modules.batchnorm._NormBase)}
    assert kinds == {nn.BatchNorm2d}
    assert not any(isinstance(mod, (nn.GroupNorm, nn.LayerNorm)) for mod in m.modules())
    for bn in (m.bn1, m.bn2):
        assert bn.eps == 1e-5 and bn.momentum == 0.1


# ─────────────────────────────────────────────────────────────────────────────
# SepConvBNAct
# ─────────────────────────────────────────────────────────────────────────────
def test_sepconv_is_depthwise_then_pointwise():
    m = SepConvBNAct(128, 128)
    assert m.dw[0].groups == 128 and m.dw[0].kernel_size == (3, 3)
    assert m.pw[0].groups == 1 and m.pw[0].kernel_size == (1, 1)
    assert m.dw[0].bias is None and m.pw[0].bias is None  # BN 이 따르므로 bias=False


def test_sepconv_shape_backward_and_nonneg():
    m = SepConvBNAct(128, 128).train()
    x = torch.randn(2, 128, 16, 16, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 128, 16, 16)
    assert (y >= 0).all()  # 마지막이 ReLU
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_sepconv_mac_ratio_vs_full_conv():
    """04 §3.3 의 '8.4배 싸다' 근거를 산술로 고정한다."""
    full = 9 * 128 * 128
    sep = 9 * 128 + 128 * 128
    assert full == 147_456 and sep == 17_536
    assert 8.3 < full / sep < 8.5


# ─────────────────────────────────────────────────────────────────────────────
# PagFM
# ─────────────────────────────────────────────────────────────────────────────
def test_pagfm_default_dot_scale_is_inv_sqrt_mid():
    assert PagFM(128, 64).dot_scale == pytest.approx(0.125)
    assert PagFM(128, 64, dot_scale=1.0).dot_scale == 1.0  # PIDNet 원본 재현 (ablation)


def test_pagfm_gate_range():
    """초기 σ 평균 0.5±0.05, std > 0.1 — dot_scale=1.0 이면 포화해서 깨진다 (04 §4.2)."""
    m = PagFM().train()
    x = torch.randn(2, 128, 16, 16)
    y = torch.randn(2, 128, 8, 8)
    k = m.f_x(x)
    q = F.interpolate(m.f_y(y), size=(16, 16), mode="bilinear", align_corners=False)
    sig = torch.sigmoid((k * q).sum(dim=1, keepdim=True) * m.dot_scale)
    assert abs(sig.mean().item() - 0.5) < 0.05
    assert sig.std().item() > 0.1


def _sigma_saturation(dot_scale: float, seed: int = 0) -> float:
    torch.manual_seed(seed)
    m = PagFM(dot_scale=dot_scale).train()
    x = torch.randn(2, 128, 16, 16)
    y = torch.randn(2, 128, 8, 8)
    k = m.f_x(x)
    q = F.interpolate(m.f_y(y), size=(16, 16), mode="bilinear", align_corners=False)
    sig = torch.sigmoid((k * q).sum(dim=1, keepdim=True) * dot_scale)
    return ((sig < 0.02) | (sig > 0.98)).float().mean().item()


def test_pagfm_dot_scale_one_saturates():
    """반례: PIDNet 원본 스케일이면 σ 가 실제로 포화한다 (04 §4.2 편차의 근거).

    실측 (5 seed): dot_scale=1.0 → 포화 픽셀 44~47 % / dot_scale=1/8 → 0.0 %.
    """
    for seed in range(3):
        assert _sigma_saturation(1.0, seed) > 0.35
        assert _sigma_saturation(0.125, seed) < 0.01


def test_pagfm_convex_combination_bounds():
    """out 은 x 와 y↑ 의 픽셀별 볼록결합이므로 두 값 사이에 있어야 한다."""
    m = PagFM().eval()
    x = torch.full((1, 128, 8, 8), -1.0)
    y = torch.full((1, 128, 4, 4), 3.0)
    with torch.no_grad():
        out = m(x, y)
    assert (out >= -1.0 - 1e-6).all() and (out <= 3.0 + 1e-6).all()


@pytest.mark.parametrize("hy", [1, 2, 4, 8])
def test_pagfm_shapes_all_res(hy):
    """x = H/4, y = H/8 의 2배 관계를 여러 크기에서. hy=1 은 H=32 극단 케이스."""
    m = PagFM().train()
    x = torch.randn(2, 128, hy * 2, hy * 2)
    y = torch.randn(2, 128, hy, hy)
    out = m(x, y)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_pagfm_skips_interpolate_when_same_res(monkeypatch):
    """(정정 B-16) PagFM_8 은 x·y 가 둘 다 H/8 이라 보간 자체가 없어야 한다."""
    calls = []
    real = F.interpolate

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(
        "bloomnet.modules.decoder_blocks.F.interpolate", counting, raising=True
    )
    m = PagFM().train()
    x = torch.randn(2, 128, 8, 8)
    y = torch.randn(2, 128, 8, 8)
    m(x, y)
    assert calls == [], "동일 해상도인데 Resize 노드가 생겼다"


def test_pagfm_y_up_reuse_is_bit_identical():
    """(정정 B-16) 호출부가 만든 c1 재사용이 내부 재계산과 bit-identical."""
    m = PagFM().eval()
    x = torch.randn(2, 128, 16, 16)
    y = torch.randn(2, 128, 8, 8)
    c1 = F.interpolate(y, size=(16, 16), mode="bilinear", align_corners=False)
    with torch.no_grad():
        a = m(x, y)
        b = m(x, y, y_up=c1)
    assert torch.equal(a, b)


def test_pagfm_out_hw_static_is_bit_identical():
    """(정정 B-14) export 경로의 정적 int 주입이 수치를 바꾸지 않는다."""
    m = PagFM().eval()
    x = torch.randn(2, 128, 16, 16)
    y = torch.randn(2, 128, 8, 8)
    with torch.no_grad():
        a = m(x, y)
        b = m(x, y, out_hw=(16, 16))
    assert torch.equal(a, b)


def test_pagfm_fy_is_applied_at_y_resolution():
    """구현 함정 회귀 — f_y 를 먼저 올려서 적용하면 MAC 4배.

    f_y 를 y 의 원해상도에서 돌렸는지는 hook 으로 입력 크기를 보면 알 수 있다.
    """
    m = PagFM().eval()
    seen = []
    m.f_y.register_forward_pre_hook(lambda mod, inp: seen.append(tuple(inp[0].shape[-2:])))
    with torch.no_grad():
        m(torch.randn(1, 128, 16, 16), torch.randn(1, 128, 8, 8))
    assert seen == [(8, 8)]


def test_pagfm_backward():
    m = PagFM().train()
    x = torch.randn(2, 128, 16, 16, requires_grad=True)
    y = torch.randn(2, 128, 8, 8, requires_grad=True)
    m(x, y).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    assert x.grad.abs().sum() > 0 and y.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# LightBag
# ─────────────────────────────────────────────────────────────────────────────
def test_lightbag_gate_is_128_channels_and_matches_reference():
    """게이트가 1채널이 아니라 128채널 elementwise 임을 참조식으로 고정한다."""
    m = LightBag().eval()
    p = torch.randn(2, 128, 8, 8)
    i = torch.randn(2, 128, 8, 8)
    d = torch.randn(2, 128, 8, 8)
    with torch.no_grad():
        out = m(p, i, d)
        sig = torch.sigmoid(d)
        ref = m.conv_p(p + (1.0 - sig) * i) + m.conv_i(i + sig * p)
    assert torch.equal(out, ref)
    assert sig.shape == d.shape  # (B,128,H,W) — 1채널로 줄이지 않는다


def test_lightbag_channelwise_gate_actually_differs_per_channel():
    """채널별로 다른 게이트가 실제로 다른 혼합을 만드는지 (1채널 구현이면 실패)."""
    m = LightBag().eval()
    p = torch.randn(1, 128, 4, 4)
    i = torch.randn(1, 128, 4, 4)
    d_hi = torch.full((1, 128, 4, 4), 20.0)
    d_mix = d_hi.clone()
    d_mix[:, :64] = -20.0
    with torch.no_grad():
        assert not torch.allclose(m(p, i, d_hi), m(p, i, d_mix))


def test_lightbag_sigma_extremes_expand_as_documented():
    """σ=1 → conv_p(p) + conv_i(i+p),  σ=0 → conv_p(p+i) + conv_i(i)  (04 §5.2)."""
    m = LightBag().eval()
    p = torch.randn(1, 128, 4, 4)
    i = torch.randn(1, 128, 4, 4)
    big = torch.full((1, 128, 4, 4), 40.0)
    with torch.no_grad():
        one = m(p, i, big)
        zero = m(p, i, -big)
        assert torch.allclose(one, m.conv_p(p) + m.conv_i(i + p), atol=1e-5)
        assert torch.allclose(zero, m.conv_p(p + i) + m.conv_i(i), atol=1e-5)


def test_lightbag_backward():
    m = LightBag().train()
    p = torch.randn(2, 128, 8, 8, requires_grad=True)
    i = torch.randn(2, 128, 8, 8, requires_grad=True)
    d = torch.randn(2, 128, 8, 8, requires_grad=True)
    m(p, i, d).sum().backward()
    for t in (p, i, d):
        assert torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# PAPPM — 정정 A-21 / B-14 의 blocker 가 여기서 잡힌다
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("H", TEST_H)
def test_pappm_shapes_all_res(H):
    """H ∈ {64,128,512} 전부에서 동작. h=2 에서 pooled 가 전부 1×1 이 되는 케이스 포함."""
    h = H // 32
    m = PAPPM().train()
    out = m(torch.randn(2, 320, h, h))
    assert out.shape == (2, 128, h, h)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("H", TEST_H)
def test_pappm_b1_train(H):
    """(04 §6.2 편차 1) B=1 + train() 에서 예외 없음.

    branch 별 BN 을 쓰면 global branch 의 BN 이 (1,320,1,1) 에 걸려
    `ValueError: Expected more than 1 value per channel when training` 이 난다.
    """
    h = H // 32
    m = PAPPM().train()
    out = m(torch.randn(1, 320, h, h))
    assert out.shape == (1, 128, h, h)


def test_pappm_has_single_shared_pre_bn():
    """편차 1 의 구조적 증거 — C_in 채널 BN 이 정확히 1개(공유 pre-BN)."""
    m = PAPPM()
    bn_320 = [b for b in m.modules() if isinstance(b, nn.BatchNorm2d) and b.num_features == 320]
    assert len(bn_320) == 1
    assert isinstance(m.pre[0], nn.BatchNorm2d) and isinstance(m.pre[1], nn.ReLU)


def test_pappm_count_include_pad_false():
    """편차 2 — k=17,p=8 이 h=16 에서 padding 감쇠를 일으키지 않게 한다."""
    m = PAPPM()
    for pool in (m.pool1, m.pool2, m.pool3):
        assert pool.count_include_pad is False
    assert (m.pool1.kernel_size, m.pool1.stride, m.pool1.padding) == (5, 2, 2)
    assert (m.pool2.kernel_size, m.pool2.stride, m.pool2.padding) == (9, 4, 4)
    assert (m.pool3.kernel_size, m.pool3.stride, m.pool3.padding) == (17, 8, 8)


def test_pappm_count_include_pad_true_would_attenuate():
    """반례: True 면 상수 입력에서도 값이 크게 줄어든다 (편차 2 의 근거)."""
    x = torch.ones(1, 4, 16, 16)
    inc = nn.AvgPool2d(17, 8, 8, count_include_pad=True)(x)
    exc = nn.AvgPool2d(17, 8, 8, count_include_pad=False)(x)
    assert torch.allclose(exc, torch.ones_like(exc))
    # 실측 [[0.2803, 0.4983], [0.4983, 0.8858]] — 감쇠가 위치마다 다르고(공간 편향)
    # 편차 1 로 BN 을 앞으로 옮겼기 때문에 이를 보정해 줄 BN 이 뒤에 없다.
    assert inc.min().item() < 0.3
    assert inc.max().item() < 0.9
    assert (inc.max() - inc.min()).item() > 0.3


def test_pappm_no_adaptive_avg_pool():
    """04 §9.3-A / 06 §10 규칙 7 — AdaptiveAvgPool2d 금지 (x.mean 으로 구현)."""
    m = PAPPM()
    assert not any(
        isinstance(mod, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)) for mod in m.modules()
    )
    # 주석이 아니라 **호출**이 없어야 한다 → ast 로 확인 (문자열 검색은 주석에 걸린다)
    called = _called_names(_SRC_DECODER)
    assert not any("adaptive" in name.lower() for name in called), sorted(called)
    assert "mean" in called  # x.mean(dim=(2,3), keepdim=True) 로 구현


@pytest.mark.parametrize("H", TEST_H)
def test_pappm_out_hw_static_is_bit_identical(H):
    """(정정 B-14) `out_hw=None` 과 정적 int 주입이 allclose(atol=0)."""
    h = H // 32
    m = PAPPM().eval()
    x = torch.randn(2, 320, h, h)
    with torch.no_grad():
        a = m(x)
        b = m(x, out_hw=(h, h))
    assert torch.equal(a, b)


@pytest.mark.parametrize("H", TEST_H)
def test_pappm_scale_mode_matches_size_mode(H):
    """export 용 up_mode='scale' 이 정본(size)과 bit-identical (배율이 정수인 한)."""
    h = H // 32
    size_m = PAPPM(up_mode="size").eval()
    scale_m = PAPPM(up_mode="scale").eval()
    scale_m.load_state_dict(size_m.state_dict())
    x = torch.randn(2, 320, h, h)
    with torch.no_grad():
        assert torch.equal(size_m(x), scale_m(x))


def test_pappm_scale_mode_rejects_non_integer_ratio():
    """h=20 (H=640) 은 pooled=ceil(20/8)=3 이라 정수 배율이 아니다 → 조용히 틀리지 않게 막는다."""
    m = PAPPM(up_mode="scale").eval()
    with pytest.raises(RuntimeError, match="정수 배율"):
        with torch.no_grad():
            m(torch.randn(1, 320, 20, 20))
    # 같은 입력이 정본 경로에서는 정상 동작해야 한다
    with torch.no_grad():
        assert PAPPM().eval()(torch.randn(1, 320, 20, 20)).shape == (1, 128, 20, 20)


def test_pappm_fixed_scale_factor_would_break_at_h2():
    """정정 A-21/B-14 의 **원인** 회귀 고정.

    h=2 에서 pooled 는 전부 1×1 이므로 필요 배율이 전부 2 다. 고정 (2,4,8) 을 쓰면
    (B,96,4,4) + (B,96,2,2) 가 되어 broadcast 실패로 터진다 — 04 §6.3 CPU 실측 재현.
    """
    xr = torch.randn(1, 96, 2, 2)
    pooled = nn.AvgPool2d(9, 4, 4, count_include_pad=False)(xr)
    assert pooled.shape[-2:] == (1, 1)
    wrong = F.interpolate(pooled, scale_factor=4, mode="bilinear", align_corners=False)
    assert wrong.shape[-2:] == (4, 4) != xr.shape[-2:]
    with pytest.raises(RuntimeError):
        _ = wrong + xr


def test_pappm_up_mode_validation():
    with pytest.raises(ValueError, match="up_mode"):
        PAPPM(up_mode="bilinear")


def test_pappm_backward():
    m = PAPPM().train()
    x = torch.randn(2, 320, 4, 4, requires_grad=True)
    m(x).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_pappm_grouped_proc_conv():
    """분석 §2.3 — proc 는 groups=4 의 3×3 (병렬 branch 를 섞지 않는다)."""
    m = PAPPM()
    conv = m.proc[2]
    assert conv.groups == 4 and conv.kernel_size == (3, 3)
    assert conv.in_channels == conv.out_channels == 4 * 96


# ─────────────────────────────────────────────────────────────────────────────
# MAC 회귀 — 정정 A-14 의 모듈 단위 값 (T24 가 소비할 선행 고정)
# ─────────────────────────────────────────────────────────────────────────────
def _gmac_1024(module: nn.Module, inputs, *, base: int = 64) -> float:
    """base² 에서 1회 측정하고 (1024/base)² 로 스케일한다 (정정 A-24).

    conv MAC 은 출력 화소 수에 선형이라 정확하다 — 유일한 예외는 PAPPM 의 global branch
    (해상도 무관 상수 30,720 MAC) 이며 그 항은 전체의 5e-5 % 다.
    """
    from bloomnet.utils.flops import count_macs_hooks, scale_macs

    total = count_macs_hooks(module, inputs).total
    return scale_macs(total, from_hw=(base, base), to_hw=(1024, 1024)) / 1e9


def test_module_macs_match_correction_a14():
    """04 §10.2 / 정정 A-14 항목표 (@1024²). 구조가 바뀌면 예산표보다 여기서 먼저 깨진다."""
    S = 64
    got = {
        "LightBag": _gmac_1024(LightBag().eval(), (torch.randn(1, 128, S // 4, S // 4),) * 3),
        "PBlock8": _gmac_1024(BasicBlock(128).eval(), torch.randn(1, 128, S // 8, S // 8)),
        "IBlock3": _gmac_1024(BasicBlock(128).eval(), torch.randn(1, 128, S // 16, S // 16)),
        "DBlock": _gmac_1024(BasicBlock(32).eval(), torch.randn(1, 32, S // 4, S // 4)),
        "PSep": _gmac_1024(SepConvBNAct(128, 128).eval(), torch.randn(1, 128, S // 4, S // 4)),
        "ISep": _gmac_1024(SepConvBNAct(128, 128).eval(), torch.randn(1, 128, S // 8, S // 8)),
        "PagFM_4": _gmac_1024(
            PagFM().eval(),
            (torch.randn(1, 128, S // 4, S // 4), torch.randn(1, 128, S // 8, S // 8)),
        ),
        "PagFM_8": _gmac_1024(
            PagFM().eval(),
            (torch.randn(1, 128, S // 8, S // 8), torch.randn(1, 128, S // 8, S // 8)),
        ),
    }
    want = {
        "LightBag": 2.1475,
        "PBlock8": 4.8318,
        "IBlock3": 1.2080,
        "DBlock": 1.2080,
        "PSep": 1.1492,
        "ISep": 0.2873,
        "PagFM_4": 0.6711,
        "PagFM_8": 0.2684,
    }
    for k, v in want.items():
        assert got[k] == pytest.approx(v, abs=5e-4), f"{k}: {got[k]:.4f} != {v}"


def test_pappm_mac_0p486():
    """PAPPM 0.4864 @1024² (정정 A-14). 스케일링 오차의 출처까지 고정한다.

    실측 정수: @1024² = 486,406,144 MAC / @256² = 30,429,184 MAC.
    256² 측정 × 16 = 486,866,944 로 **460,800 만큼 크다** — 그 전부가 global branch
    (해상도 무관 상수 30,720 MAC × (16−1)) 다. 정정 A-24 가 "유일한 비선형 항" 이라 한 그 항목.
    """
    from bloomnet.utils.flops import count_macs_hooks

    at256 = count_macs_hooks(PAPPM().eval(), torch.randn(1, 320, 8, 8)).total
    assert at256 == 30_429_184
    assert at256 * 16 - 486_406_144 == 30_720 * 15 == 460_800
    scaled = _gmac_1024(PAPPM().eval(), torch.randn(1, 320, 8, 8), base=256)
    assert scaled == pytest.approx(0.4864, abs=1e-3)  # A-14 의 0.486 과 일치


# ─────────────────────────────────────────────────────────────────────────────
# 통합: §2.3 배선 일부를 소형으로 재현
# ─────────────────────────────────────────────────────────────────────────────
def test_decoder_wiring_smoke_h64():
    """04 §2.3 의 I/P/D + 융합을 H=64 (헌법 C-5.1 필수 크기) 로 돌린다."""
    B, H = 2, 64
    F1 = torch.randn(B, 32, H // 4, H // 4, requires_grad=True)
    F2 = torch.randn(B, 64, H // 8, H // 8, requires_grad=True)
    F3 = torch.randn(B, 160, H // 16, H // 16, requires_grad=True)
    F4 = torch.randn(B, 320, H // 32, H // 32, requires_grad=True)

    from bloomnet.modules.common import ConvBN

    pappm = PAPPM().train()
    lat3, lat2, lat1 = ConvBN(1, 160, 128), ConvBN(1, 64, 128), ConvBN(1, 32, 128)
    iblk3, isep = BasicBlock(128).train(), SepConvBNAct(128, 128).train()
    pag8, pag4 = PagFM().train(), PagFM().train()
    pblk8, psep = BasicBlock(128).train(), SepConvBNAct(128, 128).train()
    dlat, diff3, diff4 = ConvBN(1, 32, 32), ConvBN(3, 128, 32), ConvBN(1, 128, 32)
    dblk, dexp = BasicBlock(32).train(), ConvBN(1, 32, 128)
    bag = LightBag().train()

    up = lambda t, f: F.interpolate(t, scale_factor=f, mode="bilinear", align_corners=False)

    c4 = pappm(F4)
    c3 = iblk3(lat3(F3) + up(c4, 2))
    c2 = isep(up(c3, 2))
    c1 = up(c2, 2)

    p8 = pag8(lat2(F2), c2)
    p8 = pblk8(p8)
    p4 = psep(pag4(up(p8, 2) + lat1(F1), c2, y_up=c1))

    d = dlat(F1) + up(diff3(c3), 4)
    d = dblk(d) + up(diff4(c4), 8)
    d32 = d
    d128 = dexp(F.relu(d))

    F_dec = bag(p=p4, i=c1, d=d128)

    assert F_dec.shape == (B, 128, H // 4, H // 4)
    assert d32.shape == (B, 32, H // 4, H // 4)
    assert p8.shape == (B, 128, H // 8, H // 8)
    assert torch.isfinite(F_dec).all()

    F_dec.sum().backward()
    for t in (F1, F2, F3, F4):
        assert torch.isfinite(t.grad).all()
        assert t.grad.abs().sum() > 0
