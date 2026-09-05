"""T12/T13 — `modules/blocks_biospec.py` · `modules/blocks_physlite.py` 계약 테스트.

검증 범위 (06 §7.2 T12/T13, 02 §9 표)
* **파라미터 회귀 상수**: MSPA 4,585/16,337/94,601/368,401, BioGate 329/617/3,461/13,321,
  MLP 19,232/71,232/212,000/833,600, BioSpecBlock 24,338/88,570/311,022/1,217,242,
  PhysLite 5,576/19,336/118,740/455,080, path 소계 5,694,110 / 1,289,952.
* **MSPA**: `att ∈ (0,1)` 전 원소, 초기 att = 0.5 정확히(sigmoid **1회**),
  반환값이 `u` 가 아니라 `s`, dilation 3·5 가 2×2 입력에서도 shape 보존.
* **BioGate**: 초기 gain == 1.0 **비트 단위**, `b=None` → 항등, `β = softplus(β̂) = 0.1`.
* **PhysLite**: SE gate ∈ [0,1] (Hardsigmoid), 초기 0.5, shape 보존.
* **(정정 A-37/B-1) gn8 파라메트릭**: 전 stage 생성 성공 + `num_groups = gcd(8,C)`
  (BioGate `C_r=20` → 4 그룹) + 파라미터 수가 norm 과 무관.
* **초기화 특칙 인계 계약**: `apply(init_encoder)` 가 gate zero-init 을 지우고
  `reset_gate_()` 가 복구한다.
* **MAC**: hook 실측 == 손계산식, 손계산식 @512² == 06 §6.1/02 §7.1 표.

헌법 C-5: CPU 전용, B<=2, 소형 텐서(<=64²). GPU 금지.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from bloomnet.modules.blocks_biospec import BioGate, BioSpecBlock, BioSpecMLP, MSPA
from bloomnet.modules.blocks_physlite import PhysLiteBlock
from bloomnet.modules.common import Downsample, init_encoder
from bloomnet.utils.flops import count_macs_hooks


@pytest.fixture(autouse=True)
def _no_gpu() -> None:
    """헌법 C-5.2 — 어떤 테스트도 GPU 를 잡지 않는다."""
    assert not torch.cuda.is_available(), "CUDA_VISIBLE_DEVICES='' 로 실행해야 한다"
    torch.manual_seed(0)
    torch.set_num_threads(4)


# ── stage 스케줄 (02 §4.2 dilation / §4.4 mlp_ratio / §5.2 dw_kernel / §6.1 depth) ──
CHANNELS = (32, 64, 160, 320)
DILATIONS = ((1, 3, 5), (1, 3, 5), (1, 2, 4), (1, 2, 3))
MLP_RATIO = (8, 8, 4, 4)
DW_KERNEL = (3, 3, 5, 5)
SPEC_DEPTHS = (2, 2, 4, 3)
PHYS_DEPTHS = (1, 1, 2, 1)
FEATURE_HW = (16, 8, 4, 2)  # 64² 입력 기준 stage1~4 (CPU 테스트 계약 02 §1.7)

MSPA_PARAMS = (4_585, 16_337, 94_601, 368_401)
BIOGATE_PARAMS = (329, 617, 3_461, 13_321)
MLP_PARAMS = (19_232, 71_232, 212_000, 833_600)
BIOSPEC_PARAMS = (24_338, 88_570, 311_022, 1_217_242)
PHYSLITE_PARAMS = (5_576, 19_336, 118_740, 455_080)
DOWNSAMPLE_PARAMS = (18_560, 92_480, 461_440)  # 32→64, 64→160, 160→320 (02 §6.2)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def make_biospec(i: int, **kw) -> BioSpecBlock:
    """1-based stage 번호로 BioSpecBlock 을 만든다."""
    return BioSpecBlock(
        CHANNELS[i - 1], mlp_ratio=MLP_RATIO[i - 1], dilations=DILATIONS[i - 1], **kw
    )


def make_physlite(i: int, **kw) -> PhysLiteBlock:
    return PhysLiteBlock(CHANNELS[i - 1], dw_kernel=DW_KERNEL[i - 1], **kw)


# ═══════════════════════════════════════════════════════════════════════════
# 1. 파라미터 회귀 상수 (06 §6.1/§10, 02 §7.1)
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_mspa_param_count(i: int) -> None:
    m = MSPA(CHANNELS[i - 1], dilations=DILATIONS[i - 1], se_ratio=4)
    assert n_params(m) == MSPA_PARAMS[i - 1]


@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_biogate_param_count(i: int) -> None:
    assert n_params(BioGate(CHANNELS[i - 1])) == BIOGATE_PARAMS[i - 1]


@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_mlp_param_count(i: int) -> None:
    m = BioSpecMLP(CHANNELS[i - 1], mlp_ratio=MLP_RATIO[i - 1])
    assert n_params(m) == MLP_PARAMS[i - 1]


@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_biospec_block_param_count(i: int) -> None:
    assert n_params(make_biospec(i)) == BIOSPEC_PARAMS[i - 1]


@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_physlite_param_count(i: int) -> None:
    assert n_params(make_physlite(i)) == PHYSLITE_PARAMS[i - 1]


def test_biospec_block_equals_sum_of_parts_plus_6c() -> None:
    """`BioSpecBlock = MSPA + BioGate + MLP + 6C` (BN_a 2C + BN_b 2C + γ_a C + γ_b C)."""
    for i, c in enumerate(CHANNELS, start=1):
        blk = make_biospec(i)
        parts = (
            n_params(blk.mspa) + n_params(blk.biogate) + n_params(blk.mlp) + 6 * c
        )
        assert n_params(blk) == parts == BIOSPEC_PARAMS[i - 1]


def test_c_r_schedule_differs_between_mspa_and_biogate() -> None:
    """MSPA 는 `C//4` (8/16/40/80), BioGate 는 `C//8` (8/8/20/40). 혼동하면 표가 깨진다."""
    assert [MSPA(c).c_r for c in CHANNELS] == [8, 16, 40, 80]
    assert [BioGate(c).c_r for c in CHANNELS] == [8, 8, 20, 40]


def test_spec_and_phys_path_subtotals() -> None:
    """06 §6.1 소계: spec path 5,694,110 / phys path 1,289,952 (downsample 포함)."""
    ds = sum(DOWNSAMPLE_PARAMS)
    assert ds == sum(
        n_params(Downsample(a, b))
        for a, b in zip(CHANNELS[:-1], CHANNELS[1:])
    )
    spec = sum(d * p for d, p in zip(SPEC_DEPTHS, BIOSPEC_PARAMS)) + ds
    phys = sum(d * p for d, p in zip(PHYS_DEPTHS, PHYSLITE_PARAMS)) + ds
    assert spec == 5_694_110
    assert phys == 1_289_952


# ═══════════════════════════════════════════════════════════════════════════
# 2. MSPA — gate 규약
# ═══════════════════════════════════════════════════════════════════════════
def test_mspa_initial_attention_is_exactly_half() -> None:
    """T12: 초기 att 평균 ≈ 0.5 (±0.02). 실제로는 gate conv zero-init 이라 **정확히** 0.5.

    `sigmoid(spatial)·sigmoid(channel)` 로 구현하면 0.25 가 되어 여기서 잡힌다.
    """
    m = MSPA(64, dilations=(1, 3, 5)).eval()
    _, att = m.forward_parts(torch.randn(2, 64, 16, 16))
    assert att.shape == (2, 64, 16, 16)
    assert torch.equal(att, torch.full_like(att, 0.5))
    assert abs(att.mean().item() - 0.5) < 0.02


def test_mspa_attention_is_strictly_in_unit_interval() -> None:
    """gate conv 를 흔들어도 `att ∈ (0,1)` (무한계 곱셈 gate 차단, D-BIO-02).

    극단 logit 에서는 float32 sigmoid 가 정확히 0.0/1.0 으로 포화하므로,
    포화 영역은 **닫힌** [0,1] 로 검사한다 — 어느 쪽이든 발산은 불가능하다.
    """
    m = MSPA(32).eval()
    with torch.no_grad():
        for p in (m.spatial.weight, m.spatial.bias, m.fc2.weight, m.fc2.bias):
            p.normal_(0.0, 0.3)
    _, att = m.forward_parts(torch.randn(2, 32, 16, 16))
    assert torch.isfinite(att).all()
    assert (att > 0.0).all() and (att < 1.0).all()

    with torch.no_grad():  # 포화 영역
        for p in (m.spatial.weight, m.spatial.bias, m.fc2.weight, m.fc2.bias):
            p.normal_(0.0, 50.0)
    _, att = m.forward_parts(torch.randn(2, 32, 16, 16) * 10.0)
    assert torch.isfinite(att).all()
    assert (att >= 0.0).all() and (att <= 1.0).all()


def test_mspa_uses_single_sigmoid_over_summed_logits() -> None:
    """logit 공간 합산 규약: bias 만 남기면 `att == sigmoid(b_sp + b_ch)` 로 균일해진다."""
    m = MSPA(32).eval()
    with torch.no_grad():
        m.spatial.bias.fill_(0.7)
        m.fc2.bias.fill_(-1.2)
    _, att = m.forward_parts(torch.randn(2, 32, 8, 8))
    expected = torch.sigmoid(torch.tensor(0.7 - 1.2))
    assert torch.allclose(att, expected.expand_as(att), atol=1e-6)


def test_mspa_returns_s_not_gated_input() -> None:
    """02 §4.2 근거 3 — 반환값은 atrous branch 출력 `s` 이지 입력 `u` 가 아니다."""
    m = MSPA(32).eval()
    u = torch.randn(2, 32, 16, 16)
    s, att = m.forward_parts(u)
    out = m(u)
    assert torch.allclose(out, s * att, atol=0)
    assert not torch.allclose(out, u * att, atol=1e-3)


@pytest.mark.parametrize("dil", DILATIONS)
def test_mspa_shape_preserved_at_2x2(dil) -> None:
    """T12: dilation 3(및 5)이 stage4 의 2×2 입력에서도 shape 을 보존한다."""
    m = MSPA(32, dilations=dil).eval()
    x = torch.randn(2, 32, 2, 2)
    assert m(x).shape == x.shape


def test_mspa_rejects_bad_dilations() -> None:
    with pytest.raises(ValueError):
        MSPA(32, dilations=(1, 3))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MSPA(32, dilations=(0, 1, 2))


def test_mspa_gap_uses_mean_not_adaptive_pool() -> None:
    """04 §9.3-A / 06 §10-7: `AdaptiveAvgPool2d` 는 ONNX 에 shape 의존 노드를 만든다."""
    m = MSPA(32)
    assert not any(isinstance(sub, nn.AdaptiveAvgPool2d) for sub in m.modules())


# ═══════════════════════════════════════════════════════════════════════════
# 3. BioGate — 크기 보존 gate
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_biogate_initial_gain_is_exactly_one(i: int) -> None:
    """T12: 초기 gain == 1.0 **정확히** (atol 1e-7 이 아니라 비트 단위로 동일).

    `2·sigmoid(0) − 1 == 0.0` 이므로 β 값과 무관하다.
    """
    c = CHANNELS[i - 1]
    g = BioGate(c).eval()
    x = torch.randn(2, c, 8, 8)
    b = torch.randn(2, 2, 8, 8)
    gain = g.gain(b)
    assert gain.shape == (2, c, 8, 8)
    assert torch.equal(gain, torch.ones_like(gain))
    assert torch.equal(g(x, b), x)


def test_biogate_none_is_identity_object() -> None:
    """T12/정정 B-9: `b=None` 은 연산 0 의 항등. zeros 로 채우면 bias 상수 gain 이 된다."""
    g = BioGate(64)
    x = torch.randn(2, 64, 8, 8)
    assert g(x, None) is x
    assert g(x) is x  # 기본값도 None


def test_biogate_none_differs_from_zeros_once_trained() -> None:
    """zeros 대체가 왜 틀린지 고정: fc2.bias 가 학습되면 zeros 는 상수 gain 을 만든다."""
    g = BioGate(32).eval()
    with torch.no_grad():
        g.fc2.bias.fill_(3.0)  # 학습 후 상태를 흉내
    x = torch.randn(2, 32, 8, 8)
    zeros_gain = g(x, torch.zeros(2, 2, 8, 8))
    assert torch.equal(g(x, None), x)
    assert not torch.allclose(zeros_gain, x)


def test_biogate_beta_is_softplus_of_beta_hat() -> None:
    """β̂ init = −2.25217 → β = softplus(β̂) = 0.1 (02 §1.5 표)."""
    g = BioGate(160, beta_init=0.1)
    assert g.beta_hat.numel() == 1
    assert abs(g.beta_hat.item() - (-2.25217)) < 1e-4
    assert abs(g.beta.item() - 0.1) < 1e-6


def test_biogate_gain_bounded_by_beta() -> None:
    """`gain ∈ (1−β, 1+β)` — 크기 폭주 없이 매 block 반복 적용할 수 있는 근거.

    원설계의 `softmax(bio_mask)` 는 ×1/44.8 (채널 수 의존)로 크기를 파괴했다.
    포화 영역에서는 float32 sigmoid 가 0/1 이 되므로 닫힌 구간으로 검사한다.
    """
    g = BioGate(32, beta_init=0.1).eval()
    with torch.no_grad():
        g.fc2.weight.normal_(0.0, 0.5)
        g.fc2.bias.normal_(0.0, 0.5)
    gain = g.gain(torch.randn(2, 2, 16, 16))
    assert (gain > 0.9).all() and (gain < 1.1).all()

    with torch.no_grad():
        g.fc2.weight.normal_(0.0, 50.0)
        g.fc2.bias.normal_(0.0, 50.0)
    gain = g.gain(torch.randn(2, 2, 16, 16) * 10.0)
    assert (gain >= 0.9 - 1e-6).all() and (gain <= 1.1 + 1e-6).all()


def test_biogate_rejects_mismatched_bio() -> None:
    g = BioGate(32)
    x = torch.randn(2, 32, 16, 16)
    with pytest.raises(ValueError):
        g(x, torch.randn(2, 2, 8, 8))  # 해상도 불일치 (pyramid 미적용)
    with pytest.raises(ValueError):
        g(x, torch.randn(2, 3, 16, 16))  # 채널 수 불일치
    with pytest.raises(ValueError):
        BioGate(32, beta_init=0.0)  # softplus 치역 밖


# ═══════════════════════════════════════════════════════════════════════════
# 4. BioSpecBlock / PhysLiteBlock — forward/backward 계약 (02 §1.7)
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_biospec_block_forward_backward(i: int) -> None:
    c, hw = CHANNELS[i - 1], FEATURE_HW[i - 1]
    blk = make_biospec(i, drop_path=0.1)
    blk.train()
    x = torch.randn(2, c, hw, hw, requires_grad=True)
    b = torch.randn(2, 2, hw, hw)
    y = blk(x, b)
    assert y.shape == (2, c, hw, hw)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    grads = [p.grad for p in blk.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    blk.eval()
    assert torch.isfinite(blk(x.detach(), b)).all()


@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_biospec_block_without_bio(i: int) -> None:
    """spec path 가 살아 있어도 지수 전부 결측(F3)일 수 있다 — block 은 그대로 동작한다."""
    c, hw = CHANNELS[i - 1], FEATURE_HW[i - 1]
    blk = make_biospec(i).eval()
    x = torch.randn(2, c, hw, hw)
    y = blk(x, None)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_biospec_block_initial_gate_neutrality() -> None:
    """초기 block 은 gate 중립(att 0.5, gain 1.0) + γ=0.01 이라 사실상 항등에 가깝다."""
    blk = make_biospec(1).eval()
    x = torch.randn(2, 32, 16, 16)
    b = torch.randn(2, 2, 16, 16)
    assert torch.equal(blk.biogate.gain(b), torch.ones(2, 32, 16, 16))
    assert torch.allclose(blk.ls_a.gamma, torch.full((32,), 0.01))
    assert torch.allclose(blk.ls_b.gamma, torch.full((32,), 0.01))
    # 잔차 branch 가 γ=0.01 로 눌려 있으므로 출력은 입력과 같은 크기 규모다.
    y = blk(x, b)
    assert 0.5 < y.std().item() / x.std().item() < 2.0


@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_physlite_forward_backward(i: int) -> None:
    c, hw = CHANNELS[i - 1], FEATURE_HW[i - 1]
    blk = make_physlite(i)
    blk.train()
    x = torch.randn(2, c, hw, hw, requires_grad=True)
    y = blk(x)
    assert y.shape == (2, c, hw, hw)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        torch.isfinite(p.grad).all() for p in blk.parameters() if p.grad is not None
    )


def test_physlite_se_gate_initially_half_and_bounded() -> None:
    """T13: SE gate ∈ [0,1] (Hardsigmoid), 초기 0.5. gate 값은 SE 입출력비로 관측한다."""
    blk = make_physlite(1).eval()
    seen = {}

    def hook(_mod, inp, out):
        seen["in"], seen["out"] = inp[0], out

    h = blk.se.register_forward_hook(hook)
    try:
        blk(torch.randn(2, 32, 16, 16))
        assert torch.allclose(seen["out"], 0.5 * seen["in"], atol=0)
        with torch.no_grad():  # gate 를 흔들어도 [0,1] 을 벗어나지 않는다
            blk.se.fc2.weight.normal_(0.0, 30.0)
            blk.se.fc2.bias.normal_(0.0, 30.0)
        blk(torch.randn(2, 32, 16, 16) * 3.0)
        # out = in * gate 이므로 in != 0 인 곳의 비가 곧 gate 다 (부호 무관).
        mask = seen["in"].abs() > 1e-2
        ratio = (seen["out"][mask] / seen["in"][mask]).detach()
        assert (ratio >= -1e-5).all() and (ratio <= 1.0 + 1e-5).all()
        assert ratio.max() > ratio.min()  # 실제로 변조가 일어났다
    finally:
        h.remove()


def test_physlite_drop_path_is_zero_by_default() -> None:
    """06 §3.3.5 ★ — phys path 는 5 block 뿐이라 stochastic depth 를 쓰지 않는다."""
    blk = make_physlite(3)
    assert blk.dp.p == 0.0
    blk.train()
    x = torch.randn(2, 160, 4, 4)
    assert torch.equal(blk(x), blk(x))  # 확률적 요소 없음


def test_physlite_rejects_even_kernel() -> None:
    with pytest.raises(ValueError):
        PhysLiteBlock(32, dw_kernel=4)


def test_physlite_uses_hardsigmoid_gate_not_raw_dot() -> None:
    """[분석] D-PHY-01 — 활성화 없는 채널 스케일링(발산 장치)이 아님을 구조로 고정."""
    blk = make_physlite(1)
    assert any(isinstance(m, nn.Hardsigmoid) for m in blk.se.modules())
    assert any(isinstance(m, nn.Hardswish) for m in blk.modules())
    assert not any(isinstance(m, nn.AdaptiveAvgPool2d) for m in blk.modules())


# ═══════════════════════════════════════════════════════════════════════════
# 5. (정정 A-37 / B-1) gn8 파라메트릭
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_gn8_builds_every_stage(i: int) -> None:
    """`GroupNorm(8, 20)` 은 ValueError 다 — 빌더가 `gcd(8,C)` 를 쓰는지 확인한다."""
    blk = make_biospec(i, norm="gn8")
    phys = make_physlite(i, norm="gn8")
    for m in list(blk.modules()) + list(phys.modules()):
        if isinstance(m, nn.GroupNorm):
            assert m.num_channels % m.num_groups == 0
            assert m.num_groups == math.gcd(8, m.num_channels)
        assert not isinstance(m, nn.BatchNorm2d), "norm='gn8' 인데 BN 이 직접 생성됐다"
    hw = FEATURE_HW[i - 1]
    x = torch.randn(2, CHANNELS[i - 1], hw, hw)
    y = blk(x, torch.randn(2, 2, hw, hw))
    z = phys(x)
    assert torch.isfinite(y).all() and torch.isfinite(z).all()
    (y.sum() + z.sum()).backward()


def test_gn8_stage3_biogate_uses_four_groups() -> None:
    """유일한 비-8배수 폭: BioGate `C_r = max(8, 160//8) = 20` → `gcd(8,20) = 4` 그룹."""
    g = BioGate(160, norm="gn8")
    assert g.c_r == 20
    assert isinstance(g.norm, nn.GroupNorm)
    assert (g.norm.num_groups, g.norm.num_channels) == (4, 20)
    with pytest.raises(ValueError):
        nn.GroupNorm(8, 20)  # 초판 구현이 여기서 죽었다


@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_gn8_preserves_param_counts(i: int) -> None:
    """BN(C)=GN(g,C)=2C 이므로 §6.1 회귀 상수는 norm 과 무관하다."""
    assert n_params(make_biospec(i, norm="gn8")) == BIOSPEC_PARAMS[i - 1]
    assert n_params(make_physlite(i, norm="gn8")) == PHYSLITE_PARAMS[i - 1]


def test_unknown_norm_kind_raises() -> None:
    with pytest.raises(ValueError):
        make_biospec(1, norm="layernorm")


# ═══════════════════════════════════════════════════════════════════════════
# 6. 초기화 특칙 인계 계약 (02 §1.5)
# ═══════════════════════════════════════════════════════════════════════════
def test_apply_init_encoder_destroys_gate_zeros_and_reset_restores() -> None:
    """상위 모델이 `apply(init_encoder)` 를 뒤에 부르면 zero-init 이 지워진다.

    이 테스트는 그 사실 자체를 고정한다 — encoder 조립자는 반드시
    `apply(init_encoder)` **다음에** `reset_gate_()` 를 돌려야 한다.
    """
    blk = make_biospec(3).eval()
    b = torch.randn(2, 2, 4, 4)
    assert torch.equal(blk.biogate.gain(b), torch.ones(2, 160, 4, 4))

    blk.apply(init_encoder)
    assert not torch.allclose(blk.biogate.gain(b), torch.ones(2, 160, 4, 4))
    _, att = blk.mspa.forward_parts(torch.randn(2, 160, 4, 4))
    assert not torch.equal(att, torch.full_like(att, 0.5))

    for m in blk.modules():
        if hasattr(m, "reset_gate_"):
            m.reset_gate_()
    assert torch.equal(blk.biogate.gain(b), torch.ones(2, 160, 4, 4))
    _, att = blk.mspa.forward_parts(torch.randn(2, 160, 4, 4))
    assert torch.equal(att, torch.full_like(att, 0.5))


def test_physlite_reset_gate_restores_se_neutrality() -> None:
    blk = make_physlite(2).eval()
    blk.apply(init_encoder)
    assert not torch.allclose(blk.se.fc2.weight, torch.zeros_like(blk.se.fc2.weight))
    blk.reset_gate_()
    assert torch.equal(blk.se.fc2.weight, torch.zeros_like(blk.se.fc2.weight))
    assert torch.equal(blk.se.fc2.bias, torch.zeros_like(blk.se.fc2.bias))


def test_layer_scale_gamma_is_per_channel() -> None:
    """02 §1.4 — γ 는 스칼라가 아니라 per-channel (CMNeXt 1e-2 근거)."""
    blk = make_biospec(2, layer_scale_init=0.05)
    assert blk.ls_a.gamma.shape == (64,)
    assert torch.allclose(blk.ls_a.gamma, torch.full((64,), 0.05))
    p = make_physlite(2, layer_scale_init=0.05)
    assert p.ls.gamma.shape == (64,)


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAC — hook 실측 vs 손계산식 vs 06 §6.1 / 02 §7.1 표
# ═══════════════════════════════════════════════════════════════════════════
def mspa_macs(c: int, n: int, c_r: int) -> int:
    return n * (27 * c + 3 * c * c + c) + 2 * c * c_r


def biogate_macs(c: int, n: int, c_r: int, bio_ch: int = 2) -> int:
    return n * (bio_ch * c_r + c_r * c)


def mlp_macs(c: int, n: int, r: int) -> int:
    return n * (2 * r * c * c + 9 * r * c)


def physlite_macs(c: int, n: int, k: int, e: int, c_se: int) -> int:
    return n * (2 * e * c * c + k * k * e * c) + 2 * c * c_se


@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_biospec_block_macs_match_formula(i: int) -> None:
    """hook 실측(B=1, 소형 텐서)이 02 §4.5 의 MAC 산식과 정확히 일치."""
    c, r, hw = CHANNELS[i - 1], MLP_RATIO[i - 1], 8
    blk = make_biospec(i)
    n = hw * hw
    expect = (
        mspa_macs(c, n, blk.mspa.c_r)
        + biogate_macs(c, n, blk.biogate.c_r)
        + mlp_macs(c, n, r)
    )
    rep = count_macs_hooks(blk, (torch.randn(1, c, hw, hw), torch.randn(1, 2, hw, hw)))
    assert rep.total == expect


@pytest.mark.parametrize("i", (1, 2, 3, 4))
def test_physlite_macs_match_formula(i: int) -> None:
    c, k, hw = CHANNELS[i - 1], DW_KERNEL[i - 1], 8
    blk = make_physlite(i)
    c_se = max(8, c // 8)
    expect = physlite_macs(c, hw * hw, k, 2, c_se)
    rep = count_macs_hooks(blk, torch.randn(1, c, hw, hw))
    assert rep.total == expect


def test_mac_formula_reproduces_spec_table_at_512() -> None:
    """정정 A-24 규약: 실측은 소형에서 하고 @512² 표 대조는 산식으로 한다 (텐서 미생성).

    02 §7.1 block 당 MMAC — BioSpec 375.7/347.1/302.2/296.9, PhysLite 76.5/71.8/113.1/109.0.
    """
    n_at_512 = (128 * 128, 64 * 64, 32 * 32, 16 * 16)
    bio_expect = (375.7, 347.1, 302.2, 296.9)
    phy_expect = (76.5, 71.8, 113.1, 109.0)
    for i, c in enumerate(CHANNELS):
        n, r, k = n_at_512[i], MLP_RATIO[i], DW_KERNEL[i]
        bio = (
            mspa_macs(c, n, max(8, c // 4))
            + biogate_macs(c, n, max(8, c // 8))
            + mlp_macs(c, n, r)
        )
        phy = physlite_macs(c, n, k, 2, max(8, c // 8))
        assert abs(bio / 1e6 - bio_expect[i]) < 0.1, (i, bio / 1e6)
        assert abs(phy / 1e6 - phy_expect[i]) < 0.1, (i, phy / 1e6)


def test_spec_path_total_macs_at_512() -> None:
    """spec path 3,833.0 MMAC / phys path 771.3 MMAC (downsample 제외분 합산 검증)."""
    n_at_512 = (128 * 128, 64 * 64, 32 * 32, 16 * 16)
    ds_macs = (18_432 * 64 * 64, 92_160 * 32 * 32, 460_800 * 16 * 16)  # 3·3·Ci·Co·N_out
    spec = sum(
        SPEC_DEPTHS[i]
        * (
            mspa_macs(c, n_at_512[i], max(8, c // 4))
            + biogate_macs(c, n_at_512[i], max(8, c // 8))
            + mlp_macs(c, n_at_512[i], MLP_RATIO[i])
        )
        for i, c in enumerate(CHANNELS)
    ) + sum(ds_macs)
    phys = sum(
        PHYS_DEPTHS[i]
        * physlite_macs(c, n_at_512[i], DW_KERNEL[i], 2, max(8, c // 8))
        for i, c in enumerate(CHANNELS)
    ) + sum(ds_macs)
    assert abs(spec / 1e6 - 3_833.0) < 1.0, spec / 1e6
    assert abs(phys / 1e6 - 771.3) < 1.0, phys / 1e6
