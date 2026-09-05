"""T11 — `modules/blocks_ela.py` (RGB path) 계약 테스트.

정본: 02 §3 / §9, 06 §3.3.3 · T11, 정정 B-1(gn8) · B-5(qkv 인터리브) · B-6(fp32 강제).

헌법 C-5.1/C-5.2: **CPU 전용**, B<=2, 64x64 이하 소형 텐서.
중점:
  (a) 파라미터 수가 §7.1 회귀 상수와 **정확히** 일치,
  (b) qkv 채널 배치가 head 별 인터리브임을 naive 참조로 고정 (조용한 오모델 차단),
  (c) attention 이 autocast 하에서도 fp32 로 계산됨을 **판별력 있게** 검증,
  (d) ε 가 0-division 을 막고 큰 입력에서도 finite,
  (e) fewer-norm / residual 형태 / 초기화 규약.
"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Tuple

import pytest
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # cwd 와 무관하게 bloomnet 패키지를 찾게 한다
    sys.path.insert(0, str(_ROOT))

from bloomnet.modules import blocks_ela  # noqa: E402
from bloomnet.modules.blocks_ela import (  # noqa: E402
    ELAGlobal,
    ELALocal,
    LiteMLA,
    MBConv,
    relu_linear_attention,
)


# 02 §3.3 head_dim 스케줄: stage 1/2 = 16, stage 3/4 = 32
STAGE_CD = ((32, 16), (64, 16), (160, 32), (320, 32))


@pytest.fixture(autouse=True)
def _cpu_only() -> None:
    """헌법 C-5.2 — GPU 절대 금지. CUDA_VISIBLE_DEVICES="" 로 실행할 것."""
    assert not torch.cuda.is_available(), "GPU 가 보인다 — CUDA_VISIBLE_DEVICES='' 로 실행하라"


def nparam(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def mbconv_params(c: int, e: int = 4, k: int = 3) -> int:
    """`e·C²  + eC  (1x1 bias 포함) + k²·eC + eC (DW bias) + e·C² + 2C`."""
    return 2 * e * c * c + (k * k + 2) * e * c + 2 * c


# ─────────────────────────────────────────────────────────────────────────────
# 1. 파라미터 수 (T11 / 02 §7.1 회귀 상수)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("c", [32, 64, 160, 320])
def test_mbconv_params_formula(c: int) -> None:
    """02 §3.2: params = 8C² + 46C (e=4, k=3)."""
    assert mbconv_params(c) == 8 * c * c + 46 * c  # 손 계산식 자체의 무결성
    assert nparam(MBConv(c)) == 8 * c * c + 46 * c


def test_ela_local_params_t11() -> None:
    """T11 회귀 상수: ELALocal(32)=9,696 / ELALocal(64)=35,776 (= 8C²+47C)."""
    assert nparam(ELALocal(32)) == 9_696
    assert nparam(ELALocal(64)) == 35_776
    for c in (32, 64, 160, 320):
        assert nparam(ELALocal(c)) == 8 * c * c + 47 * c


@pytest.mark.parametrize("c,d", STAGE_CD)
def test_litemla_params_formula(c: int, d: int) -> None:
    """02 §3.3: params = 6C² + 6·C·d + 104C."""
    assert nparam(LiteMLA(c, head_dim=d)) == 6 * c * c + 6 * c * d + 104 * c


def test_litemla_params_literals() -> None:
    """02 §3.3 표의 리터럴 (stage3/4) 및 §3.1 비용표의 stage1/2 리터럴."""
    assert nparam(LiteMLA(160, head_dim=32)) == 200_960
    assert nparam(LiteMLA(320, head_dim=32)) == 709_120
    assert nparam(LiteMLA(32, head_dim=16)) == 12_544
    assert nparam(LiteMLA(64, head_dim=16)) == 37_376


def test_ela_global_params_t11() -> None:
    """T11 회귀 상수: ELAGlobal(160,32)=413,440 / (320,32)=1,543,680 (= 14C²+6Cd+152C)."""
    assert nparam(ELAGlobal(160, head_dim=32)) == 413_440
    assert nparam(ELAGlobal(320, head_dim=32)) == 1_543_680
    for c, d in STAGE_CD:
        assert nparam(ELAGlobal(c, head_dim=d)) == 14 * c * c + 6 * c * d + 152 * c


def test_rgb_path_block_totals_match_encoder_table() -> None:
    """02 §7.1 RGB path stage 합(downsample 제외)과 정확히 일치해야 한다."""
    depths = (2, 2, 4, 3)
    per_stage = (
        nparam(ELALocal(32)),
        nparam(ELALocal(64)),
        nparam(ELAGlobal(160, head_dim=32)),
        nparam(ELAGlobal(320, head_dim=32)),
    )
    stage_sums = [p * n for p, n in zip(per_stage, depths)]
    assert stage_sums == [19_392, 71_552, 1_653_760, 4_631_040]
    # §7.1 RGB 합 6,948,224 − downsample(18,560 + 92,480 + 461,440 = 572,480)
    assert sum(stage_sums) == 6_948_224 - 572_480 == 6_375_744


def test_attn_at_stage12_extra_cost() -> None:
    """02 §3.1 비용 논증: stage1–2 에 LiteMLA 를 켰을 때의 추가 파라미터.

    주의(스펙 인계): §3.1 본문의 합계 **+100,224 은 산술 오류**다.
    같은 표가 제시하는 성분(`12,544×2 + LS 64`, `37,376×2 + LS 128`)을 그대로 더하면
    25,152 + 74,880 = **100,032** 이다(오차 192). 성분값은 본 구현과 정확히 일치한다.
    이 값은 비용 논증용이며 attn_stages=(3,4) 기본에서는 어떤 회귀 상수에도 영향이 없다.
    """
    extra_s1 = (nparam(ELAGlobal(32, head_dim=16)) - nparam(ELALocal(32))) * 2
    extra_s2 = (nparam(ELAGlobal(64, head_dim=16)) - nparam(ELALocal(64))) * 2
    assert extra_s1 == 12_544 * 2 + 64  # LiteMLA ×2 + LayerScale(32) ×2
    assert extra_s2 == 37_376 * 2 + 128
    assert extra_s1 + extra_s2 == 100_032


# ─────────────────────────────────────────────────────────────────────────────
# 2. shape / finite / backward (02 §1.7 공통 스모크)
# ─────────────────────────────────────────────────────────────────────────────
_SMOKE = {
    # 02 §6.4 stage 출력 표를 64² 입력으로 축소한 것 (H/4=16, H/8=8, H/16=4, H/32=2)
    "ELALocal_s1": (lambda: ELALocal(32), (2, 32, 16, 16)),
    "ELALocal_s2": (lambda: ELALocal(64), (2, 64, 8, 8)),
    "ELAGlobal_s3": (lambda: ELAGlobal(160, head_dim=32), (2, 160, 4, 4)),
    "ELAGlobal_s4": (lambda: ELAGlobal(320, head_dim=32), (2, 320, 2, 2)),
    "MBConv": (lambda: MBConv(32), (2, 32, 16, 16)),
    "LiteMLA": (lambda: LiteMLA(160, head_dim=32), (2, 160, 4, 4)),
}


@pytest.mark.parametrize("name", sorted(_SMOKE))
def test_smoke_forward_backward(name: str) -> None:
    factory, shape = _SMOKE[name]
    torch.manual_seed(0)
    mod = factory()
    mod.train()
    x = torch.randn(*shape, requires_grad=True)
    y = mod(x)
    assert y.shape == x.shape, f"{name}: {tuple(y.shape)} != {shape}"
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    grads = [p.grad for p in mod.parameters() if p.grad is not None]
    assert len(grads) == len(list(mod.parameters())), f"{name}: grad 가 없는 파라미터가 있다"
    assert all(torch.isfinite(g).all() for g in grads)
    mod.eval()
    with torch.no_grad():
        assert torch.isfinite(mod(x)).all()


@pytest.mark.parametrize("hw", [(8, 16), (12, 4), (6, 6)])
def test_non_square_and_odd_sizes(hw: Tuple[int, int]) -> None:
    """PE 가 없으므로(02 §3.4) 임의 H,W 에서 shape 이 보존되어야 한다."""
    h, w = hw
    m = ELAGlobal(64, head_dim=16).eval()
    x = torch.randn(2, 64, h, w)
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_eval_is_deterministic() -> None:
    m = ELAGlobal(160, head_dim=32, drop_path=0.5).eval()
    x = torch.randn(2, 160, 4, 4)
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.equal(a, b), "eval 에서 DropPath 가 항등이 아니다"


# ─────────────────────────────────────────────────────────────────────────────
# 3. (정정 B-5) qkv 채널 배치 = head 별 인터리브
# ─────────────────────────────────────────────────────────────────────────────
def _naive_attention_from_module(
    mla: LiteMLA, x: torch.Tensor, *, interleaved: bool
) -> torch.Tensor:
    """모듈의 실제 weight 로 명시 슬라이스 참조 구현을 만든다.

    interleaved=True  : head 별 [q_i, k_i, v_i] 연속 블록  (정본)
    interleaved=False : [Q_all | K_all | V_all] 블록 연접  (금지된 오모델)
    """
    d = mla.head_dim
    base = mla.qkv(x)
    ms = torch.cat([base, *(agg(base) for agg in mla.aggreg)], dim=1)
    b, c_ms, h, w = ms.shape
    n = h * w
    outs = []
    if interleaved:
        for g in range(c_ms // (3 * d)):
            blk = ms[:, g * 3 * d : (g + 1) * 3 * d].reshape(b, 3 * d, n).transpose(-1, -2)
            q, k, v = blk[..., :d], blk[..., d : 2 * d], blk[..., 2 * d :]
            outs.append(relu_linear_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)))
    else:
        # scale 덩어리(3C)마다 [Q_all|K_all|V_all] 로 잘못 해석했을 때
        c3 = c_ms // (1 + len(mla.scales))  # = 3C
        cc = c3 // 3
        for s in range(1 + len(mla.scales)):
            chunk = ms[:, s * c3 : (s + 1) * c3].reshape(b, c3, n).transpose(-1, -2)
            for hh in range(mla.heads):
                q = chunk[..., hh * d : (hh + 1) * d]
                k = chunk[..., cc + hh * d : cc + (hh + 1) * d]
                v = chunk[..., 2 * cc + hh * d : 2 * cc + (hh + 1) * d]
                outs.append(relu_linear_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)))
    out = torch.cat(outs, dim=1)
    out = out.transpose(-1, -2).reshape(b, -1, h, w)
    return mla.proj_norm(mla.proj(out))


@pytest.mark.parametrize("c,d", [(32, 32), (64, 32), (160, 32)])
def test_qkv_layout_matches_interleaved_reference(c: int, d: int) -> None:
    """정정 B-5: reshape/chunk 결과가 head 별 인터리브 슬라이스와 동일해야 한다."""
    torch.manual_seed(1)
    mla = LiteMLA(c, head_dim=d).eval()
    x = torch.randn(2, c, 4, 4)
    with torch.no_grad():
        ref = _naive_attention_from_module(mla, x, interleaved=True)
        got = mla(x)
    assert torch.allclose(ref, got, atol=1e-5), f"maxdiff={(ref - got).abs().max():.3e}"


def test_qkv_layout_test_is_discriminating() -> None:
    """위 테스트가 판별력이 있음을 증명 — 잘못된 [Q|K|V] 블록 배치는 다른 값을 낸다."""
    torch.manual_seed(2)
    mla = LiteMLA(64, head_dim=32).eval()  # h=2 (h=1 이면 두 배치가 동일해진다)
    x = torch.randn(2, 64, 4, 4)
    with torch.no_grad():
        good = _naive_attention_from_module(mla, x, interleaved=True)
        bad = _naive_attention_from_module(mla, x, interleaved=False)
    assert not torch.allclose(good, bad, atol=1e-3), "판별력 없음 — 테스트가 무의미하다"


def test_ones_column_trick_gives_exact_one() -> None:
    """v ≡ 1 이면 분자/분모가 같아져 출력이 정확히 1 이어야 한다 (02 §3.3 (b) 안)."""
    torch.manual_seed(3)
    q = torch.rand(2, 3, 7, 8)  # ReLU 후에도 양수가 남도록 uniform
    k = torch.rand(2, 3, 7, 8)
    v = torch.ones(2, 3, 7, 8)
    out = relu_linear_attention(q, k, v, eps=1e-12)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_attention_matches_explicit_sum_formula() -> None:
    """O_i = ReLU(q_i)·Σ_j ReLU(k_j)ᵀv_j / (ReLU(q_i)·Σ_j ReLU(k_j) + ε) 를 루프로 재현."""
    torch.manual_seed(4)
    b, g, n, d = 1, 2, 5, 3
    q, k, v = (torch.randn(b, g, n, d) for _ in range(3))
    eps = 1e-5
    got = relu_linear_attention(q, k, v, eps)
    qr, kr = torch.relu(q), torch.relu(k)
    ref = torch.empty_like(got)
    for bb in range(b):
        for gg in range(g):
            num = torch.zeros(d, d)
            den = torch.zeros(d)
            for j in range(n):
                num += kr[bb, gg, j].unsqueeze(1) * v[bb, gg, j].unsqueeze(0)
                den += kr[bb, gg, j]
            for i in range(n):
                ref[bb, gg, i] = (qr[bb, gg, i] @ num) / (qr[bb, gg, i] @ den + eps)
    assert torch.allclose(ref, got, atol=1e-5), f"maxdiff={(ref - got).abs().max():.3e}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. ε / fp32 강제 / 수치 안정성 (T11, 정정 B-6)
# ─────────────────────────────────────────────────────────────────────────────
def test_litemla_eps_blocks_zero_division() -> None:
    """T11: x=zeros 로 ReLU 출력을 전부 0 으로 만들어도 출력 finite."""
    m = LiteMLA(160, head_dim=32)
    for mode in ("train", "eval"):
        m.train(mode == "train")
        z = torch.zeros(2, 160, 4, 4, requires_grad=True)
        y = m(z)
        assert torch.isfinite(y).all(), f"{mode}: 0-division 이 막히지 않았다"
        y.sum().backward()
        assert torch.isfinite(z.grad).all()


def test_attention_zero_input_is_exactly_zero() -> None:
    q = k = v = torch.zeros(2, 2, 4, 8)
    out = relu_linear_attention(q, k, v)
    assert torch.isfinite(out).all() and float(out.abs().max()) == 0.0


def test_attention_forced_fp32_under_bf16_autocast() -> None:
    """정정 B-6: autocast 하에서도 두 matmul 과 나눗셈이 float32 여야 한다."""
    torch.manual_seed(5)
    q, k, v = (torch.randn(2, 4, 64, 32) for _ in range(3))
    ref = relu_linear_attention(q, k, v)  # autocast 밖 = 순수 fp32
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got = relu_linear_attention(q, k, v)
    assert got.dtype is torch.float32
    assert torch.allclose(ref, got, atol=1e-6, rtol=0), f"maxdiff={(ref - got).abs().max():.3e}"


def test_fp32_forcing_test_is_discriminating() -> None:
    """같은 계산을 autocast 를 끄지 않고 하면 bf16 로 강등되어 실제로 값이 달라진다."""
    torch.manual_seed(5)
    q, k, v = (torch.randn(2, 4, 64, 32) for _ in range(3))
    ref = relu_linear_attention(q, k, v)
    qr, kr = torch.relu(q), torch.relu(k)
    vp = torch.nn.functional.pad(v, (0, 1), value=1.0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        kv = kr.transpose(-1, -2) @ vp  # ← autocast 가 bf16 으로 강등
        unforced = qr @ kv
        unforced = unforced[..., :-1] / (unforced[..., -1:] + 1e-5)
    assert unforced.dtype is torch.bfloat16
    diff = (ref - unforced.float()).abs().max()
    assert diff > 1e-6, f"판별력 없음 (diff={diff:.3e}) — fp32 강제 테스트가 무의미하다"


def test_litemla_forward_finite_under_bf16_autocast() -> None:
    """T11: fp32 강제 경로가 bf16 autocast 하에서도 finite."""
    m = LiteMLA(160, head_dim=32).eval()
    x = torch.randn(2, 160, 8, 8)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = m(x)
    assert torch.isfinite(y.float()).all()


@pytest.mark.parametrize("scale", [1.0, 20.0, 200.0])
def test_large_input_stability(scale: float) -> None:
    """fewer-norm 이라 입력 std 가 1 을 넘는 것이 정상 — 큰 값에서도 finite 해야 한다."""
    torch.manual_seed(6)
    m = ELAGlobal(160, head_dim=32)
    m.train()
    x = torch.randn(2, 160, 8, 8) * scale
    x.requires_grad_(True)
    y = m(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_attention_output_is_bounded_by_v_range() -> None:
    """볼록결합 성질: 출력은 v 의 [min,max] 범위를 벗어날 수 없다(ε 무시 가능 조건)."""
    torch.manual_seed(7)
    q = torch.rand(2, 3, 16, 8) + 0.5  # 분모가 0 에서 충분히 떨어지도록
    k = torch.rand(2, 3, 16, 8) + 0.5
    v = torch.randn(2, 3, 16, 8) * 5
    out = relu_linear_attention(q, k, v)
    lo, hi = v.amin(dim=2, keepdim=True), v.amax(dim=2, keepdim=True)
    assert bool((out >= lo - 1e-4).all()) and bool((out <= hi + 1e-4).all())


# ─────────────────────────────────────────────────────────────────────────────
# 5. (정정 B-1) norm 파라메트릭 — bn / gn8
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("c,d", STAGE_CD)
def test_gn8_parametric_params_and_forward(c: int, d: int) -> None:
    """norm 을 바꿔도 파라미터 수가 동일하고 forward/backward 가 finite 해야 한다."""
    bn = ELAGlobal(c, head_dim=d, norm="bn")
    gn = ELAGlobal(c, head_dim=d, norm="gn8")
    assert nparam(bn) == nparam(gn), "gn8 이 파라미터 수를 바꾸면 §7.1 회귀 상수가 깨진다"
    assert any(isinstance(m, nn.GroupNorm) for m in gn.modules())
    for m in gn.modules():
        if isinstance(m, nn.GroupNorm):
            assert m.num_groups == math.gcd(8, m.num_channels)
    x = torch.randn(2, c, 4, 4, requires_grad=True)
    y = gn(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_gn8_handles_non_multiple_of_8_width() -> None:
    """정정 B-1 의 gcd 빌더가 실제로 쓰이는지 — C=20 은 GroupNorm(4,20) 이어야 한다."""
    m = MBConv(20, norm="gn8")
    gns = [g for g in m.modules() if isinstance(g, nn.GroupNorm)]
    assert len(gns) == 1 and gns[0].num_groups == 4 and gns[0].num_channels == 20
    assert nparam(m) == nparam(MBConv(20, norm="bn"))
    y = m(torch.randn(2, 20, 8, 8))
    assert torch.isfinite(y).all()


def test_syncbn_is_constructible() -> None:
    """`model.norm == "syncbn"` 스위치가 모델 생성 단계에서 죽지 않아야 한다."""
    m = ELALocal(32, norm="syncbn")
    assert any(isinstance(x, nn.SyncBatchNorm) for x in m.modules())
    assert nparam(m) == 9_696


def test_unknown_norm_raises() -> None:
    with pytest.raises(ValueError):
        MBConv(32, norm="layernorm")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 구조 계약 — fewer-norm / bias / residual / 초기화
# ─────────────────────────────────────────────────────────────────────────────
def test_mbconv_fewer_norm_and_bias_contract() -> None:
    """02 §3.2: use_bias=(T,T,F), norm=(None,None,BN), act=(hswish,hswish,None)."""
    m = MBConv(32)
    assert m.inverted_conv.bias is not None
    assert m.depth_conv.bias is not None
    assert m.point_conv.bias is None
    assert m.depth_conv.groups == m.depth_conv.in_channels == 4 * 32
    assert isinstance(m.act1, nn.Hardswish) and isinstance(m.act2, nn.Hardswish)
    assert sum(isinstance(x, (nn.BatchNorm2d, nn.GroupNorm)) for x in m.modules()) == 1


def test_litemla_fewer_norm_and_bias_contract() -> None:
    """qkv/aggreg/proj 는 전부 bias=False, norm 은 proj 뒤 1 개뿐."""
    m = LiteMLA(160, head_dim=32)
    assert m.qkv.bias is None and m.proj.bias is None
    assert m.qkv.out_channels == 3 * 160 and m.proj.in_channels == 3 * 160
    assert sum(isinstance(x, (nn.BatchNorm2d, nn.GroupNorm)) for x in m.modules()) == 1
    assert m.heads == 5
    for seq, s in zip(m.aggreg, m.scales):
        dw, pw = seq[0], seq[1]
        assert dw.bias is None and pw.bias is None
        assert dw.kernel_size == (s, s) and dw.padding == (s // 2, s // 2)
        assert dw.groups == 3 * 160  # depthwise
        assert pw.groups == 3 * m.heads  # per-head grouped mixing (그룹 크기 = head_dim)
        assert pw.in_channels // pw.groups == m.head_dim


def test_ela_global_has_exactly_two_residual_branches() -> None:
    m = ELAGlobal(160, head_dim=32)
    assert sum(isinstance(x, (nn.BatchNorm2d, nn.GroupNorm)) for x in m.modules()) == 2
    assert m.ls_ctx.gamma.shape == (160,) and m.ls_local.gamma.shape == (160,)
    assert m.drop_path_ctx is not m.drop_path_local, "두 분기는 독립 DropPath 여야 한다"


@pytest.mark.parametrize("factory", [lambda: ELALocal(32), lambda: ELAGlobal(32, head_dim=16)])
def test_layer_scale_init_is_0p01_per_channel(factory) -> None:
    m = factory()
    gammas = [p for n, p in m.named_parameters() if n.endswith("gamma")]
    assert gammas, "LayerScale gamma 가 없다"
    for g in gammas:
        assert g.shape == (32,), "gamma 는 per-channel 이어야 한다 (스칼라 금지)"
        assert torch.allclose(g, torch.full_like(g, 0.01), atol=0)


def test_layer_scale_init_is_configurable() -> None:
    m = ELAGlobal(64, head_dim=16, layer_scale_init=0.1)
    for n, p in m.named_parameters():
        if n.endswith("gamma"):
            assert torch.allclose(p, torch.full_like(p, 0.1), atol=0)


@pytest.mark.parametrize(
    "factory,c", [(lambda: ELALocal(32), 32), (lambda: ELAGlobal(160, head_dim=32), 160)]
)
def test_zero_gamma_makes_block_identity(factory, c: int) -> None:
    """residual 형태 `x + DropPath(gamma * Branch(x))` 의 구조적 증명."""
    m = factory().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("gamma"):
                p.zero_()
    x = torch.randn(2, c, 4, 4)
    with torch.no_grad():
        y = m(x)
    assert torch.equal(y, x), "gamma=0 인데 항등이 아니다 — 잔차 배선이 틀렸다"


def test_init_encoder_applied() -> None:
    """02 §1.5: 1x1 -> trunc_normal(0.02), k>=3 -> kaiming fan_out, BN -> (1, 0)."""
    torch.manual_seed(0)
    m = ELAGlobal(160, head_dim=32)
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            std = float(mod.weight.detach().std())
            if mod.kernel_size == (1, 1):
                assert 0.008 < std < 0.032, f"{name}: 1x1 std={std:.4f}"
            else:
                # torch 의 fan_out = out_channels × k² (groups 를 나누지 않는다)
                fan_out = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
                want = math.sqrt(2.0 / fan_out)
                assert 0.6 * want < std < 1.6 * want, f"{name}: {std:.5f} vs want {want:.5f}"
            if mod.bias is not None:
                assert float(mod.bias.detach().abs().max()) == 0.0
        elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
            assert torch.allclose(mod.weight.detach(), torch.ones_like(mod.weight))
            assert float(mod.bias.detach().abs().max()) == 0.0


def test_drop_path_is_stochastic_in_train_and_finite() -> None:
    torch.manual_seed(0)
    m = ELALocal(32, drop_path=0.5).train()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("gamma"):
                p.fill_(1.0)  # 분기를 크게 만들어 drop 여부가 보이게
    x = torch.randn(2, 32, 8, 8)
    outs = [m(x) for _ in range(24)]
    assert all(torch.isfinite(o).all() for o in outs)
    assert any(not torch.equal(outs[0], o) for o in outs[1:]), "train 에서 DropPath 가 작동하지 않는다"


# ─────────────────────────────────────────────────────────────────────────────
# 7. 인자 검증 / MAC 대조
# ─────────────────────────────────────────────────────────────────────────────
def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError):
        LiteMLA(160, head_dim=48)  # 160 % 48 != 0
    with pytest.raises(ValueError):
        LiteMLA(160, head_dim=32, scales=(2, 4))  # 짝수 커널 -> 비대칭 padding
    with pytest.raises(ValueError):
        MBConv(32, kernel_size=4)
    with pytest.raises(ValueError):
        MBConv(0)
    with pytest.raises(ValueError):
        relu_linear_attention(
            torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8), torch.randn(1, 1, 3, 8)
        )


def _spec_litemla_macs(c: int, d: int, n: int, scales: Tuple[int, ...] = (3, 5)) -> int:
    """02 §3.3 MAC 식: qkv + DW + grouped 1x1 + attention + proj."""
    qkv = 3 * c * c
    dw = sum(3 * c * s * s for s in scales)
    grouped = len(scales) * 3 * c * d
    attn = 2 * (3 * c) * (d + 1)  # 2 matmul × G=3h groups × d(d+1) 를 토큰당으로 (3h·d = 3C)
    proj = c * (1 + len(scales)) * c
    return n * (qkv + dw + grouped + attn + proj)


def test_mac_matches_spec_formula() -> None:
    """utils/flops 로 실측한 MAC 이 02 §3.3 / §3.2 손 계산식과 일치해야 한다."""
    from bloomnet.utils.flops import count_macs_flop_counter

    c, d, hw = 160, 32, 4
    n = hw * hw
    got = count_macs_flop_counter(LiteMLA(c, head_dim=d), torch.randn(1, c, hw, hw)).total
    assert got == _spec_litemla_macs(c, d, n), f"LiteMLA {got} != {_spec_litemla_macs(c, d, n)}"

    got_mb = count_macs_flop_counter(MBConv(c), torch.randn(1, c, hw, hw)).total
    assert got_mb == n * (8 * c * c + 36 * c)

    got_g = count_macs_flop_counter(ELAGlobal(c, head_dim=d), torch.randn(1, c, hw, hw)).total
    assert got_g == _spec_litemla_macs(c, d, n) + n * (8 * c * c + 36 * c)


def test_stage3_block_mac_at_512_matches_table() -> None:
    """02 §3.3 표: stage3 block 453.5 MMAC / stage4 410.3 MMAC @512² (N 선형 스케일)."""
    from bloomnet.utils.flops import count_macs_flop_counter, scale_macs

    m3 = count_macs_flop_counter(ELAGlobal(160, head_dim=32), torch.randn(1, 160, 8, 8))
    at512 = scale_macs(m3.total, from_hw=(8, 8), to_hw=(32, 32))  # stage3 @512² = 32²
    assert abs(at512 / 1e6 - 453.5) < 0.5, at512 / 1e6

    m4 = count_macs_flop_counter(ELAGlobal(320, head_dim=32), torch.randn(1, 320, 4, 4))
    at512_4 = scale_macs(m4.total, from_hw=(4, 4), to_hw=(16, 16))  # stage4 @512² = 16²
    assert abs(at512_4 / 1e6 - 410.3) < 0.5, at512_4 / 1e6


def test_relu_linear_attention_is_module_level_hookable() -> None:
    """LiteMLA 가 모듈 전역 함수를 호출해야 한다(테스트·프로파일러가 감쌀 수 있도록)."""
    calls: list[torch.dtype] = []
    orig = blocks_ela.relu_linear_attention

    def spy(q, k, v, eps=1e-5):
        calls.append(q.dtype)
        return orig(q, k, v, eps)

    blocks_ela.relu_linear_attention = spy  # type: ignore[assignment]
    try:
        LiteMLA(64, head_dim=32).eval()(torch.randn(2, 64, 4, 4))
    finally:
        blocks_ela.relu_linear_attention = orig  # type: ignore[assignment]
    assert calls == [torch.float32]
