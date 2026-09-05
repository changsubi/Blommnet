"""S0-Spec 사전학습 테스트 — 01 §6.5~§6.9, 06 §3.6 / T22.

헌법 C-5.1/C-5.2: **CPU 전용**, 소형 텐서. GPU 사용 금지.

중점
    (a) ``SpecMLP`` 파라미터 1,089 / 1,122 와 초기화 특칙(첫 예측 = 전역 평균)
    (b) LOSO 가 site 를 **절대** 섞지 않는다 (01 §6.4 [S1]/[S4])
    (c) 이식 게이트 G1/G2/G3 가 임계값대로 판정한다 (경계값 포함)
    (d) 이식 T1/T1′/T2/T4/T4′/T5 의 수치 계약 + **T3 이 복사되지 않음**
    (e) 게이트 실패·shape 불일치 시 **모델이 한 바이트도 바뀌지 않음** (T7 / 원자성)
    (f) T0 실패 시 raw 밴드 열이 제외되고 bio 열만 이식됨 (정정 A-3)
"""

from __future__ import annotations

import math
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterator, List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # cwd 와 무관하게 bloomnet 패키지를 찾게 한다
    sys.path.insert(0, str(_ROOT))

from bloomnet.constants import (  # noqa: E402
    K235_ALL_SITES,
    K235_LOG1P_MEAN,
    K235_TRAIN_SITES,
    K235_VAL_SITES,
    MSI_SLOTS,
)
from bloomnet.modules.heads import ChlHead, RegTrunk  # noqa: E402
from bloomnet.modules.stems import SPS  # noqa: E402
from bloomnet.pretrain import loso as loso_mod  # noqa: E402
from bloomnet.pretrain.loso import (  # noqa: E402
    GATE_DEFAULTS,
    RUN_DEFAULTS,
    format_gate_report,
    load_k235_arrays,
    run_holdout,
    run_loso,
    transplant_gate,
)
from bloomnet.pretrain.spec_mlp import (  # noqa: E402
    HEAD_BIAS_DEFAULT,
    SpecMLP,
    init_spec_mlp,
    softplus_inv,
    spec_loss,
)
from bloomnet.pretrain.transplant import (  # noqa: E402
    T0_TOL,
    check_t0,
    find_chl_head,
    find_sps,
    summarize,
    transplant_to_bloomnet,
)

REPO_ROOT = pathlib.Path(_ROOT)
REAL_NPZ = REPO_ROOT / "bloomnet" / "data" / "cache" / "k235_ms.npz"
def _needs_data(missing: bool, reason: str):
    """실데이터 의존 표시 + 부재 시 skip (06 §5.0 — CI 기본 게이트는 ``-m "not data"``)."""
    def deco(fn):
        return pytest.mark.data(pytest.mark.skipif(missing, reason=reason)(fn))
    return deco


needs_real_npz = _needs_data(
    not REAL_NPZ.exists(), f"missing {REAL_NPZ} (tools/build_k235_cache 로 생성)"
)

#: 01 §6.8 [M9] 베이스라인 표의 fold 별 n (QC 필터 후). 재현되면 파이프라인 전체가 맞다.
M9_FOLD_N: Dict[str, int] = {
    "BES": 2894,
    "CNH": 2842,
    "CSC": 3104,
    "GDK": 1687,
    "HNM": 2508,
    "JAD": 2944,
    "SHC": 2986,
    "YJD": 2546,
}


@pytest.fixture(autouse=True)
def _no_gpu() -> None:
    """헌법 C-5.2 — 구현·테스트 중 GPU 를 절대 쓰지 않는다."""
    assert not torch.cuda.is_available(), "GPU must be invisible (CUDA_VISIBLE_DEVICES='')"


@pytest.fixture()
def kw_tmp() -> Iterator[pathlib.Path]:
    base = REPO_ROOT / ".pytest_tmp"
    base.mkdir(exist_ok=True)
    path = pathlib.Path(tempfile.mkdtemp(dir=str(base)))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        try:
            base.rmdir()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  합성 npz (K235ChipDataset 이 읽는 캐시 규격 — 01 §7.6)
# ═══════════════════════════════════════════════════════════════════════════
def make_npz(
    path: pathlib.Path,
    *,
    sites: List[str],
    n_per_site: int = 24,
    seed: int = 0,
) -> pathlib.Path:
    """site 별로 **분광 신호가 확연히 다른** 합성 캐시.

    site k 의 red 밴드에 ``0.001*(k+1)`` 의 오프셋을 넣어, 학습 배치에 어떤 site 가
    들어갔는지 텐서만 보고 역추적할 수 있게 한다(누수 테스트용 marker).
    """
    rng = np.random.default_rng(seed)
    n = len(sites) * n_per_site
    spectra = 0.03 + 0.004 * rng.standard_normal((n, 5))
    site_arr, date_arr, chla = [], [], []
    for k, s in enumerate(sites):
        sl = slice(k * n_per_site, (k + 1) * n_per_site)
        spectra[sl, 2] = 0.03 + 0.001 * (k + 1)  # red = site marker
        site_arr += [s] * n_per_site
        date_arr += [f"2022{7 + (k % 3):02d}{1 + (k % 20):02d}"] * n_per_site
        chla += list(np.abs(1.0 + 2.0 * (k + 1) + rng.standard_normal(n_per_site)))
    spectra = np.clip(spectra, 1e-3, 0.9).astype(np.float32)

    payload: Dict[str, np.ndarray] = {
        "spectra": spectra,
        "spectra_sd": np.zeros_like(spectra),
        "site": np.array(site_arr, dtype="<U3"),
        "date": np.array(date_arr, dtype="<U8"),
        "time": np.array(["120000"] * n, dtype="<U6"),
        "stem": np.array([f"M_{i:05d}" for i in range(n)], dtype="<U32"),
        "split_src": np.array(["Training"] * n, dtype="<U10"),
        "lat": np.zeros(n),
        "lon": np.zeros(n),
        "band_order": np.array("blue,green,red,rededge1,nir", dtype="<U40"),
        "chla": np.asarray(chla, dtype=np.float32),
        "phyco": np.zeros(n, dtype=np.float32),
        "watertemp": np.full(n, 20.0, dtype=np.float32),
        "odomg": np.full(n, 8.0, dtype=np.float32),
        "ec": np.zeros(n, dtype=np.float32),
        "odos": np.zeros(n, dtype=np.float32),
        "temp": np.zeros(n, dtype=np.float32),
        "humidity": np.zeros(n, dtype=np.float32),
        "windspeed": np.zeros(n, dtype=np.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


FAST: Dict[str, Any] = dict(
    epochs=2, batch_size=32, warmup_epochs=1, band_order_check="off", verbose=False
)


def make_model() -> nn.Module:
    """이식 대상 최소 모델 — SPS + RegTrunk + ChlHead 만 있으면 계약이 닫힌다."""

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.sps = SPS()
            self.reg_trunk = RegTrunk()
            self.chl_head = ChlHead()

    return Toy()


def snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def diff_keys(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(k for k in a if not torch.equal(a[k], b[k]))


# ═══════════════════════════════════════════════════════════════════════════
#  1. SpecMLP — 구조와 회귀 상수
# ═══════════════════════════════════════════════════════════════════════════
def test_param_count_1089_and_1122() -> None:
    """01 [M12] / 06 §3.6 `[측정]`. 초판 961(c_in=6)은 폐기됐다 (정정 A-9)."""
    assert sum(p.numel() for p in SpecMLP().parameters()) == 1089
    assert sum(p.numel() for p in SpecMLP(head_out=2).parameters()) == 1122


def test_param_shape_table_matches_spec() -> None:
    m = SpecMLP()
    want = {
        "proj.weight": (16, 8, 1, 1),
        "proj.bias": (16,),
        "c_abs": (6, 16),
        "n1.weight": (16,),
        "dw.weight": (16, 1, 3, 3),
        "mix.weight": (32, 16, 1, 1),
        "mix.bias": (32,),
        "head.weight": (1, 32, 1, 1),
        "head.bias": (1,),
    }
    got = dict(m.named_parameters())
    for k, shape in want.items():
        assert tuple(got[k].shape) == shape, k
    # n1 은 GroupNorm(1, 16) — LayerNorm 등가 (01 §6.5 동결)
    assert isinstance(m.n1, nn.GroupNorm) and m.n1.num_groups == 1


def test_forward_backward_shapes() -> None:
    m = SpecMLP()
    x = torch.randn(2, 8, 3, 3)
    mm = torch.tensor([[0.0, 1, 1, 1, 0, 1]] * 2)
    out = m(x, mm)
    assert out.shape == (2, 1)
    out.sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_forward_accepts_non_3x3_spatial() -> None:
    """칩은 3×3 이지만 op 은 SPS 와 같아야 하므로 임의 H,W 를 받아야 한다."""
    m = SpecMLP()
    mm = torch.ones(2, 6)
    assert m(torch.randn(2, 8, 1, 1), mm).shape == (2, 1)
    assert m(torch.randn(2, 8, 5, 7), mm).shape == (2, 1)


@pytest.mark.parametrize(
    "kwargs, x, mm",
    [
        (dict(), torch.randn(2, 6, 3, 3), torch.ones(2, 6)),  # 채널 불일치
        (dict(), torch.randn(2, 8, 3), torch.ones(2, 6)),  # dim 불일치
        (dict(), torch.randn(2, 8, 3, 3), torch.ones(2, 5)),  # slot 수 불일치
        (dict(), torch.randn(2, 8, 3, 3), torch.ones(3, 6)),  # 배치 불일치
    ],
)
def test_forward_rejects_bad_input(kwargs: Dict[str, Any], x: Any, mm: Any) -> None:
    with pytest.raises(ValueError):
        SpecMLP(**kwargs)(x, mm)


@pytest.mark.parametrize("bad", [0, 3, -1])
def test_head_out_must_be_1_or_2(bad: int) -> None:
    with pytest.raises(ValueError):
        SpecMLP(head_out=bad)


# ═══════════════════════════════════════════════════════════════════════════
#  2. SpecMLP — 초기화 특칙 (01 §6.5)
# ═══════════════════════════════════════════════════════════════════════════
def test_initial_prediction_is_exactly_global_mean() -> None:
    """head.weight=0 + head.bias=1.5662 → 입력과 무관하게 softplus(b) = 1.75589."""
    m = SpecMLP().eval()
    for _ in range(3):
        z = m(torch.randn(4, 8, 3, 3) * 10.0, torch.rand(4, 6).round())
        y_hat = F.softplus(z).detach()
        assert torch.allclose(z, torch.full_like(z, HEAD_BIAS_DEFAULT), atol=1e-6)
        assert abs(float(y_hat.mean()) - K235_LOG1P_MEAN) < 1e-4


def test_softplus_inv_roundtrip_matches_frozen_bias() -> None:
    """01 §6.5 검산: softplus⁻¹(1.7559) = 1.5662 (오차 1e-4)."""
    assert abs(softplus_inv(K235_LOG1P_MEAN) - HEAD_BIAS_DEFAULT) < 1e-4
    assert abs(float(F.softplus(torch.tensor(HEAD_BIAS_DEFAULT))) - K235_LOG1P_MEAN) < 1e-4
    with pytest.raises(ValueError):
        softplus_inv(0.0)


def test_bio_gain_applies_to_columns_6_7_only() -> None:
    """정정 A-9: bio 열 index 는 canonical 8ch 기준 **6,7** 이다 (초판 4,5 가 아니다)."""
    torch.manual_seed(0)
    a = SpecMLP()
    init_spec_mlp(a, bio_gain=1.0)
    w1 = a.proj.weight.detach().clone()
    torch.manual_seed(0)
    b = SpecMLP()
    init_spec_mlp(b, bio_gain=4.0)
    w4 = b.proj.weight.detach()
    assert torch.equal(w1[:, :6], w4[:, :6])  # raw 밴드 열은 불변
    assert torch.allclose(w4[:, 6:], w1[:, 6:] * 4.0, atol=1e-7)
    assert not torch.allclose(w4[:, 6:], w1[:, 6:])  # gain 이 실제로 적용됐다


def test_c_abs_zero_init_is_identity_for_missing_slots() -> None:
    """C_abs = 0 이면 결측 slot 이 있어도 순수 zero-fill 과 동일하다."""
    m = SpecMLP().eval()
    assert float(m.c_abs.detach().abs().max()) == 0.0
    with torch.no_grad():
        m.head.weight.normal_()  # 출력이 입력에 반응하게 (init 은 weight=0 이라 항상 상수)
    x = torch.randn(2, 8, 3, 3)
    full = m(x, torch.ones(2, 6))
    part = m(x, torch.tensor([[0.0, 1, 1, 1, 0, 1]] * 2))
    assert torch.equal(full, part)  # C_abs=0 → m 은 출력에 전혀 영향이 없다
    with torch.no_grad():
        m.c_abs.normal_()
    assert not torch.equal(m(x, torch.ones(2, 6)), m(x, torch.tensor([[0.0, 1, 1, 1, 0, 1]] * 2)))


def test_head_out2_logvar_channel_starts_at_zero() -> None:
    m = SpecMLP(head_out=2).eval()
    out = m(torch.randn(3, 8, 3, 3), torch.ones(3, 6))
    assert torch.allclose(out[:, 0], torch.full((3,), HEAD_BIAS_DEFAULT), atol=1e-6)
    assert torch.allclose(out[:, 1], torch.zeros(3), atol=1e-6)  # σ² = 1 에서 출발


def test_norm_affine_starts_at_identity() -> None:
    m = SpecMLP()
    for norm in (m.n1, m.bn, m.bn2):
        assert torch.equal(norm.weight, torch.ones_like(norm.weight))
        assert torch.equal(norm.bias, torch.zeros_like(norm.bias))


# ═══════════════════════════════════════════════════════════════════════════
#  3. SpecMLP ↔ SPS op 대응 (이식의 존재 이유)
# ═══════════════════════════════════════════════════════════════════════════
def test_shapes_match_sps_counterparts() -> None:
    """T1/T1′/T2/T4 대상 텐서의 shape 이 SPS 와 정확히 대응하는지 — 이식 전제조건."""
    src, sps = SpecMLP(), SPS()
    body = sps.body
    assert tuple(src.proj.weight.shape) == tuple(body.proj.weight.shape) == (16, 8, 1, 1)
    assert tuple(src.proj.bias.shape) == tuple(body.proj.bias.shape) == (16,)
    assert tuple(src.c_abs.shape) == tuple(body.c_abs.shape) == (len(MSI_SLOTS), 16)
    assert tuple(src.n1.weight.shape) == tuple(body.norm1.weight.shape) == (16,)
    assert tuple(src.dw.weight.shape) == tuple(body.dw.weight.shape) == (16, 1, 3, 3)
    conv = body.patch_embed[0]
    assert tuple(src.mix.weight.shape[:2]) == tuple(conv.weight.shape[:2]) == (32, 16)
    assert tuple(conv.weight.shape[2:]) == (5, 5) and conv.stride == (4, 4)
    assert conv.bias is None  # 정정 A-3 — T4 bias 이식이 불가능한 근본 이유


# ═══════════════════════════════════════════════════════════════════════════
#  4. spec_loss (01 §6.6 / X-11)
# ═══════════════════════════════════════════════════════════════════════════
def test_spec_loss_reduces_to_smooth_l1() -> None:
    out = torch.tensor([[0.5], [1.5], [-0.2]])
    y = torch.tensor([1.0, 2.0, 0.3])
    d = spec_loss(out, y, rank_weight=0.0, beta=0.3)
    ref = F.smooth_l1_loss(F.softplus(out), y.view(-1, 1), beta=0.3)
    assert torch.allclose(d["loss"], ref, atol=1e-7)
    assert torch.allclose(d["reg"], ref, atol=1e-7)
    assert float(d["rank"]) == 0.0
    assert torch.allclose(d["y_hat"], F.softplus(out))


def test_spec_loss_rank_term_is_graph_connected_zero() -> None:
    """rank_weight=0 에서도 tensor(0.0) 상수가 아니라 그래프에 연결된 0 이어야 한다."""
    out = torch.tensor([[0.5], [1.5]], requires_grad=True)
    d = spec_loss(out, torch.tensor([1.0, 2.0]), rank_weight=0.0)
    assert d["rank"].requires_grad and float(d["rank"].detach()) == 0.0
    d["loss"].backward()
    assert torch.isfinite(out.grad).all()


def test_spec_loss_perfect_prediction_is_zero() -> None:
    y = torch.tensor([1.0, 2.0, 3.0])
    z = torch.log(torch.expm1(y)).view(-1, 1)  # softplus(z) == y
    d = spec_loss(z, y, rank_weight=0.0)
    assert float(d["loss"]) < 1e-9


def test_spec_loss_rank_penalizes_inverted_order() -> None:
    y = torch.tensor([0.5, 3.0])
    good = torch.log(torch.expm1(y)).view(-1, 1)
    bad = torch.log(torch.expm1(y.flip(0))).view(-1, 1)
    torch.manual_seed(0)
    a = float(spec_loss(good, y, rank_weight=1.0)["rank"])
    torch.manual_seed(0)
    b = float(spec_loss(bad, y, rank_weight=1.0)["rank"])
    assert b > a


def test_spec_loss_aleatoric_requires_head_out_2() -> None:
    with pytest.raises(ValueError):
        spec_loss(torch.zeros(4, 1), torch.zeros(4), use_aleatoric=True)


def test_spec_loss_aleatoric_backward_is_finite() -> None:
    m = SpecMLP(head_out=2)
    out = m(torch.randn(4, 8, 3, 3), torch.ones(4, 6))
    d = spec_loss(out, torch.tensor([1.0, 2.0, 0.5, 3.0]), use_aleatoric=True, u=1.0)
    d["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_spec_loss_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        spec_loss(torch.zeros(4), torch.zeros(4))
    with pytest.raises(ValueError):
        spec_loss(torch.zeros(4, 1), torch.zeros(3))


# ═══════════════════════════════════════════════════════════════════════════
#  5. LOSO — 분할 무결성 (01 §6.4 [S1]/[S4])
# ═══════════════════════════════════════════════════════════════════════════
def test_loso_never_mixes_sites(kw_tmp: pathlib.Path, monkeypatch: Any) -> None:
    """★ 핵심: fold 의 학습 배치에 holdout site 의 표본이 **한 개도** 들어가면 안 된다.

    site marker 를 red 밴드(canonical slot 2)에 심어 두고, ``train_spec_mlp`` 가 실제로
    받은 텐서에서 marker 집합을 역추출해 holdout 과의 교집합이 공집합인지 본다.
    """
    sites = ["AAA", "BBB", "CCC", "DDD"]
    npz = make_npz(kw_tmp / "syn.npz", sites=sites, n_per_site=16)
    seen: List[set] = []
    real = loso_mod.train_spec_mlp

    def spy(x, m, y, **kw):  # type: ignore[no-untyped-def]
        marks = torch.unique((x[:, 2, 0, 0] * 1000.0).round()).tolist()
        seen.append({int(v) for v in marks})
        return real(x, m, y, **kw)

    monkeypatch.setattr(loso_mod, "train_spec_mlp", spy)
    r = run_loso(str(npz), sites=sites, **FAST)

    assert len(seen) == len(sites)
    for k, hold in enumerate(sites):
        hold_mark = int(round((0.03 + 0.001 * (k + 1)) * 1000))
        assert hold_mark not in seen[k], f"{hold} 표본이 자기 fold 의 학습셋에 있다 (누수)"
        assert len(seen[k]) == len(sites) - 1
        assert r["n"][hold]["test"] == 16
        assert r["n"][hold]["train"] == 16 * (len(sites) - 1)


def test_loso_fold_keys_and_metric_fields(kw_tmp: pathlib.Path) -> None:
    sites = ["AAA", "BBB", "CCC"]
    npz = make_npz(kw_tmp / "syn.npz", sites=sites)
    r = run_loso(str(npz), sites=sites, **FAST)
    assert sorted(r["folds"]) == sorted(sites)
    for f in r["folds"].values():
        for k in ("rmse_log", "mae_log", "r2_log", "spearman", "rmse_mgm3", "f1_at_15"):
            assert k in f
        assert f["f1_at_100"] is None  # 정정 A-30/B-30 — 표본 0
    assert r["mean"]["n_folds"] == 3
    assert math.isfinite(r["mean"]["rmse_log"])


def test_loso_is_deterministic(kw_tmp: pathlib.Path) -> None:
    sites = ["AAA", "BBB"]
    npz = make_npz(kw_tmp / "syn.npz", sites=sites)
    a = run_loso(str(npz), sites=sites, **FAST)
    b = run_loso(str(npz), sites=sites, **FAST)
    for s in sites:
        assert a["folds"][s]["rmse_log"] == pytest.approx(b["folds"][s]["rmse_log"], abs=1e-9)


def test_loso_requires_two_sites(kw_tmp: pathlib.Path) -> None:
    npz = make_npz(kw_tmp / "syn.npz", sites=["AAA"])
    with pytest.raises(ValueError):
        run_loso(str(npz), sites=["AAA"], **FAST)


def test_unknown_cfg_key_is_rejected(kw_tmp: pathlib.Path) -> None:
    """오타 하나가 학습 설정을 조용히 되돌리는 것을 막는다 (06 §4.2 미지 키 거부와 동일 정신)."""
    npz = make_npz(kw_tmp / "syn.npz", sites=["AAA", "BBB"])
    with pytest.raises(TypeError, match="미지 설정 키"):
        run_loso(str(npz), sites=["AAA", "BBB"], epoch=2)  # epochs 오타


def test_cuda_device_is_rejected(kw_tmp: pathlib.Path) -> None:
    npz = make_npz(kw_tmp / "syn.npz", sites=["AAA", "BBB"])
    with pytest.raises(ValueError, match="cpu"):
        run_loso(str(npz), sites=["AAA", "BBB"], device="cuda", **FAST)


def test_aleatoric_requires_head_out_2_in_cfg(kw_tmp: pathlib.Path) -> None:
    npz = make_npz(kw_tmp / "syn.npz", sites=["AAA", "BBB"])
    with pytest.raises(ValueError):
        run_loso(str(npz), sites=["AAA", "BBB"], use_aleatoric=True, **FAST)


def test_aleatoric_loso_runs_and_reports_sigma(kw_tmp: pathlib.Path) -> None:
    """head_out=2 경로: log_var 가 평가기까지 흘러 NLL/σ 지표가 살아난다."""
    sites = ["AAA", "BBB"]
    npz = make_npz(kw_tmp / "syn.npz", sites=sites)
    r = run_loso(str(npz), sites=sites, head_out=2, use_aleatoric=True, **FAST)
    for f in r["folds"].values():
        assert math.isfinite(f["nll"]) and math.isfinite(f["sigma_mean"])


def test_u_ramp_schedule_matches_recipe() -> None:
    """05 §5.3.4: 300 에폭에서 E_warm=90 / E_ramp=60 이어야 한다."""
    from bloomnet.losses.regression import u_ramp

    assert u_ramp(89, 300) == 0.0
    assert u_ramp(90, 300) == 0.0
    assert u_ramp(120, 300) == pytest.approx(0.5)
    assert u_ramp(150, 300) == 1.0


def test_holdout_rejects_overlapping_sites(kw_tmp: pathlib.Path) -> None:
    npz = make_npz(kw_tmp / "syn.npz", sites=["AAA", "BBB"])
    with pytest.raises(ValueError, match="중복"):
        run_holdout(str(npz), train_sites=["AAA", "BBB"], val_sites=["BBB"], **FAST)


def test_holdout_runs_and_reports_sites(kw_tmp: pathlib.Path) -> None:
    sites = ["AAA", "BBB", "CCC"]
    npz = make_npz(kw_tmp / "syn.npz", sites=sites)
    h = run_holdout(str(npz), train_sites=["AAA", "BBB"], val_sites=["CCC"], **FAST)
    assert h["val_sites"] == ["CCC"] and h["n_valid"] == 24
    assert math.isfinite(h["rmse_log"])


def test_load_k235_arrays_shapes(kw_tmp: pathlib.Path) -> None:
    npz = make_npz(kw_tmp / "syn.npz", sites=["AAA", "BBB"], n_per_site=10)
    d = load_k235_arrays(str(npz), ["AAA", "BBB"], band_order_check="off")
    assert d["x"].shape == (20, 8, 3, 3)
    assert d["m"].shape == (6,)
    # use_blue=False (X-03) → slot (1,2,3,5) 만 채워진다 = M3M 과 동일 취급
    assert d["m"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    assert d["y"].shape == (20,) and float(d["y"].min().detach()) > 0.0
    assert set(d["group"][:1][0].split("|")) and len(d["site"]) == 20


def test_run_defaults_match_spec_yaml() -> None:
    """06 §4.2 `spec:` 블록 기본값과 어긋나면 CLI/config 전달이 조용히 달라진다."""
    assert RUN_DEFAULTS["huber_beta"] == 0.3
    assert RUN_DEFAULTS["rank_weight"] == 0.2
    assert RUN_DEFAULTS["rank_margin"] == 0.1
    assert RUN_DEFAULTS["bio_gain"] == 4.0
    assert RUN_DEFAULTS["epochs"] == 300
    assert RUN_DEFAULTS["batch_size"] == 512
    assert RUN_DEFAULTS["lr"] == 3.0e-3
    assert RUN_DEFAULTS["weight_decay"] == 1.0e-4
    assert RUN_DEFAULTS["head_out"] == 1
    assert RUN_DEFAULTS["device"] == "cpu"


# ═══════════════════════════════════════════════════════════════════════════
#  6. 이식 게이트 (01 §6.9)
# ═══════════════════════════════════════════════════════════════════════════
def mk_loso(rmse: float, rhos: List[float]) -> Dict[str, Any]:
    folds = {f"S{i}": {"spearman": r, "rmse_log": rmse} for i, r in enumerate(rhos)}
    return {
        "folds": folds,
        "mean": {"rmse_log": rmse, "min_spearman": min(rhos), "n_folds": len(rhos)},
    }


def test_gate_thresholds_are_frozen() -> None:
    assert GATE_DEFAULTS == {"g1_rmse": 0.750, "g2_spearman": 0.30, "g3_f1_at15": 0.60}


def test_gate_all_pass() -> None:
    ok, d = transplant_gate(mk_loso(0.70, [0.5] * 8), {"f1_at_15": 0.8})
    assert ok and d == {"G1": True, "G2": True, "G3": True}


def test_gate_boundary_values_are_inclusive() -> None:
    """G1 은 ``<=``, G2/G3 은 ``>=`` 다. 경계에서 부호가 뒤집히면 판정이 바뀐다."""
    ok, d = transplant_gate(mk_loso(0.750, [0.30] * 8), {"f1_at_15": 0.60})
    assert ok and all(d.values())
    ok, d = transplant_gate(mk_loso(0.7500001, [0.30] * 8), {"f1_at_15": 0.60})
    assert not ok and d["G1"] is False
    ok, d = transplant_gate(mk_loso(0.750, [0.30] * 7 + [0.2999999]), {"f1_at_15": 0.60})
    assert not ok and d["G2"] is False
    ok, d = transplant_gate(mk_loso(0.750, [0.30] * 8), {"f1_at_15": 0.5999999})
    assert not ok and d["G3"] is False


def test_gate_g2_is_all_folds_not_mean() -> None:
    """평균 ρ 는 0.5 지만 한 fold 가 0.1 이면 **실패**여야 한다 (01 §6.9 '8개 fold 전부')."""
    rhos = [0.9] * 7 + [0.1]
    assert sum(rhos) / 8 > 0.30
    ok, d = transplant_gate(mk_loso(0.70, rhos), {"f1_at_15": 0.9})
    assert not ok and d["G2"] is False


def test_gate_treats_undecidable_as_failure() -> None:
    """nan / None 은 '좋다' 가 아니라 '판정 불가' 다 → 이식 금지."""
    ok, d = transplant_gate(mk_loso(0.70, [float("nan")] * 8), {"f1_at_15": 0.9})
    assert not ok and d["G2"] is False
    # f1_at_15 = None (양성 표본 0개, 정정 A-30)
    ok, d = transplant_gate(mk_loso(0.70, [0.5] * 8), {"f1_at_15": None})
    assert not ok and d["G3"] is False
    ok, d = transplant_gate({"mean": {"rmse_log": float("nan")}}, {"f1_at_15": 0.9})
    assert not ok and d["G1"] is False


def test_gate_custom_thresholds() -> None:
    ok, _ = transplant_gate(mk_loso(0.9, [0.1] * 3), {"f1_at_15": 0.1}, g1_rmse=1.0,
                            g2_spearman=0.0, g3_f1_at15=0.0)
    assert ok


def test_gate_report_is_honest_about_failure() -> None:
    lo = mk_loso(0.845, [0.51, -0.13, -0.28, 0.58, 0.41, 0.73, 0.59, 0.45])
    for k, f in lo["folds"].items():
        f.update(mae_log=0.7, r2_log=-1.6, n_valid=2894)
    txt = format_gate_report(lo, {"f1_at_15": 0.4}, transplant_gate(lo, {"f1_at_15": 0.4}))
    assert "FAIL" in txt and "이식 금지" in txt
    assert "0.827" in txt and "0.810" in txt  # 01 [M9] 베이스라인이 리포트에 남는다


# ═══════════════════════════════════════════════════════════════════════════
#  7. 이식 (01 §6.7 / T22)
# ═══════════════════════════════════════════════════════════════════════════
def test_t1_t1p_t2_are_bit_identical() -> None:
    src, model = SpecMLP(), make_model()
    with torch.no_grad():  # 학습된 것처럼 만들어 0-init 과 구분되게
        for p in src.parameters():
            p.normal_()
    rep = transplant_to_bloomnet(src, model)
    body = model.encoder.sps.body
    assert torch.equal(body.proj.weight, src.proj.weight)
    assert torch.equal(body.proj.bias, src.proj.bias)
    assert torch.equal(body.c_abs, src.c_abs)
    assert torch.equal(body.norm1.weight, src.n1.weight)
    assert torch.equal(body.norm1.bias, src.n1.bias)
    assert rep[f"encoder.sps.body.proj.weight"] == "copied"
    assert rep["encoder.sps.body.c_abs"] == "copied"
    assert rep["encoder.sps.body.norm1.weight"] == "copied"


def test_t3_dw_is_never_copied() -> None:
    """★ T3 — 3×3 칩의 8/9 가 zero-pad 오염 + GSD 6~7.6배 불일치."""
    src, model = SpecMLP(), make_model()
    with torch.no_grad():
        src.dw.weight.fill_(0.12345)
    before = model.encoder.sps.body.dw.weight.detach().clone()
    rep = transplant_to_bloomnet(src, model)
    after = model.encoder.sps.body.dw.weight
    assert torch.equal(after, before), "T3 위반 — dw 가 이식됐다"
    assert not torch.allclose(after, src.dw.weight)
    assert rep["encoder.sps.body.dw.weight"] == "skipped"


def test_t4_avg_tap_and_bn_absorption() -> None:
    src, model = SpecMLP(), make_model()
    with torch.no_grad():
        src.mix.weight.normal_()
        src.mix.bias.normal_()
    pe = model.encoder.sps.body.patch_embed
    rm_before = pe[1].running_mean.detach().clone()
    rep = transplant_to_bloomnet(src, model)

    # 25탭 균등: 공간합 == 원래 1×1 값 (06 T22 assert)
    assert torch.allclose(pe[0].weight.sum(dim=(2, 3)), src.mix.weight[:, :, 0, 0], atol=1e-6)
    # 모든 탭이 같은 값 = 5×5 box filter
    w = pe[0].weight
    assert torch.allclose(w, w[:, :, :1, :1].expand_as(w), atol=0)
    assert pe[0].bias is None  # 정정 A-3
    assert torch.allclose(pe[1].running_mean, rm_before - src.mix.bias, atol=1e-7)
    assert rep["encoder.sps.body.patch_embed.0.weight"] == "avg_tap"
    assert rep["encoder.sps.body.patch_embed.1.running_mean"] == "bn_absorbed"


def test_t4p_dw_branch_bn_weight_is_zeroed() -> None:
    """T4′ — mix 가 학습 때 보던 입력 분포에서 출발하도록 잔차를 항등으로."""
    src, model = SpecMLP(), make_model()
    body = model.encoder.sps.body
    assert float(body.norm2.weight.detach().abs().max()) > 0.0  # 이식 전에는 1
    rep = transplant_to_bloomnet(src, model)
    assert float(body.norm2.weight.detach().abs().max()) == 0.0
    assert rep["encoder.sps.body.norm2.weight"] == "zeroed"

    # T4 를 끄면 T4′ 도 꺼진다 (둘은 짝이다)
    model2 = make_model()
    transplant_to_bloomnet(src, model2, do_patch_embed=False)
    assert float(model2.encoder.sps.body.norm2.weight.detach().abs().max()) > 0.0


def test_t4p_makes_dw_residual_exactly_identity() -> None:
    """norm2.weight=0 의 **의미**: 잔차 분기가 정확히 0 을 더한다 (eval 기준)."""
    src, model = SpecMLP(), make_model()
    transplant_to_bloomnet(src, model)
    body = model.encoder.sps.body.eval()
    z = torch.randn(2, 16, 8, 8)
    branch = body.act(body.norm2(body.dw(z)))
    assert torch.allclose(branch, torch.zeros_like(branch), atol=1e-6)


def test_t5_bias_only_and_in_ch_64() -> None:
    src, model = SpecMLP(), make_model()
    with torch.no_grad():
        src.head.bias.fill_(1.9)
        src.head.weight.normal_()
    rep = transplant_to_bloomnet(src, model)
    out = model.chl_head.out
    assert int(out.weight.shape[1]) == 64  # 정정 A-3 (초판 128 은 오기)
    assert float(out.weight.detach().abs().max()) == 0.0
    assert float(out.bias[0].detach()) == pytest.approx(1.9, abs=1e-7)
    assert rep["chl_head.out.bias"] == "bias_only"
    assert rep["chl_head.out.weight"] == "zeroed"


def test_t5_makes_chl_prediction_the_learned_prior() -> None:
    src, model = SpecMLP(), make_model()
    transplant_to_bloomnet(src, model)
    model.eval()
    u = model.chl_head(torch.randn(2, 64, 4, 4))
    want = float(F.softplus(src.head.bias[0]).detach())
    assert torch.allclose(u, torch.full_like(u, want), atol=1e-6)


def test_t7_gate_failure_changes_nothing() -> None:
    """★ 게이트 실패 시 어떤 텐서도 변경되지 않는다 (06 T22)."""
    src, model = SpecMLP(), make_model()
    with torch.no_grad():
        for p in src.parameters():
            p.normal_()
    before = snapshot(model)
    rep = transplant_to_bloomnet(src, model, gate_ok=False)
    assert diff_keys(before, snapshot(model)) == []
    assert set(rep.values()) == {"skipped"}


def test_t0_failure_copies_bio_columns_only() -> None:
    """정정 A-3 T0 — raw 밴드 스케일이 다르면 raw 열과 C_abs 를 이식하지 않는다."""
    src, model = SpecMLP(), make_model()
    with torch.no_grad():
        src.proj.weight.normal_()
        src.c_abs.normal_()
    body = model.encoder.sps.body
    raw_before = body.proj.weight[:, :6].detach().clone()
    cabs_before = body.c_abs.detach().clone()
    rep = transplant_to_bloomnet(src, model, raw_bands=False)
    assert torch.equal(body.proj.weight[:, :6], raw_before)
    assert torch.equal(body.proj.weight[:, 6:], src.proj.weight[:, 6:])
    assert torch.equal(body.c_abs, cabs_before)
    assert rep["encoder.sps.body.proj.weight"] == "copied_bio_only"
    assert rep["encoder.sps.body.c_abs"] == "skipped"


def test_t6_bio_kind_mismatch_copies_raw_columns_only() -> None:
    """T6 — ``source=rgb_proxy`` 의 bio 열은 ExG/NGRDI 라 NDCI/MCI_norm 과 다른 물리량이다."""
    src, model = SpecMLP(), make_model()
    with torch.no_grad():
        src.proj.weight.normal_()
    body = model.encoder.sps.body
    bio_before = body.proj.weight[:, 6:].detach().clone()
    rep = transplant_to_bloomnet(src, model, bio_kind_src="rgb_proxy", bio_kind_dst="mci")
    assert torch.equal(body.proj.weight[:, :6], src.proj.weight[:, :6])
    assert torch.equal(body.proj.weight[:, 6:], bio_before)
    assert rep["encoder.sps.body.proj.weight"] == "copied_raw_only"
    assert rep["encoder.sps.body.proj.bias"] == "skipped"


def test_t0_fail_and_t6_mismatch_skips_proj_entirely() -> None:
    src, model = SpecMLP(), make_model()
    with torch.no_grad():
        src.proj.weight.normal_()
    before = model.encoder.sps.body.proj.weight.detach().clone()
    rep = transplant_to_bloomnet(src, model, raw_bands=False, bio_kind_src="rgb_proxy")
    assert torch.equal(model.encoder.sps.body.proj.weight, before)
    assert rep["encoder.sps.body.proj.weight"] == "skipped"


def test_shape_mismatch_is_atomic_refusal() -> None:
    """★ shape 불일치 → ValueError 이고 **모델은 한 바이트도 바뀌지 않는다**."""
    src, model = SpecMLP(), make_model()
    with torch.no_grad():
        model.encoder.sps.body.proj = nn.Conv2d(9, 16, 1)  # 채널을 망가뜨린다
    before = snapshot(model)
    with pytest.raises(ValueError, match="preflight"):
        transplant_to_bloomnet(src, model)
    assert diff_keys(before, snapshot(model)) == []


def test_shape_mismatch_non_strict_skips_only_that_item() -> None:
    src, model = SpecMLP(), make_model()
    with torch.no_grad():
        model.encoder.sps.body.proj = nn.Conv2d(9, 16, 1)
        src.n1.weight.fill_(3.0)
    rep = transplant_to_bloomnet(src, model, strict=False)
    assert rep["encoder.sps.body.proj.weight"] == "skipped"
    assert rep["encoder.sps.body.norm1.weight"] == "copied"  # 나머지는 진행
    assert torch.equal(model.encoder.sps.body.norm1.weight, src.n1.weight)


def test_chl_head_wrong_in_ch_is_refused() -> None:
    """T5 in_ch 계약(64)을 어긴 모델은 조용히 통과시키지 않는다."""
    src, model = SpecMLP(), make_model()
    model.chl_head.out = nn.Conv2d(128, 1, 1)  # 초판의 오기값
    with pytest.raises(ValueError, match="64"):
        transplant_to_bloomnet(src, model)


def test_src_type_is_checked() -> None:
    with pytest.raises(TypeError):
        transplant_to_bloomnet(nn.Linear(2, 2), make_model())  # type: ignore[arg-type]


def test_find_helpers_reject_zero_and_multiple() -> None:
    with pytest.raises(ValueError, match="찾지 못했다"):
        find_sps(nn.Sequential(nn.Conv2d(3, 3, 1)))
    two = nn.Module()
    two.a, two.b = SPS(), SPS()
    with pytest.raises(ValueError, match="모호"):
        find_sps(two)
    with pytest.raises(ValueError, match="찾지 못했다"):
        find_chl_head(nn.Sequential(nn.Conv2d(3, 3, 1)))


def test_explicit_targets_bypass_search() -> None:
    src = SpecMLP()
    sps, head = SPS(), ChlHead()
    holder = nn.Module()  # SPS/ChlHead 를 품지 않은 모델이어도 명시 지정으로 동작
    rep = transplant_to_bloomnet(src, holder, sps=sps, chl_head=head)
    assert torch.equal(sps.body.proj.weight, src.proj.weight)
    assert any(v == "avg_tap" for v in rep.values())


def test_do_patch_embed_false_leaves_patch_embed_untouched() -> None:
    src, model = SpecMLP(), make_model()
    pe = model.encoder.sps.body.patch_embed
    w_before = pe[0].weight.detach().clone()
    rm_before = pe[1].running_mean.detach().clone()
    rep = transplant_to_bloomnet(src, model, do_patch_embed=False)
    assert torch.equal(pe[0].weight, w_before)
    assert torch.equal(pe[1].running_mean, rm_before)
    assert rep["encoder.sps.body.patch_embed.0.weight"] == "skipped"


def test_transplanted_sps_still_runs_forward() -> None:
    """이식 후에도 SPS forward/backward 가 살아 있어야 한다 (구조 파괴 금지)."""
    src, model = SpecMLP(), make_model()
    transplant_to_bloomnet(src, model)
    sps = model.encoder.sps
    x = torch.rand(2, 4, 32, 32) * 0.05
    out = sps(x, [1, 2, 3, 5])
    assert out.spec_stem.shape == (2, 32, 8, 8)
    out.spec_stem.sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in sps.parameters() if p.grad is not None
    )


def test_summarize_lists_every_target() -> None:
    src, model = SpecMLP(), make_model()
    txt = summarize(transplant_to_bloomnet(src, model))
    assert "변경된 텐서 10 / 11" in txt  # dw 만 skipped
    assert "dw.weight" in txt


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("bloomnet.models.bloomnet") is None,
    reason="models/bloomnet.py (L5) 미도착",
)
def test_transplant_into_real_bloomnet() -> None:
    """★ 실제 ``build_bloomnet`` 산출물에 이식이 되는지 — 계약이 닫혔는지 확인.

    ``find_sps``/``find_chl_head`` 가 attribute 이름이 아니라 **타입**으로 찾으므로
    모델 조립 방식이 바뀌어도 동작해야 한다.
    """
    from bloomnet.config import default_config, from_dict, merge_dicts, to_dict
    from bloomnet.models.bloomnet import build_bloomnet

    cfg = from_dict(
        merge_dicts(
            to_dict(default_config()),
            dict(
                mode="s1_rgb_ms4",
                data=dict(
                    modalities=["rgb", "msi", "bio"],
                    sensor="m3m",
                    num_classes=2,
                    bio=dict(kind="mci", source="msi"),
                ),
            ),
        )
    )
    model = build_bloomnet(cfg)
    src = SpecMLP()
    with torch.no_grad():
        for p in src.parameters():
            p.normal_()
    rep = transplant_to_bloomnet(src, model)

    sps, head = find_sps(model), find_chl_head(model)
    assert torch.equal(sps.body.proj.weight, src.proj.weight)
    assert torch.equal(sps.body.c_abs, src.c_abs)
    assert float(sps.body.norm2.weight.detach().abs().max()) == 0.0  # T4′
    assert float(head.out.bias[0].detach()) == pytest.approx(
        float(src.head.bias[0].detach()), abs=1e-6
    )
    assert sum(1 for v in rep.values() if v == "skipped") == 1  # dw 만

    # 이식 후에도 전체 forward 가 산다
    model.eval()
    out = model(
        rgb=torch.randn(1, 3, 64, 64),
        msi=torch.rand(1, len(cfg.data.band_ids), 64, 64) * 0.08 + 0.01,
    )
    assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v))


# ═══════════════════════════════════════════════════════════════════════════
#  8. T0 판정식
# ═══════════════════════════════════════════════════════════════════════════
def test_check_t0_numeric() -> None:
    assert T0_TOL == 0.5
    assert check_t0(0.030, 0.030)
    assert check_t0(0.030, 0.090)  # log10 3배 = 0.477 < 0.5
    assert not check_t0(0.030, 0.100)  # 3.33배 = 0.523 > 0.5
    # 정정 A-2 실측: 미보정 rho_rel 기하평균 ≈ 1.09e-5 (G 8.638e-6 … NIR 1.665e-5)
    # vs 235 median 0.0302 → |Δlog10| ≈ 3.44 → 명백히 불충족 (T0 를 신설한 이유)
    rho_rel, k235_med = 1.09e-5, 0.0302
    assert not check_t0(rho_rel, k235_med)
    # R4′(k_sensor = 3.5e3) 적용 후 0.0382 → |Δlog10| ≈ 0.10 → 통과
    assert check_t0(rho_rel * 3.5e3, k235_med)
    with pytest.raises(ValueError):
        check_t0(0.0, 0.03)


# ═══════════════════════════════════════════════════════════════════════════
#  9. 실데이터 회귀 (npz 캐시가 있을 때만)
# ═══════════════════════════════════════════════════════════════════════════
@needs_real_npz
def test_real_k235_fold_sizes_match_m9_table() -> None:
    """01 §6.8 [M9] 표의 fold 별 n 재현 — QC 필터·site 매핑 전체의 회귀 고정."""
    d = load_k235_arrays(str(REAL_NPZ), K235_ALL_SITES, band_order_check="off")
    got = {s: int((d["site"] == s).sum()) for s in K235_ALL_SITES}
    assert got == M9_FOLD_N
    assert sum(got.values()) == 21511  # 01 §6.3 [L1] 필터 후 총계
    assert d["x"].shape == (21511, 8, 3, 3)
    # [H1] 235 는 절대 반사율이라 k_sensor=1.0 으로 그대로 통과해야 한다
    med = float(np.median(d["x"][:, 1:4].numpy()[d["x"][:, 1:4].numpy() > 0]))
    assert 1e-3 < med < 1.0


@needs_real_npz
def test_real_k235_group_count_is_66() -> None:
    """01 [M8] — (site,date) 군 66개. 유효 독립표본 수가 21,511 이 아니라 ≈66 인 근거."""
    d = load_k235_arrays(str(REAL_NPZ), K235_ALL_SITES, band_order_check="off")
    assert len(set(d["group"].tolist())) == 66


@needs_real_npz
def test_real_transplant_gate_fails_as_predicted() -> None:
    """★ 정직성 회귀: **실데이터에서 게이트는 통과하지 못한다** (06 R-2 사전 등록).

    본 세션 전수 실측 (300 epoch, 05 §5.3.4 레시피 그대로, CPU LOSO 828 s + holdout 85 s):

    ==== ====== ========= ======== ========= ========
    site n      RMSE_log  MAE_log  R2_log    rho
    ==== ====== ========= ======== ========= ========
    BES  2,894  1.0420    0.8207   -0.9812   +0.0693
    CNH  2,842  1.1119    0.8933   -2.4899   -0.0407
    CSC  3,104  1.1092    0.9586   -1.7666   +0.1556
    GDK  1,687  0.9497    0.7214   -2.2386   +0.6716
    HNM  2,508  0.5220    0.4185   -1.1253   +0.3184
    JAD  2,944  0.5330    0.4098   -0.4217   +0.6405
    SHC  2,986  0.9821    0.8011   -3.1174   +0.5134
    YJD  2,546  0.4775    0.3369   -0.2974   +0.4497
    평균 —      0.8409    0.6700   -1.5548   —
    ==== ====== ========= ======== ========= ========

    → **G1 FAIL** (0.8409 > 0.750. 전역평균 0.827·ridge 0.810 둘 다 못 이긴다) /
      **G2 FAIL** (min rho = -0.0407 < 0.30, CNH) /
      **G3 FAIL** ([S2] val{CSC,JAD} F1@15 = 0.1342 < 0.60. holdout RMSE_log 0.8272 / R2 +0.1329).
    평균 R2 -1.5548 는 01 [M9] ridge 실측 -1.52 를 **모델 종류가 다른데도 독립 재현**한다.

    여기서는 6 epoch 만 돌려 G1 실패를 빠르게 고정한다. 이 테스트가 언젠가 실패한다면
    그것은 버그가 아니라 **뉴스**다 — 전수 재실행 후 이식 여부를 재판정해야 한다.
    """
    r = run_loso(
        str(REAL_NPZ),
        sites=K235_ALL_SITES,
        epochs=6,
        batch_size=512,
        warmup_epochs=1,
        band_order_check="off",
    )
    h = {"f1_at_15": None}  # 학습 없이 G3 를 판정하지 않는다 → 판정 불가 = 실패
    ok, detail = transplant_gate(r, h)
    assert not ok
    assert r["mean"]["rmse_log"] > GATE_DEFAULTS["g1_rmse"], r["mean"]
    assert detail["G1"] is False
    # 전역 평균 예측 베이스라인(0.827)조차 이기지 못하는 것이 [M9] 의 결론이다
    assert r["mean"]["r2_log"] < 0.0
