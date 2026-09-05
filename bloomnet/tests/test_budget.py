"""T24 — 배포 예산 회귀 고정 (06 §6, 정정 A-14/A-24).

**이 테스트가 깨지면 설계 문서(02/03/04/06)를 먼저 갱신해야 한다.** 코드가 문서를
앞서 나가는 것을 구조적으로 막는 장치다.

MAC 측정 규약 (정정 A-24)
-------------------------
256² 에서 **1회** 측정하고 ×4(512²)·×16(1024²)로 스케일한다. 1024² S2-Full 정방향
1회는 62.678 GMAC = 125.4 GFLOP 로 4-스레드 CPU 에서 수 초~10초대이고 중간 활성이
GB 급이라 06 §5 의 "기본 B=2, H=W=64 / 전 테스트 합 < 30 s" 와 양립하지 않는다.

``@pytest.mark.slow`` 로 옮기지 않는다 — 회귀 고정 목적이 사라진다.

LiteMLA 주의
------------
hook 카운터는 ``F.conv2d``/``torch.matmul`` 같은 functional 을 못 센다.
LiteMLA 의 ``q @ kv`` 를 포함하려면 ``count_macs_flop_counter`` 를 써야 한다
(둘의 차 = attention matmul, S2-Full @512² 에서 0.178 GMAC).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bloomnet.config import load_config  # noqa: E402
from bloomnet.constants import AUX_TAP_CH, AUX_TAP_MID, CHANNELS  # noqa: E402
from bloomnet.models.bloomnet import build_bloomnet  # noqa: E402
from bloomnet.models.encoder import BloomNetEncoder  # noqa: E402
from bloomnet.modules.heads import AuxSegHead, EdgeHead  # noqa: E402
from bloomnet.modules.pid_decoder import PIDDecoder  # noqa: E402
from bloomnet.utils.flops import (  # noqa: E402
    count_macs_flop_counter,
    count_macs_hooks,
    scale_macs,
)

MEAS_HW = 256  # 정정 A-24 측정 해상도

# ── §6.1 / §6.2 파라미터 회귀 ───────────────────────────────────────────────
ENCODER_PARAMS = 13_968_915
BMEF_PARAMS = 460_008
DECODER_PARAMS = 1_382_272
MODEL_S2_K2_PARAMS = 15_969_023

# ── 상한 게이트 ─────────────────────────────────────────────────────────────
PARAM_CEILING = 18_000_000
MAC_CEILING_1024 = 70.0  # GMAC


def _macs_gmac(model, kw: Dict[str, torch.Tensor], to_hw: Tuple[int, int], *, hooks=False):
    rep = (count_macs_hooks if hooks else count_macs_flop_counter)(model, kw)
    return scale_macs(rep.total, from_hw=(MEAS_HW, MEAS_HW), to_hw=to_hw) / 1e9, rep


def _deploy_model(name: str):
    model = build_bloomnet(load_config(str(REPO_ROOT / "configs" / name)))
    model.deploy()
    return model


def _model_inputs(model, h: int = MEAS_HW) -> Dict[str, torch.Tensor]:
    ch = {"rgb": 3, "msi": len(model.msi_band_ids), "bio": 2, "ir": 1, "pol": 3}
    return {n: torch.zeros(1, ch[n], h, h) for n in model.active_modalities}


# ─────────────────────────────────────────────────────────────────────────────
# 파라미터
# ─────────────────────────────────────────────────────────────────────────────
def test_encoder_bmef_decoder_params() -> None:
    """§6.1 소계 3종."""
    enc = BloomNetEncoder(active_paths=("rgb", "spec", "phys"), use_pol=True)
    assert sum(p.numel() for p in enc.parameters()) == ENCODER_PARAMS

    model = build_bloomnet(load_config(str(REPO_ROOT / "configs" / "s2_full.yaml")))
    assert sum(p.numel() for p in model.backbone.bmef.parameters()) == BMEF_PARAMS
    assert sum(p.numel() for p in PIDDecoder().parameters()) == DECODER_PARAMS


def test_inference_head_params_subtotal() -> None:
    """§6.1 '디코더+추론헤드 1,540,100' (K=2)."""
    model = _deploy_model("s2_full.yaml")
    heads = (
        sum(p.numel() for p in model.seg_head.parameters())
        + sum(p.numel() for p in model.reg_trunk.parameters())
        + sum(p.numel() for p in model.chl_head.parameters())
        + sum(p.numel() for p in model.unc_head.parameters())
    )
    dec = sum(p.numel() for p in model.decoder.parameters())
    assert dec == DECODER_PARAMS
    assert dec + heads == 1_540_100, f"디코더+추론헤드 {dec + heads}"


def test_aux_seg_head_params() -> None:
    """정정 B-21 — ``AUX_TAP_MID`` 리터럴 표 (``C_in//2`` 규칙이면 T16 이 깨진다)."""
    expect = {"enc_s8": 19_020, "enc_s16": 93_388, "enc_s32": 371_084, "p8": 74_892}
    for tap, want in expect.items():
        head = AuxSegHead(AUX_TAP_CH[tap], AUX_TAP_MID[tap], 12)
        assert sum(p.numel() for p in head.parameters()) == want, tap
    assert sum(expect[t] for t in ("enc_s8", "enc_s16", "enc_s32")) == 483_492


@pytest.mark.parametrize(
    "name,train_params,infer_params",
    [
        ("s0_rgb_aihub092.yaml", 9_447_195, 8_954_390),
        ("s1_rgb_ms4.yaml", 15_151_057, 14_660_522),
        ("s2_full.yaml", 16_459_558, MODEL_S2_K2_PARAMS),
    ],
)
def test_mode_param_budget(name: str, train_params: int, infer_params: int) -> None:
    """§6.2 모드별 학습/추론 파라미터."""
    train = build_bloomnet(load_config(str(REPO_ROOT / "configs" / name)))
    assert sum(p.numel() for p in train.parameters()) == train_params
    infer = _deploy_model(name)
    assert sum(p.numel() for p in infer.parameters()) == infer_params


# ─────────────────────────────────────────────────────────────────────────────
# MAC
# ─────────────────────────────────────────────────────────────────────────────
def test_s2_full_total_mac_matches_frozen_table() -> None:
    """★ 헤드라인 수치 — S2-Full(K=2, 추론) **15.670 @512² / 62.678 @1024²**."""
    model = _deploy_model("s2_full.yaml")
    kw = _model_inputs(model)
    g512, _ = _macs_gmac(model, kw, (512, 512))
    g1024, _ = _macs_gmac(model, kw, (1024, 1024))
    assert g512 == pytest.approx(15.670, abs=5e-3), g512
    assert g1024 == pytest.approx(62.678, abs=2e-2), g1024


#: ★ 실측 vs 06 §10 요약 카드. **S0/S1 은 문서 값이 BMEF 를 과다 계상했다.**
#: BMEF MAC 은 실행 path 수에 정확히 비례한다 (실측 @512²: 0.0964 / 0.1927 / 0.2890).
#: 문서는 S0/S1 에도 3-path 값 0.289 를 그대로 썼다:
#:   S1  14.291(문서) − 14.195(실측) = 0.096 = BMEF 의 phys 몫 (정확히 1/3)
#:   S0  10.198(문서) − 10.005(실측) = 0.193 = BMEF 의 spec+phys 몫 (정확히 2/3)
#: S2 는 3-path 라 차이가 없어 문서와 정확히 일치한다.
MODE_MAC_512 = {
    "s0_rgb_aihub092.yaml": (10.0048, 10.198),
    "s1_rgb_ms4.yaml": (14.1947, 14.291),
    "s2_full.yaml": (15.6696, 15.670),
}


@pytest.mark.parametrize("name", sorted(MODE_MAC_512))
def test_mode_mac_measured(name: str) -> None:
    measured, documented = MODE_MAC_512[name]
    model = _deploy_model(name)
    g512, _ = _macs_gmac(model, _model_inputs(model), (512, 512))
    assert g512 == pytest.approx(measured, abs=5e-3), (
        f"{name}: 실측 {g512:.4f} != 회귀값 {measured} (문서 {documented})"
    )


def test_bmef_mac_scales_with_active_paths() -> None:
    """★ 위 불일치의 원인 — BMEF MAC 은 실행 path 수에 **정확히 비례**한다."""
    got = {}
    for name in ("s0_rgb_aihub092.yaml", "s1_rgb_ms4.yaml", "s2_full.yaml"):
        model = _deploy_model(name)
        _, rep = _macs_gmac(model, _model_inputs(model), (512, 512), hooks=True)
        v = sum(
            m for k, m in rep.by_module.items()
            if k == "backbone.bmef" or k.startswith("backbone.bmef.")
        )
        got[name] = scale_macs(v, from_hw=(MEAS_HW, MEAS_HW), to_hw=(512, 512)) / 1e9
    full = got["s2_full.yaml"]
    assert full == pytest.approx(0.2890, abs=1e-3)
    assert got["s1_rgb_ms4.yaml"] == pytest.approx(full * 2 / 3, rel=1e-3)
    assert got["s0_rgb_aihub092.yaml"] == pytest.approx(full * 1 / 3, rel=1e-3)


def test_decoder_mac_subtotal() -> None:
    """정정 A-14 — 디코더 **3.309 @512² / 13.235 @1024²** (초판 3.36/13.44 폐기)."""
    dec = PIDDecoder().eval()
    feats = tuple(
        torch.zeros(1, c, MEAS_HW // s, MEAS_HW // s) for c, s in zip(CHANNELS, (4, 8, 16, 32))
    )
    rep = count_macs_hooks(dec, feats)
    g512 = scale_macs(rep.total, from_hw=(MEAS_HW, MEAS_HW), to_hw=(512, 512)) / 1e9
    g1024 = scale_macs(rep.total, from_hw=(MEAS_HW, MEAS_HW), to_hw=(1024, 1024)) / 1e9
    assert g512 == pytest.approx(3.309, abs=1e-3)
    assert g1024 == pytest.approx(13.235, abs=1e-3)


def test_aux_seg_head_mac_subtotal() -> None:
    """정정 B-21 — AuxSegHead×3 = **0.267 @512² / 1.068 @1024²** (초판 0.409/1.635 폐기)."""
    total = 0
    for tap, s in (("enc_s8", 8), ("enc_s16", 16), ("enc_s32", 32)):
        head = AuxSegHead(AUX_TAP_CH[tap], AUX_TAP_MID[tap], 12).eval()
        x = torch.zeros(1, AUX_TAP_CH[tap], MEAS_HW // s, MEAS_HW // s)
        total += count_macs_hooks(head, x).total
    assert scale_macs(total, from_hw=(MEAS_HW, MEAS_HW), to_hw=(512, 512)) / 1e9 == pytest.approx(
        0.267, abs=1e-3
    )
    assert scale_macs(total, from_hw=(MEAS_HW, MEAS_HW), to_hw=(1024, 1024)) / 1e9 == pytest.approx(
        1.068, abs=1e-3
    )


def test_edge_head_mac() -> None:
    """``EdgeHead`` 는 ``F_dec``(128ch) 이 아니라 **D 분기 ``d32``(32ch)** 에서 분기한다.

    04 §8.3 / [분석] D-DEC-10. ``in_ch=128`` 로 잘못 배선하면 MAC 이 4배(0.606 → 2.418
    @1024²)가 되고 params 도 9,313 이 아니게 된다 — 여기서 잡는다.
    """
    head = EdgeHead().eval()  # 기본 in_ch = BD_CH = 32
    assert sum(p.numel() for p in head.parameters()) == 9_313
    rep = count_macs_hooks(head, torch.zeros(1, 32, MEAS_HW // 4, MEAS_HW // 4))
    g1024 = scale_macs(rep.total, from_hw=(MEAS_HW, MEAS_HW), to_hw=(1024, 1024)) / 1e9
    assert g1024 == pytest.approx(0.606, abs=2e-3), g1024


def test_training_configuration_total_mac() -> None:
    """학습 구성 총계 = 추론 + EdgeHead + AuxSegHead×3 (siam 은 λ=0 이라 모듈 없음).

    ★ 두 MAC 카운터 모두 결정론을 위해 내부에서 ``model.eval()`` 을 강제하므로
    학습 전용 헤드는 모델 통째 측정으로는 잡히지 않는다. 그래서 **합성**으로 구한다.

    실측 합성값 **16.086 @512² / 64.343 @1024²** (K=2).
    06 §10 요약 카드는 16.110 / 64.435 로, 각각 +0.024 / +0.092 크다. 문서 값은
    구성요소의 반올림 합산이며 어느 모듈에도 대응하지 않는다 → 정정 대상으로 기록.
    """
    infer = _deploy_model("s2_full.yaml")
    g512_i, _ = _macs_gmac(infer, _model_inputs(infer), (512, 512))

    edge = count_macs_hooks(
        EdgeHead().eval(), torch.zeros(1, 32, MEAS_HW // 4, MEAS_HW // 4)
    ).total
    aux = 0
    for tap, s in (("enc_s8", 8), ("enc_s16", 16), ("enc_s32", 32)):
        head = AuxSegHead(AUX_TAP_CH[tap], AUX_TAP_MID[tap], 2).eval()
        aux += count_macs_hooks(
            head, torch.zeros(1, AUX_TAP_CH[tap], MEAS_HW // s, MEAS_HW // s)
        ).total

    extra512 = scale_macs(edge + aux, from_hw=(MEAS_HW, MEAS_HW), to_hw=(512, 512)) / 1e9
    total512 = g512_i + extra512
    assert total512 == pytest.approx(16.086, abs=1e-2), total512
    assert total512 * 4 == pytest.approx(64.343, abs=4e-2)
    # 문서 값보다 작다 = 예산 초과가 아니다. 방향까지 고정한다.
    assert total512 < 16.110 and total512 * 4 < 64.435


# ─────────────────────────────────────────────────────────────────────────────
# 상한 게이트
# ─────────────────────────────────────────────────────────────────────────────
def test_budget_ceiling() -> None:
    """★ 총 params ≤ 18 M, 총 MAC@1024² ≤ 70 GMAC (06 §6.5 상한 게이트).

    목표치는 02/03/04 확정치를 본 뒤 설정했으므로 자기충족적이다. 의미 있는 것은
    **여유율** 이며, 향후 모듈 추가로 이 선을 넘으면 재설계 신호다.
    """
    model = _deploy_model("s2_full.yaml")
    params = sum(p.numel() for p in model.parameters())
    g1024, _ = _macs_gmac(model, _model_inputs(model), (1024, 1024))
    assert params <= PARAM_CEILING, f"{params:,} > {PARAM_CEILING:,}"
    assert g1024 <= MAC_CEILING_1024, f"{g1024:.3f} > {MAC_CEILING_1024}"
    # 여유율 (리포트용 — 값이 아니라 부호만 계약이다).
    assert (PARAM_CEILING - params) / PARAM_CEILING > 0.10
    assert (MAC_CEILING_1024 - g1024) / MAC_CEILING_1024 > 0.05


def test_hook_vs_flop_counter_gap_is_attention_only() -> None:
    """hook 과 FlopCounterMode 의 차 = LiteMLA 의 functional matmul (실측 0.178 @512²)."""
    model = _deploy_model("s2_full.yaml")
    kw = _model_inputs(model)
    g_fc, _ = _macs_gmac(model, kw, (512, 512))
    g_hk, _ = _macs_gmac(model, kw, (512, 512), hooks=True)
    gap = g_fc - g_hk
    assert gap == pytest.approx(0.178, abs=5e-3), f"gap {gap:.4f}"
    # S0-RGB 도 stage3/4 LiteMLA 를 갖는다 → gap 이 남아 있어야 한다.
    m0 = _deploy_model("s0_rgb_aihub092.yaml")
    kw0 = _model_inputs(m0)
    assert _macs_gmac(m0, kw0, (512, 512))[0] - _macs_gmac(m0, kw0, (512, 512), hooks=True)[0] > 0.1
