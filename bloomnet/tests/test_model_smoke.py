"""T23 — 전체 모델 forward/backward 스모크 (06 §5.1 T23, 필수 게이트).

대상: ``modules/pid_decoder.py`` (L3), ``models/encoder.py`` (L3),
``models/backbone.py`` (L4), ``models/bloomnet.py`` (L5).

헌법 C-5.1/C-5.2: **CPU 전용**, B=2, H∈{64,128}. 최대 텐서 (2,320,4,4).

검증 항목:
  (a) S0-RGB / (b) S1-RGB+MS4 / (c) S2-Full 전 모달 forward 의 출력 shape 완전성
  (d) ``L.backward()`` 후 전 파라미터 grad 유한 + **죽은 파라미터 목록이 예상과 정확히 일치**
  (e) train / eval 모드 출력 키 집합 (04 §9.1)
  (f) 파라미터 총계가 06 §6.1/§6.2 표와 **정확히** 일치
  + deploy() 후 `edge_head|aux_|siam` 키 0개, num_classes 교체 로드, ExportWrapper 계약,
    정정 A-12 (downsample 배선) 채널 스케줄 회귀.

★ 죽은 파라미터에 대한 실측 결론 (본 테스트가 회귀 고정한다):
  1. ``bmef.<i>.log_tau0`` 은 **어느 모드에서도 gradient 를 받지 못한다**. ``log_tau0`` 은
     ``vacuity = sigmoid(log_tau0 − log_tau)`` 에만 등장하고, ``vacuity`` 는 X-12 로
     **진단 전용(손실 미소비)** 이 동결됐기 때문이다. 05 §5.1.2 는 이 파라미터를
     lr×50 physics 그룹에 넣어 두었다 — 스펙 owner 판단이 필요하다.
  2. ``bmef.<i>.kappa_raw`` 는 ``g_pol is None`` (= ``use_pol=False``, 즉 S0/S1) 에서
     gradient 가 없다. ``g_pol`` 이 있는 S2 에서만 살아난다.
  3. 따라서 06 §6.2 의 S0-RGB "live 8,670,350" 은 **16 파라미터만큼 낙관적**이다
     (kappa_raw 3×4 + log_tau0 1×4 = 16). 참값 8,670,334.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, List, Set, Tuple

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:  # cwd 와 무관하게 bloomnet 패키지를 찾게 한다
    sys.path.insert(0, str(_ROOT))

from bloomnet.config import (  # noqa: E402
    BloomNetConfig,
    default_config,
    from_dict,
    merge_dicts,
    to_dict,
)
from bloomnet.constants import (  # noqa: E402
    CHANNELS,
    OUT_AUX,
    OUT_CHL,
    OUT_EDGE,
    OUT_LOGVAR,
    OUT_SEG,
    OUT_VACUITY,
    PATHS,
)
from bloomnet.models import (  # noqa: E402
    BloomNet,
    BloomNetBackbone,
    BloomNetEncoder,
    ExportWrapper,
    build_bloomnet,
)
from bloomnet.modules.pid_decoder import PIDDecoder  # noqa: E402
from bloomnet.utils.checkpoint import (  # noqa: E402
    load_state_dict_shape_tolerant,
    prune_train_only_keys,
)
from bloomnet.utils.flops import count_macs_hooks, scale_macs  # noqa: E402

B = 2
HW = 64


@pytest.fixture(autouse=True)
def _cpu_only() -> None:
    """헌법 C-5.2 — GPU 절대 금지. 매 테스트가 직접 확인한다."""
    assert not torch.cuda.is_available(), "GPU 가 보이면 안 된다 (CUDA_VISIBLE_DEVICES='')"
    torch.manual_seed(0)
    torch.set_num_threads(4)


# ═══════════════════════════════════════════════════════════════════════════
#  프리셋
# ═══════════════════════════════════════════════════════════════════════════
def _cfg(**override) -> BloomNetConfig:
    return from_dict(merge_dicts(to_dict(default_config()), override))


def cfg_s0(num_classes: int = 12) -> BloomNetConfig:
    return _cfg(mode="s0_rgb", data=dict(modalities=["rgb"], num_classes=num_classes))


def cfg_s1(num_classes: int = 2) -> BloomNetConfig:
    return _cfg(
        mode="s1_rgb_ms4",
        data=dict(
            modalities=["rgb", "msi", "bio"], sensor="m3m", num_classes=num_classes,
            bio=dict(kind="mci", source="msi"),
        ),
    )


def cfg_s2(num_classes: int = 2) -> BloomNetConfig:
    return _cfg(
        mode="s2_full",
        data=dict(
            modalities=["rgb", "msi", "bio", "ir", "pol"], sensor="m3m",
            num_classes=num_classes, bio=dict(kind="mci", source="msi"),
            augment=dict(geometric=dict(rot90=False)),
        ),
        model=dict(ppn=dict(use_pol=True)),
    )


PRESETS = {
    "s0_rgb_k12": cfg_s0(12),
    "s0_rgb_k2": cfg_s0(2),
    "s1_k2": cfg_s1(2),
    "s2_k2": cfg_s2(2),
    "s2_k12": cfg_s2(12),
}


def make_inputs(cfg: BloomNetConfig, *, h: int = HW, b: int = B) -> Dict[str, torch.Tensor]:
    """모드가 요구하는 modality 만 만든다 (A4: 비활성 key 는 존재하지 않는다)."""
    mods = cfg.data.modalities
    out: Dict[str, torch.Tensor] = {"rgb": torch.randn(b, 3, h, h)}
    if "msi" in mods:
        # [H1] 스케일 계약을 만족하는 값 (01 §2.6 R4′ 적용 후 반사율 O(1e-2))
        out["msi"] = torch.rand(b, len(cfg.data.band_ids), h, h) * 0.08 + 0.01
    if "ir" in mods:
        out["ir"] = torch.rand(b, 1, h, h)
    if "pol" in mods:
        out["pol"] = torch.rand(b, 3, h, h)
    return out


def expected_train_keys(cfg: BloomNetConfig) -> Set[str]:
    keys = {OUT_SEG, OUT_CHL, OUT_LOGVAR, OUT_VACUITY}
    if cfg.model.heads.enable_edge_head:
        keys.add(OUT_EDGE)
    for t in cfg.model.aux_taps:
        keys.add(OUT_AUX[t])
    return keys


EVAL_KEYS = {OUT_SEG, OUT_CHL, OUT_LOGVAR, OUT_VACUITY}


# ═══════════════════════════════════════════════════════════════════════════
#  (a)(b)(c) forward — 모드별 출력 shape
# ═══════════════════════════════════════════════════════════════════════════
# H=128 은 두 모드에서만 돈다 (전 테스트 합 < 30초 규약, 06 §5 공통 규약).
_SHAPE_CASES = [(n, 64) for n in sorted(PRESETS)] + [("s0_rgb_k12", 128), ("s2_k2", 128)]


@pytest.mark.parametrize("name,h", _SHAPE_CASES)
def test_forward_shapes_all_modes(name: str, h: int) -> None:
    cfg = PRESETS[name]
    model = build_bloomnet(cfg).train()
    out = model(**make_inputs(cfg, h=h))
    k = cfg.data.num_classes

    assert set(out) == expected_train_keys(cfg)
    assert out[OUT_SEG].shape == (B, k, h // 4, h // 4)
    assert out[OUT_CHL].shape == (B, 1, h // 4, h // 4)
    assert out[OUT_LOGVAR].shape == (B, 1, h // 4, h // 4)
    assert out[OUT_VACUITY].shape == (B, 1, h // 4, h // 4)
    assert out[OUT_EDGE].shape == (B, 1, h // 4, h // 4)
    for tap, s in (("enc_s8", 8), ("enc_s16", 16), ("enc_s32", 32)):
        assert out[OUT_AUX[tap]].shape == (B, k, h // s, h // s)
    for key, v in out.items():
        assert torch.isfinite(v).all(), key

    # ★ 모델 forward 는 절대 업샘플하지 않는다 (04 §9.1)
    assert out[OUT_SEG].shape[-1] == h // 4 != h


def test_chl_and_logvar_initial_values() -> None:
    """ChlHead bias=softplus⁻¹(1.7559) · UncHead zero-init 의 초기 상태 (X-10)."""
    model = build_bloomnet(cfg_s0()).eval()
    out = model(**make_inputs(cfg_s0()))
    assert out[OUT_CHL].allclose(torch.full_like(out[OUT_CHL], 1.7559), atol=1e-4)
    assert torch.equal(out[OUT_LOGVAR], torch.zeros_like(out[OUT_LOGVAR]))


def test_unc_enabled_false_zeroes_log_var() -> None:
    model = build_bloomnet(cfg_s0()).eval()
    # weight 를 흔들어 s != 0 인 상태를 만든 뒤 enabled=False 를 확인한다
    torch.nn.init.normal_(model.unc_head.out.weight, std=0.5)
    torch.nn.init.constant_(model.unc_head.out.bias, 0.3)
    on = model(**make_inputs(cfg_s0()), unc_enabled=True)[OUT_LOGVAR]
    off = model(**make_inputs(cfg_s0()), unc_enabled=False)[OUT_LOGVAR]
    assert on.abs().sum() > 0
    assert torch.equal(off, torch.zeros_like(off))


# ═══════════════════════════════════════════════════════════════════════════
#  (e) train / eval 모드 출력 키
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("name", sorted(PRESETS))
def test_eval_mode_has_no_train_only_keys(name: str) -> None:
    cfg = PRESETS[name]
    model = build_bloomnet(cfg)
    x = make_inputs(cfg)
    model.train()
    train_keys = set(model(**x))
    model.eval()
    eval_keys = set(model(**x))
    assert eval_keys == EVAL_KEYS
    assert train_keys - eval_keys == {OUT_EDGE} | {OUT_AUX[t] for t in cfg.model.aux_taps}


def test_siam_keys_only_when_enabled() -> None:
    # V11 이 teacher 파일 존재를 요구하므로 config 를 우회해 직접 조립한다 (05 D4 — 1차년도 미사용).
    cfg = cfg_s0(12)
    base = build_bloomnet(cfg)
    model = BloomNet(
        base.backbone, base.decoder, num_classes=cfg.data.num_classes,
        siam_channels={"s3": (160, 320), "s4": (320, 512), "dec": (128, 256)},
    ).train()
    out = model(**make_inputs(cfg))
    assert {"siam_s3", "siam_s4", "siam_dec"} <= set(out)
    assert out["siam_s3"].shape == (B, 320, HW // 16, HW // 16)
    assert out["siam_s4"].shape == (B, 512, HW // 32, HW // 32)
    assert out["siam_dec"].shape == (B, 256, HW // 4, HW // 4)
    model.eval()
    assert set(model(**make_inputs(cfg))) == EVAL_KEYS


# ═══════════════════════════════════════════════════════════════════════════
#  (d) backward + 죽은 파라미터
# ═══════════════════════════════════════════════════════════════════════════
def _dead_names(model: BloomNet) -> Set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad and p.grad is None}


def _predicted_dead(model: BloomNet) -> Set[str]:
    """설계상 gradient 를 받을 수 없는 파라미터 (위 docstring 1~3 참조)."""
    live = set(model.active_paths)
    has_gpol = model.encoder.ppn.gate_mode != "none"
    dead: Set[str] = set()
    for i, bm in model.backbone.bmef.items():
        pre = f"backbone.bmef.{i}"
        dead.add(f"{pre}.log_tau0")  # vacuity 가 손실에 안 들어간다 (X-12)
        if not has_gpol:
            dead.add(f"{pre}.kappa_raw")  # g_pol is None -> log_r 항 자체가 없다
        for m in PATHS:
            if m in live:
                continue
            # 미실행 path 의 evidence 헤드 (06 §6.2 각주의 141,764/stage-set)
            dead |= {
                f"{pre}.mean_conv.{m}.weight",
                f"{pre}.mean_norm.{m}.weight", f"{pre}.mean_norm.{m}.bias",
                f"{pre}.prec_dw.{m}.weight",
                f"{pre}.prec_norm.{m}.weight", f"{pre}.prec_norm.{m}.bias",
                f"{pre}.prec_pw.{m}.weight", f"{pre}.prec_pw.{m}.bias",
                f"{pre}.p_raw.{m}",
            }
            if m != "rgb" and bm.enable_feedback:
                dead.add(f"{pre}.fb_gamma.{m}")  # 소비되지 않는 feedback
    if model.encoder.null_embedding:
        dead |= {f"backbone.encoder.null_emb.{m}" for m in PATHS}
    return dead


def _criterion_like_loss(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """criterion 규약을 흉내낸다 — ``vacuity`` 는 **소비하지 않는다** (X-12)."""
    return sum(v.float().pow(2).mean() for k, v in out.items() if k != OUT_VACUITY)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_backward_grads_finite_and_dead_set_exact(name: str) -> None:
    cfg = PRESETS[name]
    model = build_bloomnet(cfg).train()
    out = model(**make_inputs(cfg))
    _criterion_like_loss(out).backward()

    bad = [n for n, p in model.named_parameters() if p.grad is not None
           and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grad: {bad[:5]}"

    dead = _dead_names(model)
    assert dead == _predicted_dead(model), (
        f"예상 밖 죽은 파라미터: {sorted(dead - _predicted_dead(model))[:8]} / "
        f"살아난 예상치: {sorted(_predicted_dead(model) - dead)[:8]}"
    )
    # 죽지 않은 파라미터는 실제로 0 이 아닌 gradient 를 받아야 한다 (배선 확인)
    alive = [p for n, p in model.named_parameters() if n not in dead]
    assert sum(float(p.grad.abs().sum()) for p in alive) > 0


def test_s0_dead_parameter_count_matches_corrected_budget() -> None:
    """06 §6.2 의 'S0-RGB live 8,670,350' 을 실측으로 정정한다 (본 파일 docstring 3)."""
    cfg = cfg_s0(12)
    model = build_bloomnet(cfg).train()
    _criterion_like_loss(model(**make_inputs(cfg))).backward()
    dead = _dead_names(model)
    numel = {n: p.numel() for n, p in model.named_parameters()}
    dead_numel = sum(numel[n] for n in dead)

    # 06 §6.2 각주가 센 것: spec/phys mean·prec 헤드 2×141,764 + fb_gamma 512
    assert 284_040 == 2 * 141_764 + 512
    # 실측: 거기에 kappa_raw 3×4 + log_tau0 1×4 = 16 이 더 있다
    assert dead_numel == 284_056, dead_numel
    assert sum(numel[n] for n in dead if n.endswith("kappa_raw")) == 12
    assert sum(numel[n] for n in dead if n.endswith("log_tau0")) == 4

    deployed = build_bloomnet(cfg).deploy()
    infer = sum(p.numel() for p in deployed.parameters())
    assert infer == 8_954_390  # 06 §6.2 S0-RGB 추론
    assert infer - dead_numel == 8_670_334  # 문서값 8,670,350 대비 −16


def test_input_grad_reaches_every_present_path() -> None:
    """세 modality 입력 전부에 실제 gradient 가 흐르는지 (배선 단절 탐지)."""
    cfg = cfg_s2(2)
    model = build_bloomnet(cfg).train()
    x = make_inputs(cfg)
    for v in x.values():
        v.requires_grad_(True)
    _criterion_like_loss(model(**x)).backward()
    for k, v in x.items():
        assert v.grad is not None and float(v.grad.abs().sum()) > 0, k


def test_null_embedding_is_the_only_extra_dead_parameter() -> None:
    """``null_embedding=True`` 는 +1,728 params 이고 현재 **소비되지 않는다** (02 §2.6)."""
    base = build_bloomnet(cfg_s2(2))
    cfg = cfg_s2(2)
    cfg.model.null_embedding = True
    model = build_bloomnet(cfg).train()
    assert sum(p.numel() for p in model.parameters()) - sum(
        p.numel() for p in base.parameters()
    ) == 3 * sum(CHANNELS) == 1728
    _criterion_like_loss(model(**make_inputs(cfg))).backward()
    assert _dead_names(model) == _predicted_dead(model)


# ═══════════════════════════════════════════════════════════════════════════
#  (f) 파라미터 회귀 — 06 §6.1 / §6.2
# ═══════════════════════════════════════════════════════════════════════════
def test_pid_decoder_params() -> None:
    assert sum(p.numel() for p in PIDDecoder().parameters()) == 1_382_272


def test_pid_decoder_branch_params() -> None:
    """06 §6.1 디코더 행의 분기별 소계."""
    d = PIDDecoder()

    def n(*mods) -> int:
        return sum(p.numel() for m in mods for p in m.parameters())

    assert n(d.pappm) == 590_144
    assert n(d.lat3, d.iblock3, d.isep) == 334_208  # I
    assert n(d.lat2, d.pag8, d.pblock8, d.lat1, d.pag4, d.psep) == 359_552  # P
    assert n(d.dlat, d.diff3, d.dblock, d.diff4, d.dexp) == 65_088  # D
    assert n(d.bag) == 33_280


def test_pid_decoder_mac_regression() -> None:
    """정정 A-14 — 디코더 MAC 3.309 @512² / 13.235 @1024². 초판 3.36/13.44 는 산술 오류.

    정정 A-24 대로 256² 에서 1회 측정하고 ×4 / ×16 로 스케일한다.
    """
    d = PIDDecoder().eval()
    h = 256
    feats = tuple(torch.zeros(1, c, h // s, h // s) for c, s in zip(CHANNELS, (4, 8, 16, 32)))
    rep = count_macs_hooks(d, feats)
    assert scale_macs(rep.total, from_hw=(h, h), to_hw=(512, 512)) / 1e9 == pytest.approx(
        3.309, abs=1e-3
    )
    assert scale_macs(rep.total, from_hw=(h, h), to_hw=(1024, 1024)) / 1e9 == pytest.approx(
        13.235, abs=1e-3
    )

    def grp(*pref: str) -> float:
        v = sum(
            m for k, m in rep.by_module.items()
            if any(k == p or k.startswith(p + ".") for p in pref)
        )
        return scale_macs(v, from_hw=(h, h), to_hw=(1024, 1024)) / 1e9

    assert grp("lat3", "iblock3", "isep") == pytest.approx(1.579, abs=1e-3)
    assert grp("lat2", "pag8", "pblock8", "lat1", "pag4", "psep") == pytest.approx(
        7.323, abs=1e-3
    )
    assert grp("dlat", "diff3", "dblock", "diff4", "dexp") == pytest.approx(1.699, abs=1e-3)
    assert grp("bag") == pytest.approx(2.147, abs=1e-3)
    # PAPPM 은 global branch 가 해상도 무관 상수라 스케일링에 +460,800 MAC 오차가 있다
    # (04 §6.4 직접 측정 0.4864 vs 스케일 0.4869). 04 §10.2 각주와 정합.
    assert grp("pappm") == pytest.approx(0.4869, abs=1e-3)


def test_encoder_params_regression() -> None:
    """02 §7.3 / 06 §6.1 인코더 총계."""
    full = BloomNetEncoder(active_paths=("rgb", "spec", "phys"), use_pol=True)
    assert sum(p.numel() for p in full.parameters()) == 13_968_915
    assert sum(p.numel() for p in full.ppn.parameters()) == 10_101
    assert sum(p.numel() for p in full.sps.parameters()) == 13_312
    assert sum(p.numel() for p in full.tps.parameters()) == 13_216
    for m, want in (("rgb", 6_948_224), ("spec", 5_694_110), ("phys", 1_289_952)):
        got = sum(p.numel() for p in full.blocks[m].parameters())
        got += sum(p.numel() for p in full.down[m].parameters())
        assert got == want, (m, got)

    rgb_only = BloomNetEncoder(active_paths=("rgb",), use_pol=False)
    assert sum(p.numel() for p in rgb_only.parameters()) == 6_952_992
    assert rgb_only.sps is None and rgb_only.tps is None


def test_inactive_path_modules_are_not_created() -> None:
    """헌법 C-3 / 정정 A-17 — S0 체크포인트에는 spec/phys 키가 **애초에 없다**."""
    sd = build_bloomnet(cfg_s0()).state_dict()
    assert not [k for k in sd if ".sps." in k or ".tps." in k]
    assert not [k for k in sd if k.startswith("backbone.encoder.blocks.spec")]
    assert not [k for k in sd if k.startswith("backbone.encoder.blocks.phys")]
    # ★ BMEF 는 active_paths 와 무관하게 3 path 헤드를 항상 생성한다 (06 §6.2 각주)
    assert [k for k in sd if "bmef.1.mean_conv.phys" in k]


def test_bmef_params_regression() -> None:
    enc = BloomNetEncoder(active_paths=("rgb", "spec", "phys"), use_pol=True)
    bb = BloomNetBackbone(enc)
    per = [sum(p.numel() for p in bb.bmef[str(i)].parameters()) for i in (1, 2, 3, 4)]
    assert per == [5_135, 16_399, 90_907, 347_567]
    assert sum(per) == 460_008


@pytest.mark.parametrize(
    "cfg_fn,train_total,infer_total",
    [
        (lambda: cfg_s0(12), 9_447_195, 8_954_390),
        (lambda: cfg_s1(2), 15_151_057, 14_660_522),
        (lambda: cfg_s2(2), 16_459_558, 15_969_023),
        (lambda: cfg_s2(12), 16_463_118, 15_970_313),
    ],
)
def test_mode_param_budget(cfg_fn, train_total: int, infer_total: int) -> None:
    """06 §6.2 모드별 실행 파라미터 (정정 A-15/A-17)."""
    cfg = cfg_fn()
    model = build_bloomnet(cfg)
    assert sum(p.numel() for p in model.parameters()) == train_total
    model.deploy()
    assert sum(p.numel() for p in model.parameters()) == infer_total


def test_head_params_regression() -> None:
    m = build_bloomnet(cfg_s2(2))
    assert sum(p.numel() for p in m.seg_head.parameters()) == 147_970
    assert sum(p.numel() for p in m.reg_trunk.parameters()) == 9_728
    assert sum(p.numel() for p in m.chl_head.parameters()) == 65
    assert sum(p.numel() for p in m.unc_head.parameters()) == 65
    assert sum(p.numel() for p in m.edge_head.parameters()) == 9_313
    m12 = build_bloomnet(cfg_s2(12))
    assert sum(p.numel() for p in m12.seg_head.parameters()) == 149_260
    assert sum(p.numel() for p in m12.aux_heads.parameters()) == 483_492


# ═══════════════════════════════════════════════════════════════════════════
#  구조 계약 — 정정 A-12 / A-26 / X-17
# ═══════════════════════════════════════════════════════════════════════════
def test_channel_schedule_holds_on_every_path() -> None:
    """정정 A-12 — downsample 이 rgb·spec 을 건너뛰면 여기서 즉시 실패한다."""
    cfg = cfg_s2(2)
    model = build_bloomnet(cfg)
    x = make_inputs(cfg)
    out = model.backbone(
        x["rgb"], x_msi=x["msi"], band_ids=cfg.data.band_ids, x_ir=x["ir"], x_pol=x["pol"],
        phys_slot_ids=tuple(cfg.data.phys_slot_ids),
    )
    for m in PATHS:
        got = tuple(int(t.shape[1]) for t in out.feats[m])
        assert got == CHANNELS, (m, got)
        hw = tuple(int(t.shape[-1]) for t in out.feats[m])
        assert hw == (HW // 4, HW // 8, HW // 16, HW // 32), (m, hw)
    assert tuple(int(t.shape[1]) for t in out.fused) == CHANNELS
    assert out.present.shape == (B, 3) and bool(out.present.all())


def test_backbone_is_bit_identical_to_private_streams_at_init() -> None:
    """``fb_gamma`` zero-init 이면 피드백 유무가 **bit-identical** (02 §6.3, atol=0)."""
    cfg = cfg_s2(2)
    torch.manual_seed(7)
    model = build_bloomnet(cfg).eval()
    x = make_inputs(cfg)
    kw = dict(x_msi=x["msi"], band_ids=cfg.data.band_ids, x_ir=x["ir"], x_pol=x["pol"],
              phys_slot_ids=tuple(cfg.data.phys_slot_ids))
    with_fb = model.backbone(x["rgb"], **kw)
    for bm in model.backbone.bmef.values():
        bm.enable_feedback = False  # forward 에서 feedback dict 가 비게 만든다
        bm.fb_gamma = None
    without = model.backbone(x["rgb"], **kw)
    for a, b_ in zip(with_fb.fused, without.fused):
        assert torch.equal(a, b_)


def test_encoder_standalone_forward_matches_backbone_paths() -> None:
    """X-17 — 인코더 단독 경로도 동작하고 present/bio 계약을 지킨다."""
    cfg = cfg_s1(2)
    enc = build_bloomnet(cfg).encoder
    x = make_inputs(cfg)
    eo = enc(x["rgb"], x_msi=x["msi"], band_ids=cfg.data.band_ids)
    assert eo.feats["phys"] is None
    assert eo.g_pol is None  # use_pol=False (정정 B-4)
    assert torch.equal(eo.present, torch.tensor([[True, True, False]] * B))
    assert [tuple(t.shape) for t in eo.bio] == [
        (B, 2, HW // s, HW // s) for s in (4, 8, 16, 32)
    ]
    assert eo.bio_valid is not None and eo.bio_valid.shape == (B, 2)


def test_encoder_without_msi_has_none_bio_pyramid() -> None:
    """정정 B-9 — x_msi 부재 시 bio 는 zeros 가 아니라 ``[None]*4`` 다."""
    enc = build_bloomnet(cfg_s0()).encoder
    eo = enc(torch.randn(B, 3, HW, HW))
    assert eo.bio == [None, None, None, None]
    assert eo.feats["spec"] is None and eo.bio_valid is None


def test_identity_stage_when_not_in_bmef_stages() -> None:
    """정정 A-26 — bmef.stages 에 없는 stage 는 항등(mean-only). stage1 은 필수(V19)."""
    enc = BloomNetEncoder(active_paths=("rgb",))
    bb = BloomNetBackbone(enc, bmef_stages=(1, 2))
    assert set(bb.bmef) == {"1", "2"}
    out = bb(torch.randn(B, 3, HW, HW))
    assert tuple(int(t.shape[1]) for t in out.fused) == CHANNELS
    assert torch.equal(out.vacuity[2], torch.ones_like(out.vacuity[2]))
    with pytest.raises(ValueError, match="V19"):
        BloomNetBackbone(BloomNetEncoder(), bmef_stages=(2, 3, 4))


def test_stage1_identity_flag_drops_the_module() -> None:
    enc = BloomNetEncoder(active_paths=("rgb",))
    bb = BloomNetBackbone(enc, stage1_identity=True)
    assert "1" not in bb.bmef and set(bb.bmef) == {"2", "3", "4"}
    out = bb(torch.randn(B, 3, HW, HW))
    assert out.fused[0].shape == (B, CHANNELS[0], HW // 4, HW // 4)


def test_run_stage_rejects_wrong_channels() -> None:
    """정정 A-12 회귀 — downsample 을 건너뛴 텐서를 넣으면 조용히 통과하면 안 된다."""
    enc = BloomNetEncoder(active_paths=("rgb",))
    with pytest.raises(ValueError, match="채널 스케줄 위반"):
        enc.run_stage("rgb", 2, torch.randn(B, 32, 8, 8))


def test_non_multiple_of_32_rejected() -> None:
    model = build_bloomnet(cfg_s0())
    with pytest.raises(ValueError, match="32"):
        model(rgb=torch.randn(B, 3, 48, 48))


# ═══════════════════════════════════════════════════════════════════════════
#  modality dropout / avail
# ═══════════════════════════════════════════════════════════════════════════
def test_drop_modal_batch_level(monkeypatch) -> None:
    """X-21 — 배치 단위 path off. rgb 는 drop 할 수 없다."""
    cfg = cfg_s2(2)
    model = build_bloomnet(cfg).eval()
    x = make_inputs(cfg)
    only_rgb = model(rgb=x["rgb"])
    dropped = model(**x, drop_modal={"spec": True, "phys": True})
    assert torch.equal(only_rgb[OUT_SEG], dropped[OUT_SEG])
    with pytest.raises(ValueError, match="rgb"):
        model(**x, drop_modal={"rgb": True})
    with pytest.raises(ValueError, match="drop_modal"):
        model(**x, drop_modal={"msi": True})


def test_per_sample_avail_is_accepted_and_finite() -> None:
    """정정 A-25 — 샘플 수준 결측(avail) 이 present/stem 마스킹까지 전파된다."""
    cfg = cfg_s2(2)
    model = build_bloomnet(cfg).eval()
    x = make_inputs(cfg)
    for k in ("msi", "ir", "pol"):
        x[k][1] = 0.0  # A1: avail=0 ⟺ 텐서 전부 정확히 0.0
    avail = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.float32)
    out = model(**x, avail=avail)
    for v in out.values():
        assert torch.isfinite(v).all()
    # 결측 샘플의 spec/phys stem 출력은 정확히 0 (A7-ii)
    stem, _, _ = model.encoder.stems(
        x["rgb"], x_msi=x["msi"], band_ids=cfg.data.band_ids, x_ir=x["ir"], x_pol=x["pol"],
        phys_slot_ids=tuple(cfg.data.phys_slot_ids),
        avail_msi=avail[:, 1], avail_phys=avail[:, 3] * avail[:, 4],
    )
    with torch.no_grad():
        assert float(stem["spec"][1].abs().max()) == 0.0
        assert float(stem["phys"][1].abs().max()) == 0.0
        assert float(stem["spec"][0].abs().max()) > 0.0


def test_present_all_missing_does_not_nan() -> None:
    """03 T5 의 모델 수준 대응 — present=0 (전부 결측) 에서 NaN 이 나오면 안 된다."""
    cfg = cfg_s2(2)
    model = build_bloomnet(cfg).eval()
    x = make_inputs(cfg)
    avail = torch.zeros(B, 5)
    avail[:, 0] = 1.0  # rgb 만 (A2)
    out = model(**x, avail=avail)
    for k, v in out.items():
        assert torch.isfinite(v).all(), k


# ═══════════════════════════════════════════════════════════════════════════
#  deploy / 체크포인트 / export
# ═══════════════════════════════════════════════════════════════════════════
def test_deploy_removes_train_only_keys() -> None:
    model = build_bloomnet(cfg_s2(12)).deploy()
    sd = model.state_dict()
    bad = [k for k in sd if "edge_head" in k or "aux_" in k or "siam" in k]
    assert bad == []
    assert model.is_deployed and not model.training
    assert all(not p.requires_grad for p in model.parameters())
    # prune_train_only_keys 와 동일 결과여야 한다 (utils/checkpoint 계약)
    train_sd = build_bloomnet(cfg_s2(12)).state_dict()
    assert set(prune_train_only_keys(train_sd)) == set(sd)


def test_num_classes_swap_missing_keys_are_exactly_cls() -> None:
    """T23 — K=12 ckpt → K=2 모델. 누락 키는 정확히 cls weight/bias 뿐."""
    src = build_bloomnet(cfg_s2(12))
    dst = build_bloomnet(cfg_s2(2))
    rep = load_state_dict_shape_tolerant(dst, src.state_dict(), verbose=False)
    expect = {"seg_head.cls.weight", "seg_head.cls.bias"} | {
        f"aux_heads.{t}.cls.{p}" for t in ("enc_s8", "enc_s16", "enc_s32")
        for p in ("weight", "bias")
    }
    assert set(rep["missing"]) == expect
    assert set(rep["shape_mismatch"]) == expect
    assert not rep["unexpected"]


def test_mode_upgrade_partial_load_s0_to_s1() -> None:
    """정정 A-17 — 모드 전환 = shape-tolerant partial load + 신규 모듈 랜덤 초기화."""
    s0 = build_bloomnet(cfg_s0(2))
    s1 = build_bloomnet(cfg_s1(2))
    rep = load_state_dict_shape_tolerant(s1, s0.state_dict(), verbose=False)
    assert not rep["unexpected"] and not rep["shape_mismatch"]
    missing = set(rep["missing"])
    assert all(("sps" in k) or (".spec" in k) or ("blocks.spec" in k) for k in missing), (
        sorted(missing)[:5]
    )


def test_export_wrapper_contract() -> None:
    """T25 선행분 — 3-tuple, 입력 수 == active modality 수, conf/chl 범위."""
    for cfg, n_inputs in ((cfg_s0(12), 1), (cfg_s1(2), 3), (cfg_s2(2), 5)):
        model = build_bloomnet(cfg).deploy()
        wrap = ExportWrapper(model, input_hw=(HW, HW))
        assert len(wrap.input_names) == n_inputs
        outs = wrap(*wrap.dummy_inputs(1))
        assert isinstance(outs, tuple) and len(outs) == 3
        seg, chl, conf = outs
        assert seg.shape == (1, cfg.data.num_classes, HW, HW)
        assert chl.shape == (1, 1, HW, HW) and conf.shape == (1, 1, HW, HW)
        assert float(chl.min()) >= 0.0 and float(chl.max()) <= 500.0
        assert 0.0293 <= float(conf.min()) and float(conf.max()) <= 0.9705


def test_export_wrapper_requires_deploy() -> None:
    model = build_bloomnet(cfg_s0())
    with pytest.raises(AssertionError, match="deploy"):
        ExportWrapper(model)


def test_export_static_hw_is_bit_identical_to_training_path() -> None:
    """정정 A-21/B-14 — 정적 ``out_hw`` 주입이 수치를 바꾸지 않는다."""
    torch.manual_seed(3)
    dec = PIDDecoder().eval()
    feats = [torch.randn(1, c, HW // s, HW // s) for c, s in zip(CHANNELS, (4, 8, 16, 32))]
    ref = dec(*feats)
    dec.set_export_hw((HW, HW))
    got = dec(*feats)
    for a, b_ in zip(ref, got):
        assert torch.equal(a, b_)
    dec.set_export_hw(None)
    assert dec.pappm.up_mode == "size"


# ═══════════════════════════════════════════════════════════════════════════
#  optimizer 그룹 계약 (05 §5.1.2 / 02 §11 / 03 §13.1)
# ═══════════════════════════════════════════════════════════════════════════
def test_no_weight_decay_covers_the_frozen_list() -> None:
    model = build_bloomnet(cfg_s2(2))
    nd = model.no_weight_decay()
    names = {n for n, _ in model.named_parameters()}
    assert nd <= names
    must: List[str] = [
        "backbone.encoder.ppn.a_hat", "backbone.encoder.ppn.b",
        "backbone.encoder.ppn.inp_head.bias", "backbone.encoder.sps.body.c_abs",
        "backbone.encoder.tps.body.c_abs",
        "backbone.encoder.blocks.spec.1.0.biogate.beta_hat",
        "backbone.bmef.1.kappa_raw", "backbone.bmef.1.log_tau0",
        "backbone.bmef.1.p_raw.rgb", "backbone.bmef.1.fb_gamma.spec",
        "backbone.encoder.blocks.rgb.1.0.ls_local.gamma",
        "seg_head.bn.weight", "seg_head.bn.bias", "seg_head.cls.bias",
    ]
    for name in must:
        assert name in names, f"이름 규약이 바뀌었다: {name}"
        assert name in nd, name
    # decay 를 **걸어야 하는** 것이 새어 들어오지 않았는지
    for name in ("seg_head.conv.weight", "seg_head.cls.weight",
                 "backbone.bmef.1.mean_conv.rgb.weight"):
        assert name not in nd, name


def test_physics_params_are_the_frozen_five() -> None:
    model = build_bloomnet(cfg_s2(2))
    phys = model.physics_params()
    assert phys <= model.no_weight_decay()
    leaves = {n.rsplit(".", 1)[-1] for n in phys}
    assert leaves == {"a_hat", "b", "beta_hat", "kappa_raw", "log_tau0"}
    assert sum(1 for n in phys if n.endswith("kappa_raw")) == 4
    assert "backbone.encoder.ppn.b" in phys
    # ppn 이 아닌 곳의 'b' 를 끌어오지 않는다
    assert all("ppn" in n.split(".") for n in phys if n.endswith(".b"))


# ═══════════════════════════════════════════════════════════════════════════
#  jit/trace 없이도 export 목록이 config 와 정합한지
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "cfg_fn,expect",
    [
        (cfg_s0, ("rgb",)),
        (cfg_s1, ("rgb", "msi", "bio")),
        (cfg_s2, ("rgb", "msi", "bio", "ir", "pol")),
    ],
)
def test_active_modalities_order(cfg_fn, expect: Tuple[str, ...]) -> None:
    """정정 A-35 — deploy.input_names 의 정본."""
    assert build_bloomnet(cfg_fn()).active_modalities == expect


def test_aux_tap_p8_option() -> None:
    """X-06 대조군 (a5_aux_p8) — p8 tap 이 디코더 P 분기에서 나온다."""
    cfg = _cfg(model=dict(aux_taps=["p8"]), loss=dict(lambda_aux={"p8": 0.4}))
    model = build_bloomnet(cfg).train()
    out = model(**make_inputs(cfg))
    assert OUT_AUX["p8"] in out
    assert out[OUT_AUX["p8"]].shape == (B, cfg.data.num_classes, HW // 8, HW // 8)
    assert sum(p.numel() for p in model.aux_heads.parameters()) == 74_892
