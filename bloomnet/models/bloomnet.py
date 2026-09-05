"""BloomNet 최상위 모델 · ExportWrapper · build_bloomnet (04 §9, 06 §3.4.3, 레벨 L5).

세 겹의 학습 전용 모듈 제거 (04 §9.2):

1. **계산 게이팅** — ``self.training`` 일 때만 edge/aux/siam 을 계산한다.
2. **구조 제거** — :meth:`BloomNet.deploy` 가 ``TRAIN_ONLY`` 를 ``None`` 으로 치환해
   ``state_dict`` 에서 없앤다.
3. **export 전용 forward** — :class:`ExportWrapper` 가 tuple 을 반환하고 업샘플·역변환을
   수행한다. 원본 ``forward`` 를 몽키패치하지 않는다.

★ ``forward`` 는 **절대 업샘플하지 않는다**. 업샘플은 criterion(학습) 또는
ExportWrapper(배포) 의 책임이다 (04 §9.1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bloomnet.constants import (
    AUX_TAP_CH,
    AUX_TAP_MID,
    DEC_CH,
    K235_LOG1P_MEAN,
    MODALITY_ORDER,
    OUT_AUX,
    OUT_CHL,
    OUT_EDGE,
    OUT_LOGVAR,
    OUT_SEG,
    OUT_SIAM,
    OUT_VACUITY,
    PATHS,
    TRAIN_ONLY_MODULES,
)
from bloomnet.data.bundle import derive_present
from bloomnet.models.backbone import BackboneOut, BloomNetBackbone
from bloomnet.models.encoder import BloomNetEncoder
from bloomnet.modules.common import LayerScale
from bloomnet.modules.heads import (
    AuxSegHead,
    ChlHead,
    EdgeHead,
    RegTrunk,
    SegHead,
    SiamProjection,
    UncHead,
)
from bloomnet.modules.pid_decoder import PIDDecoder

__all__ = ["BloomNet", "ExportWrapper", "build_bloomnet"]

# ── no_weight_decay 규약 (02 §11, 03 §13.1, 05 §5.1.2) ────────────────────
#   leaf 이름이 이 집합에 있으면 decay 를 걸지 않는다.
_NO_DECAY_LEAF: Tuple[str, ...] = (
    "bias", "gamma", "a_hat", "b", "c_hat", "c_abs", "beta_hat",
    "kappa_raw", "log_tau0",
)
#   dotted path 에 이 세그먼트가 있으면 decay 를 걸지 않는다.
_NO_DECAY_SEG: Tuple[str, ...] = ("p_raw", "fb_gamma", "inp1", "inp2", "inp3", "inp_head")
#   lr × physics_lr_mult 그룹 (05 §5.1.2)
_PHYSICS_LEAF: Tuple[str, ...] = ("a_hat", "b", "beta_hat", "kappa_raw", "log_tau0")

_NORM_TYPES = (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm)
# fused[i] 중 어느 것이 어떤 tap 인지 (AUX_TAP_STRIDE 와 1:1)
_ENC_TAP_INDEX = {"enc_s8": 1, "enc_s16": 2, "enc_s32": 3}
_SIAM_SOURCE = {"s3": 2, "s4": 3}  # fused index; "dec" 는 F_dec


class BloomNet(nn.Module):
    """backbone + PIDDecoder + 헤드 5종."""

    TRAIN_ONLY: Tuple[str, ...] = TRAIN_ONLY_MODULES  # ("edge_head","aux_heads","siam_proj")

    def __init__(
        self,
        backbone: BloomNetBackbone,
        decoder: PIDDecoder,
        *,
        num_classes: int = 2,
        seg_head_mid: int = DEC_CH,
        reg_trunk_mid: int = 64,
        chl_prior_log1p_mean: float = K235_LOG1P_MEAN,
        edge_prior_pos: float = 0.03,
        unc_use_vacuity: bool = False,
        aux_taps: Sequence[str] = ("enc_s8", "enc_s16", "enc_s32"),
        enable_edge_head: bool = True,
        siam_channels: Optional[Dict[str, Tuple[int, int]]] = None,
        msi_band_ids: Sequence[int] = (1, 2, 3, 5),
        phys_slot_ids: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.num_classes = int(num_classes)
        self.msi_band_ids: Tuple[int, ...] = tuple(int(x) for x in msi_band_ids)
        self.phys_slot_ids: Tuple[int, ...] = tuple(int(x) for x in phys_slot_ids)
        self.unc_use_vacuity = bool(unc_use_vacuity)

        dec_ch = decoder.dec_ch
        self.seg_head = SegHead(dec_ch, seg_head_mid, self.num_classes)
        self.reg_trunk = RegTrunk(dec_ch, reg_trunk_mid)
        self.chl_head = ChlHead(reg_trunk_mid, prior_log1p_mean=chl_prior_log1p_mean)
        self.unc_head = UncHead(reg_trunk_mid, use_vacuity=self.unc_use_vacuity)

        # ── 학습 전용 (TRAIN_ONLY) ─────────────────────────────────────────
        self.edge_head: Optional[nn.Module] = (
            EdgeHead(decoder.bd_ch, decoder.bd_ch, prior_pos=edge_prior_pos)
            if enable_edge_head
            else None
        )

        taps = tuple(aux_taps or ())
        unknown = set(taps) - set(OUT_AUX)
        if unknown:
            raise ValueError(f"aux_taps 는 {sorted(OUT_AUX)} 의 부분집합: {sorted(unknown)}")
        self.aux_taps: Tuple[str, ...] = taps
        self.aux_heads: Optional[nn.ModuleDict] = (
            nn.ModuleDict(
                {t: AuxSegHead(AUX_TAP_CH[t], AUX_TAP_MID[t], self.num_classes) for t in taps}
            )
            if taps
            else None
        )

        self.siam_proj: Optional[nn.ModuleDict] = None
        if siam_channels:
            unknown = set(siam_channels) - set(OUT_SIAM)
            if unknown:
                raise ValueError(f"siam_channels 키는 {sorted(OUT_SIAM)}: {sorted(unknown)}")
            self.siam_proj = nn.ModuleDict(
                {k: SiamProjection(int(v[0]), int(v[1])) for k, v in siam_channels.items()}
            )

    # ------------------------------------------------------------------ 편의
    @property
    def encoder(self) -> BloomNetEncoder:
        return self.backbone.encoder

    @property
    def active_paths(self) -> Tuple[str, ...]:
        return self.backbone.active_paths

    @property
    def active_modalities(self) -> Tuple[str, ...]:
        """ExportWrapper 그래프 입력 순서 (정정 A-35). rgb 는 항상 첫 번째."""
        names = ["rgb"]
        if "spec" in self.active_paths:
            names += ["msi", "bio"]
        if "phys" in self.active_paths:
            names += ["ir", "pol"]
        return tuple(names)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        *,
        rgb: Tensor,
        msi: Optional[Tensor] = None,
        bio: Optional[Tensor] = None,
        ir: Optional[Tensor] = None,
        pol: Optional[Tensor] = None,
        avail: Optional[Tensor] = None,
        band_ids: Optional[Sequence[int]] = None,
        phys_slot_ids: Optional[Sequence[int]] = None,
        drop_modal: Optional[Dict[str, bool]] = None,
        prec_ramp: float = 1.0,
        unc_enabled: bool = True,
        bio_valid: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """모든 출력은 **H/4 이하 해상도** 다 — forward 는 업샘플하지 않는다 (04 §9.1)."""
        b = int(rgb.shape[0])

        # ── batch 단위 modality dropout (★X-21) ────────────────────────────
        if drop_modal:
            unknown = set(drop_modal) - set(PATHS)
            if unknown:
                raise ValueError(f"drop_modal 키는 {PATHS}: {sorted(unknown)}")
            if drop_modal.get("rgb"):
                raise ValueError("rgb 는 drop 할 수 없다 (헌법 C-3 에 RGB 부재 모드가 없다)")
            if drop_modal.get("spec"):
                msi, bio = None, None
            if drop_modal.get("phys"):
                ir, pol = None, None

        # ── avail → present (A7/A8, 정정 A-25) ─────────────────────────────
        if avail is None:
            av = torch.ones(b, len(MODALITY_ORDER), dtype=torch.float32, device=rgb.device)
        else:
            av = avail.to(device=rgb.device, dtype=torch.float32)
            if av.shape != (b, len(MODALITY_ORDER)):
                raise ValueError(f"avail 은 (B,{len(MODALITY_ORDER)}): {tuple(av.shape)}")
        slots = (
            self.phys_slot_ids if phys_slot_ids is None
            else tuple(int(s) for s in phys_slot_ids)
        )
        present = derive_present(av, phys_slot_ids=slots)
        avail_msi = av[:, MODALITY_ORDER.index("msi")]
        avail_phys = present[:, PATHS.index("phys")].to(av.dtype)

        out_b: BackboneOut = self.backbone(
            rgb,
            x_pol=pol,
            x_msi=msi,
            band_ids=self.msi_band_ids if band_ids is None else band_ids,
            x_bio=bio,
            x_ir=ir,
            phys_slot_ids=slots,
            present=present,
            prec_ramp=prec_ramp,
            avail_msi=avail_msi,
            avail_phys=avail_phys,
            bio_valid=bio_valid,
        )
        f1, f2, f3, f4 = out_b.fused
        f_dec, d32, p8 = self.decoder(f1, f2, f3, f4)

        vac1 = out_b.vacuity[0]  # (B,1,H/4,W/4) — stage1 이 디코더와 같은 해상도
        out: Dict[str, Tensor] = {
            OUT_SEG: self.seg_head(f_dec),
            OUT_VACUITY: vac1,
        }
        r = self.reg_trunk(f_dec)
        out[OUT_CHL] = self.chl_head(r)
        out[OUT_LOGVAR] = self.unc_head(r, vacuity=vac1, enabled=unc_enabled)

        if self.training:
            if self.edge_head is not None:
                out[OUT_EDGE] = self.edge_head(d32)
            if self.aux_heads is not None:
                for t in self.aux_taps:
                    src = p8 if t == "p8" else out_b.fused[_ENC_TAP_INDEX[t]]
                    out[OUT_AUX[t]] = self.aux_heads[t](src)
            if self.siam_proj is not None:
                for k, proj in self.siam_proj.items():
                    src = f_dec if k == "dec" else out_b.fused[_SIAM_SOURCE[k]]
                    out[OUT_SIAM[k]] = proj(src)
        return out

    # ----------------------------------------------------------------- deploy
    def deploy(self) -> "BloomNet":
        """TRAIN_ONLY 모듈 제거 → ``eval()`` → ``requires_grad_(False)`` (04 §9.2)."""
        for name in self.TRAIN_ONLY:
            if getattr(self, name, None) is not None:
                setattr(self, name, None)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    @property
    def is_deployed(self) -> bool:
        return all(getattr(self, n, None) is None for n in self.TRAIN_ONLY)

    # ------------------------------------------------------- optimizer 그룹
    def no_weight_decay(self) -> Set[str]:
        """decay 를 걸지 않을 파라미터 **이름** 집합 (02 §11, 03 §13.1, 05 §5.1.2).

        norm affine · 모든 bias · LayerScale γ · PPN ``a_hat``/``b``/inpainter ·
        SPS/TPS ``c_abs`` · BioGate ``β̂`` · BMEF ``p_raw``/``kappa_raw``/``log_tau0``/``fb_gamma``.
        """
        names: Set[str] = set()
        for mod_name, mod in self.named_modules():
            if isinstance(mod, _NORM_TYPES):
                for pn, _ in mod.named_parameters(recurse=False):
                    names.add(f"{mod_name}.{pn}" if mod_name else pn)
            elif isinstance(mod, LayerScale):
                names.add(f"{mod_name}.gamma" if mod_name else "gamma")
        for n, _ in self.named_parameters():
            segs = n.split(".")
            if segs[-1] in _NO_DECAY_LEAF or any(s in _NO_DECAY_SEG for s in segs):
                names.add(n)
        return names

    def physics_params(self) -> Set[str]:
        """lr × ``physics_lr_mult`` 그룹 (05 §5.1.2). ``ppn.b`` 는 게이트 절편이다."""
        out: Set[str] = set()
        for n, _ in self.named_parameters():
            segs = n.split(".")
            if segs[-1] not in _PHYSICS_LEAF:
                continue
            if segs[-1] == "b" and "ppn" not in segs:
                continue  # 다른 모듈의 우연한 'b' 를 끌어오지 않는다
            out.add(n)
        return out

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )


# ═══════════════════════════════════════════════════════════════════════════
#  build_bloomnet
# ═══════════════════════════════════════════════════════════════════════════
def build_bloomnet(cfg: Any) -> BloomNet:
    """``cfg.data.modalities`` → active_paths/use_pol → encoder → backbone → decoder → BloomNet.

    Args:
        cfg: :class:`bloomnet.config.BloomNetConfig`.
    """
    d, m = cfg.data, cfg.model
    active = tuple(cfg.active_paths)
    use_pol = bool(m.ppn.use_pol)

    band_ids = d.band_ids if d.band_ids is not None else (1, 2, 3, 5)
    mci_c = cfg.spec.mci_c if cfg.spec.mci_c is not None else 0.380952
    bio_kind = d.bio.kind if d.bio.kind in ("mci", "gndvi") else "mci"

    encoder = BloomNetEncoder(
        channels=tuple(m.channels),
        depths={"rgb": tuple(m.depths.rgb), "spec": tuple(m.depths.spec),
                "phys": tuple(m.depths.phys)},
        ela=dict(
            attn_stages=tuple(m.ela.attn_stages), head_dim=tuple(m.ela.head_dim),
            scales=tuple(m.ela.scales), expand_ratio=m.ela.expand_ratio, eps=m.ela.eps,
        ),
        biospec=dict(
            mlp_ratio=tuple(m.biospec.mlp_ratio), se_ratio=m.biospec.se_ratio,
            dilations=tuple(tuple(r) for r in m.biospec.dilations),
            beta_init=m.biospec.beta_init,
        ),
        phys=dict(
            expand_ratio=m.physlite.expand_ratio, dw_kernel=tuple(m.physlite.dw_kernel),
            se_ratio=m.physlite.se_ratio, se_min=m.physlite.se_min,
        ),
        ppn=dict(
            always_build=m.ppn.always_build, inpaint_ch=m.ppn.inpaint_ch,
            dilations=tuple(m.ppn.dilations), gate_a_init=m.ppn.gate_a_init,
            gate_b_init=m.ppn.gate_b_init, rgb_proxy_gate=m.ppn.rgb_proxy_gate,
        ),
        sps=dict(bio_init_gain=m.sps.bio_init_gain),
        msi_slots=m.sps.num_slots,
        phys_slots=m.tps.num_slots,
        active_paths=active,
        use_pol=use_pol,
        bio_kind=bio_kind,
        mci_c=float(mci_c),
        layer_scale_init=tuple(m.layer_scale_init),
        drop_path_rate=m.drop_path_rate,
        norm=m.norm,
        null_embedding=m.null_embedding,
    )

    backbone = BloomNetBackbone(
        encoder,
        bmef=dict(
            se_reduction=m.bmef.se_reduction, se_min_channels=m.bmef.se_min_channels,
            logtau_range=m.bmef.logtau_range, chan_logtau_range=m.bmef.chan_logtau_range,
            r_floor=m.bmef.r_floor, kappa_init=tuple(m.bmef.kappa_init),
            gpol_pool=m.bmef.gpol_pool, norm_mode=m.bmef.norm_mode,
            norm_layer=m.bmef.norm_layer, enable_feedback=m.bmef.enable_feedback,
        ),
        bmef_stages=tuple(m.bmef.stages),
        stage1_identity=m.bmef.stage1_identity,
    )

    decoder = PIDDecoder(
        tuple(m.channels),
        dec_ch=m.decoder.dec_ch,
        bd_ch=m.decoder.bd_ch,
        ppm_branch=m.decoder.ppm_branch,
        zero_init_residual=m.decoder.zero_init_residual,
        dot_scale=m.decoder.dot_scale,
    )

    siam_channels = None
    if m.siam.enabled:
        siam_channels = {k: (int(v[0]), int(v[1])) for k, v in m.siam.channels.items()}

    return BloomNet(
        backbone,
        decoder,
        num_classes=d.num_classes,
        seg_head_mid=m.heads.seg_mid,
        reg_trunk_mid=m.heads.reg_trunk_mid,
        chl_prior_log1p_mean=m.heads.chl_prior_log1p_mean,
        edge_prior_pos=m.heads.edge_prior_pos,
        unc_use_vacuity=m.heads.unc_use_vacuity,
        aux_taps=tuple(m.aux_taps),
        enable_edge_head=m.heads.enable_edge_head,
        siam_channels=siam_channels,
        msi_band_ids=tuple(band_ids),
        phys_slot_ids=tuple(d.phys_slot_ids),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  ExportWrapper
# ═══════════════════════════════════════════════════════════════════════════
class ExportWrapper(nn.Module):
    """ONNX/TRT 전용 래퍼 (★X-24, 정정 A-21/A-35).

    ``band_ids``/``phys_slot_ids``/``present`` 를 **생성 시점에 고정**하고 forward 는
    텐서만 받는다. 입력 목록·순서는 ``model.active_modalities`` 로 결정된다 —
    S0-RGB ``(x_rgb,)`` / S1 ``(x_rgb, x_msi, x_bio)`` / S2 ``(x_rgb, x_msi, x_bio, x_ir, x_pol)``.

    Args:
        model: :meth:`BloomNet.deploy` 를 **먼저 호출한** 모델이어야 한다.
        input_hw: ``deploy.input_hw``. 주면 ``PagFM``/``PAPPM`` 의 ``out_hw`` 를 파이썬 int
            리터럴로 굽고 ``PAPPM.up_mode="scale"`` 로 바꾼다 → export 그래프에서
            ``Shape``/``Gather`` 0개 (04 §9.4 step 3). ``None`` 이면 학습 정본(``size=``)
            그대로라 step 3 이 **반드시 실패**한다. 06 §3.4.3 동결 시그니처에 없는
            keyword-only 추가분이며 04 §9.4 가 이 값을 요구한다.
    """

    def __init__(self, model: BloomNet, *, input_hw: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        assert isinstance(model, BloomNet), "ExportWrapper 는 BloomNet 만 감싼다"
        assert model.is_deployed, (
            "ExportWrapper 는 model.deploy() 이후여야 한다 — "
            f"{[n for n in model.TRAIN_ONLY if getattr(model, n, None) is not None]} 가 남아 있다"
        )
        assert not model.training, "export 전 model.eval() 필수 (04 §9.3-J)"
        self.model = model
        self.input_names: Tuple[str, ...] = model.active_modalities
        self.input_hw: Optional[Tuple[int, int]] = (
            None if input_hw is None else (int(input_hw[0]), int(input_hw[1]))
        )
        model.decoder.set_export_hw(self.input_hw)
        # present 는 생성 시점 상수 (X-24). 배치 축은 forward 에서 확장한다.
        pres = torch.zeros(1, len(PATHS), dtype=torch.bool)
        for k, p in enumerate(PATHS):
            pres[0, k] = p in model.active_paths
        self.register_buffer("_present", pres, persistent=False)

    @torch.no_grad()
    def forward(self, *tensors: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            ``(seg (B,K,H,W), chl (B,1,H,W) ∈[0,500] mg/m³, conf (B,1,H,W) ∈[0.0293,0.9705])``.
            dict 이 아니라 **tuple** 이다 (04 §9.3-K).
        """
        if len(tensors) != len(self.input_names):
            raise ValueError(
                f"입력 {len(self.input_names)}개 {self.input_names} 를 기대했는데 "
                f"{len(tensors)}개를 받았다 (정정 A-35)"
            )
        kw: Dict[str, Optional[Tensor]] = dict(zip(self.input_names, tensors))
        m = self.model
        present = self._present.expand(int(tensors[0].shape[0]), len(PATHS))

        out_b = m.backbone(
            kw["rgb"],
            x_pol=kw.get("pol"),
            x_msi=kw.get("msi"),
            band_ids=m.msi_band_ids,
            x_bio=kw.get("bio"),
            x_ir=kw.get("ir"),
            phys_slot_ids=m.phys_slot_ids,
            present=present,
            prec_ramp=1.0,
        )
        f1, f2, f3, f4 = out_b.fused
        f_dec, _, _ = m.decoder(f1, f2, f3, f4)

        seg = m.seg_head(f_dec)
        r = m.reg_trunk(f_dec)
        chl = m.chl_head(r, return_mgm3=True)  # ★ exp(u)-1 (expm1 금지, 정정 B-13)
        s = m.unc_head(r, vacuity=out_b.vacuity[0])
        conf = 1.0 / (1.0 + torch.exp(0.5 * s))  # ★X-25 동결 정의

        seg = F.interpolate(seg, scale_factor=4.0, mode="bilinear", align_corners=False)
        chl = F.interpolate(chl, scale_factor=4.0, mode="bilinear", align_corners=False)
        conf = F.interpolate(conf, scale_factor=4.0, mode="bilinear", align_corners=False)
        return seg, chl, conf

    def dummy_inputs(self, batch: int = 1, *, msi_channels: Optional[int] = None) -> List[Tensor]:
        """``torch.onnx.export`` 용 더미 입력 (순서 = :attr:`input_names`)."""
        if self.input_hw is None:
            raise ValueError("dummy_inputs 는 input_hw 가 필요하다")
        h, w = self.input_hw
        km = len(self.model.msi_band_ids) if msi_channels is None else int(msi_channels)
        ch = {"rgb": 3, "msi": km, "bio": 2, "ir": 1, "pol": 3}
        return [torch.zeros(batch, ch[n], h, w) for n in self.input_names]
