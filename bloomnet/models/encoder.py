"""BloomNet 인코더 — stem 3종 + path 3종 (02 §6, 06 §3.4.1, 레벨 L3).

★X-17: **BMEF 를 소유하지 않는다.** 03 §5.5 의 aux 피드백 때문에 stage 를 한 번에 다 돌 수
없으므로, 본선(`models/backbone.py`)이 :meth:`BloomNetEncoder.run_stage` /
:meth:`BloomNetEncoder.downsample` 을 BMEF 와 **교대 호출**한다. :meth:`forward` 는 피드백
없이 4 stage 를 끝까지 도는 단독 실행 경로(단위 테스트·ablation 용)다.

핵심 계약:

* ``active_paths`` 에 없는 path 의 stem·block 은 **생성하지 않는다** (헌법 C-3, 02 §2.6).
  따라서 S0-RGB 체크포인트에는 spec/phys 키가 애초에 없다 — 모드 전환은 shape-tolerant
  partial load + 신규 모듈 랜덤 초기화다 (정정 A-17).
* 결측 path 는 ``feats[m] = None`` 이다. zeros 를 만들지 않는다 — BioGate 가 bias 만으로
  동작해 학습 후 상수 gain 이 되어 Novelty N2 의 근거가 사라진다 (정정 B-9).
* ``g_pol`` 은 ``Optional`` 이며 **full resolution** (B,1,H,W) 이다 (정정 B-4, 03 §13.1).
* 채널 스케줄 32→64→160→320 이 **세 path 모두에서** 성립한다 (정정 A-12).

06 §3.4.1 동결 시그니처 대비 추가분 (전부 keyword-only, 기본값 = 동결 동작):

``ppn`` / ``sps`` dict
    ``config.model.ppn.*`` (``always_build``/``inpaint_ch``/``dilations``/``gate_*_init``/
    ``rgb_proxy_gate``) 과 ``config.model.sps.bio_init_gain`` 을 전달할 통로가 동결
    시그니처에 없다. ``ela``/``biospec``/``phys`` 와 동일한 dict 패턴을 따른다.
``stems(..., avail_msi=, avail_phys=, return_bio_valid=)``
    A7(정정 A-25) 의 샘플 수준 결측 전파와 ``bio_valid``(정정 B-3) 를 넘길 통로.
    ``return_bio_valid=False`` 기본이라 동결된 3-tuple 반환 계약은 그대로다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from bloomnet.constants import CHANNELS, MSI_SLOTS, PATHS, PHYS_SLOTS
from bloomnet.modules.blocks_biospec import BioSpecBlock
from bloomnet.modules.blocks_ela import ELAGlobal, ELALocal
from bloomnet.modules.blocks_physlite import PhysLiteBlock
from bloomnet.modules.common import Downsample, build_bio_pyramid, init_encoder
from bloomnet.modules.stems import PPN, SPS, TPS

__all__ = ["EncoderOut", "BloomNetEncoder", "DEFAULT_DEPTHS"]

DEFAULT_DEPTHS: Dict[str, Tuple[int, ...]] = {
    "rgb": (2, 2, 4, 3),
    "spec": (2, 2, 4, 3),
    "phys": (1, 1, 2, 1),
}
_DEFAULT_ELA = dict(attn_stages=(3, 4), head_dim=(16, 16, 32, 32), scales=(3, 5),
                    expand_ratio=4, eps=1e-5)
_DEFAULT_BIOSPEC = dict(mlp_ratio=(8, 8, 4, 4), se_ratio=4, beta_init=0.1,
                        dilations=((1, 3, 5), (1, 3, 5), (1, 2, 4), (1, 2, 3)))
_DEFAULT_PHYS = dict(expand_ratio=2, dw_kernel=(3, 3, 5, 5), se_ratio=8, se_min=8)
_DEFAULT_PPN = dict(always_build=False, inpaint_ch=16, dilations=(1, 2, 4),
                    gate_a_init=20.0, gate_b_init=-5.0, rgb_proxy_gate=False)
_DEFAULT_SPS = dict(bio_init_gain=4.0, mci_c_re2=None, allow_mci_re2=True)

# 02 §2.4 — phys canonical slot 0 은 IR(x_ir ch0), 1..3 은 x_pol 의 ch0..2.
_PHYS_SLOT_SOURCE = {0: ("ir", 0), 1: ("pol", 0), 2: ("pol", 1), 3: ("pol", 2)}


@dataclass
class EncoderOut:
    """02 §6.5 / 06 §3.4.1 반환 계약."""

    feats: Dict[str, Optional[List[Tensor]]]  # 'rgb'|'spec'|'phys' -> [F^(1..4)] 또는 None
    present: Tensor  # (B,3) bool, 순서 = PATHS
    g_pol: Optional[Tensor]  # (B,1,H,W) 또는 None                      ★X-05/X-18
    bio: List[Optional[Tensor]]  # [(B,2,H_i,W_i)] i=1..4, 없으면 [None]*4  ★정정 B-9
    # ★ (정정 B-3) 02 §6.5 가 추가한 5번째 필드. 06 §3.4.1 에는 없어 기본값을 둔다 —
    #   앞 4개 위치 생성 계약은 불변이다.
    bio_valid: Optional[Tensor] = None  # (B,2), x_msi 부재 시 None
    x_bio: Optional[Tensor] = None  # (B,2,H,W) 원해상도 (진단·재사용)


def _as_tuple(v, n: int, name: str) -> Tuple:
    if isinstance(v, (int, float)):
        return tuple([v] * n)
    t = tuple(v)
    if len(t) != n:
        raise ValueError(f"{name} 은 {n} 원소여야 한다: {t}")
    return t


def _dpr(rate: float, n: int) -> List[float]:
    """``linspace(0, rate, n)`` (02 §1.4). ``n == 1`` 이면 ``[0.0]``."""
    if n <= 1:
        return [0.0] * n
    return [rate * i / (n - 1) for i in range(n)]


class BloomNetEncoder(nn.Module):
    """stem 3종 + path 3종. BMEF 없음 (★X-17)."""

    def __init__(
        self,
        *,
        channels: Tuple[int, ...] = CHANNELS,
        depths: Optional[Dict[str, Tuple[int, ...]]] = None,
        ela: Optional[dict] = None,
        biospec: Optional[dict] = None,
        phys: Optional[dict] = None,
        ppn: Optional[dict] = None,
        sps: Optional[dict] = None,
        msi_slots: int = len(MSI_SLOTS),
        phys_slots: int = len(PHYS_SLOTS),
        active_paths: Tuple[str, ...] = ("rgb",),
        use_pol: bool = False,
        bio_kind: str = "mci",
        mci_c: float = 0.380952,
        layer_scale_init: Union[float, Sequence[float]] = 0.01,
        drop_path_rate: float = 0.1,
        norm: str = "bn",
        null_embedding: bool = False,
    ) -> None:
        super().__init__()
        ch = tuple(int(c) for c in channels)
        if len(ch) != 4:
            raise ValueError(f"channels 는 4원소: {ch}")
        act = tuple(active_paths)
        if "rgb" not in act:
            raise ValueError("active_paths 에는 'rgb' 가 반드시 있어야 한다 (02 §6.5)")
        unknown = set(act) - set(PATHS)
        if unknown:
            raise ValueError(f"active_paths 는 {PATHS} 의 부분집합: {sorted(unknown)}")

        self.channels = ch
        self.active_paths: Tuple[str, ...] = tuple(m for m in PATHS if m in act)
        self.use_pol = bool(use_pol)
        self.norm = norm
        self.null_embedding = bool(null_embedding)

        d = dict(DEFAULT_DEPTHS)
        d.update(depths or {})
        self.depths = {m: tuple(int(x) for x in d[m]) for m in PATHS}
        for m in PATHS:
            if len(self.depths[m]) != 4:
                raise ValueError(f"depths[{m!r}] 는 4원소: {self.depths[m]}")

        e = dict(_DEFAULT_ELA); e.update(ela or {})
        b = dict(_DEFAULT_BIOSPEC); b.update(biospec or {})
        p = dict(_DEFAULT_PHYS); p.update(phys or {})
        pn = dict(_DEFAULT_PPN); pn.update(ppn or {})
        sp = dict(_DEFAULT_SPS); sp.update(sps or {})
        self.attn_stages = tuple(int(i) for i in e["attn_stages"])

        ls = _as_tuple(layer_scale_init, 4, "layer_scale_init")
        head_dim = _as_tuple(e["head_dim"], 4, "ela.head_dim")
        mlp_ratio = _as_tuple(b["mlp_ratio"], 4, "biospec.mlp_ratio")
        dils = tuple(tuple(int(x) for x in row) for row in b["dilations"])
        if len(dils) != 4:
            raise ValueError(f"biospec.dilations 는 4행: {dils}")
        dw_kernel = _as_tuple(p["dw_kernel"], 4, "phys.dw_kernel")

        # ── stem ───────────────────────────────────────────────────────────
        self.ppn = PPN(
            in_chans=3,
            embed_dim=ch[0],
            use_pol=self.use_pol,
            inpaint_ch=int(pn["inpaint_ch"]),
            dilations=tuple(int(x) for x in pn["dilations"]),
            gate_a_init=float(pn["gate_a_init"]),
            gate_b_init=float(pn["gate_b_init"]),
            rgb_proxy_gate=bool(pn["rgb_proxy_gate"]),
            always_build=bool(pn["always_build"]),
            norm=norm,
        )
        self.sps = (
            SPS(
                num_slots=int(msi_slots),
                bio_ch=2,
                embed_dim=ch[0],
                bio_init_gain=float(sp["bio_init_gain"]),
                bio_kind=bio_kind,
                mci_c=float(mci_c),
                mci_c_re2=sp["mci_c_re2"],
                allow_mci_re2=bool(sp["allow_mci_re2"]),
                norm=norm,
            )
            if "spec" in self.active_paths
            else None
        )
        self.tps = (
            TPS(num_slots=int(phys_slots), embed_dim=ch[0], norm=norm)
            if "phys" in self.active_paths
            else None
        )

        # ── block / downsample ─────────────────────────────────────────────
        self.blocks = nn.ModuleDict()
        self.down = nn.ModuleDict()
        for m in self.active_paths:
            n_total = sum(self.depths[m])
            # PhysLite 는 전체 5 block 이라 stochastic depth 의 통계적 의미가 없다 (02 §1.4).
            rates = [0.0] * n_total if m == "phys" else _dpr(float(drop_path_rate), n_total)
            per_stage = nn.ModuleDict()
            k = 0
            for i in range(1, 5):
                lst = nn.ModuleList()
                for _ in range(self.depths[m][i - 1]):
                    lst.append(
                        self._make_block(
                            m, i, ch[i - 1],
                            ls=ls[i - 1], dp=rates[k], head_dim=head_dim[i - 1],
                            ela=e, mlp_ratio=mlp_ratio[i - 1], dil=dils[i - 1],
                            se_ratio=int(b["se_ratio"]), beta_init=float(b["beta_init"]),
                            phys_cfg=p, dw_k=dw_kernel[i - 1],
                        )
                    )
                    k += 1
                per_stage[str(i)] = lst
            self.blocks[m] = per_stage

            ds = nn.ModuleDict()
            for i in range(1, 4):  # stage i -> i+1
                dsm = Downsample(ch[i - 1], ch[i], norm=norm)
                dsm.apply(init_encoder)  # Downsample 은 자체 초기화가 없다
                ds[str(i)] = dsm
            self.down[m] = ds

        # ── (옵션) null embedding — 기본 비활성 (02 §2.6) ────────────────────
        # ⚠ 현재 forward 는 이 파라미터를 **소비하지 않는다**. 유일한 용도인 per-path aux
        #   seg head (02 §12-U8) 가 구현되지 않았기 때문이다. 켜면 grad=None 인
        #   파라미터 3×4개(합 1,728)가 생긴다 — 의도된 상태이며 T23 이 그 사실을 고정한다.
        if self.null_embedding:
            self.null_emb = nn.ParameterDict(
                {m: nn.Parameter(torch.zeros(sum(ch))) for m in PATHS}
            )
        else:
            self.null_emb = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ build
    def _make_block(
        self, path: str, i: int, dim: int, *, ls: float, dp: float, head_dim: int,
        ela: dict, mlp_ratio: int, dil: Tuple[int, int, int], se_ratio: int,
        beta_init: float, phys_cfg: dict, dw_k: int,
    ) -> nn.Module:
        if path == "rgb":
            if i in self.attn_stages:
                return ELAGlobal(
                    dim, head_dim=int(head_dim),
                    scales=tuple(int(s) for s in ela["scales"]),
                    expand_ratio=int(ela["expand_ratio"]), eps=float(ela["eps"]),
                    layer_scale_init=ls, drop_path=dp, norm=self.norm,
                )
            return ELALocal(
                dim, expand_ratio=int(ela["expand_ratio"]),
                layer_scale_init=ls, drop_path=dp, norm=self.norm,
            )
        if path == "spec":
            return BioSpecBlock(
                dim, mlp_ratio=int(mlp_ratio), dilations=dil, bio_ch=2,
                se_ratio=int(se_ratio), layer_scale_init=ls, drop_path=dp,
                beta_init=beta_init, norm=self.norm,
            )
        return PhysLiteBlock(
            dim, expand_ratio=int(phys_cfg["expand_ratio"]), dw_kernel=int(dw_k),
            se_ratio=int(phys_cfg["se_ratio"]), se_min=int(phys_cfg["se_min"]),
            layer_scale_init=ls, drop_path=0.0, norm=self.norm,
        )

    # ------------------------------------------------------------------ stems
    def build_phys_input(
        self, x_ir: Optional[Tensor], x_pol: Optional[Tensor],
        phys_slot_ids: Sequence[int],
    ) -> Tuple[Optional[Tensor], Tuple[int, ...]]:
        """``(x_ir, x_pol)`` → TPS 입력 ``(B,K_in,H,W)`` 와 실제 slot id 목록.

        가용하지 않은 slot 은 목록에서 빠진다 — ``CanonicalScatter`` 가 ``m`` 으로
        결측을 표현하고 ``C_abs`` 가 보상한다 (02 §2.1).
        """
        pieces: List[Tensor] = []
        ids: List[int] = []
        for s in phys_slot_ids:
            s = int(s)
            if s not in _PHYS_SLOT_SOURCE:
                raise ValueError(f"phys slot id 는 0..3 (X-04): {s}")
            src, j = _PHYS_SLOT_SOURCE[s]
            t = x_ir if src == "ir" else x_pol
            if t is None:
                continue
            if t.shape[1] <= j:
                raise ValueError(
                    f"phys slot {s} 는 {src} 의 채널 {j} 를 요구하는데 "
                    f"입력 채널이 {t.shape[1]} 개다 (data.pol.encoding 확인)"
                )
            pieces.append(t[:, j : j + 1])
            ids.append(s)
        if not pieces:
            return None, ()
        return torch.cat(pieces, dim=1), tuple(ids)

    def stems(
        self,
        x_rgb: Tensor,
        *,
        x_pol: Optional[Tensor] = None,
        x_msi: Optional[Tensor] = None,
        band_ids: Optional[Sequence[int]] = None,
        x_bio: Optional[Tensor] = None,
        x_ir: Optional[Tensor] = None,
        phys_slot_ids: Optional[Sequence[int]] = None,
        avail_msi: Optional[Tensor] = None,
        avail_phys: Optional[Tensor] = None,
        bio_valid: Optional[Tensor] = None,
        return_bio_valid: bool = False,
    ):
        """3 stem 을 실행한다.

        Returns:
            ``({"rgb":…, "spec":…|None, "phys":…|None}, g_pol|None, x_bio_full|None)``
            — ``return_bio_valid=True`` 면 ``bio_valid`` 를 붙인 4-tuple.
        """
        out: Dict[str, Optional[Tensor]] = {"rgb": None, "spec": None, "phys": None}
        rgb_stem, g_pol = self.ppn(x_rgb, x_pol)
        out["rgb"] = rgb_stem

        x_bio_full: Optional[Tensor] = None
        bv: Optional[Tensor] = None
        if self.sps is not None and x_msi is not None:
            if band_ids is None:
                raise ValueError("x_msi 를 주면 band_ids 도 함께 줘야 한다 (02 §2.1)")
            sout = self.sps(
                x_msi, band_ids, x_bio, avail_msi, bio_valid=bio_valid
            )
            out["spec"], x_bio_full, bv = sout.spec_stem, sout.x_bio_full, sout.bio_valid

        if self.tps is not None:
            ids = (0, 1, 2, 3) if phys_slot_ids is None else tuple(int(s) for s in phys_slot_ids)
            x_phys, real_ids = self.build_phys_input(x_ir, x_pol, ids)
            if x_phys is not None:
                out["phys"] = self.tps(x_phys, real_ids, avail_phys)

        if return_bio_valid:
            return out, g_pol, x_bio_full, bv
        return out, g_pol, x_bio_full

    # ------------------------------------------------------------ stage 단위 API
    def run_stage(
        self, path: str, i: int, x: Tensor, bio_i: Optional[Tensor] = None
    ) -> Tensor:
        """``path`` 의 stage ``i`` block 을 순차 실행한다. ``bio_i`` 는 spec 만 쓴다."""
        if path not in self.blocks:
            raise KeyError(f"path {path!r} 는 생성되지 않았다 (active_paths={self.active_paths})")
        if i not in (1, 2, 3, 4):
            raise ValueError(f"stage i 는 1..4: {i}")
        expect = self.channels[i - 1]
        if x.shape[1] != expect:
            raise ValueError(
                f"채널 스케줄 위반: path={path} stage={i} 는 C={expect} 를 기대하는데 "
                f"입력이 {x.shape[1]} 이다 (정정 A-12 downsample 배선 확인)"
            )
        for blk in self.blocks[path][str(i)]:
            x = blk(x, bio_i) if path == "spec" else blk(x)
        return x

    def downsample(self, path: str, i: int, x: Tensor) -> Tensor:
        """stage ``i`` → ``i+1`` (``i ∈ 1..3``)."""
        if i not in (1, 2, 3):
            raise ValueError(f"downsample i 는 1..3: {i}")
        return self.down[path][str(i)](x)

    def bio_pyramid(self, x_bio: Optional[Tensor], h: int, w: int) -> List[Optional[Tensor]]:
        """원해상도 ``x_bio`` → stage 별 area-pooled 목록 (정정 B-7)."""
        sizes = [(h // s, w // s) for s in (4, 8, 16, 32)]
        return build_bio_pyramid(x_bio, sizes)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x_rgb: Tensor,
        *,
        x_pol: Optional[Tensor] = None,
        x_msi: Optional[Tensor] = None,
        band_ids: Optional[Sequence[int]] = None,
        x_bio: Optional[Tensor] = None,
        x_ir: Optional[Tensor] = None,
        phys_slot_ids: Optional[Sequence[int]] = None,
        present: Optional[Tensor] = None,
        avail_msi: Optional[Tensor] = None,
        avail_phys: Optional[Tensor] = None,
    ) -> EncoderOut:
        """피드백 없이 4 stage 를 끝까지 도는 **단독 실행 경로**.

        학습·추론 본선은 :class:`~bloomnet.models.backbone.BloomNetBackbone` 이
        ``run_stage``/``downsample`` 을 BMEF 와 교대 호출한다.
        """
        h, w = int(x_rgb.shape[-2]), int(x_rgb.shape[-1])
        if h % 32 or w % 32:
            raise ValueError(f"H,W 는 32 의 배수여야 한다 (헌법 C-4): {(h, w)}")

        stem, g_pol, x_bio_full, bio_valid = self.stems(
            x_rgb, x_pol=x_pol, x_msi=x_msi, band_ids=band_ids, x_bio=x_bio,
            x_ir=x_ir, phys_slot_ids=phys_slot_ids, avail_msi=avail_msi,
            avail_phys=avail_phys, return_bio_valid=True,
        )
        bio = self.bio_pyramid(x_bio_full, h, w)

        live = [m for m in self.active_paths if stem[m] is not None]
        feats: Dict[str, Optional[List[Tensor]]] = {m: None for m in PATHS}
        x = {m: stem[m] for m in live}
        for m in live:
            feats[m] = []
        for i in range(1, 5):
            for m in live:
                f = self.run_stage(m, i, x[m], bio[i - 1] if m == "spec" else None)
                feats[m].append(f)  # type: ignore[union-attr]
                if i < 4:
                    x[m] = self.downsample(m, i, f)

        pres = self.default_present(x_rgb.shape[0], live, device=x_rgb.device)
        if present is not None:
            pres = pres & present.to(device=x_rgb.device).bool()
        return EncoderOut(
            feats=feats, present=pres, g_pol=g_pol, bio=bio,
            bio_valid=bio_valid, x_bio=x_bio_full,
        )

    def default_present(
        self, batch: int, live: Sequence[str], *, device=None
    ) -> Tensor:
        """실행된 path 만 True 인 (B,3) 마스크. 데이터 의존 분기가 없다 (export 안전)."""
        pres = torch.zeros(batch, len(PATHS), dtype=torch.bool, device=device)
        for k, m in enumerate(PATHS):
            if m in live:
                pres[:, k] = True
        return pres

    # ------------------------------------------------------------------ 잡다
    def extra_repr(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"active_paths={self.active_paths}, channels={self.channels}, "
            f"use_pol={self.use_pol}, norm={self.norm!r}"
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
