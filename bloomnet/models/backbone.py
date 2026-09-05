"""BloomNet 백본 — encoder + BMEF×4 **교대 실행** + aux feedback 배선 (레벨 L4).

출처: 02 §6.3, 03 §5.5, 06 §3.4.2 (★X-17, ★정정 A-12/A-26).

실행 순서 (동결, 정정 A-12) — **피드백 루프와 downsample 루프를 분리**한다::

    P = encoder.active_paths ∩ 실행 가능한 path
    x = {m: stems(...)[m] for m in P}
    for i in 1..4:
        F      = {m: encoder.run_stage(m, i, x[m], bio[i]) for m in P}
        out_i  = bmef[i](F, present, g_pol, prec_ramp=…)
        fused[i], vacuity[i] = out_i.fused, out_i.vacuity
        if i < 4:
            for m in ("spec", "phys"):          # rgb 는 피드백 없음 (pre-fusion 스트림)
                if F.get(m) is not None: F[m] = F[m] + out_i.feedback[m]
            for m in P:                          # ★ rgb 포함 **전 path** 를 downsample
                x[m] = encoder.downsample(m, i, F[m])

초판 의사코드는 downsample 을 피드백 루프의 루프 변수 ``m`` 으로 호출해 rgb/spec 이 한 번도
축소되지 않았다 — 그대로 구현하면 stage2 에서 (B,32,·) 를 C=64 블록에 넣어 즉시 터진다.
``run_stage`` 가 채널 스케줄을 assert 하므로 이 회귀는 구조적으로 차단된다.

``fb_gamma`` 는 zero-init 이므로 학습 초기에는 완전한 private stream 과 **bit-identical** 이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from torch import Tensor, nn

from bloomnet.constants import PATHS
from bloomnet.models.encoder import BloomNetEncoder
from bloomnet.modules.bmef import FEEDBACK_PATHS, BMEF, BMEFOutput, identity_fuse

__all__ = ["BackboneOut", "BloomNetBackbone"]


@dataclass
class BackboneOut:
    """06 §3.4.2 반환 계약."""

    fused: List[Tensor]  # [F_fused^(1..4)] 채널 32/64/160/320
    vacuity: List[Tensor]  # [(B,1,H_i,W_i)] i=1..4
    present: Tensor  # (B,3) bool
    g_pol: Optional[Tensor]  # (B,1,H,W) 또는 None
    feats: Dict[str, Optional[List[Tensor]]]  # per-path (진단·per-path aux 용)
    weights: Optional[List[Tensor]] = None  # [(B,3,H_i,W_i)] — return_weights=True 시
    # ★ (정정 B-3) 추가분. 앞 6개 위치 생성 계약은 불변이다.
    bio_valid: Optional[Tensor] = None
    log_tau: Optional[List[Tensor]] = None  # 진단 (03 §8.5 모니터링 의무)


class BloomNetBackbone(nn.Module):
    """encoder 와 BMEF 4개를 교대 실행한다.

    Args:
        encoder: 소유하지 않고 **주입받는다** (★X-17 — BMEF 소유자는 백본이다).
        bmef: :class:`~bloomnet.modules.bmef.BMEF` 생성자 kwargs.
        bmef_stages: BMEF 를 실제로 두는 stage. **1 은 반드시 포함** (V19) — PIDDecoder 의
            필수 입력 ``F_fused^(1)`` 이 정의되지 않으면 헌법 C-3 이 그 stage 에서 깨진다.
            여기 없는 stage 는 항등(mean-only) 모드로 동작한다 (정정 A-26).
        stage1_identity: 03 U-8 ablation. ``True`` 면 stage 1 BMEF 모듈을 **생성하지 않고**
            항등 모드로 돌린다 (동결 시그니처에 없는 keyword-only 추가분 —
            06 §3.4.2 가 ``bmef.stage1_identity`` 플래그를 요구한다).
    """

    def __init__(
        self,
        encoder: BloomNetEncoder,
        *,
        bmef: Optional[dict] = None,
        bmef_stages: Tuple[int, ...] = (1, 2, 3, 4),
        stage1_identity: bool = False,
    ) -> None:
        super().__init__()
        stages = tuple(sorted({int(i) for i in bmef_stages}))
        if any(i not in (1, 2, 3, 4) for i in stages):
            raise ValueError(f"bmef_stages 는 1..4 의 부분집합: {stages}")
        if 1 not in stages:
            raise ValueError(
                "V19: bmef.stages 에는 1 이 반드시 포함되어야 한다 (정정 A-26). "
                "stage 1 을 항등으로 돌리려면 stage1_identity=True 를 쓴다."
            )
        self.encoder = encoder
        self.bmef_stages = stages
        self.stage1_identity = bool(stage1_identity)

        kw = dict(bmef or {})
        kw.pop("stages", None)
        kw.pop("stage1_identity", None)
        kw.pop("warmup_epochs", None)  # trainer 소관 (prec_ramp 산출용)
        self.bmef = nn.ModuleDict()
        for i in stages:
            if i == 1 and self.stage1_identity:
                continue  # 03 U-8 — 모듈 자체를 만들지 않는다
            self.bmef[str(i)] = BMEF(encoder.channels[i - 1], i, **kw)

    # ------------------------------------------------------------------ 편의
    @property
    def active_paths(self) -> Tuple[str, ...]:
        return self.encoder.active_paths

    def _fuse(
        self, i: int, feats: Dict[str, Optional[Tensor]], present: Tensor,
        g_pol: Optional[Tensor], *, prec_ramp: float, return_weights: bool,
    ) -> BMEFOutput:
        key = str(i)
        if key in self.bmef:
            return self.bmef[key](
                feats, present, g_pol, prec_ramp=prec_ramp, return_weights=return_weights
            )
        # 정정 A-26 — 항등(mean-only) 모드. feedback 이 zero 텐서라 배선이 그대로 동작한다.
        return identity_fuse(
            feats, present, stage_idx=i, enable_feedback=True, return_weights=return_weights
        )

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
        prec_ramp: float = 1.0,
        return_weights: bool = False,
        avail_msi: Optional[Tensor] = None,
        avail_phys: Optional[Tensor] = None,
        bio_valid: Optional[Tensor] = None,
    ) -> BackboneOut:
        h, w = int(x_rgb.shape[-2]), int(x_rgb.shape[-1])
        if h % 32 or w % 32:
            raise ValueError(f"H,W 는 32 의 배수여야 한다 (헌법 C-4): {(h, w)}")

        enc = self.encoder
        stem, g_pol, x_bio_full, bv = enc.stems(
            x_rgb, x_pol=x_pol, x_msi=x_msi, band_ids=band_ids, x_bio=x_bio,
            x_ir=x_ir, phys_slot_ids=phys_slot_ids, avail_msi=avail_msi,
            avail_phys=avail_phys, bio_valid=bio_valid, return_bio_valid=True,
        )
        bio = enc.bio_pyramid(x_bio_full, h, w)

        # P = active_paths ∩ 실제로 stem 을 만들 수 있었던 path.
        # ★ present 의 `.any()` 를 보지 않는다 — GPU 동기화를 유발하고(06 §10 규칙 15)
        #   export 그래프에 데이터 의존 분기를 만든다. 배치 단위 강제 off 는 호출부가
        #   해당 입력 텐서를 None 으로 주는 방식(X-21)으로 표현한다.
        live: List[str] = [m for m in enc.active_paths if stem[m] is not None]

        pres = enc.default_present(x_rgb.shape[0], live, device=x_rgb.device)
        if present is not None:
            pres = pres & present.to(device=x_rgb.device).bool()

        x: Dict[str, Tensor] = {m: stem[m] for m in live}  # type: ignore[misc]
        feats: Dict[str, Optional[List[Tensor]]] = {m: ([] if m in live else None) for m in PATHS}
        fused: List[Tensor] = []
        vac: List[Tensor] = []
        logt: List[Tensor] = []
        weights: List[Tensor] = []

        for i in range(1, 5):
            f_i: Dict[str, Optional[Tensor]] = {m: None for m in PATHS}
            for m in live:
                f_i[m] = enc.run_stage(m, i, x[m], bio[i - 1] if m == "spec" else None)
                feats[m].append(f_i[m])  # type: ignore[union-attr]

            out_i = self._fuse(
                i, f_i, pres, g_pol, prec_ramp=prec_ramp, return_weights=return_weights
            )
            fused.append(out_i.fused)
            vac.append(out_i.vacuity)
            logt.append(out_i.log_tau)
            if return_weights and out_i.weights is not None:
                weights.append(out_i.weights)

            if i < 4:
                # (1) 피드백 가산 — spec/phys 만. rgb 는 pre-fusion 스트림을 그대로 잇는다.
                for m in FEEDBACK_PATHS:
                    if f_i.get(m) is not None and m in out_i.feedback:
                        f_i[m] = f_i[m] + out_i.feedback[m]  # type: ignore[operator]
                # (2) downsample — ★ rgb 포함 **전 path** (정정 A-12)
                for m in live:
                    x[m] = enc.downsample(m, i, f_i[m])  # type: ignore[arg-type]

        return BackboneOut(
            fused=fused,
            vacuity=vac,
            present=pres,
            g_pol=g_pol,
            feats=feats,
            weights=weights if (return_weights and weights) else None,
            bio_valid=bv,
            log_tau=logt,
        )

    def extra_repr(self) -> str:  # pragma: no cover - 디버깅 편의
        return (
            f"bmef_stages={self.bmef_stages}, stage1_identity={self.stage1_identity}, "
            f"active_paths={self.active_paths}"
        )


def _assert_channel_schedule(out: BackboneOut, channels: Sequence[int]) -> None:
    """T18 보조 — 세 path 모두에서 32→64→160→320 이 성립하는지 확인한다."""
    for m, lst in out.feats.items():
        if lst is None:
            continue
        got = tuple(int(t.shape[1]) for t in lst)
        if got != tuple(int(c) for c in channels):
            raise AssertionError(f"채널 스케줄 위반: path={m} {got} != {tuple(channels)}")
    for i, t in enumerate(out.fused):
        if int(t.shape[1]) != int(channels[i]):
            raise AssertionError(f"fused[{i + 1}] 채널 {t.shape[1]} != {channels[i]}")


