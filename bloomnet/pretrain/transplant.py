"""S0-Spec → BloomNet 가중치 이식 (T0~T7) — 01 §6.7, 06 §3.6 (레벨 L6).

이식 규칙 요약 (정정 A-3 / A-9 / A-11 반영):

==== ================================================== ==========================
T0   방사 스케일 전제조건 ``|Δlog10 median| < 0.5``      실패 → **raw 밴드 열 제외**
T1   ``SpecMLP.proj`` → ``SPS.body.proj``                직접 복사 (16,8,1,1)
T1′  ``SpecMLP.c_abs`` → ``SPS.body.c_abs``              직접 복사 (6,16), T0 선행
T2   ``SpecMLP.n1`` → ``SPS.body.norm1``                 affine 직접 복사
T3   ``SpecMLP.dw`` → ``SPS.body.dw``                    **★ 이식 금지**
T4   ``SpecMLP.mix`` → ``SPS.body.patch_embed[0]``       25탭 균등 주입 + BN 흡수
T4′  ``SPS.body.norm2.weight = 0``                       T4 동반 필수 (항등 출발)
T5   ``SpecMLP.head.bias`` → ``ChlHead.out.bias``        bias 만, weight = 0
T6   ``bio_kind`` 불일치 → proj 의 bio 열(6,7) 제외      재초기화 유지
T7   **게이트 실패 → T1~T5 를 전부 수행하지 않는다**     텐서 무변경
==== ================================================== ==========================

T3 이 금지인 이유(정량): 235 칩은 3×3 이라 3×3 DW conv 출력 9픽셀 중 완전한 이웃을 갖는 것은
중앙 1개뿐이다(8/9 = 88.9 % 가 zero-pad 오염). 게다가 235 GSD 0.2788 m/px 대 M3M 0.0369 m/px 로
**6.0~7.6배** 차이라 공간 커널의 물리적 의미가 전혀 다르다.

**원자성**: 모든 shape 검증을 먼저 끝내고(preflight) 통과했을 때만 텐서를 만진다. 검증 실패 시
``strict=True`` 는 ``ValueError`` 를 내며 **모델은 한 바이트도 바뀌지 않는다**. ``strict=False``
는 해당 항목만 ``"skipped"`` 로 남기고 나머지를 진행한다.

레벨 L6 — L−1(`constants`), L2(`modules.stems`, `modules.heads`, `pretrain.spec_mlp`) 만 import 한다.
``models/bloomnet.py`` 는 타입 힌트용으로만(TYPE_CHECKING) 참조한다 — 런타임은 duck typing 이라
모델 조립 방식이 바뀌어도 이 파일은 깨지지 않는다.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from bloomnet.modules.heads import ChlHead
from bloomnet.modules.stems import SPS
from bloomnet.pretrain.spec_mlp import SpecMLP

if TYPE_CHECKING:  # pragma: no cover - 순환/미존재 모듈 방어
    from bloomnet.models.bloomnet import BloomNet

__all__ = [
    "T0_TOL",
    "STATUSES",
    "check_t0",
    "find_sps",
    "find_chl_head",
    "transplant_to_bloomnet",
]

log = logging.getLogger(__name__)

#: T0 판정 허용 오차 — ``|log10(median(msi)) − log10(median(k235))| < 0.5`` (01 §6.7 T0).
T0_TOL: float = 0.5

#: 반환 dict 의 값 어휘. 앞 4개는 06 §3.6 동결 어휘이고, 나머지는 정정 A-3/A-9 가
#: 만든 **부분 이식**을 표현하기 위한 확장분이다 (동결 어휘로는 표현 자체가 불가능하다).
STATUSES: Tuple[str, ...] = (
    "copied",  # 전량 복사
    "avg_tap",  # T4 25탭 균등 주입
    "bias_only",  # T5
    "skipped",  # 규칙상 금지(T3) 또는 전제조건 미충족
    "copied_raw_only",  # T6 불일치 → bio 열 제외
    "copied_bio_only",  # T0 실패 → raw 밴드 열 제외
    "bn_absorbed",  # T4 의 mix.bias → patch_embed BN running_mean
    "zeroed",  # T4′ / T5 의 weight = 0
)


# ─────────────────────────────────────────────────────────────────────────────
# T0 — 방사 스케일 전제조건
# ─────────────────────────────────────────────────────────────────────────────
def check_t0(msi_median: float, k235_median: float, *, tol: float = T0_TOL) -> bool:
    """01 §6.7 T0: 두 데이터셋의 반사율 스케일이 같은 자릿수인가.

    ``|log10(median(msi_train[msi_train>0])) − log10(median(k235.spectra))| < tol``

    Args:
        msi_median: 학습 msi 의 양수 픽셀 중앙값 (R4′ ``k_sensor`` 적용 **후**).
        k235_median: 235 칩 분광 중앙값 (절대 반사율, ``k_sensor = 1.0``).
        tol: 기본 0.5 (약 3.2배).

    Returns:
        통과 여부. **게이트 G1~G3 와 독립**이다 — G 는 성능 게이트이지 스케일 게이트가 아니다.

    Note:
        정정 A-2 시점의 미보정 값(``rho_rel`` ≈ 1e-5 vs 235 ≈ 1e-2)은
        ``|Δlog10| ≈ 3.5`` 로 **명백히 불충족**이었다. 이것이 R4′(``k_sensor``)를 신설하고
        T0 를 게이트와 분리한 이유다.
    """
    if not (msi_median > 0.0 and k235_median > 0.0):
        raise ValueError(
            f"T0 는 양수 중앙값을 요구한다: msi={msi_median!r}, k235={k235_median!r}"
        )
    return abs(math.log10(float(msi_median)) - math.log10(float(k235_median))) < float(tol)


# ─────────────────────────────────────────────────────────────────────────────
# 대상 모듈 탐색 (attribute 이름이 아니라 **타입**으로 찾는다)
# ─────────────────────────────────────────────────────────────────────────────
def _find_unique(model: nn.Module, cls: type, what: str) -> nn.Module:
    hits = [(n, mod) for n, mod in model.named_modules() if isinstance(mod, cls)]
    if not hits:
        raise ValueError(
            f"transplant: 모델에서 {what} 를 찾지 못했다 ({cls.__name__} 인스턴스 0개). "
            f"명시적으로 넘겨라 (예: transplant_to_bloomnet(..., {what}=model.encoder.sps))"
        )
    if len(hits) > 1:
        raise ValueError(
            f"transplant: {what} 후보가 {len(hits)}개다 {[n for n, _ in hits]} — "
            "어느 것에 이식할지 모호하다. 명시적으로 넘겨라."
        )
    return hits[0][1]


def find_sps(model: nn.Module) -> SPS:
    """모델 안의 유일한 :class:`~bloomnet.modules.stems.SPS` 를 찾는다."""
    return _find_unique(model, SPS, "sps")  # type: ignore[return-value]


def find_chl_head(model: nn.Module) -> ChlHead:
    """모델 안의 유일한 :class:`~bloomnet.modules.heads.ChlHead` 를 찾는다."""
    return _find_unique(model, ChlHead, "chl_head")  # type: ignore[return-value]


def _qual_name(model: nn.Module, target: nn.Module, leaf: str) -> str:
    for name, mod in model.named_modules():
        if mod is target:
            return f"{name}.{leaf}" if name else leaf
    return leaf


# ─────────────────────────────────────────────────────────────────────────────
# preflight
# ─────────────────────────────────────────────────────────────────────────────
def _preflight(
    src: SpecMLP, sps: nn.Module, chl_head: nn.Module
) -> Tuple[Dict[str, str], Any]:
    """shape 계약을 전부 검사한다. ``(문제 dict, sps.body)`` 를 돌려준다 — **무변경**."""
    problems: Dict[str, str] = {}
    body = getattr(sps, "body", None)
    if body is None:
        raise ValueError("transplant: SPS 에 `body`(_SlotStem) 가 없다 — 구조가 바뀌었다")

    # T1 / T1′
    sw, dw_ = src.proj.weight, body.proj.weight
    if tuple(sw.shape) != tuple(dw_.shape):
        problems["proj"] = f"T1 shape 불일치: src {tuple(sw.shape)} vs dst {tuple(dw_.shape)}"
    elif tuple(sw.shape) != (16, 8, 1, 1):
        problems["proj"] = f"T1 canonical 위반: {tuple(sw.shape)} != (16, 8, 1, 1) (정정 A-9)"
    if tuple(src.c_abs.shape) != tuple(body.c_abs.shape):
        problems["c_abs"] = (
            f"T1' shape 불일치: src {tuple(src.c_abs.shape)} vs dst {tuple(body.c_abs.shape)}"
        )

    # T2
    if getattr(body.norm1, "weight", None) is None:
        problems["norm1"] = "T2: dst norm1 에 affine 파라미터가 없다"
    elif tuple(src.n1.weight.shape) != tuple(body.norm1.weight.shape):
        problems["norm1"] = (
            f"T2 shape 불일치: src {tuple(src.n1.weight.shape)} "
            f"vs dst {tuple(body.norm1.weight.shape)}"
        )

    # T4
    pe = body.patch_embed
    conv = pe[0] if isinstance(pe, nn.Sequential) else getattr(pe, "conv", None)
    if not isinstance(conv, nn.Conv2d):
        problems["patch_embed"] = "T4: patch_embed 의 Conv2d 를 찾지 못했다"
    else:
        if conv.bias is not None:
            problems["patch_embed"] = (
                "T4: patch_embed.bias 가 None 이 아니다 — 02 §2.3 Step5 는 bias=False 로 "
                "동결되어 있다(정정 A-3). 예산표(SPS 13,312)와도 어긋난다."
            )
        elif tuple(conv.weight.shape[:2]) != tuple(src.mix.weight.shape[:2]):
            problems["patch_embed"] = (
                f"T4 채널 불일치: src mix {tuple(src.mix.weight.shape)} "
                f"vs dst {tuple(conv.weight.shape)}"
            )

    # T4′
    if getattr(body.norm2, "weight", None) is None:
        problems["norm2"] = "T4': dst norm2(dw 잔차 분기) 에 affine weight 가 없다"

    # T5
    out = getattr(chl_head, "out", None)
    if not isinstance(out, nn.Conv2d):
        problems["chl_head"] = "T5: ChlHead.out (Conv2d) 를 찾지 못했다"
    elif out.bias is None:
        problems["chl_head"] = "T5: ChlHead.out.bias 가 None 이다"
    elif int(out.weight.shape[1]) != 64:
        # 초판의 in=128 은 오기다. 04 §8.2 / 06 §3.3.8 동결 구조는 RegTrunk(128->64) -> 1x1(64->1).
        problems["chl_head"] = (
            f"T5 in_ch 계약 위반: {int(out.weight.shape[1])} != 64 (정정 A-3)"
        )
    return problems, body


# ─────────────────────────────────────────────────────────────────────────────
# 본체
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def transplant_to_bloomnet(
    src: SpecMLP,
    model: "BloomNet",
    *,
    do_patch_embed: bool = True,
    bio_kind_src: str = "mci",
    bio_kind_dst: str = "mci",
    # ── 06 동결표에 없는 keyword-only 추가분 (정정 A-3/A-19/A-40, 01 T0/T7) ──
    raw_bands: bool = True,
    gate_ok: Optional[bool] = None,
    strict: bool = True,
    sps: Optional[nn.Module] = None,
    chl_head: Optional[nn.Module] = None,
) -> Dict[str, str]:
    """S0-Spec 가중치를 BloomNet 에 이식한다 (01 §6.7).

    Args:
        src: 학습이 끝난 :class:`~bloomnet.pretrain.spec_mlp.SpecMLP`.
        model: 대상 BloomNet (혹은 SPS/ChlHead 를 품은 임의의 ``nn.Module``).
        do_patch_embed: T4(+T4′) 수행 여부. ``spec.transplant.patch_embed`` 키에 대응한다.
            **끄면 T4′ 도 함께 꺼진다** — 둘은 짝이다(정정 A-3).
        bio_kind_src / bio_kind_dst: S0 학습 시점과 대상의 bio 지수 종류. 다르면 T6 이
            발동해 ``proj`` 의 bio 열(index 6,7)을 이식하지 않는다. 판정 근거는 ``kind``
            이지만 실제 금지 조건은 ``source == "rgb_proxy"`` 다(정정 A-11 / 06 V12) —
            호출자가 ``bio_kind_src="rgb_proxy"`` 로 넘겨 표현한다.
        raw_bands: **T0 결과**. False 면 canonical slot 0~5 열과 ``C_abs`` 를 이식하지 않는다.
            ``spec.transplant.raw_bands`` 키 / 06 V21 과 대응하며, manifest 에
            ``t0_failed: true`` 를 남기는 것은 호출자 책임이다.
        gate_ok: :func:`~bloomnet.pretrain.loso.transplant_gate` 의 첫 반환값.
            **False 면 T1~T5 를 전부 건너뛰고 모델을 전혀 건드리지 않는다** (T7).
            None 이면 "호출자가 이미 판정했다" 로 보고 진행한다.
        strict: shape 계약 위반 시 ``ValueError``(무변경). False 면 해당 항목만 skip.
        sps / chl_head: 자동 탐색 대신 명시 지정.

    Returns:
        ``{param_name: status}``. ``param_name`` 은 **대상 모델 기준의 정규화된 경로**이며
        ``status`` 는 :data:`STATUSES` 중 하나다. 전부 로그로도 출력한다.

    Raises:
        ValueError: ``strict=True`` 에서 shape 계약 위반, 또는 대상 모듈 탐색 실패.

    Note:
        T4 의 BN 흡수(``running_mean -= mix.bias``)는 **eval 모드에서만** 정확하다 —
        학습 중 BN 은 배치 통계를 쓰므로 이식된 bias 항은 첫 forward 부터 배치 평균에
        흡수되어 사라진다. 01 §6.7 의 "수학적 등가 흡수" 는 이식 직후 추론(및 BN 이
        얼기 전 초기 스텝)에 대한 진술이다.
    """
    if not isinstance(src, SpecMLP):
        raise TypeError(f"src must be SpecMLP, got {type(src).__name__}")

    if gate_ok is False:
        # T7 — 게이트 실패. 어떤 텐서도 만지지 않는다. 01 §6.9 "실패 사실을 그대로 기록".
        log.warning(
            "transplant: 이식 게이트 실패 → T1~T5 전부 미수행. "
            "SPS/ChlHead 는 랜덤 초기화로 진행한다 (01 §6.9)."
        )
        return {k: "skipped" for k in _ALL_TARGET_KEYS}

    dst_sps = sps if sps is not None else find_sps(model)
    dst_head = chl_head if chl_head is not None else find_chl_head(model)

    problems, body = _preflight(src, dst_sps, dst_head)
    if problems and strict:
        raise ValueError(
            "transplant preflight 실패 — 모델은 변경되지 않았다:\n  "
            + "\n  ".join(f"[{k}] {v}" for k, v in sorted(problems.items()))
        )
    for k, v in sorted(problems.items()):
        log.warning("transplant: %s 를 건너뛴다 — %s", k, v)

    n = f"{_qual_name(model, dst_sps, 'body')}"
    hn = _qual_name(model, dst_head, "out")
    rep: Dict[str, str] = {}

    bio_ok = str(bio_kind_src) == str(bio_kind_dst)
    if not bio_ok:
        log.warning(
            "transplant T6: bio_kind 불일치 (src=%r, dst=%r) → proj 의 bio 열 6,7 은 "
            "이식하지 않고 대상의 랜덤 초기화를 유지한다 (01 §6.7 T6).",
            bio_kind_src,
            bio_kind_dst,
        )
    if not raw_bands:
        log.warning(
            "transplant T0 실패: raw 밴드 열(canonical slot 0~5)과 C_abs 를 이식하지 않는다. "
            "manifest 에 t0_failed: true 를 남겨라 (정정 A-3 / 06 V21)."
        )

    # ── T1 / T1' ────────────────────────────────────────────────────────────
    b = int(src.bio_start)  # = num_slots = 6
    if "proj" in problems:
        rep[f"{n}.proj.weight"] = "skipped"
        rep[f"{n}.proj.bias"] = "skipped"
    elif raw_bands and bio_ok:
        body.proj.weight.copy_(src.proj.weight)
        rep[f"{n}.proj.weight"] = "copied"
        if body.proj.bias is not None and src.proj.bias is not None:
            body.proj.bias.copy_(src.proj.bias)
            rep[f"{n}.proj.bias"] = "copied"
        else:
            rep[f"{n}.proj.bias"] = "skipped"
    elif raw_bands and not bio_ok:
        body.proj.weight[:, :b].copy_(src.proj.weight[:, :b])
        rep[f"{n}.proj.weight"] = "copied_raw_only"
        # bias 는 출력 채널 단위라 열 부분집합만 이식하면 의미가 정의되지 않는다.
        rep[f"{n}.proj.bias"] = "skipped"
    elif (not raw_bands) and bio_ok:
        body.proj.weight[:, b:].copy_(src.proj.weight[:, b:])
        rep[f"{n}.proj.weight"] = "copied_bio_only"
        rep[f"{n}.proj.bias"] = "skipped"
    else:
        rep[f"{n}.proj.weight"] = "skipped"
        rep[f"{n}.proj.bias"] = "skipped"

    if "c_abs" in problems or not raw_bands:
        rep[f"{n}.c_abs"] = "skipped"  # C_abs 는 raw 밴드 스케일에 종속 (T1' 선행조건)
    else:
        body.c_abs.copy_(src.c_abs)
        rep[f"{n}.c_abs"] = "copied"

    # ── T2 ──────────────────────────────────────────────────────────────────
    if "norm1" in problems:
        rep[f"{n}.norm1.weight"] = "skipped"
        rep[f"{n}.norm1.bias"] = "skipped"
    else:
        body.norm1.weight.copy_(src.n1.weight)
        body.norm1.bias.copy_(src.n1.bias)
        rep[f"{n}.norm1.weight"] = "copied"
        rep[f"{n}.norm1.bias"] = "copied"

    # ── T3 (이식 금지) ──────────────────────────────────────────────────────
    rep[f"{n}.dw.weight"] = "skipped"

    # ── T4 / T4' ────────────────────────────────────────────────────────────
    if do_patch_embed and "patch_embed" not in problems:
        pe = body.patch_embed
        conv = pe[0] if isinstance(pe, nn.Sequential) else pe.conv
        kh, kw = int(conv.weight.shape[2]), int(conv.weight.shape[3])
        taps = float(kh * kw)
        conv.weight.copy_(
            src.mix.weight[:, :, 0:1, 0:1].expand(-1, -1, kh, kw) / taps
        )
        rep[f"{n}.patch_embed.0.weight"] = "avg_tap"

        norm = pe[1] if isinstance(pe, nn.Sequential) and len(pe) > 1 else getattr(pe, "bn", None)
        rm = getattr(norm, "running_mean", None)
        if rm is not None and src.mix.bias is not None:
            rm -= src.mix.bias.to(rm.dtype)
            rep[f"{n}.patch_embed.1.running_mean"] = "bn_absorbed"
        else:
            # GroupNorm 등 running_mean 이 없는 norm 이면 흡수할 곳이 없다.
            log.warning(
                "transplant T4: patch_embed 뒤 norm 에 running_mean 이 없어 mix.bias 를 "
                "흡수할 수 없다 (norm=%s). bias 항은 소실된다.",
                type(norm).__name__,
            )
            rep[f"{n}.patch_embed.1.running_mean"] = "skipped"

        if "norm2" in problems:
            rep[f"{n}.norm2.weight"] = "skipped"
        else:
            body.norm2.weight.zero_()  # T4′ — dw 잔차 분기를 항등에서 출발시킨다
            rep[f"{n}.norm2.weight"] = "zeroed"
    else:
        rep[f"{n}.patch_embed.0.weight"] = "skipped"
        rep[f"{n}.patch_embed.1.running_mean"] = "skipped"
        rep[f"{n}.norm2.weight"] = "skipped"

    # ── T5 ──────────────────────────────────────────────────────────────────
    if "chl_head" in problems:
        rep[f"{hn}.weight"] = "skipped"
        rep[f"{hn}.bias"] = "skipped"
    else:
        dst_head.out.weight.zero_()
        dst_head.out.bias.fill_(float(src.head.bias[0]))
        rep[f"{hn}.weight"] = "zeroed"
        rep[f"{hn}.bias"] = "bias_only"

    for k in sorted(rep):
        log.info("transplant %-44s %s", k, rep[k])
    return rep


#: ``gate_ok=False`` 조기 반환용 논리 키 (실제 경로는 모델 조립에 따라 달라진다).
_ALL_TARGET_KEYS: Tuple[str, ...] = (
    "sps.proj.weight",
    "sps.proj.bias",
    "sps.c_abs",
    "sps.norm1.weight",
    "sps.norm1.bias",
    "sps.dw.weight",
    "sps.patch_embed.0.weight",
    "sps.patch_embed.1.running_mean",
    "sps.norm2.weight",
    "chl_head.out.weight",
    "chl_head.out.bias",
)


def summarize(report: Dict[str, str]) -> str:
    """이식 리포트를 사람이 읽는 한 덩어리로. manifest/SAT 보고서에 그대로 붙인다."""
    lines: List[str] = ["S0-Spec -> BloomNet 이식 결과 (01 §6.7)"]
    for k in sorted(report):
        lines.append(f"  {k:<48s} {report[k]}")
    n_copied = sum(1 for v in report.values() if v != "skipped")
    lines.append(f"  변경된 텐서 {n_copied} / {len(report)}")
    return "\n".join(lines)
