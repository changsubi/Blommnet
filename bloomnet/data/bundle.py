"""ModalityBundle 자료구조 계약 — 01 §2 / 06 §3.2.1.

* :data:`Sample` — ``Dataset.__getitem__`` 반환 스키마 (배치 전 CHW).
* :func:`bloom_collate` — 규칙 C1~C7.
* :func:`assert_availability_contract` — 규칙 A1~A6 (하드).
* :func:`derive_present` — ``avail (B,5)`` → ``present (B,3)``. phys 는 **AND** (정정 A-25).

레벨 L1. `bloomnet.constants` (L−1 예외) 외에는 `bloomnet.*` 를 import 하지 않는다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from bloomnet.constants import (
    MODALITY_CHANNELS,
    MODALITY_ORDER,
    PATHS,
    PHYS_SLOTS,
)

__all__ = [
    "Sample",
    "MODALITY_KEYS",
    "SAMPLE_TARGET_KEYS",
    "SAMPLE_META_KEY",
    "SAMPLE_REQUIRED_META",
    "OPTIONAL_MODALITY_KEYS",
    "PHYS_SLOT_TO_MODALITY_INDEX",
    "bloom_collate",
    "assert_availability_contract",
    "derive_present",
    "phys_modality_indices",
]

# ── 스키마 상수 ──────────────────────────────────────────────────────────────
MODALITY_KEYS: Tuple[str, ...] = tuple(MODALITY_ORDER)          # (rgb, msi, bio, ir, pol)
OPTIONAL_MODALITY_KEYS: Tuple[str, ...] = ("msi", "bio", "ir", "pol")   # A5
SAMPLE_TARGET_KEYS: Tuple[str, ...] = (
    "y_seg",
    "y_edge",
    "y_edge_valid",
    "y_chl",
    "y_chl_valid",
    "y_chl_scalar",
    "y_chl_scalar_valid",
)
SAMPLE_META_KEY: str = "meta"
SAMPLE_REQUIRED_META: Tuple[str, ...] = (
    "stem", "scene", "group_key", "split", "site", "date", "flight_line",
    "alt_m", "gsd_m", "sensor", "band_ids", "phys_slot_ids", "band_centers_nm",
    "bio_kind", "chl_space", "aug",
)

# PHYS canonical slot → avail 인덱스.  slot 0(ir) → modality 'ir'(3),
# slot 1..3(dolp, aolp_sin, aolp_cos) → modality 'pol'(4).   (X-04)
_IR_IDX = MODALITY_ORDER.index("ir")
_POL_IDX = MODALITY_ORDER.index("pol")
PHYS_SLOT_TO_MODALITY_INDEX: Tuple[int, ...] = tuple(
    _IR_IDX if name == "ir" else _POL_IDX for name in PHYS_SLOTS
)

try:  # TypedDict 는 문서화 목적. 런타임 검증은 아래 함수들이 한다.
    from typing import TypedDict

    Sample = TypedDict(
        "Sample",
        {
            "rgb": Tensor,
            "msi": Tensor,
            "bio": Tensor,
            "ir": Tensor,
            "pol": Tensor,
            "avail": Tensor,
            "y_seg": Tensor,
            "y_edge": Tensor,
            "y_edge_valid": Tensor,
            "y_chl": Tensor,
            "y_chl_valid": Tensor,
            "y_chl_scalar": Tensor,
            "y_chl_scalar_valid": Tensor,
            "meta": dict,
        },
        total=False,   # A4: 모드 비활성 modality key 는 아예 없다
    )
except ImportError:  # pragma: no cover
    Sample = dict  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# collate
# ─────────────────────────────────────────────────────────────────────────────
def bloom_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """전용 콜레이트 (01 §2.5 C1~C7). ``default_collate`` 는 meta dict 때문에 쓸 수 없다.

    Raises:
        RuntimeError: C1(공간 크기 불일치) 또는 C2(key 집합 불일치) 위반.
    """
    if not samples:
        raise RuntimeError("bloom_collate: empty sample list")

    # C2 key 집합 동일 강제
    ref_keys = set(samples[0].keys())
    for i, s in enumerate(samples[1:], start=1):
        if set(s.keys()) != ref_keys:
            missing = sorted(ref_keys - set(s.keys()))
            extra = sorted(set(s.keys()) - ref_keys)
            raise RuntimeError(
                f"bloom_collate C2: sample[{i}] key set differs from sample[0]; "
                f"missing={missing} extra={extra}"
            )

    # C1 (H,W) 동일 강제 — 조용한 resize 금지
    ref_hw = _spatial_hw(samples[0])
    if ref_hw is not None:
        for i, s in enumerate(samples[1:], start=1):
            hw = _spatial_hw(s)
            if hw != ref_hw:
                raise RuntimeError(
                    f"bloom_collate C1: sample[{i}] spatial size {hw} != sample[0] {ref_hw} "
                    "(silent resize is forbidden)"
                )

    batch: Dict[str, Any] = {}
    for key in samples[0]:
        if key == SAMPLE_META_KEY:
            batch[key] = [s[key] for s in samples]          # C4
            continue
        vals = [s[key] for s in samples]
        if not isinstance(vals[0], Tensor):
            batch[key] = vals
            continue
        ref = vals[0]
        for i, v in enumerate(vals[1:], start=1):
            if v.shape != ref.shape:
                raise RuntimeError(
                    f"bloom_collate C1: '{key}' shape {tuple(v.shape)} at sample[{i}] "
                    f"!= {tuple(ref.shape)} at sample[0]"
                )
            if v.dtype != ref.dtype:                        # C7 dtype 승격 금지
                raise RuntimeError(
                    f"bloom_collate C7: '{key}' dtype {v.dtype} at sample[{i}] != {ref.dtype}"
                )
        batch[key] = torch.stack(vals, dim=0)               # C3 / C5

    # C6 배치 전원 결측 플래그
    all_missing: Dict[str, bool] = {}
    if "avail" in batch:
        avail = batch["avail"]
        for k, name in enumerate(MODALITY_ORDER):
            if name in batch:
                all_missing[name] = bool((avail[:, k] < 0.5).all().item())
    batch["all_missing"] = all_missing
    return batch


def _spatial_hw(sample: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """샘플의 기준 공간 크기 (H, W). 우선순위 rgb → y_seg → 첫 modality."""
    for key in ("rgb",) + OPTIONAL_MODALITY_KEYS:
        t = sample.get(key)
        if isinstance(t, Tensor) and t.dim() == 3:
            return int(t.shape[-2]), int(t.shape[-1])
    t = sample.get("y_seg")
    if isinstance(t, Tensor) and t.dim() == 2:
        return int(t.shape[-2]), int(t.shape[-1])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# availability 계약
# ─────────────────────────────────────────────────────────────────────────────
def assert_availability_contract(
    sample: Dict[str, Any],
    *,
    bio_source: str = "none",
    active_modalities: Optional[Sequence[str]] = None,
) -> None:
    """01 §2.4 A1~A6 하드 검증. 위반 시 규칙 ID + key 를 담은 ``AssertionError``.

    Args:
        sample: ``Dataset.__getitem__`` 반환 dict (배치 전).
        bio_source: ``cfg.data.bio.source``. ``"rgb_proxy"`` 면 A3 가 완화된다 (정정 A-11).
        active_modalities: 주어지면 A4(모드 비활성 key 부재)를 함께 검사한다.

    Note:
        A7(결측의 stem slot 전파)·A8(phys present AND)은 텐서 하나로 판정할 수 없는
        **모델 측 계약**이므로 :func:`derive_present` 와 ``CanonicalScatter`` 가 소유한다.
        여기서는 A7 의 데이터 측 전제(A1)만 강제한다.
    """
    avail = sample.get("avail")
    assert isinstance(avail, Tensor), "A1: sample['avail'] must be a Tensor"
    assert avail.shape == (len(MODALITY_ORDER),), (
        f"A1: avail must be ({len(MODALITY_ORDER)},), got {tuple(avail.shape)}"
    )
    assert avail.dtype == torch.float32, f"A1: avail dtype must be float32, got {avail.dtype}"

    present_mods = [k for k in MODALITY_ORDER if k in sample]

    # A4 — 모드 비활성 modality key 는 존재하지 않는다
    if active_modalities is not None:
        want = set(active_modalities)
        got = set(present_mods)
        assert got == want, f"A4: modality keys {sorted(got)} != active {sorted(want)}"

    # A2 — rgb 는 항상 가용
    assert "rgb" in sample, "A2: key 'rgb' must always exist"
    assert float(avail[0]) == 1.0, f"A2: avail[0](rgb) must be 1.0, got {float(avail[0])}"

    for k, name in enumerate(MODALITY_ORDER):
        if name not in sample:
            continue
        t = sample[name]
        assert isinstance(t, Tensor), f"A1: sample['{name}'] must be a Tensor"
        cm = MODALITY_CHANNELS[name]
        if cm > 0:
            assert t.shape[0] == cm, (
                f"A1: '{name}' must have {cm} channels, got {int(t.shape[0])}"
            )
        a = float(avail[k])
        assert a in (0.0, 1.0), f"A1: avail[{k}]('{name}') must be 0.0 or 1.0, got {a}"
        is_zero = bool((t == 0).all().item())               # atol = 0 (계약)
        if a == 0.0:
            assert is_zero, f"A1: avail['{name}']==0 but tensor is not exactly all-zero"
            # A5 — 샘플 수준 결측이 허용되는 key
            assert name in OPTIONAL_MODALITY_KEYS, (
                f"A5: '{name}' may not be missing at sample level"
            )
        else:
            assert not is_zero, (
                f"A1: avail['{name}']==1 but tensor is exactly all-zero "
                "(⟺ contract, 01 §2.4 A1)"
            )

    # A3 (정정 A-11 완화) — bio 는 msi 의 부속. rgb_proxy 경로만 예외
    if "bio" in sample and float(avail[2]) == 1.0:
        ok = float(avail[1]) == 1.0 or bio_source == "rgb_proxy"
        assert ok, (
            "A3: avail['bio']==1 requires avail['msi']==1 or bio.source=='rgb_proxy' "
            f"(bio_source={bio_source!r})"
        )
        if bio_source == "rgb_proxy":
            meta = sample.get(SAMPLE_META_KEY, {})
            assert meta.get("bio_kind") == "rgb_proxy", (
                "A3/V17: bio.source=='rgb_proxy' requires meta['bio_kind']=='rgb_proxy'"
            )

    # A6 — 타깃 결측은 0 채움 + *_valid=False, y_seg 는 전부 255
    _assert_target_pair(sample, "y_edge", "y_edge_valid")
    _assert_target_pair(sample, "y_chl", "y_chl_valid")
    _assert_target_pair(sample, "y_chl_scalar", "y_chl_scalar_valid")
    if "y_seg" in sample:
        assert sample["y_seg"].dtype == torch.int64, (
            f"A6: y_seg dtype must be int64, got {sample['y_seg'].dtype}"
        )


def _assert_target_pair(sample: Dict[str, Any], key: str, valid_key: str) -> None:
    if key not in sample or valid_key not in sample:
        return
    y = sample[key]
    v = sample[valid_key]
    assert v.dtype == torch.bool, f"A6: '{valid_key}' dtype must be bool, got {v.dtype}"
    if not bool(v.any().item()):
        assert bool((y == 0).all().item()), (
            f"A6: '{valid_key}' is all False but '{key}' is not zero-filled"
        )


# ─────────────────────────────────────────────────────────────────────────────
# present 유도
# ─────────────────────────────────────────────────────────────────────────────
def phys_modality_indices(phys_slot_ids: Sequence[int]) -> Tuple[int, ...]:
    """phys canonical slot 집합이 **실제로 요구하는** modality 인덱스 집합 (01 §2.4 A8).

    예: ``(0,1,2,3) → (3(ir), 4(pol))``, ``(0,) → (3,)``, ``(1,2,3) → (4,)``.
    """
    idxs = sorted({PHYS_SLOT_TO_MODALITY_INDEX[int(s)] for s in phys_slot_ids})
    return tuple(idxs)


def derive_present(
    avail: Tensor, *, phys_slot_ids: Sequence[int] = tuple(range(len(PHYS_SLOTS)))
) -> Tensor:
    """``avail (B,5) float32`` → ``present (B,3) bool``. 순서 = ``PATHS``.

    ``present[:,0] = avail[rgb] > 0.5``
    ``present[:,1] = avail[msi] > 0.5``            (bio 는 msi 의 부속이라 미반영)
    ``present[:,2] = AND over mode_phys_keys``     ★ (정정 A-25) OR 가 **아니다**

    ``phys_slot_ids`` 는 ``BloomNet.__init__`` 에서 생성 시점 고정(X-24)되는 값이며,
    키워드 전용 인자라 06 §3.2.1 의 위치 시그니처는 그대로 유지된다.
    """
    if avail.dim() != 2 or avail.shape[1] != len(MODALITY_ORDER):
        raise ValueError(f"avail must be (B,{len(MODALITY_ORDER)}), got {tuple(avail.shape)}")
    present = torch.zeros(avail.shape[0], len(PATHS), dtype=torch.bool, device=avail.device)
    present[:, 0] = avail[:, 0] > 0.5
    present[:, 1] = avail[:, 1] > 0.5
    keys = phys_modality_indices(phys_slot_ids)
    if keys:
        acc = avail[:, keys[0]] > 0.5
        for k in keys[1:]:
            acc = acc & (avail[:, k] > 0.5)
        present[:, 2] = acc
    return present
