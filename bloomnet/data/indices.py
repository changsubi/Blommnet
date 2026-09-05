"""분광/녹색도 지수와 M3M 방사 보정 — 01 §2.6, §4 / 06 §3.2.2 (★X-01/X-02).

동결 사항
    * ``x_bio = [NDCI, MCI_norm]`` (X-01 / X-27). 02 초판의 ``[NDCI, GNDVI]`` 는 폐기됐다.
      GNDVI 는 ``kind="gndvi"`` ablation 대조군(a4_bio_gndvi.yaml)으로만 남는다.
    * MCI 는 681 nm 형광 밴드를 요구하지 **않는다**. baseline-subtracted red-edge peak height
      ``MCI = RE1 − R − c·(NIR − R)``, ``c = (λ_RE1 − λ_R)/(λ_NIR − λ_R)`` 형태이며
      M3M 4밴드로 계산 가능하다 (정정 A/B: 02 의 "681 nm 필수" 기각 논거는 사실 오류).
    * msi 는 R1→R4 뒤 **R4′(`k_sensor`)** 까지 적용해야 의미가 있다 (정정 A-2 / X-28).
      로더 진입부 하드 검증은 :func:`assert_msi_scale_contract`.

레벨 L1. `bloomnet.constants` (L−1 예외) 외에는 `bloomnet.*` 를 import 하지 않는다.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from bloomnet.constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    K_SENSOR,
    MSI_MEDIAN_RANGE,
    MSI_SLOT_ID,
)

__all__ = [
    "mci_coefficient",
    "canonical_scatter_np",
    "compute_bio_canonical",
    "compute_rgb_proxy",
    "specular_proxy",
    "normalize_imagenet",
    "denormalize_imagenet",
    "m3m_dn_to_relative_reflectance",
    "apply_k_sensor",
    "assert_msi_scale_contract",
    "coregister_m3m",
]

_SLOT_GREEN = MSI_SLOT_ID["green"]
_SLOT_RED = MSI_SLOT_ID["red"]
_SLOT_RE1 = MSI_SLOT_ID["rededge1"]
_SLOT_NIR = MSI_SLOT_ID["nir"]

_DN_MAX: int = 65535


# ─────────────────────────────────────────────────────────────────────────────
# 지수 계수
# ─────────────────────────────────────────────────────────────────────────────
def mci_coefficient(lam_red: float, lam_rededge1: float, lam_nir: float) -> float:
    """MCI baseline 계수 ``c = (λ_RE1 − λ_R) / (λ_NIR − λ_R)`` (01 §4.2.2).

    m3m (730−650)/(860−650) = 0.380952 / rededge_mx (717−668)/(840−668) = 0.284884 /
    rededge_p (705−665)/(783−665) = 0.338983 (정정 A-27: 0.475410 은 오기).
    """
    denom = float(lam_nir) - float(lam_red)
    if denom == 0.0:
        raise ValueError("lam_nir must differ from lam_red")
    return (float(lam_rededge1) - float(lam_red)) / denom


# ─────────────────────────────────────────────────────────────────────────────
# canonical scatter (numpy 판, 학습 전처리 전용)
# ─────────────────────────────────────────────────────────────────────────────
def canonical_scatter_np(
    x_raw: np.ndarray, band_ids: Sequence[int], num_slots: int
) -> Tuple[np.ndarray, np.ndarray]:
    """센서 밴드를 canonical slot 으로 흩뿌린다 (02 §2.1 의 CPU/numpy 판).

    Args:
        x_raw: ``(K, H, W)`` 센서 원 밴드 (밴드 정합 R6 + 일사 정규화 R4/R4′ 완료).
        band_ids: 길이 K. ``band_ids[k]`` = 밴드 k 가 차지하는 canonical slot 인덱스.
        num_slots: canonical slot 수 (msi = 6, phys = 4).

    Returns:
        ``(S (num_slots,H,W) float32, m (num_slots,) float32)``. 결측 slot 은 0, ``m`` 은 0.
    """
    if x_raw.ndim != 3:
        raise ValueError(f"x_raw must be (K,H,W), got {x_raw.shape}")
    if len(band_ids) != x_raw.shape[0]:
        raise ValueError(f"len(band_ids)={len(band_ids)} != K={x_raw.shape[0]}")
    ids = [int(i) for i in band_ids]
    if len(set(ids)) != len(ids):
        raise ValueError(f"band_ids must be unique, got {ids}")
    if any(i < 0 or i >= num_slots for i in ids):
        raise ValueError(f"band_ids out of range [0,{num_slots}): {ids}")

    s = np.zeros((num_slots,) + x_raw.shape[1:], dtype=np.float32)
    m = np.zeros((num_slots,), dtype=np.float32)
    for k, slot in enumerate(ids):
        s[slot] = x_raw[k].astype(np.float32, copy=False)
        m[slot] = 1.0
    return s, m


# ─────────────────────────────────────────────────────────────────────────────
# x_bio
# ─────────────────────────────────────────────────────────────────────────────
def compute_bio_canonical(
    S: Tensor,
    m: Tensor,
    *,
    mci_c: float,
    kind: str = "mci",
    eps: float = 1e-6,
) -> Tensor:
    """canonical slot 텐서에서 ``x_bio`` 2채널을 만든다 (01 §4.2.2, ★X-01/X-02 단일 구현).

    Args:
        S: ``(B, 6, H, W)`` canonical scatter 결과 (R4/R4′ + R6 완료).
        m: ``(B, 6)`` slot presence (0/1 float).
        mci_c: :func:`mci_coefficient` 값. 센서별 상수.
        kind: ``"mci"`` (정본) 또는 ``"gndvi"`` (ablation 전용).
        eps: 분모 안정화 항.

    Returns:
        ``(B, 2, H, W)`` = ``clamp([ch0, ch1], -1, 1)``.
        필수 slot 이 결측인 **샘플**의 해당 채널은 정확히 0 이다.
    """
    if S.dim() != 4 or S.shape[1] != len(MSI_SLOT_ID):
        raise ValueError(f"S must be (B,{len(MSI_SLOT_ID)},H,W), got {tuple(S.shape)}")
    if m.dim() != 2 or m.shape[0] != S.shape[0] or m.shape[1] != S.shape[1]:
        raise ValueError(f"m must be (B,{S.shape[1]}), got {tuple(m.shape)}")
    if kind not in ("mci", "gndvi"):
        raise ValueError(f"kind must be 'mci' or 'gndvi', got {kind!r}")

    mf = m.to(S.dtype)
    g = S[:, _SLOT_GREEN : _SLOT_GREEN + 1]
    r = S[:, _SLOT_RED : _SLOT_RED + 1]
    re1 = S[:, _SLOT_RE1 : _SLOT_RE1 + 1]
    nir = S[:, _SLOT_NIR : _SLOT_NIR + 1]

    def _gate(*slots: int) -> Tensor:
        out = mf[:, slots[0]]
        for s in slots[1:]:
            out = out * mf[:, s]
        return out.view(-1, 1, 1, 1)

    # ch0: NDCI (Mishra & Mishra 2012). 절대 캘리브레이션(R5)에 불변.
    ndci = (re1 - r) / (re1 + r + eps)
    ndci = ndci * _gate(_SLOT_RED, _SLOT_RE1)

    if kind == "mci":
        # ch1: MCI_norm — 정규화형이라 스케일 불변 (01 §4.2.3 근거 3).
        mci = re1 - r - float(mci_c) * (nir - r)
        ch1 = mci / (r + re1 + nir + eps)
        ch1 = ch1 * _gate(_SLOT_RED, _SLOT_RE1, _SLOT_NIR)
    else:
        ch1 = (nir - g) / (nir + g + eps)
        ch1 = ch1 * _gate(_SLOT_GREEN, _SLOT_NIR)

    return torch.clamp(torch.cat([ndci, ch1], dim=1), -1.0, 1.0)


def compute_rgb_proxy(rgb01: Tensor, *, eps: float = 1e-8) -> Tensor:
    """RGB 3밴드 **녹색도 지수** ``[ExG, clamp(NGRDI,-1,1)]`` (01 §4.1).

    ★ NDCI 의 대체물이 **아니다**. 8-site pooled 상관은 ExG +0.098 / NGRDI −0.219 로
    부호조차 일정하지 않다. 문서·보고서에서는 반드시 "녹색도 지수" 라고만 부른다.

    Args:
        rgb01: ``(B,3,H,W)`` ∈ [0,1] (ImageNet 정규화 **이전** 값).

    Returns:
        ``(B,2,H,W)``. ch0 = ExG ∈ [−1,2], ch1 = clamp(NGRDI, −1, 1).
    """
    if rgb01.dim() != 4 or rgb01.shape[1] != 3:
        raise ValueError(f"rgb01 must be (B,3,H,W), got {tuple(rgb01.shape)}")
    r = rgb01[:, 0:1]
    g = rgb01[:, 1:2]
    b = rgb01[:, 2:3]
    tot = r + g + b + eps
    exg = (2.0 * g - r - b) / tot
    ngrdi = torch.clamp((g - r) / (g + r + eps), -1.0, 1.0)
    return torch.cat([exg, ngrdi], dim=1)


def specular_proxy(rgb01: Tensor, *, eps: float = 1e-6) -> Tensor:
    """이색성 반사 모형의 무채색 성분 대리값 ``Vmax·(1 − Sat)`` ∈ [0,1] (01 §5.2.1).

    ★ 물리적 glint 가 아니다. 기본 경로는 PPN 우회이며 이 값은 진단 오버레이 용도가 우선이다.
    """
    if rgb01.dim() != 4 or rgb01.shape[1] != 3:
        raise ValueError(f"rgb01 must be (B,3,H,W), got {tuple(rgb01.shape)}")
    vmax = rgb01.amax(dim=1, keepdim=True)
    vmin = rgb01.amin(dim=1, keepdim=True)
    sat = (vmax - vmin) / (vmax + eps)
    return vmax * (1.0 - sat)


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet 정규화 (aihub092 와 bit-exact 여야 한다 — 헌법 C-4)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_imagenet(rgb01: Tensor) -> Tensor:
    """``(x − MEAN) / STD``. ``rgb01`` 은 ``(...,3,H,W)`` ∈ [0,1]."""
    mean = IMAGENET_MEAN.to(device=rgb01.device, dtype=rgb01.dtype)
    std = IMAGENET_STD.to(device=rgb01.device, dtype=rgb01.dtype)
    return (rgb01 - mean) / std


def denormalize_imagenet(rgb: Tensor) -> Tensor:
    """``clamp(x*STD + MEAN, 0, 1)``. 비정규화 RGB 를 따로 저장하지 않기 위한 복원 (01 §2.1.4)."""
    mean = IMAGENET_MEAN.to(device=rgb.device, dtype=rgb.dtype)
    std = IMAGENET_STD.to(device=rgb.device, dtype=rgb.dtype)
    return torch.clamp(rgb * std + mean, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# M3M DN → 상대 반사율 (01 §2.6 R1~R4) + R4′ + [H1]
# ─────────────────────────────────────────────────────────────────────────────
def m3m_dn_to_relative_reflectance(
    dn: np.ndarray,
    *,
    black_level: int,
    vignetting_coeffs: Sequence[float],
    optical_center: Tuple[float, float],
    exposure_time: float,
    sensor_gain: float,
    sensor_gain_adj: float,
    irradiance: float,
) -> np.ndarray:
    """M3M uint16 DN → 상대 반사율 ``rho_rel`` (R1 → R2 → R3 → R4).

    R5(절대 반사율, ``k_panel``)는 [U-3] 이므로 여기서 적용하지 않는다.
    **R4′(``k_sensor``)도 여기서 적용하지 않는다** — :func:`apply_k_sensor` 로 분리해
    "상대" 와 "센서 정합" 의 책임을 나눈다 (정정 A-2).

    Args:
        dn: ``(H, W)`` 또는 ``(K, H, W)`` raw DN.
        black_level: XMP ``BlackLevel`` (M3M = 3200).
        vignetting_coeffs: ``VignettingData`` 6계수 ``c_1..c_6``.
        optical_center: ``(cx, cy)`` **픽셀 좌표** (col, row). r 은 픽셀 단위 반경.
        exposure_time / sensor_gain / sensor_gain_adj: XMP 노출·게인.
        irradiance: 밴드별 DLS ``Irradiance``.

    Returns:
        ``dn`` 과 같은 shape 의 float64 ``rho_rel``. 실측 자릿수는 ``O(1e−5)`` 다 [M13].
    """
    arr = np.asarray(dn, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError(f"dn must be (H,W) or (K,H,W), got {arr.shape}")
    if len(vignetting_coeffs) != 6:
        raise ValueError(f"vignetting_coeffs must have 6 entries, got {len(vignetting_coeffs)}")
    denom_gain = float(exposure_time) * float(sensor_gain) * float(sensor_gain_adj)
    if denom_gain == 0.0:
        raise ValueError("exposure_time * sensor_gain * sensor_gain_adj must be non-zero")
    if float(irradiance) == 0.0:
        raise ValueError("irradiance must be non-zero (R4 는 생략 불가)")

    # R1 dark + range
    x = np.clip((arr - float(black_level)) / float(_DN_MAX - black_level), 0.0, 1.0)

    # R2 vignetting.  V = 1 + Σ_{k=1..6} c_k r^k
    h, w = arr.shape[-2:]
    cx, cy = float(optical_center[0]), float(optical_center[1])
    uu = np.arange(w, dtype=np.float64)[None, :] - cx
    vv = np.arange(h, dtype=np.float64)[:, None] - cy
    r = np.sqrt(uu * uu + vv * vv)
    v_field = np.ones_like(r)
    rk = np.ones_like(r)
    for c in vignetting_coeffs:
        rk = rk * r
        v_field = v_field + float(c) * rk
    if np.any(v_field == 0.0):
        raise ValueError("vignetting field has zeros; check VignettingData / optical_center")
    x = x / v_field

    # R3 노출/게인
    x = x / denom_gain

    # R4 밴드간 일사 정규화 — 생략하면 지수가 무의미해진다 (01 §2.6 [J])
    return x / float(irradiance)


def apply_k_sensor(
    msi: Union[Tensor, np.ndarray], sensor: str, *, k_sensor: float | None = None
) -> Union[Tensor, np.ndarray]:
    """R4′ 센서 스케일 정합 ``rho = rho_rel · k_sensor[sensor]`` (정정 A-2 / X-28).

    m3m 의 ``3.5e3`` 은 **잠정값** [01 U-11] 이다. ``tools/calibrate_k_sensor.py`` 로
    수면 픽셀 median 기반 재산출 전에는 S1 학습을 시작하지 않는다.
    """
    k = K_SENSOR[sensor] if k_sensor is None else float(k_sensor)
    return msi * k


def assert_msi_scale_contract(msi: Union[Tensor, np.ndarray]) -> float:
    """로더 하드 검증 [H1] — ``1e-3 < median(msi[msi>0]) < 1.0`` (01 §2.6, 정정 A-2).

    "조용한 열화" 를 허용하지 않는 유일한 예외다. 스케일이 틀리면 지수·bio gain·C_abs·
    이식이 전부 무의미해지므로 즉시 죽인다.

    Returns:
        측정된 median. 위반 시 ``RuntimeError``.
    """
    if isinstance(msi, Tensor):
        pos = msi[msi > 0]
        if pos.numel() == 0:
            raise RuntimeError(
                "msi radiometric scale contract violated: no positive pixel. "
                "01 §2.6 R4'/k_sensor"
            )
        med = float(pos.to(torch.float64).median().item())
    else:
        arr = np.asarray(msi, dtype=np.float64)
        pos_np = arr[arr > 0]
        if pos_np.size == 0:
            raise RuntimeError(
                "msi radiometric scale contract violated: no positive pixel. "
                "01 §2.6 R4'/k_sensor"
            )
        med = float(np.median(pos_np))
    lo, hi = MSI_MEDIAN_RANGE
    if not (lo < med < hi):
        raise RuntimeError(
            f"msi radiometric scale contract violated: median={med:.3e}, "
            f"expected ({lo:g}, {hi:g}). 01 §2.6 R4'/k_sensor"
        )
    return med


def coregister_m3m(
    bands: Sequence[np.ndarray], homographies: Sequence[np.ndarray]
) -> np.ndarray:
    """R6 밴드 정합 — 각 밴드를 G 밴드 격자로 warp 한다 (01 §2.6, 지수 계산 전 필수).

    Args:
        bands: 길이 K 의 ``(H, W)`` 배열. ``bands[0]`` 이 기준 격자(G 밴드) 라고 가정한다.
        homographies: 길이 K 의 ``(3,3)``. **목적지(G 격자) → 원본 밴드** 로 가는
            역방향 매핑이다(샘플링에 그대로 쓰인다). 기준 밴드는 단위행렬.

    Returns:
        ``(K, H, W)`` float32. 격자 밖은 0 으로 채운다.

    Note:
        cv2 의존을 만들지 않기 위해 numpy 이중선형 역워프로 구현했다.
    """
    if len(bands) != len(homographies):
        raise ValueError(f"len(bands)={len(bands)} != len(homographies)={len(homographies)}")
    if len(bands) == 0:
        raise ValueError("bands must not be empty")
    ref = np.asarray(bands[0], dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError(f"each band must be (H,W), got {ref.shape}")
    h, w = ref.shape

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij"
    )
    ones = np.ones_like(xx)
    dst = np.stack([xx, yy, ones], axis=0).reshape(3, -1)  # (3, H*W)

    out = np.zeros((len(bands), h, w), dtype=np.float32)
    for k, (band, hom) in enumerate(zip(bands, homographies)):
        src_img = np.asarray(band, dtype=np.float64)
        if src_img.shape != (h, w):
            raise ValueError(f"band {k} shape {src_img.shape} != reference {(h, w)}")
        hm = np.asarray(hom, dtype=np.float64)
        if hm.shape != (3, 3):
            raise ValueError(f"homographies[{k}] must be (3,3), got {hm.shape}")
        p = hm @ dst
        wgt = p[2]
        wgt = np.where(np.abs(wgt) < 1e-12, 1e-12, wgt)
        sx = (p[0] / wgt).reshape(h, w)
        sy = (p[1] / wgt).reshape(h, w)
        out[k] = _bilinear_sample(src_img, sx, sy).astype(np.float32, copy=False)
    return out


def _bilinear_sample(img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """격자 밖 0 채움 이중선형 샘플링 (numpy). ``img (H,W)``, ``sx/sy (H,W)``."""
    h, w = img.shape
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = sx - x0
    fy = sy - y0

    def _gather(yi: np.ndarray, xi: np.ndarray) -> np.ndarray:
        ok = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
        yc = np.clip(yi, 0, h - 1)
        xc = np.clip(xi, 0, w - 1)
        return np.where(ok, img[yc, xc], 0.0)

    v00 = _gather(y0, x0)
    v01 = _gather(y0, x1)
    v10 = _gather(y1, x0)
    v11 = _gather(y1, x1)
    top = v00 * (1.0 - fx) + v01 * fx
    bot = v10 * (1.0 - fx) + v11 * fx
    return top * (1.0 - fy) + bot * fy
