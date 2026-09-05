"""증강 파이프라인 — 01 §7.4 / 06 §3.2.4.

원칙 (하드):
    * 기하 변환은 **전 모달 + 전 타깃에 동일 파라미터**로 적용한다.
    * 광도 변환은 **RGB 에만**. msi/bio/ir/pol 에는 어떤 광도 변환도 금지 (06 §10 규칙 2).
      근거(정량): ``NDCI=(a−b)/(a+b)`` 에서 밴드 이득 오차 ε → ``ΔNDCI ≈ 0.5ε``.
      실측 NDCI IQR 0.155 이므로 10 % jitter 만으로 IQR 의 32 % 가 이동한다 = 라벨 노이즈 주입.
    * 편광 flip/rot90 시 AoLP 부호 변환은 **필수**다 (06 §10 규칙 10, R-10).
      누락하면 shape·finite 테스트를 모두 통과하는 **조용한 물리 오염 버그**가 된다.

파이프라인 순서 (01 §7.4.3): 로드 → msi R4/R6 → **기하변환** → x_bio 계산 → 광도(rgb) →
정규화 → 타깃 생성 → avail assert. 즉 ``bio`` 와 ``y_edge`` 는 이 클래스가 볼 필요가 없다.

레벨 L1. `bloomnet.constants` (L−1 예외) 외에는 `bloomnet.*` 를 import 하지 않는다.
(``normalize_imagenet`` 은 06 §3.2.2 API 동결표에 따라 ``data/indices.py`` 가 소유한다.
 L1→L1 import 가 금지되어 있으므로 여기서 재정의하지 않는다.)
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import IGNORE_INDEX, MODALITY_ORDER

__all__ = ["JointGeometricTransform", "PhotometricRGB", "transform_aolp"]

# 최근접으로만 리샘플해야 하는 타깃 (라벨 값을 섞으면 안 된다)
_NEAREST_TARGETS: Tuple[str, ...] = ("y_seg", "y_chl", "y_chl_valid")
# 기하변환 시점에 존재하면 안 되는 타깃 (01 §7.4.3 step 7 에서 생성된다)
_POST_GEOMETRIC_TARGETS: Tuple[str, ...] = ("y_edge", "y_edge_valid")
# 공간 축이 없는 타깃 (그대로 통과)
_SCALAR_TARGETS: Tuple[str, ...] = ("y_chl_scalar", "y_chl_scalar_valid")


# ─────────────────────────────────────────────────────────────────────────────
# AoLP
# ─────────────────────────────────────────────────────────────────────────────
def transform_aolp(pol: Tensor, *, hflip: bool, vflip: bool, rot90_k: int) -> Tensor:
    """편광 채널의 기하 변환 부호 보정 (01 §7.4.1).

    Args:
        pol: ``(..., 3, H, W)`` = ``[DoLP, sin2θ, cos2θ]``. 공간축은 이미 변환된 상태여야 한다.
        hflip / vflip: 적용된 flip 여부. 둘의 **홀짝**이 의미를 갖는다
            (flip 1회 → ``θ → −θ``, 2회 → 180° 회전이므로 ``θ → θ``).
        rot90_k: 적용된 rot90 횟수 ``k``. ``θ → θ + 90k°``.

    Returns:
        같은 shape 의 텐서. DoLP 채널은 불변, ``(sin2θ, cos2θ)`` 만 부호가 바뀐다.

    Note:
        ``sin(2θ+180k) = (−1)^k sin2θ``, ``cos(2θ+180k) = (−1)^k cos2θ``.
    """
    if pol.shape[-3] != 3:
        raise ValueError(f"pol must be (...,3,H,W)=[DoLP,sin2θ,cos2θ], got {tuple(pol.shape)}")
    flip_parity = (int(bool(hflip)) + int(bool(vflip))) % 2
    rot_sign = -1.0 if (int(rot90_k) % 2) else 1.0
    sin_sign = (-1.0 if flip_parity else 1.0) * rot_sign
    cos_sign = rot_sign
    if sin_sign == 1.0 and cos_sign == 1.0:
        return pol
    out = pol.clone()
    out[..., 1, :, :] = out[..., 1, :, :] * sin_sign
    out[..., 2, :, :] = out[..., 2, :, :] * cos_sign
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 기하 변환
# ─────────────────────────────────────────────────────────────────────────────
class JointGeometricTransform:
    """기하 변환을 전 모달 + 전 타깃에 **동일 파라미터**로 적용 (01 §7.4.1).

    순서(하드): random scale → random crop(``cat_max_ratio`` 재추출) → hflip → vflip → rot90.

    Note:
        기본 ``scale_range=(0.5, 2.0)`` 은 06 §3.2.4 동결값이다. 05 정정 B-27 은
        aihub092 원본이 정확히 512² 이므로 ``s<1`` 이 결정론적 우/하단 패딩(전체의 약 1/3,
        평균 42 % ignore)을 만든다고 지적하고 ``(1.0, 2.0)`` 을 권고한다 —
        그 선택은 config(`data.augment.geometric.scale_range`)에서 한다.
    """

    def __init__(
        self,
        *,
        crop_size: Tuple[int, int] = (512, 512),
        scale_range: Tuple[float, float] = (0.5, 2.0),
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rot90: bool = True,
        allow_rot90_with_pol: bool = False,
        cat_max_ratio: float = 0.75,
        cat_max_attempts: int = 10,
        pad_label: int = IGNORE_INDEX,
    ) -> None:
        if crop_size[0] <= 0 or crop_size[1] <= 0:
            raise ValueError(f"crop_size must be positive, got {crop_size}")
        if not (0.0 < scale_range[0] <= scale_range[1]):
            raise ValueError(f"invalid scale_range {scale_range}")
        self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.rot90 = bool(rot90)
        self.allow_rot90_with_pol = bool(allow_rot90_with_pol)
        self.cat_max_ratio = float(cat_max_ratio)
        self.cat_max_attempts = int(cat_max_attempts)
        self.pad_label = int(pad_label)

    # ── public ───────────────────────────────────────────────────────────────
    def __call__(
        self,
        tensors: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        rng: random.Random,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]:
        """Args:
            tensors: modality dict. 값은 ``(C,H,W)``. 키는 ``MODALITY_ORDER`` 부분집합.
            targets: 타깃 dict. ``y_seg (H,W) int64`` / ``y_chl (1,H,W)`` / ``y_chl_valid`` 등.
            rng: 샘플 로컬 ``random.Random``. 재현성의 유일한 원천이다.

        Returns:
            ``(tensors, targets, aug)``. ``aug`` = ``{scale, crop_box, hflip, vflip, rot90_k}``.
        """
        for k in _POST_GEOMETRIC_TARGETS:
            if k in targets:
                raise ValueError(
                    f"'{k}' must be generated AFTER geometric transform "
                    "(01 §7.4.3 step 7); passing it here breaks the H/4 stride contract"
                )
        for k in tensors:
            if k not in MODALITY_ORDER:
                raise ValueError(f"unknown modality key {k!r}; expected {MODALITY_ORDER}")

        src_hw = self._check_and_get_hw(tensors, targets)
        ch, cw = self.crop_size

        # 1) random scale
        scale = rng.uniform(*self.scale_range)
        nh = max(1, int(round(src_hw[0] * scale)))
        nw = max(1, int(round(src_hw[1] * scale)))
        tensors = {k: self._resize_image(v, nh, nw) for k, v in tensors.items()}
        targets = {k: self._resize_target(k, v, nh, nw) for k, v in targets.items()}

        # 2) pad (우/하단) → random crop with cat_max_ratio 재추출
        pad_h = max(0, ch - nh)
        pad_w = max(0, cw - nw)
        if pad_h or pad_w:
            tensors = {k: _pad_br(v, pad_h, pad_w, 0.0) for k, v in tensors.items()}
            targets = {
                k: self._pad_target(k, v, pad_h, pad_w) for k, v in targets.items()
            }
            nh, nw = nh + pad_h, nw + pad_w

        top, left = self._sample_crop(targets.get("y_seg"), nh, nw, rng)
        tensors = {k: v[..., top : top + ch, left : left + cw] for k, v in tensors.items()}
        targets = {
            k: (v if k in _SCALAR_TARGETS else v[..., top : top + ch, left : left + cw])
            for k, v in targets.items()
        }

        # 3) flip
        hflip = rng.random() < self.hflip_p
        vflip = rng.random() < self.vflip_p
        if hflip:
            tensors = {k: torch.flip(v, dims=(-1,)) for k, v in tensors.items()}
            targets = {
                k: (v if k in _SCALAR_TARGETS else torch.flip(v, dims=(-1,)))
                for k, v in targets.items()
            }
        if vflip:
            tensors = {k: torch.flip(v, dims=(-2,)) for k, v in tensors.items()}
            targets = {
                k: (v if k in _SCALAR_TARGETS else torch.flip(v, dims=(-2,)))
                for k, v in targets.items()
            }

        # 4) rot90 — pol 존재 시 기본 금지 (태양-관측 기하 파괴). 비정방 crop 은 k∈{0,2}.
        rot90_k = 0
        if self._rot90_allowed(tensors):
            choices = (0, 1, 2, 3) if ch == cw else (0, 2)
            rot90_k = rng.choice(choices)
            if rot90_k:
                tensors = {k: torch.rot90(v, rot90_k, dims=(-2, -1)) for k, v in tensors.items()}
                targets = {
                    k: (v if k in _SCALAR_TARGETS else torch.rot90(v, rot90_k, dims=(-2, -1)))
                    for k, v in targets.items()
                }

        # 5) AoLP 부호 보정 (필수)
        if "pol" in tensors:
            tensors["pol"] = transform_aolp(
                tensors["pol"], hflip=hflip, vflip=vflip, rot90_k=rot90_k
            )

        tensors = {k: v.contiguous() for k, v in tensors.items()}
        targets = {k: v.contiguous() for k, v in targets.items()}
        aug: Dict[str, Any] = {
            "scale": scale,
            "crop_box": (top, left, ch, cw),
            "hflip": hflip,
            "vflip": vflip,
            "rot90_k": rot90_k,
        }
        return tensors, targets, aug

    # ── internals ────────────────────────────────────────────────────────────
    def _rot90_allowed(self, tensors: Dict[str, Tensor]) -> bool:
        if not self.rot90:
            return False
        if "pol" in tensors and not self.allow_rot90_with_pol:
            return False
        return True

    @staticmethod
    def _check_and_get_hw(
        tensors: Dict[str, Tensor], targets: Dict[str, Tensor]
    ) -> Tuple[int, int]:
        hw: Optional[Tuple[int, int]] = None
        for k, v in tensors.items():
            if v.dim() != 3:
                raise ValueError(f"tensors['{k}'] must be (C,H,W), got {tuple(v.shape)}")
            cur = (int(v.shape[-2]), int(v.shape[-1]))
            if hw is None:
                hw = cur
            elif cur != hw:
                raise ValueError(f"tensors['{k}'] spatial {cur} != {hw}")
        for k, v in targets.items():
            if k in _SCALAR_TARGETS:
                continue
            cur = (int(v.shape[-2]), int(v.shape[-1]))
            if hw is None:
                hw = cur
            elif cur != hw:
                raise ValueError(f"targets['{k}'] spatial {cur} != {hw}")
        if hw is None:
            raise ValueError("nothing to transform: tensors and targets are both empty")
        return hw

    @staticmethod
    def _resize_image(t: Tensor, nh: int, nw: int) -> Tensor:
        if (int(t.shape[-2]), int(t.shape[-1])) == (nh, nw):
            return t
        out = F.interpolate(
            t[None].to(torch.float32), size=(nh, nw), mode="bilinear", align_corners=False
        )[0]
        return out.to(t.dtype)

    def _resize_target(self, key: str, t: Tensor, nh: int, nw: int) -> Tensor:
        if key in _SCALAR_TARGETS:
            return t
        if (int(t.shape[-2]), int(t.shape[-1])) == (nh, nw):
            return t
        if key in _NEAREST_TARGETS or t.dtype in (torch.int64, torch.bool):
            x = t[None] if t.dim() == 3 else t[None, None]
            out = F.interpolate(x.to(torch.float32), size=(nh, nw), mode="nearest")
            out = out.to(t.dtype)
            return out[0] if t.dim() == 3 else out[0, 0]
        return self._resize_image(t, nh, nw)

    def _pad_target(self, key: str, t: Tensor, pad_h: int, pad_w: int) -> Tensor:
        if key in _SCALAR_TARGETS:
            return t
        if key == "y_seg":
            return _pad_br(t, pad_h, pad_w, float(self.pad_label))
        if t.dtype == torch.bool:
            return _pad_br(t, pad_h, pad_w, 0.0)
        return _pad_br(t, pad_h, pad_w, 0.0)

    def _sample_crop(
        self, y_seg: Optional[Tensor], nh: int, nw: int, rng: random.Random
    ) -> Tuple[int, int]:
        ch, cw = self.crop_size
        max_top = max(0, nh - ch)
        max_left = max(0, nw - cw)
        top = rng.randint(0, max_top)
        left = rng.randint(0, max_left)
        if y_seg is None or self.cat_max_ratio >= 1.0 or self.cat_max_attempts <= 1:
            return top, left
        for _ in range(self.cat_max_attempts - 1):
            patch = y_seg[..., top : top + ch, left : left + cw]
            vals, cnt = torch.unique(patch, return_counts=True)
            keep = cnt[vals != self.pad_label]
            if keep.numel() > 1 and float(keep.max()) / float(keep.sum()) < self.cat_max_ratio:
                break
            top = rng.randint(0, max_top)
            left = rng.randint(0, max_left)
        return top, left


def _pad_br(t: Tensor, pad_h: int, pad_w: int, value: float) -> Tensor:
    """우/하단 패딩. ``F.pad`` 는 bool 을 지원하지 않으므로 float 경유."""
    if pad_h <= 0 and pad_w <= 0:
        return t
    if t.dtype == torch.bool:
        out = F.pad(t.to(torch.uint8), (0, pad_w, 0, pad_h), value=value)
        return out.to(torch.bool)
    return F.pad(t, (0, pad_w, 0, pad_h), value=value)


# ─────────────────────────────────────────────────────────────────────────────
# 광도 변환 (RGB 전용)
# ─────────────────────────────────────────────────────────────────────────────
class PhotometricRGB:
    """RGB 에만 적용. msi/bio/ir/pol 에는 어떤 광도 변환도 금지 (01 §7.4.2).

    입출력은 ``(3,H,W)`` ∈ [0,1] (ImageNet 정규화 **이전**). torchvision 에 의존하지 않고
    ``rng`` 하나로 완전히 재현 가능하게 구현했다.
    """

    def __init__(
        self,
        *,
        brightness: float = 0.25,
        contrast: float = 0.25,
        saturation: float = 0.25,
        hue: float = 0.05,
        p: float = 0.5,
        blur_p: float = 0.0,
        blur_sigma: Tuple[float, float] = (0.1, 1.0),
        noise_std: float = 0.0,
    ) -> None:
        for name, v in (("brightness", brightness), ("contrast", contrast),
                        ("saturation", saturation)):
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        if not (0.0 <= hue <= 0.5):
            raise ValueError(f"hue must be in [0, 0.5], got {hue}")
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        self.hue = float(hue)
        self.p = float(p)
        self.blur_p = float(blur_p)
        self.blur_sigma = (float(blur_sigma[0]), float(blur_sigma[1]))
        self.noise_std = float(noise_std)

    def __call__(self, rgb01: Tensor, rng: random.Random) -> Tensor:
        if rgb01.dim() != 3 or rgb01.shape[0] != 3:
            raise ValueError(f"rgb01 must be (3,H,W), got {tuple(rgb01.shape)}")
        x = rgb01.to(torch.float32)

        if rng.random() < self.p:
            ops = ["brightness", "contrast", "saturation", "hue"]
            rng.shuffle(ops)
            for op in ops:
                if op == "brightness" and self.brightness > 0:
                    x = x * rng.uniform(max(0.0, 1 - self.brightness), 1 + self.brightness)
                elif op == "contrast" and self.contrast > 0:
                    f = rng.uniform(max(0.0, 1 - self.contrast), 1 + self.contrast)
                    mean = _luma(x).mean()
                    x = (x - mean) * f + mean
                elif op == "saturation" and self.saturation > 0:
                    f = rng.uniform(max(0.0, 1 - self.saturation), 1 + self.saturation)
                    gray = _luma(x)
                    x = (x - gray) * f + gray
                elif op == "hue" and self.hue > 0:
                    x = _adjust_hue(torch.clamp(x, 0.0, 1.0), rng.uniform(-self.hue, self.hue))
                x = torch.clamp(x, 0.0, 1.0)

        if self.blur_p > 0.0 and rng.random() < self.blur_p:
            x = _gaussian_blur(x, rng.uniform(*self.blur_sigma))
            x = torch.clamp(x, 0.0, 1.0)

        if self.noise_std > 0.0:
            gen = torch.Generator(device="cpu").manual_seed(rng.randrange(2**31))
            noise = torch.randn(x.shape, generator=gen, dtype=torch.float32) * self.noise_std
            x = torch.clamp(x + noise.to(x.device), 0.0, 1.0)

        return x.contiguous()


def _luma(x: Tensor) -> Tensor:
    """ITU-R 601 휘도 ``(1,H,W)``."""
    return (0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3])


def _adjust_hue(x: Tensor, shift: float) -> Tensor:
    if shift == 0.0:
        return x
    h, s, v = _rgb_to_hsv(x)
    h = (h + shift) % 1.0
    return _hsv_to_rgb(h, s, v)


def _rgb_to_hsv(x: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor]:
    r, g, b = x[0:1], x[1:2], x[2:3]
    maxc, _ = x.max(dim=0, keepdim=True)
    minc, _ = x.min(dim=0, keepdim=True)
    delta = maxc - minc
    safe = torch.where(delta == 0, torch.ones_like(delta), delta)
    rc = (maxc - r) / safe
    gc = (maxc - g) / safe
    bc = (maxc - b) / safe
    h = torch.where(
        maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc)
    )
    h = (h / 6.0) % 1.0
    h = torch.where(delta == 0, torch.zeros_like(h), h)
    s = delta / (maxc + eps)
    return h, s, maxc


def _hsv_to_rgb(h: Tensor, s: Tensor, v: Tensor) -> Tensor:
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i % 6).to(torch.int64)
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(
        i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(
        i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(
        i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.cat([r, g, b], dim=0)


def _gaussian_blur(x: Tensor, sigma: float) -> Tensor:
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(round(3.0 * sigma)))
    k = 2 * radius + 1
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - radius
    kernel = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    c = x.shape[0]
    kx = kernel.view(1, 1, 1, k).expand(c, 1, 1, k)
    ky = kernel.view(1, 1, k, 1).expand(c, 1, k, 1)
    y = x[None]
    y = F.pad(y, (radius, radius, 0, 0), mode="reflect")
    y = F.conv2d(y, kx, groups=c)
    y = F.pad(y, (0, 0, radius, radius), mode="reflect")
    y = F.conv2d(y, ky, groups=c)
    return y[0]


def photometric_forbidden_keys() -> Sequence[str]:
    """광도 변환이 **금지된** 키 목록 (문서화·assert 용). 01 §7.4.2."""
    return tuple(k for k in MODALITY_ORDER if k != "rgb")
