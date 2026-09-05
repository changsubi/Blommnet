"""aihub092 (AI Hub 092) 어댑터 — 01 §7.1~§7.5 / 06 §3.2.5 (레벨 L2).

담당:
    * :func:`parse_stem` / :func:`group_key` / :func:`find_pairs` — 01 §7.2 페어링 규칙
      (페어링 실패는 **조용한 스킵이 아니라 RuntimeError**).
    * :class:`AIHub092Dataset` — 01 §7.4.3 의 8단계 파이프라인을 **그 순서 그대로** 수행한다.

파이프라인 순서 (하드, 01 §7.4.3):
    1 로드 → 2 msi R4/R4′/R6 → 3 기하변환 → 4 x_bio 계산 → 5 광도(rgb 만)
    → 6 정규화 → 7 타깃 생성(y_edge) → 8 avail 계산 및 A1~A6 assert

    4 가 3 뒤에 와야 하는 이유는 지수가 비선형이기 때문이다:
    ``mean((a−b)/(a+b)) ≠ (mean a − mean b)/(mean a + mean b)``.
    지수를 먼저 계산하고 리샘플하면 "비율의 평균"이 되어 물리적으로 틀린 값이 된다.

레벨 L2 — L−1(`constants`), L0(`data.boundary`), L1(`data.bundle`/`indices`/`transforms`) 만 import 한다.
같은 L2 파일(`k235.py`, `drone_m3m.py`)은 import 하지 않는다.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from bloomnet.constants import (
    BAND_CENTERS_NM,
    IGNORE_INDEX,
    MODALITY_ORDER,
    MSI_SLOTS,
    SENSOR_BAND_IDS,
    STEM_RE,
)
from bloomnet.data.boundary import make_boundary_target
from bloomnet.data.bundle import assert_availability_contract
from bloomnet.data.indices import (
    canonical_scatter_np,
    compute_bio_canonical,
    compute_rgb_proxy,
    mci_coefficient,
    normalize_imagenet,
)
from bloomnet.data.transforms import JointGeometricTransform, PhotometricRGB

__all__ = [
    "LABEL_SUFFIX",
    "IMAGE_SUFFIX",
    "BIO_SOURCES",
    "parse_stem",
    "group_key",
    "find_pairs",
    "scan_class_presence",
    "AIHub092Dataset",
]

IMAGE_SUFFIX: str = ".png"
LABEL_SUFFIX: str = "_labelids.png"
BIO_SOURCES: Tuple[str, ...] = ("none", "msi", "rgb_proxy")

_STEM_RE = re.compile(STEM_RE)


# ─────────────────────────────────────────────────────────────────────────────
# 파일 페어링 (01 §7.2)
# ─────────────────────────────────────────────────────────────────────────────
def parse_stem(stem: str) -> Dict[str, str]:
    """``L01_11680_65_20231025_N03_00001`` → 토큰 dict (01 §7.2).

    Returns:
        ``{"river","admin","alt","date","line","seq"}`` — 전부 원본 문자열.

    Raises:
        ValueError: 정규식 불일치. ★ 조용히 넘기지 않는다.

    Note:
        `[J]` 01 §7.2: sensor-suffix 스트립 로직(`_windshield_vis` 등)은 **구현하지 않는다**.
        [M3] 전수 스캔에서 이 데이터의 stem 은 센서 접미사를 한 번도 갖지 않았다.
    """
    m = _STEM_RE.match(stem)
    if m is None:
        raise ValueError(f"stem does not match AI Hub 092 pattern: {stem!r} (01 §7.2 STEM_RE)")
    return dict(m.groupdict())


def group_key(scene: str, stem: str) -> Tuple[str, str, str, str]:
    """flight-line 군 키 ``(scene, admin, date, line)`` (01 §7.3 [M3], 총 174 군)."""
    g = parse_stem(stem)
    return (str(scene), g["admin"], g["date"], g["line"])


def find_pairs(root: Path, split: str) -> List[Tuple[Path, Path]]:
    """``images/<split>/<scene>/<stem>.png`` ↔ ``labels/<split>/<scene>/<stem>_labelids.png``.

    Returns:
        ``(scene, stem)`` 사전순으로 정렬된 (image, label) 경로 쌍 리스트.

    Raises:
        FileNotFoundError: split 디렉터리 부재.
        RuntimeError: 페어링 실패가 **1건이라도** 있을 때 (01 §7.2 하드 검증).
            현 이전 구현 구현의 "라벨 없으면 조용히 스킵"을 오류로 승격한 것이다.
    """
    root = Path(root)
    img_root = root / "images" / split
    lbl_root = root / "labels" / split
    if not img_root.is_dir():
        raise FileNotFoundError(f"image split dir not found: {img_root}")

    pairs: List[Tuple[Path, Path]] = []
    orphan_images: List[str] = []
    seen_labels: set = set()
    for scene_dir in sorted((p for p in img_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        for img in sorted(scene_dir.glob("*" + IMAGE_SUFFIX)):
            lbl = lbl_root / scene_dir.name / (img.stem + LABEL_SUFFIX)
            if lbl.exists():
                pairs.append((img, lbl))
                seen_labels.add(str(lbl))
            else:
                orphan_images.append(str(img))

    orphan_labels: List[str] = []
    if lbl_root.is_dir():
        scene_dirs = sorted((p for p in lbl_root.iterdir() if p.is_dir()), key=lambda p: p.name)
        for scene_dir in scene_dirs:
            for lbl in sorted(scene_dir.glob("*" + LABEL_SUFFIX)):
                if str(lbl) not in seen_labels:
                    orphan_labels.append(str(lbl))

    if orphan_images or orphan_labels:
        raise RuntimeError(
            "AIHub092 pairing failed (01 §7.2: silent skip is forbidden): "
            f"{len(orphan_images)} image(s) without label, "
            f"{len(orphan_labels)} label(s) without image. "
            f"first_missing_label={orphan_images[:3]} first_orphan_label={orphan_labels[:3]}"
        )
    if not pairs:
        raise RuntimeError(f"AIHub092: no image/label pair under {img_root}")
    return pairs


def scan_class_presence(
    label_paths: Sequence[Path], num_classes: int = 12, *, ignore_index: int = IGNORE_INDEX
) -> np.ndarray:
    """``(N, num_classes)`` bool 클래스 출현 행렬 (RFS·group 분할 입력).

    ``ignore_index`` 와 ``>= num_classes`` 인 값은 무시한다 (01 §7.3 [M14] 스캔과 동일 규약).
    """
    out = np.zeros((len(label_paths), int(num_classes)), dtype=bool)
    for i, p in enumerate(label_paths):
        arr = _read_label_array(Path(p))
        for v in np.unique(arr):
            iv = int(v)
            if 0 <= iv < num_classes and iv != ignore_index:
                out[i, iv] = True
    return out


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
def _read_label_array(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        # PIL 버퍼는 읽기 전용이라 torch.from_numpy 가 경고를 낸다 → 명시적 복사
        return np.array(im, copy=True)


def _load_rgb01(path: Path) -> Tensor:
    """PNG/JPG → ``(3,H,W) float32`` ∈ [0,1]. ``arr/255.0`` 은 헌법 C-4 bit-exact 규약."""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    t = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
    return t.to(torch.float32) / 255.0


def _load_label(path: Path) -> Tensor:
    arr = _read_label_array(path)
    if arr.ndim != 2:
        raise RuntimeError(f"label must be single-channel, got shape {arr.shape}: {path}")
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class AIHub092Dataset(torch.utils.data.Dataset):
    """aihub092 512² RGB + 12클래스 마스크 (01 §7, 06 §3.2.5 동결 시그니처).

    Args:
        root: 링크 트리 루트 (``bloomnet/data/aihub092_asis`` 또는 ``…_group``).
        split: ``{"train","val","test"}``.
        num_classes: S0 = 12 / S1 = 2 (헌법 R7).
        geometric / photometric: ``None`` 이면 해당 증강 없음 (val/test 규약).
        bio_source: ``{"none","msi","rgb_proxy"}``. aihub092 는 msi 원천이 없으므로
            ``"msi"`` 를 주면 msi 결측(avail=0) → bio 도 결측이 된다.
        boundary_source: ``"dataset"`` 이면 여기서 ``y_edge`` 를 만들고,
            ``"criterion"`` 이면 0 채움 + ``y_edge_valid=False`` 로 둔다 (정정 A-28:
            소유권은 criterion 에 있고 criterion 이 ``y_seg`` 에서 재생성한다).

    Keyword-only 추가 인자 (06 동결표에 없음 — 기본값이 초판 동작과 동일하므로 계약 불변):
        seed / epoch: 샘플별 ``random.Random`` 시드 원천. ``set_epoch`` 로 갱신한다.
        phys_slot_ids / band_ids / mci_c / bio_eps: ``meta`` 기록 및 지수 계산용.
        strict_contract: ``__getitem__`` 마지막에 A1~A6 assert (01 §7.4.3 step 8).
        class_presence_path: ``tools/compute_class_stats.py`` 산출 ``.npy``/``.json``.
            없으면 :attr:`class_presence` 접근 시 라벨을 전수 스캔한다(느리다).

    Note:
        ``avail[0](rgb) == 1.0`` 은 A2 로 항상 참이며, A1 은 "avail==1 ⟺ 텐서가 정확히
        전부 0 은 아니다" 를 요구한다. 완전 무채색(R=G=B) 이미지에서 ``rgb_proxy`` bio 는
        수학적으로 전부 0 이 되므로 A1 위반으로 죽는다 — 이것이 의도된 동작이다
        (조용한 열화 대신 즉시 실패, 01 §2.4).
    """

    def __init__(
        self,
        root: str,
        split: str,
        *,
        num_classes: int = 12,
        ignore_index: int = IGNORE_INDEX,
        geometric: Optional[JointGeometricTransform] = None,
        photometric: Optional[PhotometricRGB] = None,
        bio_source: str = "none",
        bio_kind: str = "mci",
        boundary_radius: int = 1,
        boundary_stride: int = 4,
        boundary_source: str = "dataset",
        rare_class_ids: Sequence[int] = (),
        rare_class_crop_prob: float = 0.0,
        rare_class_min_pixels: int = 0,
        rare_class_crop_attempts: int = 10,
        active_modalities: Sequence[str] = ("rgb",),
        sensor: str = "none",
        # ── 06 동결표에 없는 keyword-only 추가분 ────────────────────────────
        seed: int = 1234,
        epoch: int = 0,
        phys_slot_ids: Sequence[int] = (0, 1, 2, 3),
        band_ids: Optional[Sequence[int]] = None,
        mci_c: Optional[float] = None,
        bio_eps: float = 1.0e-6,
        strict_contract: bool = True,
        class_presence_path: Optional[str] = None,
    ) -> None:
        if bio_source not in BIO_SOURCES:
            raise ValueError(f"bio_source must be one of {BIO_SOURCES}, got {bio_source!r}")
        if boundary_source not in ("dataset", "criterion"):
            raise ValueError(f"boundary_source must be dataset|criterion, got {boundary_source!r}")
        if bio_kind not in ("mci", "gndvi", "rgb_proxy", "none"):
            raise ValueError(f"unknown bio_kind {bio_kind!r}")
        for m in active_modalities:
            if m not in MODALITY_ORDER:
                raise ValueError(f"unknown modality {m!r}; expected subset of {MODALITY_ORDER}")
        if "rgb" not in active_modalities:
            raise ValueError("A2: 'rgb' must always be active")

        self.root = Path(root)
        self.split = str(split)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.geometric = geometric
        self.photometric = photometric
        self.bio_source = str(bio_source)
        self.bio_kind = str(bio_kind)
        self.boundary_radius = int(boundary_radius)
        self.boundary_stride = int(boundary_stride)
        self.boundary_source = str(boundary_source)
        self.rare_class_ids = tuple(int(c) for c in rare_class_ids)
        self.rare_class_crop_prob = float(rare_class_crop_prob)
        self.rare_class_min_pixels = int(rare_class_min_pixels)
        self.rare_class_crop_attempts = int(rare_class_crop_attempts)
        self.active_modalities = tuple(active_modalities)
        self.sensor = str(sensor)
        self.seed = int(seed)
        self.phys_slot_ids = tuple(int(s) for s in phys_slot_ids)
        self.bio_eps = float(bio_eps)
        self.strict_contract = bool(strict_contract)
        self.class_presence_path = class_presence_path
        self._epoch = int(epoch)
        self._class_presence: Optional[np.ndarray] = None

        if band_ids is not None:
            self.band_ids: Tuple[int, ...] = tuple(int(b) for b in band_ids)
        elif self.sensor in SENSOR_BAND_IDS:
            self.band_ids = tuple(SENSOR_BAND_IDS[self.sensor])
        else:
            self.band_ids = ()
        self.band_centers_nm: Dict[str, float] = dict(BAND_CENTERS_NM.get(self.sensor, {}))
        if mci_c is not None:
            self.mci_c = float(mci_c)
        else:
            self.mci_c = _default_mci_c(self.sensor)

        self.pairs: List[Tuple[Path, Path]] = find_pairs(self.root, self.split)
        self.stems: List[str] = [p[0].stem for p in self.pairs]
        self.scenes: List[str] = [p[0].parent.name for p in self.pairs]

    # ── API ──────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        """증강 RNG 를 에폭에 종속시킨다 (``persistent_workers=False`` 전제)."""
        self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def class_presence(self) -> np.ndarray:
        """``(N, num_classes)`` bool — :class:`RepeatFactorSampler` 입력 (06 §3.2.5)."""
        if self._class_presence is None:
            self._class_presence = self._load_or_scan_presence()
        return self._class_presence

    def group_keys(self) -> List[Tuple[str, str, str, str]]:
        """샘플별 flight-line 군 키 (진단·`GroupBatchSampler` 용)."""
        return [group_key(sc, st) for sc, st in zip(self.scenes, self.stems)]

    # ── __getitem__ (01 §7.4.3 의 8단계) ─────────────────────────────────────
    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_path, lbl_path = self.pairs[index]
        stem, scene = self.stems[index], self.scenes[index]
        rng = random.Random((self.seed * 1_000_003 + self._epoch * 9_176 + index) % (2**63))

        # 1) 로드
        rgb01 = _load_rgb01(img_path)
        y_seg = _load_label(lbl_path)
        if y_seg.shape != rgb01.shape[-2:]:
            raise RuntimeError(
                f"image/label size mismatch for {stem}: "
                f"{tuple(rgb01.shape[-2:])} vs {tuple(y_seg.shape)}"
            )
        h, w = int(rgb01.shape[-2]), int(rgb01.shape[-1])

        # 2) msi R4/R4'/R6 — aihub092 에는 msi 원천이 없다 (샘플 수준 결측, A5)
        tensors: Dict[str, Tensor] = {"rgb": rgb01}
        msi_ch = len(self.band_ids) if self.band_ids else len(MSI_SLOTS)
        if "msi" in self.active_modalities:
            tensors["msi"] = torch.zeros(msi_ch, h, w, dtype=torch.float32)
        if "ir" in self.active_modalities:
            tensors["ir"] = torch.zeros(1, h, w, dtype=torch.float32)
        if "pol" in self.active_modalities:
            tensors["pol"] = torch.zeros(3, h, w, dtype=torch.float32)

        targets: Dict[str, Tensor] = {
            "y_seg": y_seg,
            "y_chl": torch.zeros(1, h, w, dtype=torch.float32),
            "y_chl_valid": torch.zeros(1, h, w, dtype=torch.bool),
            "y_chl_scalar": torch.zeros((), dtype=torch.float32),
            "y_chl_scalar_valid": torch.zeros((), dtype=torch.bool),
        }

        # 3) 기하 변환 (전 모달 + 전 타깃 동일 파라미터) — y_edge 는 아직 만들지 않는다
        aug: Dict[str, Any] = {}
        if self.geometric is not None:
            tensors, targets, aug = self._geometric_with_rare_class(tensors, targets, rng)

        # 4) ★ x_bio — 반드시 기하 변환 '이후' (지수는 비선형)
        rgb01 = tensors["rgb"]
        bio, bio_present = self._compute_bio(tensors)
        if bio is not None:
            tensors["bio"] = bio

        # 5) 광도 변환 — rgb 만 (msi/bio/ir/pol 전면 금지, 01 §7.4.2)
        if self.photometric is not None:
            tensors["rgb"] = self.photometric(tensors["rgb"], rng)

        # 6) 정규화
        tensors["rgb"] = normalize_imagenet(tensors["rgb"]).contiguous()

        # 7) 타깃 생성 — y_edge (정정 A-28: source == "criterion" 이면 여기서 만들지 않는다)
        y_seg_out = targets["y_seg"]
        hh, ww = int(y_seg_out.shape[-2]), int(y_seg_out.shape[-1])
        eh, ew = hh // self.boundary_stride, ww // self.boundary_stride
        if self.boundary_source == "dataset":
            edge, edge_valid = make_boundary_target(
                y_seg_out,
                ignore_index=self.ignore_index,
                radius=self.boundary_radius,
                out_stride=self.boundary_stride,
            )
        else:
            edge = torch.zeros(1, eh, ew, dtype=torch.float32)
            edge_valid = torch.zeros(1, eh, ew, dtype=torch.bool)

        # 8) avail + meta + 계약 assert
        avail = torch.zeros(len(MODALITY_ORDER), dtype=torch.float32)
        avail[0] = 1.0  # A2
        if bio_present:
            avail[MODALITY_ORDER.index("bio")] = 1.0

        sample: Dict[str, Any] = {k: v.contiguous() for k, v in tensors.items()}
        sample["avail"] = avail
        sample["y_seg"] = y_seg_out.contiguous()
        sample["y_edge"] = edge.contiguous()
        sample["y_edge_valid"] = edge_valid.contiguous()
        sample["y_chl"] = targets["y_chl"].contiguous()
        sample["y_chl_valid"] = targets["y_chl_valid"].contiguous()
        sample["y_chl_scalar"] = targets["y_chl_scalar"]
        sample["y_chl_scalar_valid"] = targets["y_chl_scalar_valid"]
        sample["meta"] = self._meta(stem, scene, aug)

        if self.strict_contract:
            assert_availability_contract(
                sample,
                bio_source=self.bio_source,
                active_modalities=self._sample_modalities(),
            )
        return sample

    # ── internals ────────────────────────────────────────────────────────────
    def _sample_modalities(self) -> Tuple[str, ...]:
        mods = list(self.active_modalities)
        if self.bio_source != "none" and "bio" not in mods:
            mods.append("bio")
        return tuple(m for m in MODALITY_ORDER if m in mods)

    def _default_gsd(self) -> Optional[float]:
        """AI Hub 092 는 GSD 메타를 제공하지 않는다 → ``None`` (R-7 은 build.py 가 검증)."""
        return None

    def _meta(self, stem: str, scene: str, aug: Dict[str, Any]) -> Dict[str, Any]:
        tok = parse_stem(stem)
        gsd = self._default_gsd()
        scale = float(aug.get("scale", 1.0))
        if gsd is not None and scale > 0:
            gsd = gsd / scale  # 확대(scale>1) → 지상 표본 간격 감소
        return {
            "stem": stem,
            "scene": scene,
            "group_key": group_key(scene, stem),
            "split": self.split,
            "site": tok["admin"],
            "date": tok["date"],
            "flight_line": tok["line"],
            "alt_m": float(tok["alt"]),
            "gsd_m": gsd,
            "sensor": self.sensor,
            "band_ids": tuple(self.band_ids),
            "phys_slot_ids": tuple(self.phys_slot_ids),
            "band_centers_nm": dict(self.band_centers_nm),
            "bio_kind": self._effective_bio_kind(),
            "chl_space": "log1p",
            "aug": dict(aug),
        }

    def _effective_bio_kind(self) -> str:
        """V17 / 정정 A-11: ``source == rgb_proxy`` 면 ``meta['bio_kind']`` 도 rgb_proxy 다."""
        if self.bio_source == "rgb_proxy":
            return "rgb_proxy"
        return self.bio_kind

    def _compute_bio(self, tensors: Dict[str, Tensor]) -> Tuple[Optional[Tensor], bool]:
        """step 4. 반환 ``(bio (2,H,W) 또는 None, present)``."""
        if self.bio_source == "none":
            return None, False
        if self.bio_source == "rgb_proxy":
            bio = compute_rgb_proxy(tensors["rgb"][None], eps=1.0e-8)[0]
            return bio.to(torch.float32), True
        # bio_source == "msi"
        msi = tensors.get("msi")
        h, w = int(tensors["rgb"].shape[-2]), int(tensors["rgb"].shape[-1])
        if msi is None or bool((msi == 0).all().item()) or not self.band_ids:
            return torch.zeros(2, h, w, dtype=torch.float32), False
        s, m = canonical_scatter_np(msi.numpy(), self.band_ids, len(MSI_SLOTS))
        bio = compute_bio_canonical(
            torch.from_numpy(s)[None],
            torch.from_numpy(m)[None],
            mci_c=self.mci_c,
            kind=self.bio_kind if self.bio_kind in ("mci", "gndvi") else "mci",
            eps=self.bio_eps,
        )[0]
        return bio.to(torch.float32), True

    def _geometric_with_rare_class(
        self,
        tensors: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        rng: random.Random,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]:
        """희소 클래스 인지 crop 재추출 ([F] 분석 §E.1.8 / 06 §3.2.5 rare_class_*)."""
        assert self.geometric is not None
        want_rare = (
            bool(self.rare_class_ids)
            and self.rare_class_crop_prob > 0.0
            and rng.random() < self.rare_class_crop_prob
            and _has_any_class(targets["y_seg"], self.rare_class_ids)
        )
        attempts = max(1, self.rare_class_crop_attempts) if want_rare else 1
        best: Optional[Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Any]]] = None
        best_count = -1
        for _ in range(attempts):
            out = self.geometric(dict(tensors), dict(targets), rng)
            if not want_rare:
                return out
            cnt = _count_any_class(out[1]["y_seg"], self.rare_class_ids)
            if cnt > best_count:
                best, best_count = out, cnt
            if cnt >= self.rare_class_min_pixels:
                return out
        assert best is not None
        return best

    def _load_or_scan_presence(self) -> np.ndarray:
        if self.class_presence_path is not None:
            return _load_presence_file(Path(self.class_presence_path), len(self.pairs))
        cache = self.root / f"class_presence_{self.split}.npy"
        if cache.exists():
            return _load_presence_file(cache, len(self.pairs))
        pres = scan_class_presence(
            [p[1] for p in self.pairs], self.num_classes, ignore_index=self.ignore_index
        )
        # 캐시는 있으면 좋고 없어도 되는 것. ★ 저장소 밖(원본 데이터 트리)에는 절대 쓰지 않는다
        # (헌법 C-5.6: 이전 구현 는 읽기 전용 참조).
        if _inside_repo(self.root):
            try:
                np.save(cache, pres)
            except OSError:
                pass
        return pres


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _inside_repo(path: Path) -> bool:
    """``path`` 가 BloomNet 저장소 안인가 (쓰기 허용 범위, 헌법 C-5.6)."""
    try:
        return Path(path).resolve().is_relative_to(_REPO_ROOT)
    except (OSError, ValueError):
        return False


def _default_mci_c(sensor: str) -> float:
    """센서 밴드 중심에서 MCI baseline 계수를 유도한다 (리터럴 복제 금지, 06 §3.1 각주)."""
    centers = BAND_CENTERS_NM.get(sensor)
    if not centers or not {"red", "rededge1", "nir"} <= set(centers):
        return 0.0
    return mci_coefficient(centers["red"], centers["rededge1"], centers["nir"])


def _has_any_class(y_seg: Tensor, class_ids: Sequence[int]) -> bool:
    if not class_ids:
        return False
    ids = torch.tensor(list(class_ids), dtype=y_seg.dtype)
    return bool(torch.isin(y_seg, ids).any().item())


def _count_any_class(y_seg: Tensor, class_ids: Sequence[int]) -> int:
    if not class_ids:
        return 0
    ids = torch.tensor(list(class_ids), dtype=y_seg.dtype)
    return int(torch.isin(y_seg, ids).sum().item())


def _load_presence_file(path: Path, n_expected: int) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        z = np.load(path)
        key = "class_presence" if "class_presence" in z else list(z.keys())[0]
        arr = z[key]
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        arr = np.asarray(data["class_presence"] if isinstance(data, dict) else data)
    else:
        raise ValueError(f"unsupported class_presence file: {path}")
    arr = np.asarray(arr).astype(bool)
    if arr.shape[0] != n_expected:
        raise RuntimeError(
            f"class_presence rows {arr.shape[0]} != dataset length {n_expected} ({path})"
        )
    return arr
