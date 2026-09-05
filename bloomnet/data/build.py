"""config → Dataset / Sampler / DataLoader 조립 — 06 §3.2.6 (레벨 L3).

`build_datasets` 와 `build_dataloaders` 만이 공개 계약이며, 나머지는 그 둘이 쓰는 조립 헬퍼다.
증강 규약(01 §7.4)의 "train 에만 적용" 을 여기서 강제한다 — val/test 는 기하·광도 변환이
**둘 다 None** 이어야 한다 (평가 재현성).

레벨 L3 — L2(`aihub092`/`k235`/`drone_m3m`) 이하만 import 한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from bloomnet.constants import SIZE_DIVISOR
from bloomnet.data.aihub092 import AIHub092Dataset
from bloomnet.data.bundle import bloom_collate
from bloomnet.data.drone_m3m import DroneM3MDataset
from bloomnet.data.k235 import K235ChipDataset
from bloomnet.data.samplers import RepeatFactorSampler
from bloomnet.data.transforms import JointGeometricTransform, PhotometricRGB
from bloomnet.utils.seed import make_generator, worker_init_fn

if TYPE_CHECKING:  # pragma: no cover
    from bloomnet.config import BloomNetConfig

__all__ = [
    "SPLITS",
    "resolve_data_root",
    "assert_resolution_contract",
    "build_geometric",
    "build_photometric",
    "build_datasets",
    "build_sampler",
    "build_dataloaders",
    "load_class_presence",
]

SPLITS: Tuple[str, ...] = ("train", "val", "test")
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# 공통
# ─────────────────────────────────────────────────────────────────────────────
def resolve_data_root(root: str) -> Path:
    """상대 경로는 저장소 루트(`<repo_root>`) 기준으로 푼다."""
    p = Path(root)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def assert_resolution_contract(cfg: "BloomNetConfig") -> None:
    """R-7 방어 — 1024² 는 '2배 확대'가 아니라 '4배 넓은 지면' (02 §1.6).

    여기서 강제할 수 있는 것은 **크기 규약**뿐이다: 두 해상도 모두 32 의 배수여야 하고
    (헌법 C-4 / `SIZE_DIVISOR`), 경계 반경 매핑에 두 해상도가 모두 있어야 한다 (정정 A-28 V10).
    실제 GSD 일치 여부는 `meta["gsd_m"]` 를 기록하는 어댑터 몫이다.
    """
    for name, hw in (("train_size", cfg.data.train_size), ("eval_size", cfg.data.eval_size)):
        for v in hw:
            if int(v) % SIZE_DIVISOR != 0:
                raise ValueError(
                    f"data.{name}={list(hw)} must be a multiple of {SIZE_DIVISOR} "
                    "(헌법 C-4 / 04 §1.2)"
                )
    for hw in (cfg.data.train_size, cfg.data.eval_size):
        size = int(hw[0])
        if size not in cfg.data.boundary.radius:
            raise ValueError(
                f"data.boundary.radius has no entry for size {size} "
                f"(정정 A-28 / V10): {dict(cfg.data.boundary.radius)}"
            )


def build_geometric(cfg: "BloomNetConfig", split: str) -> Optional[JointGeometricTransform]:
    """train 에만 기하 증강을 붙인다 (01 §7.4: val/test 는 결정론적이어야 한다)."""
    if split != "train":
        return None
    g = cfg.data.augment.geometric
    return JointGeometricTransform(
        crop_size=(int(cfg.data.train_size[0]), int(cfg.data.train_size[1])),
        scale_range=(float(g.scale_range[0]), float(g.scale_range[1])),
        hflip_p=float(g.hflip_p),
        vflip_p=float(g.vflip_p),
        rot90=bool(g.rot90),
        allow_rot90_with_pol=bool(g.allow_rot90_with_pol),
        cat_max_ratio=float(g.cat_max_ratio),
        cat_max_attempts=int(g.cat_max_attempts),
        pad_label=int(cfg.data.ignore_index),
    )


def build_photometric(cfg: "BloomNetConfig", split: str) -> Optional[PhotometricRGB]:
    """RGB 전용 광도 증강. S2(pol 활성)에서는 01 §7.4.2 에 따라 **금지**된다."""
    p = cfg.data.augment.photometric
    if split != "train" or not p.enabled:
        return None
    if "pol" in cfg.data.modalities:
        return None  # DoLP 는 S0 로 정규화된 양이라 RGB 만 jitter 하면 물리 관계가 깨진다
    return PhotometricRGB(
        brightness=float(p.brightness),
        contrast=float(p.contrast),
        saturation=float(p.saturation),
        hue=float(p.hue),
        p=float(p.p),
        blur_p=float(p.blur_p),
        noise_std=float(p.noise_std),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def build_datasets(cfg: "BloomNetConfig") -> Dict[str, Dataset]:
    """``cfg.data.dataset`` 에 따라 ``{"train","val","test"}`` 데이터셋을 만든다.

    ``drone_m3m`` 은 라벨이 없어 학습·검증 분할이 정의되지 않으므로 ``{"test"}`` 만 낸다
    (01 §8.4: 진단 전용).
    """
    kind = cfg.data.dataset
    if kind == "aihub092":
        return _build_aihub092(cfg)
    if kind == "k235":
        return _build_k235(cfg)
    if kind == "drone_m3m":
        return _build_drone(cfg)
    raise ValueError(f"unknown data.dataset {kind!r} (expected aihub092|k235|drone_m3m)")


def _build_aihub092(cfg: "BloomNetConfig") -> Dict[str, Dataset]:
    assert_resolution_contract(cfg)
    root = resolve_data_root(cfg.data.root)
    out: Dict[str, Dataset] = {}
    for split in SPLITS:
        if not (root / "images" / split).is_dir():
            continue
        size = int(cfg.data.train_size[0] if split == "train" else cfg.data.eval_size[0])
        out[split] = AIHub092Dataset(
            str(root),
            split,
            num_classes=int(cfg.data.num_classes),
            ignore_index=int(cfg.data.ignore_index),
            geometric=build_geometric(cfg, split),
            photometric=build_photometric(cfg, split),
            bio_source=cfg.data.bio.source,
            bio_kind=cfg.data.bio.kind,
            boundary_radius=cfg.boundary_radius(size),
            boundary_stride=int(cfg.data.boundary.stride),
            boundary_source=cfg.data.boundary.source,
            rare_class_ids=tuple(cfg.data.augment.rare_class.ids),
            rare_class_crop_prob=float(cfg.data.augment.rare_class.crop_prob),
            rare_class_min_pixels=int(cfg.data.augment.rare_class.min_pixels),
            rare_class_crop_attempts=int(cfg.data.augment.rare_class.attempts),
            active_modalities=tuple(cfg.data.modalities),
            sensor=cfg.data.sensor,
            seed=int(cfg.seed),
            phys_slot_ids=tuple(cfg.data.phys_slot_ids),
            band_ids=tuple(cfg.data.band_ids) if cfg.data.band_ids else None,
            mci_c=cfg.spec.mci_c,
            bio_eps=float(cfg.data.bio.eps),
            class_presence_path=cfg.data.sampler.class_stats,
        )
    if not out:
        raise RuntimeError(f"aihub092: no split directory under {root / 'images'}")
    return out


def _build_k235(cfg: "BloomNetConfig") -> Dict[str, Dataset]:
    npz = resolve_data_root(cfg.spec.npz)
    common = dict(
        band_order=tuple(cfg.spec.band_order),
        use_blue=bool(cfg.spec.use_blue),
        apply_quality_filter=bool(cfg.spec.quality_filter),
        target=cfg.data.chl.space,
    )
    if cfg.spec.mci_c is not None:
        common["mci_c"] = float(cfg.spec.mci_c)
    train = K235ChipDataset(str(npz), tuple(cfg.spec.train_sites), **common)
    val = K235ChipDataset(str(npz), tuple(cfg.spec.val_sites), **common)
    # 01 §6.4 [S2] 는 val 만 정의한다. LOSO(=정직한 보고 분할)는 pretrain/loso.py 소관이므로
    # 여기서 test 를 따로 만들지 않고 val 과 동일 객체를 가리킨다 (수치 중복 보고 금지).
    return {"train": train, "val": val, "test": val}


def _build_drone(cfg: "BloomNetConfig") -> Dict[str, Dataset]:
    root = resolve_data_root(cfg.data.root)
    ds = DroneM3MDataset(
        str(root),
        out_hw=(int(cfg.data.eval_size[0]), int(cfg.data.eval_size[1])),
        sensor=cfg.data.sensor,
        active_modalities=tuple(cfg.data.modalities),
        bio_source=cfg.data.bio.source,
        bio_kind=cfg.data.bio.kind,
        ignore_index=int(cfg.data.ignore_index),
        k_sensor=cfg.data.msi.k_sensor,
    )
    return {"test": ds}


# ─────────────────────────────────────────────────────────────────────────────
# Sampler / DataLoader
# ─────────────────────────────────────────────────────────────────────────────
def load_class_presence(path: str, n_expected: int) -> np.ndarray:
    """``tools/compute_class_stats.py`` 산출물 → ``(N,K)`` bool."""
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".npz":
        z = np.load(p)
        key = "class_presence" if "class_presence" in z else list(z.keys())[0]
        arr = z[key]
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        arr = np.asarray(data["class_presence"] if isinstance(data, dict) else data)
    arr = np.asarray(arr).astype(bool)
    if arr.shape[0] != n_expected:
        raise RuntimeError(f"class_presence rows {arr.shape[0]} != dataset length {n_expected}")
    return arr


def build_sampler(
    cfg: "BloomNetConfig", dataset: Dataset
) -> Optional[torch.utils.data.Sampler]:
    """``sampler.kind`` 가 ``rfs`` 일 때만 :class:`RepeatFactorSampler` 를 만든다."""
    kind = cfg.data.sampler.kind
    if kind == "shuffle":
        return None
    if kind != "rfs":
        raise ValueError(f"unknown data.sampler.kind {kind!r} (expected shuffle|rfs)")
    if cfg.data.sampler.class_stats:
        presence = load_class_presence(  # type: ignore[arg-type]
            cfg.data.sampler.class_stats, len(dataset)
        )
    else:
        presence = getattr(dataset, "class_presence", None)
        if presence is None:
            raise RuntimeError(
                "sampler.kind='rfs' requires class presence; set data.sampler.class_stats "
                "(tools/compute_class_stats.py) or use a dataset exposing .class_presence"
            )
    return RepeatFactorSampler(
        np.asarray(presence), t=float(cfg.data.sampler.repeat_t), seed=int(cfg.seed)
    )


def build_dataloaders(
    cfg: "BloomNetConfig", datasets: Dict[str, Dataset]
) -> Dict[str, DataLoader]:
    """train 은 shuffle/RFS + ``drop_last=True``, val/test 는 결정론 (06 §3.2.6).

    Note:
        ``persistent_workers=False`` 가 규약이다 — 에폭마다 worker 를 재생성해야
        ``Dataset.set_epoch`` 과 ``sampler.set_epoch`` 이 실제로 반영된다.
    """
    nw = int(cfg.data.num_workers)
    loaders: Dict[str, DataLoader] = {}
    for split, ds in datasets.items():
        is_train = split == "train"
        sampler = build_sampler(cfg, ds) if is_train else None
        kwargs: Dict[str, Any] = dict(
            batch_size=int(cfg.schedule.batch_size),
            num_workers=nw,
            pin_memory=bool(cfg.data.pin_memory),
            persistent_workers=False,
            collate_fn=bloom_collate,
            worker_init_fn=worker_init_fn,
            generator=make_generator(int(cfg.seed)),
            drop_last=is_train,
        )
        if nw > 0:
            kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
        if sampler is not None:
            kwargs["sampler"] = sampler
        else:
            kwargs["shuffle"] = is_train
        loaders[split] = DataLoader(ds, **kwargs)
    return loaders
