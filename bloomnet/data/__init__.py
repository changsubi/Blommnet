"""`bloomnet.data` — 데이터 계약 · 지수 · 증강 · 샘플러 (01 §2, §4, §7).

06 §2.1 원칙 3: ``__init__.py`` 는 **re-export 만** 한다 (로직 금지).
"""

from bloomnet.data.boundary import (
    POS_RATIO_ANCHOR,
    boundary_pos_weight,
    make_boundary_target,
    pos_ratio_is_sane,
)
from bloomnet.data.bundle import (
    MODALITY_KEYS,
    OPTIONAL_MODALITY_KEYS,
    PHYS_SLOT_TO_MODALITY_INDEX,
    SAMPLE_META_KEY,
    SAMPLE_REQUIRED_META,
    SAMPLE_TARGET_KEYS,
    Sample,
    assert_availability_contract,
    bloom_collate,
    derive_present,
    phys_modality_indices,
)
from bloomnet.data.indices import (
    apply_k_sensor,
    assert_msi_scale_contract,
    canonical_scatter_np,
    compute_bio_canonical,
    compute_rgb_proxy,
    coregister_m3m,
    denormalize_imagenet,
    m3m_dn_to_relative_reflectance,
    mci_coefficient,
    normalize_imagenet,
    specular_proxy,
)
from bloomnet.data.samplers import GroupBatchSampler, RepeatFactorSampler
from bloomnet.data.transforms import (
    JointGeometricTransform,
    PhotometricRGB,
    transform_aolp,
)

__all__ = [
    # boundary (L0)
    "make_boundary_target",
    "boundary_pos_weight",
    "pos_ratio_is_sane",
    "POS_RATIO_ANCHOR",
    # bundle (L1)
    "Sample",
    "MODALITY_KEYS",
    "OPTIONAL_MODALITY_KEYS",
    "SAMPLE_TARGET_KEYS",
    "SAMPLE_META_KEY",
    "SAMPLE_REQUIRED_META",
    "PHYS_SLOT_TO_MODALITY_INDEX",
    "bloom_collate",
    "assert_availability_contract",
    "derive_present",
    "phys_modality_indices",
    # indices (L1)
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
    # transforms (L1)
    "JointGeometricTransform",
    "PhotometricRGB",
    "transform_aolp",
    # samplers (L1)
    "RepeatFactorSampler",
    "GroupBatchSampler",
]
