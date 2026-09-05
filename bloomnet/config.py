"""BloomNet 설정 스키마 · YAML 로드/병합/검증 · `--set` 오버라이드 (06 §4, 레벨 L1).

정본 관계
    - **dataclass 트리가 스키마의 정본**이다. `configs/_base.yaml` 은 이 트리의 덤프본이며
      (`python -m bloomnet.config --dump-base`), 여기 없는 키는 거부된다 (이전 구현의
      unknown-key rejection 을 중첩으로 확장 — [분석] §E.3).
    - 로드 순서 (06 §4.1): 사전 파서(--config/--base) → YAML 병합(_base 체인, 리스트는 교체)
      → 스키마 검증(미지 키 ValueError) → CLI 오버라이드(--set + 1급 플래그 15종)
      → `__post_init__` 검증(V1~V22) → `config.resolved.yaml` 덤프.

주요 정정 반영
    A-11 V12 교체 / V16 · V17 신설,  A-18 warmup_epochs 1급 키 + V15 강화,
    A-19 band_order_confirmed + V18 (transplant 기본 false),  A-20 unc_clamp [-7,7],
    A-22 fp32_forced 에 litemla + V13 강화,  A-26 V19(1 ∈ bmef.stages) + stage1_identity,
    A-27 spec.mci_c 를 센서에서 유도,  A-28 boundary.radius 를 해상도 매핑 + V10,
    A-32 loss.aux_loss_stride + V22,  A-37 V20(gn8 = GroupNorm(gcd(8,C), C)),
    A-40 data.msi.k_sensor + V21,  B-27 augment.scale_range [1.0, 2.0].
"""

from __future__ import annotations

import argparse
import copy
import math
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from bloomnet.constants import (
    AUX_TAP_STRIDE,
    BAND_CENTERS_NM,
    CHANNELS,
    DEC_CH,
    IGNORE_INDEX,
    K_SENSOR,
    MODALITY_ORDER,
    MSI_SLOTS,
    PHYS_SLOTS,
    SENSOR_BAND_IDS,
    SIZE_DIVISOR,
)
from bloomnet.version import short_revision

__all__ = [
    "BloomNetConfig",
    "default_config",
    "from_dict",
    "to_dict",
    "load_config",
    "apply_overrides",
    "dump_config",
    "to_yaml",
    "merge_dicts",
    "resolve_schedule",
    "make_run_name",
    "build_parser",
    "parse_cli",
    "FIRST_CLASS_FLAGS",
    "DEFAULT_BASE_PATH",
    "SCHEMA_EXCLUDE",
]

DEFAULT_BASE_PATH: str = "configs/_base.yaml"
SCHEMA_EXCLUDE: str = "schema_exclude"  # field metadata 키: True 면 YAML 스키마에서 제외

MODES: Tuple[str, ...] = ("s0_rgb", "s0_spec", "s1_rgb_ms4", "s2_full")
# V14 — mode ↔ modalities 정합 (경고만. 명시적 override 허용)
_MODE_MODALITIES: Dict[str, Tuple[str, ...]] = {
    "s0_rgb": ("rgb",),
    "s0_spec": ("rgb",),  # 분광 MLP 단독 학습이므로 seg 모달은 rgb 만 둔다
    "s1_rgb_ms4": ("rgb", "msi", "bio"),
    "s2_full": ("rgb", "msi", "bio", "ir", "pol"),
}
_FP32_REQUIRED: Tuple[str, ...] = ("bmef.fusion", "encoder.*.litemla.attention")


# ═══════════════════════════════════════════════════════════════════════
#  데이터
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class BioConfig:
    kind: str = "mci"  # {mci, gndvi, rgb_proxy, none}   [X-01/X-27, 정정 A-11]
    source: str = "none"  # {none, msi, rgb_proxy}         [01 §4.1]
    eps: float = 1.0e-6
    clamp: List[float] = field(default_factory=lambda: [-1.0, 1.0])


@dataclass
class PolConfig:
    encoding: str = "sincos"  # {sincos(3ch), stokes(2ch)}  [X-04]


@dataclass
class MsiConfig:
    clamp: List[float] = field(default_factory=lambda: [0.0, 1.5])  # k_sensor 적용 후에만 유효
    apply_r4_irradiance: bool = True  # 지수 계산 전 필수 [01 §2.6]
    apply_r4p_k_sensor: bool = True  # R4' 센서 스케일 정합 (정정 A-40)
    k_sensor: Optional[float] = None  # null -> K_SENSOR[sensor] (V21)
    apply_r6_coregister: bool = True


@dataclass
class BoundaryConfig:
    source: str = "dataset"  # {dataset, criterion}  [X-07 / 정정 A-28]
    radius: Dict[int, int] = field(default_factory=lambda: {512: 1, 1024: 2})  # 해상도별 (V10)
    stride: int = 4  # (고정) 디코더 작업 해상도 [헌법 C-4]


@dataclass
class ChlConfig:
    space: str = "log1p"  # (고정) [X-14]
    supervision: str = "none"  # {none, sparse_points, idw_dense, scalar}
    max_dt_hours: float = 6.0
    max_dist_m: float = 100.0


@dataclass
class GeometricAugConfig:
    # ★ (정정 B-27) [0.5, 2.0] -> [1.0, 2.0]. aihub092 원본이 정확히 512² 라 축소 스케일은
    #   전체 샘플의 1/3 에 평균 42 % 결정론적 우/하단 ignore 패딩을 만든다 (05 §5.1.6).
    scale_range: List[float] = field(default_factory=lambda: [1.0, 2.0])
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rot90: bool = True  # pol 존재 시 자동 비활성 (V7) [01 §7.4.1]
    allow_rot90_with_pol: bool = False
    cat_max_ratio: float = 0.75
    cat_max_attempts: int = 10


@dataclass
class PhotometricAugConfig:
    """★ RGB 에만 적용. msi/bio/ir/pol 전면 금지 (01 §7.4.2)."""

    enabled: bool = True
    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.25
    hue: float = 0.05
    p: float = 0.5
    blur_p: float = 0.0
    noise_std: float = 0.0


@dataclass
class RareClassAugConfig:
    ids: List[int] = field(default_factory=list)
    crop_prob: float = 0.0
    min_pixels: int = 0
    attempts: int = 10


@dataclass
class AugmentConfig:
    geometric: GeometricAugConfig = field(default_factory=GeometricAugConfig)
    photometric: PhotometricAugConfig = field(default_factory=PhotometricAugConfig)
    rare_class: RareClassAugConfig = field(default_factory=RareClassAugConfig)


@dataclass
class SamplerConfig:
    kind: str = "shuffle"  # {shuffle, rfs}
    repeat_t: float = 0.05  # RFS threshold [05 §4.3]
    class_stats: Optional[str] = None  # tools/compute_class_stats.py 산출 JSON


@dataclass
class DataConfig:
    dataset: str = "aihub092"  # {aihub092, k235, drone_m3m}
    root: str = "data/aihub092_group"  # group split (누수 없음)
    split_variant: str = "group"  # {asis, group, river_holdout} [01 §7.3]
    num_classes: int = 12  # S0=12 / S1=2 [01 C-04]
    ignore_index: int = IGNORE_INDEX  # (고정) [헌법 C-4]
    modalities: List[str] = field(default_factory=lambda: ["rgb"])
    sensor: str = "none"  # {none, m3m, rededge_mx, rededge_p}
    band_ids: Optional[List[int]] = None  # null -> SENSOR_BAND_IDS[sensor]
    phys_slot_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # [X-04]
    train_size: List[int] = field(default_factory=lambda: [512, 512])
    eval_size: List[int] = field(default_factory=lambda: [1024, 1024])
    num_workers: int = 12
    prefetch_factor: int = 4
    pin_memory: bool = True
    bio: BioConfig = field(default_factory=BioConfig)
    pol: PolConfig = field(default_factory=PolConfig)
    msi: MsiConfig = field(default_factory=MsiConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    chl: ChlConfig = field(default_factory=ChlConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


# ═══════════════════════════════════════════════════════════════════════
#  모델
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class DepthsConfig:
    rgb: List[int] = field(default_factory=lambda: [2, 2, 4, 3])
    spec: List[int] = field(default_factory=lambda: [2, 2, 4, 3])
    phys: List[int] = field(default_factory=lambda: [1, 1, 2, 1])


@dataclass
class PPNConfig:
    use_pol: bool = False  # modalities 에 pol 있으면 자동 true (V6) [X-05]
    always_build: bool = False
    inpaint_ch: int = 16
    dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    gate_a_init: float = 20.0  # a = softplus(a_hat) [헌법 R3]
    gate_b_init: float = -5.0
    rgb_proxy_gate: bool = False  # RGB-only specular 게이트 (opt-in) [01 §5.2]


@dataclass
class SPSConfig:
    num_slots: int = len(MSI_SLOTS)  # 6
    bio_init_gain: float = 4.0  # [U-U5] ablation {1,2,4,8}


@dataclass
class TPSConfig:
    num_slots: int = len(PHYS_SLOTS)  # 4  [X-04]


@dataclass
class ELAConfig:
    attn_stages: List[int] = field(default_factory=lambda: [3, 4])
    head_dim: List[int] = field(default_factory=lambda: [16, 16, 32, 32])
    scales: List[int] = field(default_factory=lambda: [3, 5])
    expand_ratio: int = 4
    eps: float = 1.0e-5


@dataclass
class BioSpecConfig:
    mlp_ratio: List[int] = field(default_factory=lambda: [8, 8, 4, 4])
    se_ratio: int = 4
    dilations: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 2, 4], [1, 2, 3]]
    )
    beta_init: float = 0.1


@dataclass
class PhysLiteConfig:
    expand_ratio: int = 2
    dw_kernel: List[int] = field(default_factory=lambda: [3, 3, 5, 5])
    se_ratio: int = 8
    se_min: int = 8


@dataclass
class BMEFConfig:
    stages: List[int] = field(default_factory=lambda: [1, 2, 3, 4])  # 1 필수 (V19, 정정 A-26)
    stage1_identity: bool = False  # true -> stage1 을 항등(mean-only) 모드로 (03 U-8)
    se_reduction: int = 8
    se_min_channels: int = 8
    logtau_range: float = 4.0  # R
    chan_logtau_range: float = 2.0  # Rp
    r_floor: float = 0.05  # eps_r
    kappa_init: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.018])  # [03 §4.2]
    gpol_pool: str = "area"  # {area, max} [03 U-3]
    norm_mode: str = "l1"  # {l1, l2} [03 U-2]
    norm_layer: str = "bn"  # {bn, gn}
    enable_feedback: bool = True
    warmup_epochs: int = 10  # prec_ramp = min(1, epoch/warmup_epochs)


@dataclass
class DecoderConfig:
    dec_ch: int = DEC_CH  # (고정) [헌법 C-4]
    bd_ch: int = 32
    ppm_branch: int = 96
    dot_scale: Optional[float] = None  # null -> mid**-0.5 = 0.125 [04 §4.2]
    zero_init_residual: bool = True


@dataclass
class HeadsConfig:
    seg_mid: int = 128
    reg_trunk_mid: int = 64
    chl_prior_log1p_mean: float = 1.7559  # bias = softplus⁻¹(·) = 1.5662 [X-10]
    edge_prior_pos: float = 0.03  # [01 M4]
    unc_use_vacuity: bool = False  # [X-12]
    enable_edge_head: bool = True


@dataclass
class SiamConfig:
    enabled: bool = False  # teacher 부재 (헌법 C-5.3) [05 D4]
    teacher: Optional[str] = None
    channels: Dict[str, List[int]] = field(
        default_factory=lambda: {"s3": [160, 320], "s4": [320, 512], "dec": [128, 256]}
    )


@dataclass
class ModelConfig:
    channels: List[int] = field(default_factory=lambda: list(CHANNELS))  # (고정) [헌법 C-4]
    layer_scale_init: List[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.01])
    drop_path_rate: float = 0.1
    norm: str = "bn"  # {bn, syncbn, gn8}
    null_embedding: bool = False  # 기본 비활성 (+1,728 params) [02 §2.6]
    depths: DepthsConfig = field(default_factory=DepthsConfig)
    ppn: PPNConfig = field(default_factory=PPNConfig)
    sps: SPSConfig = field(default_factory=SPSConfig)
    tps: TPSConfig = field(default_factory=TPSConfig)
    ela: ELAConfig = field(default_factory=ELAConfig)
    biospec: BioSpecConfig = field(default_factory=BioSpecConfig)
    physlite: PhysLiteConfig = field(default_factory=PhysLiteConfig)
    bmef: BMEFConfig = field(default_factory=BMEFConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    aux_taps: List[str] = field(default_factory=lambda: ["enc_s8", "enc_s16", "enc_s32"])  # [X-06]
    siam: SiamConfig = field(default_factory=SiamConfig)


# ═══════════════════════════════════════════════════════════════════════
#  손실 · 최적화 · 스케줄 · 런타임
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class LossConfig:
    lambda_seg: float = 1.0  # (고정) [사업문서 s42/43]
    lambda_reg: float = 0.5  # (고정) [사업문서 s42/43]
    lambda_bd: float = 20.0  # [PIDNet]
    lambda_bas: float = 1.0
    lambda_aux: Dict[str, float] = field(
        default_factory=lambda: {"enc_s8": 0.2, "enc_s16": 0.4, "enc_s32": 0.4, "p8": 0.4}
    )
    lambda_siam: float = 0.0
    w_dice: float = 0.4
    ohem_thresh: float = 0.7  # S0=0.7 / S1=0.9
    ohem_keep_frac: float = 0.0625
    bas_tau: float = 0.8
    huber_beta: float = 1.0  # (고정, dense head) [X-11]
    unc_clamp: List[float] = field(default_factory=lambda: [-7.0, 7.0])  # [X-26, 정정 A-20]
    aux_loss_stride: int = 1  # {1, 4} OOM 레버 (V22, 정정 A-32)
    u_warm_frac: float = 0.30
    u_ramp_frac: float = 0.20
    class_weight: Optional[List[float]] = None
    aux_decay: bool = False
    debug_assert: bool = False  # chl 이중-log 하드가드는 이 스위치 안에서만 (정정 A-33)


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 4.0e-4  # from-scratch. S1 파인튜닝은 6.0e-5 [05 §5.1.2]
    lr_encoder: Optional[float] = None  # null -> lr 과 동일 (S0)
    physics_lr_mult: float = 50.0
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1


@dataclass
class ScheduleConfig:
    epochs: int = 60
    batch_size: int = 32  # ★ 추정치. 첫 실행 전 측정 필수 [05 §5.1.8, U-4]
    scheduler: str = "PolynomialLR"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {"power": 0.9})
    warmup_epochs: int = 2  # ★ 1급 키 (정정 A-18) [05 §5.1.3]
    warmup_iters: Optional[int] = None  # null -> round(warmup_epochs × iters_per_epoch)
    warmup_start_factor: float = 1.0e-3
    freeze_stages_epochs: int = 0  # S1 Phase A: 10 [05 §5.2.2]
    iters_per_epoch: Optional[int] = None  # ★ 런타임 유도값 (정정 A-18). resolved.yaml 에 덤프


@dataclass
class EmaConfig:
    enabled: bool = True
    decay: float = 0.9998
    start_epoch: int = 5


@dataclass
class EarlyStoppingConfig:
    metric: str = "val_miou_ema"
    patience: int = 10
    min_delta: float = 1.0e-4
    mode: str = "max"


@dataclass
class ModalityDropoutConfig:
    enabled: bool = False  # S0 에서는 path 가 1개라 무의미
    mode: str = "subset"  # {subset, bernoulli} [X-21]
    p_full: float = 0.5
    p_bernoulli: float = 0.15


@dataclass
class TrainConfig:
    amp: str = "bf16"  # {off, bf16, fp16}
    loss_dtype: str = "fp32"  # (고정) criterion 전체 autocast off
    ema: EmaConfig = field(default_factory=EmaConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    modality_dropout: ModalityDropoutConfig = field(default_factory=ModalityDropoutConfig)
    nonfinite_skip_max_per_epoch: int = 10
    init_from: Optional[str] = None  # 가중치만 (shape-tolerant partial)
    resume_from: Optional[str] = None  # 가중치+optimizer+scheduler+scaler+epoch+rng
    # False 면 cudnn.benchmark 를 켜고 use_deterministic_algorithms 를 끈다.
    # A100 실측(S0-RGB, B=32, 512²): 1128 ms/step → 429 ms/step (2.6배).
    # 대가는 run 간 bit-exact 재현성 상실이다. RNG seed 는 그대로 고정된다.
    deterministic: bool = True


@dataclass
class TTAConfig:
    enabled: bool = False
    scales: List[float] = field(default_factory=lambda: [1.0])
    hflip: bool = False


@dataclass
class EvalConfig:
    miou_convention: str = "exclude_absent"  # (고정) union==0 클래스 제외 [X-20]
    boundary_tolerances: List[int] = field(default_factory=lambda: [1, 3, 5])
    alarm_thresholds: List[float] = field(default_factory=lambda: [15.0, 25.0, 100.0])
    report_splits: List[str] = field(default_factory=lambda: ["asis", "group"])
    tta: TTAConfig = field(default_factory=TTAConfig)


# ═══════════════════════════════════════════════════════════════════════
#  S0-Spec · 배포
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class TransplantConfig:
    enabled: bool = False  # ★ (정정 A-19) 명시적 opt-in 으로만 (V12/V18)
    patch_embed: bool = True  # T4 + T4'(dw 잔차 BN.weight=0) 동시 적용 필수
    raw_bands: bool = True  # T0 실패 시 런타임에 false 로 강등 (V21, 01 T0)
    gate_g1_rmse: float = 0.750
    gate_g2_spearman: float = 0.30
    gate_g3_f1_at15: float = 0.60


@dataclass
class SpecConfig:
    npz: str = "data/cache/k235_ms.npz"
    band_order: List[str] = field(
        default_factory=lambda: ["blue", "green", "red", "rededge1", "nir"]
    )  # 가설 H2 [01 U-1]
    band_order_confirmed: bool = False  # ★ (정정 A-19) 구축활용가이드 확보 시에만 true (V18)
    use_blue: bool = False  # [X-03]
    quality_filter: bool = True  # [X-23]
    train_sites: List[str] = field(
        default_factory=lambda: ["BES", "CNH", "GDK", "HNM", "SHC", "YJD"]
    )
    val_sites: List[str] = field(default_factory=lambda: ["CSC", "JAD"])  # [X-22]
    loso: bool = True
    head_out: int = 1  # 1 -> SmoothL1+rank / 2 -> Huber-NLL [X-09]
    huber_beta: float = 0.3  # [X-11]
    rank_weight: float = 0.2
    rank_margin: float = 0.1
    bio_gain: float = 4.0
    mci_c: Optional[float] = None  # ★ (정정 A-27) null -> BAND_CENTERS_NM[sensor] 에서 유도
    epochs: int = 300
    batch_size: int = 512
    lr: float = 3.0e-3
    weight_decay: float = 1.0e-4
    device: str = "cpu"  # 1,089 params — GPU 불필요
    transplant: TransplantConfig = field(default_factory=TransplantConfig)


@dataclass
class DeployConfig:
    input_hw: List[int] = field(default_factory=lambda: [1024, 1024])  # H/W 정적 [04 §9.3-C]
    opset: int = 17
    dynamic_batch: bool = True
    precision: str = "fp16"  # {fp32, fp16, int8}
    fp32_forced: List[str] = field(default_factory=lambda: list(_FP32_REQUIRED))
    fp16_locked: List[str] = field(
        default_factory=lambda: [
            "chl_head",
            "unc_head",
            "pagfm",
            "encoder.*.litemla.attention",
        ]
    )
    input_names: Optional[List[str]] = None  # null -> active_paths 에서 유도 (X-24, 정정 A-35)
    use_dynamo: bool = False  # ★ (정정 A-29) legacy TorchScript exporter 고정
    chl_max_mgm3: float = 500.0


# ═══════════════════════════════════════════════════════════════════════
#  루트
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class BloomNetConfig:
    """전 설정의 루트. `__post_init__` 에서 유도값 해결 + V1~V22 검증을 수행한다."""

    mode: str = "s0_rgb"  # {s0_rgb, s0_spec, s1_rgb_ms4, s2_full} [헌법 C-3]
    seed: int = 1234
    device: str = "cuda"
    output_dir: str = "outputs"
    run_name: Optional[str] = None  # null -> make_run_name(cfg)
    dry_run: bool = False  # true -> 3 step 만 돌고 종료 (스모크)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)

    # ── YAML 스키마에 포함되지 않는 런타임 플래그 ──────────────────────
    allow_contract_break: bool = field(
        default=False, repr=False, metadata={SCHEMA_EXCLUDE: True}
    )  # V2 (λ 계약) 우회. CLI --allow_contract_break 전용

    def __post_init__(self) -> None:
        self.resolve()
        self.validate()

    # ── 유도값 해결 ────────────────────────────────────────────────────
    def resolve(self) -> "BloomNetConfig":
        """null 로 남은 유도 가능한 값을 채운다 (센서 의존 상수 등)."""
        d = self.data
        # boundary.radius 는 YAML 에서 문자열 키로 올 수 있다 -> int 정규화
        d.boundary.radius = {int(k): int(v) for k, v in d.boundary.radius.items()}
        if d.sensor != "none":
            if d.sensor not in SENSOR_BAND_IDS:
                raise ValueError(
                    f"Unknown data.sensor '{d.sensor}' — {sorted(SENSOR_BAND_IDS)} 중 하나여야 한다"
                )
            if d.band_ids is None:
                d.band_ids = list(SENSOR_BAND_IDS[d.sensor])
            if d.msi.k_sensor is None:
                d.msi.k_sensor = float(K_SENSOR[d.sensor])
            if self.spec.mci_c is None:
                self.spec.mci_c = _mci_c_from_centers(d.sensor)
        return self

    # ── 검증 (06 §4.3 V1~V22 + 타입/범위) ──────────────────────────────
    def validate(self) -> "BloomNetConfig":
        _validate_ranges(self)
        _validate_rules(self)
        return self

    # ── 편의 ───────────────────────────────────────────────────────────
    @property
    def active_paths(self) -> Tuple[str, ...]:
        """modalities -> 활성 path (06 §3.4.1). rgb 는 항상 존재 (V4)."""
        paths = ["rgb"]
        if "msi" in self.data.modalities:
            paths.append("spec")
        if "ir" in self.data.modalities or "pol" in self.data.modalities:
            paths.append("phys")
        return tuple(paths)

    def boundary_radius(self, size: int) -> int:
        """해상도별 경계 반경 (정정 A-28). V10 이 존재를 보장한다."""
        return int(self.data.boundary.radius[int(size)])

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)

    def copy(self) -> "BloomNetConfig":
        return from_dict(self.to_dict(), source="<copy>")


# ═══════════════════════════════════════════════════════════════════════
#  dict ↔ dataclass
# ═══════════════════════════════════════════════════════════════════════
_HINTS_CACHE: Dict[type, Dict[str, Any]] = {}


def _type_hints(dc_type: type) -> Dict[str, Any]:
    """`from __future__ import annotations` 하에서 field.type 은 문자열이므로 실제 타입을 푼다."""
    if dc_type not in _HINTS_CACHE:
        _HINTS_CACHE[dc_type] = get_type_hints(dc_type, globalns=globals())
    return _HINTS_CACHE[dc_type]


def _schema_fields(dc_type: type) -> Dict[str, Any]:
    return {f.name: f for f in fields(dc_type) if not f.metadata.get(SCHEMA_EXCLUDE, False)}


def to_dict(cfg: Any) -> Dict[str, Any]:
    """dataclass 트리 -> 순수 dict (YAML 덤프/round-trip 용). 스키마 제외 필드는 뺀다."""
    out: Dict[str, Any] = {}
    for name, f in _schema_fields(type(cfg)).items():
        value = getattr(cfg, name)
        out[name] = _to_plain(value)
    return out


def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_dict(value)
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def from_dict(data: Mapping[str, Any], *, source: str = "<dict>", **kwargs: Any) -> BloomNetConfig:
    """중첩 dict -> `BloomNetConfig`. 스키마에 없는 키는 **전체 dotted path** 와 함께 거부한다."""
    built = _build(BloomNetConfig, data, prefix="", source=source)
    built.update(kwargs)
    return BloomNetConfig(**built)


def _build(dc_type: type, data: Mapping[str, Any], *, prefix: str, source: str) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(
            f"Config value at '{prefix.rstrip('.') or '<root>'}' in {source} must be a mapping, "
            f"got {type(data).__name__}"
        )
    schema = _schema_fields(dc_type)
    hints = _type_hints(dc_type)
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in schema:
            raise ValueError(
                f"Unknown config key '{prefix}{key}' in {source} "
                f"(허용 키: {', '.join(sorted(schema))})"
            )
        kwargs[key] = _coerce(hints[key], value, path=f"{prefix}{key}", source=source)
    return kwargs


def _coerce(annotation: Any, value: Any, *, path: str, source: str) -> Any:
    """어노테이션에 맞춰 값을 변환/검증한다. 실패 시 ValueError(경로 포함)."""
    origin = get_origin(annotation)

    # Optional[X] / Union
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        if len(args) == 1:
            return _coerce(args[0], value, path=path, source=source)
        return copy.deepcopy(value)

    if is_dataclass(annotation) and isinstance(annotation, type):
        return annotation(**_build(annotation, value, prefix=f"{path}.", source=source))

    if annotation is Any:
        return copy.deepcopy(value)

    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"Config key '{path}' in {source} must be a list, got {value!r}")
        (item_t,) = get_args(annotation) or (Any,)
        return [_coerce(item_t, v, path=f"{path}[{i}]", source=source) for i, v in enumerate(value)]

    if origin in (dict, Dict):
        if not isinstance(value, Mapping):
            raise ValueError(f"Config key '{path}' in {source} must be a mapping, got {value!r}")
        key_t, val_t = get_args(annotation) or (Any, Any)
        out: Dict[Any, Any] = {}
        for k, v in value.items():
            # YAML 매핑 키가 문자열로 올 수 있다 (예: data.boundary.radius 의 "512")
            if key_t is int and isinstance(k, str) and k.lstrip("-").isdigit():
                k = int(k)
            key = _coerce(key_t, k, path=f"{path}.<key>", source=source)
            out[key] = _coerce(val_t, v, path=f"{path}.{k}", source=source)
        return out

    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"Config key '{path}' in {source} must be a bool, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Config key '{path}' in {source} must be an int, got {value!r}")
        return int(value)
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Config key '{path}' in {source} must be a float, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"Config key '{path}' in {source} must be a str, got {value!r}")
        return value
    return copy.deepcopy(value)


def default_config(**kwargs: Any) -> BloomNetConfig:
    """전 기본값 config. `configs/_base.yaml` 의 정본이다."""
    return BloomNetConfig(**kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  YAML 병합 · 로드 · 덤프
# ═══════════════════════════════════════════════════════════════════════
def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """dict 재귀 merge. **리스트는 교체**(병합 아님) — 06 §4.1."""
    out: Dict[str, Any] = {k: copy.deepcopy(v) for k, v in base.items()}
    for k, v in override.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping, got {type(data).__name__}")
    return data


def _load_chain(path: Union[str, Path]) -> Dict[str, Any]:
    """최상위 `_base:` 체인을 따라 병합한 dict 를 돌려준다 (부모 먼저, 자식이 덮어씀)."""
    p = Path(path).expanduser()
    raw = _read_yaml(p)
    parents = raw.pop("_base", None)
    merged: Dict[str, Any] = {}
    if parents is not None:
        if isinstance(parents, str):
            parents = [parents]
        if not isinstance(parents, list):
            raise ValueError(f"'_base' in {p} must be a string or list of strings")
        for parent in parents:
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = p.parent / parent_path
            merged = merge_dicts(merged, _load_chain(parent_path))
    return merge_dicts(merged, raw)


def _leaf_paths(data: Mapping[str, Any], prefix: str = "") -> set:
    """dict 의 모든 dotted 경로(중간 노드 포함)를 수집한다 — V6 '명시 여부' 판정용."""
    out = set()
    for k, v in data.items():
        path = f"{prefix}{k}"
        out.add(path)
        if isinstance(v, Mapping):
            out |= _leaf_paths(v, prefix=f"{path}.")
    return out


def apply_overrides(
    data: Dict[str, Any], overrides: Sequence[str], *, source: str = "--set"
) -> Dict[str, Any]:
    """`--set a.b.c=<yaml>` 형식 오버라이드를 dict 에 적용한다 (값은 `yaml.safe_load` 파싱).

    존재하지 않는 경로는 여기서 새로 만들되, 이후 `from_dict` 의 스키마 검증이 거부한다.
    """
    out = copy.deepcopy(dict(data))
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}' in {source} — expected 'key.path=value'")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}' in {source} — empty key")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:  # pragma: no cover - yaml 파서 오류 경로
            raise ValueError(f"Invalid override value in '{item}' ({source}): {exc}") from exc
        node: Dict[str, Any] = out
        parts = key.split(".")
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value
    return out


def load_config(
    config: Optional[Union[str, Path]] = None,
    *,
    base: Optional[Union[str, Path]] = None,
    overrides: Optional[Sequence[str]] = None,
    allow_contract_break: bool = False,
) -> BloomNetConfig:
    """06 §4.1 로드 규약 구현.

    Args:
        config: 사용자 config YAML (`_base:` 체인 지원). None 이면 base/기본값만 쓴다.
        base: 기저 YAML. None 이면 **dataclass 기본값**이 기저다
              (`configs/_base.yaml` 은 그 덤프본이므로 둘은 동일해야 한다).
        overrides: `--set` 문자열 시퀀스. CLI > YAML > 기본값.
        allow_contract_break: V2(λ 계약) 우회.
    """
    merged: Dict[str, Any] = to_dict(default_config())
    explicit: set = set()

    if base is not None:
        base_data = _load_chain(base)
        # 기저 파일은 **키·타입만** 검증한다. 규칙(V1~V22)은 병합 완료본에만 적용해야 한다
        # (부분 base 가 단독으로는 규칙을 위반해도 자식과 병합하면 유효할 수 있다).
        _build(BloomNetConfig, base_data, prefix="", source=str(base))
        merged = merge_dicts(merged, base_data)

    source = str(config) if config is not None else "<defaults>"
    if config is not None:
        cfg_data = _load_chain(config)
        merged = merge_dicts(merged, cfg_data)
        explicit |= _leaf_paths(cfg_data)

    if overrides:
        merged = apply_overrides(merged, overrides)
        explicit |= {item.partition("=")[0].strip() for item in overrides}
        source = f"{source} + --set" if config is not None else "--set"

    _autosync(merged, explicit)
    return from_dict(merged, source=source, allow_contract_break=allow_contract_break)


def _autosync(merged: Dict[str, Any], explicit: set) -> None:
    """V6 자동 동기화: `model.ppn.use_pol` 을 modalities 에서 유도한다 (명시 시엔 손대지 않음)."""
    key = "model.ppn.use_pol"
    if key in explicit:
        return
    modalities = merged.get("data", {}).get("modalities", [])
    if not isinstance(modalities, list):
        return
    merged.setdefault("model", {}).setdefault("ppn", {})["use_pol"] = "pol" in modalities


def to_yaml(cfg: BloomNetConfig) -> str:
    return yaml.safe_dump(
        to_dict(cfg), sort_keys=False, allow_unicode=True, default_flow_style=False
    )


def dump_config(cfg: BloomNetConfig, path: Union[str, Path]) -> Path:
    """병합 완료본 전체를 `config.resolved.yaml` 로 덤프한다 (06 §4.1 step 6, 재현성)."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_yaml(cfg), encoding="utf-8")
    return p


# ═══════════════════════════════════════════════════════════════════════
#  런타임 유도 (V15) · run_name
# ═══════════════════════════════════════════════════════════════════════
def resolve_schedule(cfg: BloomNetConfig, iters_per_epoch: int) -> BloomNetConfig:
    """선택된 split 의 `len(train_loader)` 로 warmup/total iteration 을 유도하고 V15 를 강제한다.

    (정정 A-18) 절대 iteration 을 스키마에 두지 않는다 — 기본 분할이 group(67,438장)인데
    asis(85,635장) 기준 상수를 쓰면 warmup 이 2 epoch 이 아니게 되고 total_iters 가 어긋나
    학습 종료 시점 LR 이 0 으로 수렴하지 않는다.
    """
    if not isinstance(iters_per_epoch, int) or isinstance(iters_per_epoch, bool):
        raise ValueError(f"iters_per_epoch must be an int, got {iters_per_epoch!r}")
    if iters_per_epoch < 1:
        raise ValueError(f"iters_per_epoch must be >= 1, got {iters_per_epoch}")

    s = cfg.schedule
    derived_warmup = int(round(s.warmup_epochs * iters_per_epoch))
    if s.warmup_iters is None:
        s.warmup_iters = derived_warmup
    elif s.warmup_iters != derived_warmup:
        raise ValueError(
            "V15: schedule.warmup_iters 는 round(warmup_epochs × iters_per_epoch) 와 일치해야 한다 "
            f"(주어진 값 {s.warmup_iters} != 유도값 {derived_warmup} "
            f"= round({s.warmup_epochs} × {iters_per_epoch}))"
        )

    total = s.epochs * iters_per_epoch - s.warmup_iters
    given = s.scheduler_kwargs.get("total_iters")
    if given is not None and int(given) != total:
        raise ValueError(
            "V15: schedule.scheduler_kwargs.total_iters 는 "
            f"epochs × iters_per_epoch − warmup_iters 와 일치해야 한다 ({given} != {total})"
        )
    s.scheduler_kwargs["total_iters"] = total
    s.iters_per_epoch = iters_per_epoch

    if not (s.warmup_iters < total):
        raise ValueError(f"V15: warmup_iters({s.warmup_iters}) < total_iters({total}) 여야 한다")
    return cfg


def make_run_name(cfg: BloomNetConfig, *, timestamp: Optional[str] = None) -> str:
    """`f"{mode}_{git_rev[:7]}_{timestamp}"` (06 §4.2). 결정론이 필요하면 timestamp 를 준다."""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{cfg.mode}_{short_revision()}_{ts}"


# ═══════════════════════════════════════════════════════════════════════
#  CLI (사전 파서 + 1급 플래그 15종 + --set)
# ═══════════════════════════════════════════════════════════════════════
#  1급 플래그 -> dotted path (06 §4.1 step 4). 내부적으로 --set 으로 변환된다.
FIRST_CLASS_FLAGS: Dict[str, str] = {
    "run_name": "run_name",
    "output_dir": "output_dir",
    "data_root": "data.root",
    "batch_size": "schedule.batch_size",
    "lr": "optim.lr",
    "epochs": "schedule.epochs",
    "seed": "seed",
    "device": "device",
    "num_workers": "data.num_workers",
    "amp": "train.amp",
    "resume_from": "train.resume_from",
    "init_from": "train.init_from",
    "num_classes": "data.num_classes",
    "mode": "mode",
    "dry_run": "dry_run",
}


def build_parser(description: str = "BloomNet") -> argparse.ArgumentParser:
    """1급 플래그 + `--set` 파서. `--config`/`--base` 는 사전 파서에서 처리한다."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="사용자 config YAML")
    p.add_argument("--base", default=DEFAULT_BASE_PATH, help="기저 config YAML")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="중첩 키 오버라이드 (반복 가능). 값은 yaml.safe_load 로 파싱",
    )
    p.add_argument("--run_name", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--amp", nargs="?", const="bf16", default=None, choices=["off", "bf16", "fp16"])
    p.add_argument("--no_amp", dest="amp", action="store_const", const="off")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--init_from", default=None)
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--mode", default=None, choices=list(MODES))
    p.add_argument("--dry_run", action="store_true", default=None)
    p.add_argument(
        "--allow_contract_break",
        action="store_true",
        help="V2(λ_seg=1.0 / λ_reg=0.5, 사업문서 확정값) 우회",
    )
    return p


def parse_cli(argv: Optional[Sequence[str]] = None) -> Tuple[BloomNetConfig, argparse.Namespace]:
    """사전 파서 → YAML 병합 → 1급 플래그를 `--set` 으로 변환 → config 생성."""
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = list(args.overrides)
    for dest, dotted in FIRST_CLASS_FLAGS.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        overrides.append(f"{dotted}={yaml.safe_dump(value).strip()}")

    base: Optional[Union[str, Path]] = args.base
    if base is not None and not Path(base).expanduser().is_file():
        if str(base) != DEFAULT_BASE_PATH:
            raise FileNotFoundError(f"--base file not found: {base}")
        base = None  # 기본 경로가 없으면 dataclass 기본값이 기저다

    cfg = load_config(
        args.config,
        base=base,
        overrides=overrides,
        allow_contract_break=bool(args.allow_contract_break),
    )
    return cfg, args


# ═══════════════════════════════════════════════════════════════════════
#  검증 구현
# ═══════════════════════════════════════════════════════════════════════
def _mci_c_from_centers(sensor: str) -> float:
    """c = (λ_RE1 − λ_R) / (λ_NIR − λ_R)  (01 §4.2.2, X-01).

    구현 정본은 `data/indices.mci_coefficient` 이지만 config(L1) 는 같은 레벨을 import 할 수
    없으므로(06 §2.1 규칙 2) 동일 식을 여기서 계산한다. 리터럴은 상수에서만 온다.
    """
    centers = BAND_CENTERS_NM[sensor]
    return (centers["rededge1"] - centers["red"]) / (centers["nir"] - centers["red"])


def _gn_num_groups(num_channels: int) -> int:
    """gn8 정책 (V20 / 정정 A-37, B-1): `num_groups = gcd(8, C)`. 빌더 정본은 modules/common.py."""
    return math.gcd(8, num_channels)


def _norm_group_widths(cfg: BloomNetConfig) -> Dict[str, int]:
    """config 가 함의하는 norm 채널 폭 (V20 점검 대상). BioGate C_r 이 유일한 비8배수다."""
    m = cfg.model
    widths: Dict[str, int] = {}
    for i, c in enumerate(m.channels, start=1):
        widths[f"stage{i}"] = int(c)
        widths[f"biogate_c_r.stage{i}"] = max(8, int(c) // 8)  # 02 §4: C_r = max(8, C//8)
        widths[f"bmef_se.stage{i}"] = max(m.bmef.se_min_channels, int(c) // m.bmef.se_reduction)
        widths[f"physlite_se.stage{i}"] = max(m.physlite.se_min, int(c) // m.physlite.se_ratio)
    widths["decoder.dec_ch"] = m.decoder.dec_ch
    widths["decoder.bd_ch"] = m.decoder.bd_ch
    widths["decoder.ppm_branch"] = m.decoder.ppm_branch
    widths["heads.seg_mid"] = m.heads.seg_mid
    widths["heads.reg_trunk_mid"] = m.heads.reg_trunk_mid
    return widths


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _in_choices(value: Any, choices: Sequence[Any], path: str) -> None:
    _check(value in choices, f"Config '{path}' must be one of {list(choices)}, got {value!r}")


def _in_range(value: float, lo: float, hi: float, path: str) -> None:
    _check(lo <= value <= hi, f"Config '{path}' must be in [{lo}, {hi}], got {value!r}")


def _validate_ranges(cfg: BloomNetConfig) -> None:
    """타입·범위·열거값 검증 (06 §4.3 표제)."""
    d, m = cfg.data, cfg.model
    _in_choices(cfg.mode, MODES, "mode")
    _check(isinstance(cfg.seed, int) and cfg.seed >= 0, "Config 'seed' must be a non-negative int")
    _check(
        cfg.device.startswith(("cuda", "cpu", "mps")),
        f"Config 'device' must start with cuda/cpu/mps, got {cfg.device!r}",
    )
    _in_choices(d.dataset, ("aihub092", "k235", "drone_m3m"), "data.dataset")
    _in_choices(d.split_variant, ("asis", "group", "river_holdout"), "data.split_variant")
    _in_choices(d.sensor, ("none", *sorted(SENSOR_BAND_IDS)), "data.sensor")
    _check(d.num_classes >= 2, f"Config 'data.num_classes' must be >= 2, got {d.num_classes}")
    _check(len(d.modalities) == len(set(d.modalities)), "data.modalities 에 중복이 있다")
    for mod in d.modalities:
        _in_choices(mod, MODALITY_ORDER, "data.modalities[*]")
    if d.band_ids is not None:
        for b in d.band_ids:
            _check(
                0 <= b < len(MSI_SLOTS),
                f"data.band_ids 원소는 0..{len(MSI_SLOTS) - 1} 여야 한다 (got {b})",
            )
    for s in d.phys_slot_ids:
        _check(
            0 <= s < len(PHYS_SLOTS),
            f"data.phys_slot_ids 원소는 0..{len(PHYS_SLOTS) - 1} 여야 한다 (got {s})",
        )
    _check(len(d.train_size) == 2 and len(d.eval_size) == 2, "train_size/eval_size 는 (H, W) 2원소")
    _in_choices(d.bio.kind, ("mci", "gndvi", "rgb_proxy", "none"), "data.bio.kind")
    _in_choices(d.bio.source, ("none", "msi", "rgb_proxy"), "data.bio.source")
    _in_choices(d.pol.encoding, ("sincos", "stokes"), "data.pol.encoding")
    _in_choices(d.boundary.source, ("dataset", "criterion"), "data.boundary.source")
    _check(d.boundary.stride == 4, "data.boundary.stride 는 4 로 고정 (헌법 C-4)")
    _in_choices(d.chl.space, ("log1p",), "data.chl.space")
    _in_choices(
        d.chl.supervision, ("none", "sparse_points", "idw_dense", "scalar"), "data.chl.supervision"
    )
    _in_choices(d.sampler.kind, ("shuffle", "rfs"), "data.sampler.kind")
    for p_name in ("hflip_p", "vflip_p", "cat_max_ratio"):
        _in_range(
            getattr(d.augment.geometric, p_name), 0.0, 1.0, f"data.augment.geometric.{p_name}"
        )
    _check(
        len(d.augment.geometric.scale_range) == 2
        and 0 < d.augment.geometric.scale_range[0] <= d.augment.geometric.scale_range[1],
        "data.augment.geometric.scale_range 는 0 < lo <= hi 인 2원소여야 한다",
    )
    _in_range(d.augment.photometric.p, 0.0, 1.0, "data.augment.photometric.p")
    _in_range(d.sampler.repeat_t, 0.0, 1.0, "data.sampler.repeat_t")

    _in_choices(m.norm, ("bn", "syncbn", "gn8"), "model.norm")
    _in_range(m.drop_path_rate, 0.0, 1.0, "model.drop_path_rate")
    for name in ("rgb", "spec", "phys"):
        _check(
            len(getattr(m.depths, name)) == 4,
            f"model.depths.{name} 는 stage 4개여야 한다",
        )
    _check(len(m.layer_scale_init) == 4, "model.layer_scale_init 은 stage 4개여야 한다")
    _check(m.sps.num_slots == len(MSI_SLOTS), f"model.sps.num_slots 는 {len(MSI_SLOTS)} 고정")
    _check(m.tps.num_slots == len(PHYS_SLOTS), f"model.tps.num_slots 는 {len(PHYS_SLOTS)} 고정 (X-04)")
    for st in m.ela.attn_stages:
        _in_choices(st, (1, 2, 3, 4), "model.ela.attn_stages[*]")
    for st in m.bmef.stages:
        _in_choices(st, (1, 2, 3, 4), "model.bmef.stages[*]")
    _in_choices(m.bmef.gpol_pool, ("area", "max"), "model.bmef.gpol_pool")
    _in_choices(m.bmef.norm_mode, ("l1", "l2"), "model.bmef.norm_mode")
    _in_choices(m.bmef.norm_layer, ("bn", "gn"), "model.bmef.norm_layer")
    _check(len(m.bmef.kappa_init) == 3, "model.bmef.kappa_init 은 (rgb, spec, phys) 3원소")
    _check(m.bmef.warmup_epochs >= 1, "model.bmef.warmup_epochs >= 1 (prec_ramp 분모)")
    _in_range(m.heads.edge_prior_pos, 1e-6, 0.5, "model.heads.edge_prior_pos")

    lo = cfg.loss
    for name in ("lambda_bd", "lambda_bas", "lambda_siam", "w_dice"):
        _check(getattr(lo, name) >= 0.0, f"loss.{name} 는 0 이상이어야 한다")
    _in_range(lo.ohem_thresh, 0.0, 1.0, "loss.ohem_thresh")
    _in_range(lo.ohem_keep_frac, 0.0, 1.0, "loss.ohem_keep_frac")
    _in_range(lo.bas_tau, 0.0, 1.0, "loss.bas_tau")
    _in_range(lo.u_warm_frac, 0.0, 1.0, "loss.u_warm_frac")
    _in_range(lo.u_ramp_frac, 0.0, 1.0, "loss.u_ramp_frac")
    _check(
        len(lo.unc_clamp) == 2 and lo.unc_clamp[0] < lo.unc_clamp[1],
        "loss.unc_clamp 은 (min, max) 2원소",
    )
    _check(
        tuple(lo.unc_clamp) == (-7.0, 7.0),
        "loss.unc_clamp 은 UncHead S_MIN/S_MAX 와 같은 [-7, 7] 이어야 한다 (X-26, 정정 A-20)",
    )

    op = cfg.optim
    _in_choices(op.name, ("adamw", "adam", "sgd"), "optim.name")
    _check(op.lr > 0, "optim.lr > 0")
    _check(op.lr_encoder is None or op.lr_encoder > 0, "optim.lr_encoder 는 null 또는 양수")
    _check(op.grad_accum_steps >= 1, "optim.grad_accum_steps >= 1")
    _check(len(op.betas) == 2, "optim.betas 는 2원소")

    s = cfg.schedule
    _check(s.epochs >= 1, "schedule.epochs >= 1")
    _check(s.batch_size >= 1, "schedule.batch_size >= 1")
    _check(s.warmup_epochs >= 0, "schedule.warmup_epochs >= 0")
    _check(s.warmup_epochs < s.epochs, "schedule.warmup_epochs < epochs")
    _check(
        s.warmup_iters is None or s.warmup_iters >= 0, "schedule.warmup_iters 는 null 또는 0 이상"
    )
    _check(s.freeze_stages_epochs >= 0, "schedule.freeze_stages_epochs >= 0")
    _in_range(s.warmup_start_factor, 0.0, 1.0, "schedule.warmup_start_factor")

    t = cfg.train
    _in_choices(t.amp, ("off", "bf16", "fp16"), "train.amp")
    _in_choices(t.loss_dtype, ("fp32",), "train.loss_dtype")
    _in_range(t.ema.decay, 0.0, 1.0, "train.ema.decay")
    _in_choices(t.early_stopping.mode, ("max", "min"), "train.early_stopping.mode")
    _in_choices(t.modality_dropout.mode, ("subset", "bernoulli"), "train.modality_dropout.mode")
    _in_range(t.modality_dropout.p_full, 0.0, 1.0, "train.modality_dropout.p_full")
    _in_range(t.modality_dropout.p_bernoulli, 0.0, 1.0, "train.modality_dropout.p_bernoulli")

    _in_choices(cfg.eval.miou_convention, ("exclude_absent",), "eval.miou_convention")
    _in_choices(cfg.spec.head_out, (1, 2), "spec.head_out")
    _check(cfg.spec.epochs >= 1 and cfg.spec.batch_size >= 1, "spec.epochs/batch_size >= 1")
    _in_choices(cfg.deploy.precision, ("fp32", "fp16", "int8"), "deploy.precision")
    _check(cfg.deploy.opset >= 11, "deploy.opset >= 11")
    _check(len(cfg.deploy.input_hw) == 2, "deploy.input_hw 는 (H, W) 2원소")


def _validate_rules(cfg: BloomNetConfig) -> None:
    """06 §4.3 V1~V22."""
    d, m, lo = cfg.data, cfg.model, cfg.loss

    # V1 — 입력 크기는 32의 배수 (04 §1.2)
    for name, hw in (("train_size", d.train_size), ("eval_size", d.eval_size),
                     ("deploy.input_hw", cfg.deploy.input_hw)):
        for v in hw:
            _check(
                int(v) % SIZE_DIVISOR == 0 and int(v) > 0,
                f"V1: {name} 의 H,W 는 {SIZE_DIVISOR} 의 배수여야 한다 (got {hw})",
            )

    # V2 — 사업문서 확정 λ (헌법 C-1)
    if not cfg.allow_contract_break:
        _check(
            lo.lambda_seg == 1.0 and lo.lambda_reg == 0.5,
            "V2: 사업문서 확정값 변경 금지 — 헌법 C-1 (lambda_seg=1.0, lambda_reg=0.5). "
            "--allow_contract_break 로만 우회한다",
        )

    # V3 — 헌법 C-4 고정값
    _check(d.ignore_index == IGNORE_INDEX, f"V3: data.ignore_index 는 {IGNORE_INDEX} 고정 (헌법 C-4)")
    _check(tuple(m.channels) == CHANNELS, f"V3: model.channels 는 {list(CHANNELS)} 고정 (헌법 C-4)")
    _check(m.decoder.dec_ch == DEC_CH, f"V3: model.decoder.dec_ch 는 {DEC_CH} 고정 (헌법 C-4)")

    # V4 — RGB 없는 모드는 헌법 C-3 에 없다 (01 A2)
    _check("rgb" in d.modalities, "V4: 'rgb' 는 data.modalities 에 반드시 있어야 한다 (01 A2)")

    # V5 — (정정 A-11 완화) bio 는 msi 의 부속이거나 rgb 프록시여야 한다
    if "bio" in d.modalities:
        _check(
            "msi" in d.modalities or d.bio.source == "rgb_proxy",
            "V5: 'bio' 는 'msi' 와 함께 있거나 data.bio.source == 'rgb_proxy' 여야 한다 (01 A3)",
        )

    # V6 — use_pol 자동 동기화 (명시 시 불일치는 에러) [X-05]
    _check(
        m.ppn.use_pol == ("pol" in d.modalities),
        f"V6: model.ppn.use_pol({m.ppn.use_pol}) 는 ('pol' in data.modalities)"
        f"({'pol' in d.modalities}) 와 같아야 한다 (X-05)",
    )

    # V7 — pol 존재 시 rot90 강제 비활성 (경고 후 수정) [01 §7.4.1]
    g = d.augment.geometric
    if g.rot90 and "pol" in d.modalities and not g.allow_rot90_with_pol:
        warnings.warn(
            "V7: pol 모달이 있으므로 rot90 을 비활성화한다 (AoLP 회전 변환 미구현 경로 차단, "
            "01 §7.4.1). allow_rot90_with_pol=true 로 명시하면 유지된다.",
            RuntimeWarning,
            stacklevel=2,
        )
        g.rot90 = False

    # V8 — bio.source == msi 의 전제
    if d.bio.source == "msi":
        _check(
            "msi" in d.modalities and d.sensor != "none",
            "V8: data.bio.source == 'msi' 이면 modalities 에 msi 가 있고 sensor != none 이어야 한다",
        )

    # V9 — aux tap 계약 [X-06]
    _check(
        set(m.aux_taps) <= set(AUX_TAP_STRIDE),
        f"V9: model.aux_taps ⊆ {sorted(AUX_TAP_STRIDE)} (got {m.aux_taps})",
    )
    missing = [t for t in m.aux_taps if t not in lo.lambda_aux]
    _check(not missing, f"V9: loss.lambda_aux 에 tap {missing} 이 없다 (X-06)")

    # V10 — (정정 A-28) boundary.radius 는 해상도 매핑
    _check(
        isinstance(d.boundary.radius, dict) and len(d.boundary.radius) > 0,
        "V10: data.boundary.radius 는 해상도 매핑 {512: 1, 1024: 2} 이어야 한다 (스칼라 금지)",
    )
    for name, hw in (("train_size", d.train_size), ("eval_size", d.eval_size)):
        _check(
            int(hw[0]) in d.boundary.radius,
            f"V10: data.boundary.radius 에 {name}[0]={hw[0]} 키가 없다 (01 C-10)",
        )

    # V11 — teacher 파일 존재 (헌법 C-5.3, 다운로드 금지)
    if m.siam.enabled:
        _check(
            m.siam.teacher is not None and Path(m.siam.teacher).expanduser().is_file(),
            f"V11: model.siam.enabled 이면 teacher 파일이 존재해야 한다 (got {m.siam.teacher!r})",
        )

    # V12 — (정정 A-11 교체) 이식은 msi 기반 mci 열에서만
    if cfg.spec.transplant.enabled:
        _check(
            d.bio.source == "msi" and d.bio.kind == "mci",
            "V12: 01 §6.7 T6 — S0 의 bio 열이 rgb_proxy 였다면 이식 금지 "
            "(spec.transplant.enabled ⟹ data.bio.source == 'msi' and data.bio.kind == 'mci')",
        )

    # V13 — (정정 A-22 강화) fp16 경로는 BMEF·LiteMLA 를 fp32 로 강제
    if cfg.train.amp == "fp16" or cfg.deploy.precision in ("fp16", "int8"):
        missing_fp32 = [k for k in _FP32_REQUIRED if k not in cfg.deploy.fp32_forced]
        _check(
            not missing_fp32,
            f"V13: fp16/int8 경로에서는 deploy.fp32_forced 에 {missing_fp32} 가 있어야 한다 "
            "(03 §5.3 BMEF subnormal + 02 §3.3 LiteMLA kv 누산 overflow)",
        )

    # V14 — mode ↔ modalities 정합 (경고. 명시적 override 허용)
    expected = _MODE_MODALITIES.get(cfg.mode)
    if expected is not None and tuple(d.modalities) != expected:
        warnings.warn(
            f"V14: mode='{cfg.mode}' 의 표준 modalities 는 {list(expected)} 인데 "
            f"{d.modalities} 가 지정됐다 (의도한 override 인지 확인).",
            RuntimeWarning,
            stacklevel=2,
        )

    # V15 — 스케줄 유도값. iters_per_epoch 는 런타임에만 알 수 있으므로 여기서는 부분 검증만
    s = cfg.schedule
    total = s.scheduler_kwargs.get("total_iters")
    if s.warmup_iters is not None and total is not None:
        _check(
            s.warmup_iters < int(total),
            f"V15: warmup_iters({s.warmup_iters}) < total_iters({total}) 여야 한다",
        )
    if s.iters_per_epoch is not None:
        resolve_schedule(cfg, int(s.iters_per_epoch))

    # V16 — (정정 A-11 신설) RE/NIR 기반 지수는 RGB 프록시로 정의할 수 없다
    if d.bio.kind in ("mci", "gndvi") and (d.bio.source != "none" or "bio" in d.modalities):
        _check(
            d.bio.source == "msi",
            f"V16: RE/NIR 기반 지수는 RGB 프록시로 정의할 수 없다 "
            f"(kind='{d.bio.kind}' ⟹ source == 'msi', got '{d.bio.source}')",
        )

    # V17 — (정정 A-11 신설) rgb_proxy 사용 사실을 소실시키지 않는다
    if d.bio.source == "rgb_proxy":
        _check(
            d.bio.kind == "rgb_proxy",
            "V17: data.bio.source == 'rgb_proxy' 이면 kind 도 'rgb_proxy' 여야 한다 "
            "(dataset 이 meta['bio_kind'] 를 강제 기록한다)",
        )

    # V18 — (정정 A-19 신설) 밴드 순서 확정 전 235 가중치 전이 금지
    if cfg.spec.transplant.enabled:
        _check(
            cfg.spec.band_order_confirmed,
            "V18: 02 §12-U2 / 05 §5.3.2 — 밴드 순서 확정 전 235 가중치 전이 금지 "
            "(spec.band_order_confirmed 는 구축활용가이드 확보 시에만 true)",
        )

    # V19 — (정정 A-26 신설) stage1 BMEF 는 필수
    _check(
        1 in m.bmef.stages,
        "V19: stage1 BMEF 없이는 PIDDecoder 의 F_fused^(1) 이 정의되지 않는다 "
        "(ablation 은 model.bmef.stage1_identity=true 로 표현한다)",
    )

    # V20 — (정정 A-37 신설) gn8 = GroupNorm(gcd(8, C), C)
    if m.norm == "gn8":
        for name, width in _norm_group_widths(cfg).items():
            _check(
                width >= 1,
                f"V20: norm 채널 폭 '{name}' 이 {width} 다 — GroupNorm 을 만들 수 없다",
            )
            groups = _gn_num_groups(width)
            _check(
                width % groups == 0,
                f"V20: gn8 은 num_groups=gcd(8,C) 를 쓴다. '{name}' C={width} 에서 "
                f"{width} % {groups} != 0",
            )

    # V21 — (정정 A-40 신설, X-28) msi 방사 스케일 정합
    if d.sensor != "none":
        _check(
            d.msi.k_sensor is not None and d.msi.k_sensor > 0,
            f"V21: data.sensor != 'none' 이면 data.msi.k_sensor > 0 이어야 한다 "
            f"(got {d.msi.k_sensor!r}; null 이면 K_SENSOR[sensor] 로 유도된다)",
        )
    # V21 의 T0(|Δlog10 median| < 0.5) 는 실데이터가 필요하므로 로더/이식 단계에서 검사한다.

    # V22 — (정정 A-32 신설) OOM 레버
    _check(
        lo.aux_loss_stride in (1, 4),
        f"V22: loss.aux_loss_stride 는 {{1, 4}} 여야 한다 (got {lo.aux_loss_stride}) — "
        "05 §2.3.2 의 criterion 활성 메모리 레버",
    )


# ═══════════════════════════════════════════════════════════════════════
#  CLI 진입점 — configs/_base.yaml 생성용
# ═══════════════════════════════════════════════════════════════════════
def _main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - 수동 도구
    p = argparse.ArgumentParser(description="BloomNet config 도구")
    p.add_argument("--dump-base", action="store_true", help="기본값 전체를 YAML 로 출력")
    p.add_argument("--out", default=None, help="출력 파일 (없으면 stdout)")
    args = p.parse_args(argv)
    if args.dump_base:
        text = to_yaml(default_config())
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0
    p.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
