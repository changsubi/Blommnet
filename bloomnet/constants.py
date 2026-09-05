"""BloomNet 전 리터럴의 단일 출처 (06 §3.1, 레벨 L-1).

레벨 규약 (06 §2.1 규칙 2 + 정정 A-23):
    이 파일은 `bloomnet.*` 를 **하나도 import 하지 않는다**. 따라서 어느 레벨의 파일이든
    이 파일을 import 할 수 있다. 리터럴 복제는 금지 — 값이 필요하면 여기서 가져온다.

주요 정정 반영:
    A-27  MCI_C["rededge_p"] 0.475410 -> 0.338983
    A-39  SPLIT_* 6종 신설 (분할 알고리즘 재작성)
    A-40  K_SENSOR / MSI_MEDIAN_RANGE 신설 (msi 방사 스케일 R4')
    A-25  PHYS present 는 slot 집합 AND (규칙 자체는 data/bundle.py)
    I-6/B-21  AUX_TAP_MID 리터럴 동결
"""

from __future__ import annotations

from typing import Dict, Final, Tuple

import torch

# ─── 모달 계약 (01 C-01) ────────────────────────────────────────────────
MODALITY_ORDER: Final[Tuple[str, ...]] = ("rgb", "msi", "bio", "ir", "pol")  # avail 인덱스 0..4
MODALITY_CHANNELS: Final[Dict[str, int]] = {"rgb": 3, "msi": -1, "bio": 2, "ir": 1, "pol": 3}
#   msi = -1 : 센서 의존(4/5/6). canonical scatter 후에는 항상 MSI_SLOTS(=6)
PATHS: Final[Tuple[str, str, str]] = ("rgb", "spec", "phys")  # present 인덱스 0..2 (03)

IMAGENET_MEAN: Final[torch.Tensor] = torch.tensor(
    (0.485, 0.456, 0.406), dtype=torch.float32
).view(3, 1, 1)  # 헌법 C-4, aihub092 bit-exact
IMAGENET_STD: Final[torch.Tensor] = torch.tensor(
    (0.229, 0.224, 0.225), dtype=torch.float32
).view(3, 1, 1)
IGNORE_INDEX: Final[int] = 255  # 헌법 C-4

# ─── 텐서 스케줄 (헌법 C-4) ─────────────────────────────────────────────
CHANNELS: Final[Tuple[int, int, int, int]] = (32, 64, 160, 320)
STRIDES: Final[Tuple[int, int, int, int]] = (4, 8, 16, 32)
DEC_CH: Final[int] = 128
BD_CH: Final[int] = 32
SIZE_DIVISOR: Final[int] = 32  # H, W ≡ 0 (mod 32)  (04 §1.2)

# ─── canonical slot 사전 (02 §2.1, X-04 반영) ───────────────────────────
MSI_SLOTS: Final[Tuple[str, ...]] = (
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "nir",
)  # 6
PHYS_SLOTS: Final[Tuple[str, ...]] = ("ir", "dolp", "aolp_sin", "aolp_cos")  # 4  ★X-04
MSI_SLOT_ID: Final[Dict[str, int]] = {n: i for i, n in enumerate(MSI_SLOTS)}
PHYS_SLOT_ID: Final[Dict[str, int]] = {n: i for i, n in enumerate(PHYS_SLOTS)}

SENSOR_BAND_IDS: Final[Dict[str, Tuple[int, ...]]] = {
    "m3m": (1, 2, 3, 5),  # G560 R650 RE730 NIR860        결측 slot 0, 4
    "rededge_mx": (1, 2, 3, 5),  # ★X-03: use_blue=False 기본. blue 포함 시 (0,1,2,3,5)
    "rededge_p": (0, 1, 2, 3, 4, 5),
}
BAND_CENTERS_NM: Final[Dict[str, Dict[str, float]]] = {
    "m3m": {"green": 560, "red": 650, "rededge1": 730, "nir": 860},
    "rededge_mx": {"blue": 475, "green": 560, "red": 668, "rededge1": 717, "nir": 840},
    "rededge_p": {
        "blue": 443,
        "green": 560,
        "red": 665,
        "rededge1": 705,
        "rededge2": 740,
        "nir": 783,
    },
}
MCI_C: Final[Dict[str, float]] = {"m3m": 0.380952, "rededge_mx": 0.284884, "rededge_p": 0.338983}
#   ★ (정정 A-27) rededge_p 0.475410 -> 0.338983.
#   c = (λ_RE1 − λ_R) / (λ_NIR − λ_R).  구현 정본은 data/indices.mci_coefficient 런타임 계산이며
#   이 dict 는 회귀 대조용이다 (06 §3.1 각주).
#     m3m        (730−650)/(860−650) = 80/210 = 0.3809524
#     rededge_mx (717−668)/(840−668) = 49/172 = 0.2848837
#     rededge_p  (705−665)/(783−665) = 40/118 = 0.3389831

# ─── msi 방사 스케일 정합 (01 §2.6 R4', ★X-28 / 정정 A-40) ──────────────
K_SENSOR: Final[Dict[str, float]] = {"m3m": 3.5e3, "rededge_mx": 1.0, "rededge_p": 1.0}
#   rho = rho_rel · K_SENSOR[sensor].  235(RedEdge)는 이미 절대 반사율이므로 1.0.
#   m3m 3.5e3 은 **잠정값** [01 U-11] — 밴드별 비(G 6,987 / R 4,979 / RE 1,693 / NIR 2,667)의
#   기하평균 3,540 에서 온 자릿수 앵커다. tools/calibrate_k_sensor.py 로 재산출 후 갱신한다.
MSI_MEDIAN_RANGE: Final[Tuple[float, float]] = (1e-3, 1.0)  # 로더 하드 검증 [H1]

# ─── 분할 (01 §7.3, ★정정 A-1) ──────────────────────────────────────────
SPLIT_SEED: Final[int] = 20260731
SPLIT_TARGET: Final[Tuple[float, float, float]] = (0.70, 0.15, 0.15)
SPLIT_TOL: Final[float] = 0.02  # |achieved − target| 상한. 완화 불가
SPLIT_BAND: Final[float] = 0.50  # 클래스 출현율 허용 배율 [0.5, 1.5]
SPLIT_MIN_GROUPS: Final[int] = 3  # split 당 클래스별 최소 독립 군 수
SPLIT_RETRIES: Final[int] = 8

# ─── AI Hub 092 (01 §7.2) ───────────────────────────────────────────────
SCENES: Final[Tuple[str, ...]] = ("01.한강", "02.낙동강", "03.금강", "04.영산강", "05.새만금")
STEM_RE: Final[str] = (
    r"^L(?P<river>\d{2})_(?P<admin>\d{5})_(?P<alt>\d+)_(?P<date>\d{8})"
    r"_(?P<line>N\d{2})_(?P<seq>\d+)$"
)
FINE_CLASS_NAMES: Final[Tuple[str, ...]] = (
    "background",
    "밭_논",
    "잔재물",
    "배수로",
    "비닐하우스",
    "과수원",
    "축사",
    "야적퇴비_가축분뇨",
    "목장",
    "분뇨개별처리시설",
    "부유쓰레기",
    "연못",
)
TRANSFER_CLASS_IDS: Final[Tuple[int, ...]] = (3, 4, 10, 11)  # transfer_score = mean IoU (01 §3.5)
S1_CLASS_NAMES: Final[Tuple[str, ...]] = ("background", "algae")

# ─── 235 (01 §6, X-22/X-23) ─────────────────────────────────────────────
K235_BAND_ORDER_DEFAULT: Final[Tuple[str, ...]] = (
    "blue",
    "green",
    "red",
    "rededge1",
    "nir",
)  # 가설 H2, [U-1]
K235_QC: Final[str] = "WaterTemp <= 40.0 and ODOMg <= 90.0"  # 23,790 → 21,511
K235_LOG1P_MEAN: Final[float] = 1.7559
K235_LOG1P_STD: Final[float] = 0.8265
K235_VAL_SITES: Final[Tuple[str, ...]] = ("CSC", "JAD")  # n = 6,048
K235_TRAIN_SITES: Final[Tuple[str, ...]] = ("BES", "CNH", "GDK", "HNM", "SHC", "YJD")  # n = 15,463
K235_ALL_SITES: Final[Tuple[str, ...]] = ("BES", "CNH", "CSC", "GDK", "HNM", "JAD", "SHC", "YJD")

ALARM_THRESHOLDS_MGM3: Final[Tuple[float, ...]] = (15.0, 25.0, 100.0)  # 사업문서 s5
ALARM_THRESHOLDS_LOG1P: Final[Tuple[float, ...]] = (2.7725887, 3.2580965, 4.6151205)
ALARM_LEVEL_NAMES: Final[Tuple[str, ...]] = ("정상", "관심", "경계", "대발생")

# ─── 모델 출력 키 (X-15). criterion 과 모델의 순환 import 차단용 ─────────
OUT_SEG: Final[str] = "seg_logits_s4"
OUT_CHL: Final[str] = "chl_u_s4"  # softplus(z), log1p 공간
OUT_LOGVAR: Final[str] = "log_var_s4"
OUT_EDGE: Final[str] = "edge_logits_s4"  # train-only
OUT_VACUITY: Final[str] = "vacuity_s4"  # 진단 전용, 손실 미소비 (X-12)
OUT_AUX: Final[Dict[str, str]] = {
    "enc_s8": "aux_seg_s8",
    "enc_s16": "aux_seg_s16",
    "enc_s32": "aux_seg_s32",
    "p8": "aux_p_s8",
}
AUX_TAP_STRIDE: Final[Dict[str, int]] = {"enc_s8": 8, "enc_s16": 16, "enc_s32": 32, "p8": 8}
AUX_TAP_CH: Final[Dict[str, int]] = {"enc_s8": 64, "enc_s16": 160, "enc_s32": 320, "p8": 128}
AUX_TAP_MID: Final[Dict[str, int]] = {"enc_s8": 32, "enc_s16": 64, "enc_s32": 128, "p8": 64}
#   ★ (I-6 / 정정 B-21) `C_mid = C_in // 2` 규칙은 폐기됐다. 이 리터럴이 정본이다 —
#   규칙대로면 enc_s16 93,388 -> 116,652 / enc_s32 371,084 -> 463,692 로 T16 이 즉시 깨진다.
OUT_SIAM: Final[Dict[str, str]] = {"s3": "siam_s3", "s4": "siam_s4", "dec": "siam_dec"}

TRAIN_ONLY_MODULES: Final[Tuple[str, ...]] = (
    "edge_head",
    "aux_heads",
    "siam_proj",
)  # deploy() 제거 대상
