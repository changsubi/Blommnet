#!/usr/bin/env python3
"""BloomNet 추론 시각화 — 이미지 / 이미지 디렉터리 / 동영상 (12클래스 하천오염 seg).

════════════════════════════════════════════════════════════════════════════
★ 이 파일은 **쌍(pair) 스크립트**다 ★
════════════════════════════════════════════════════════════════════════════
짝:  tools/predict_media.py (이전 구현 교사/학생)
나:  <repo_root>/bloomnet/tools/predict_media.py   (BloomNet)

두 스크립트의 존재 이유는 **같은 영상에 대해 두 모델 결과를 나란히 비교**하는
것이다. 따라서 모델 자체를 제외한 모든 것이 동일해야 하며, **한쪽을 고치면
반드시 다른 쪽도 같이 고쳐야 한다.** 두 저장소는 서로를 import 하지 않으므로
공통 코드는 아래 "SHARED CONTRACT BLOCK" 에 **바이트 단위로 복제**해 두었다.

동일해야 하는 항목:
  1. 팔레트 / 클래스 이름   PALETTE_12·CLASS_NAMES_12·PALETTE_4·CLASS_NAMES_4
  2. 전처리                 resize_frames() — PIL BILINEAR(uint8, CPU) + ImageNet 정규화
  3. 마스크 업샘플          upscale_ids() — PIL NEAREST (torch "nearest" 금지)
  4. 범례                   draw_legend() — 좌상단(0,0) 흰 박스 + 검정 글씨,
                            내용은 항상 paint 집합(=legend_class_ids)과 일치,
                            표시 시점은 --legend (기본 all=매 프레임)
  5. 인코더                 libx264 / quality=6 / macro_block_size=1 / yuv420p /
                            짝수 크롭(even_size·crop_to) / cv2 fourcc="mp4v".
                            실제 적용값은 VideoWriter.encoder_summary() 가 찍는다
  6. CLI                    add_common_arguments() 의 인자 이름·타입·기본값·help
                            (--classes 는 번호 또는 한글 이름 문자열)
  7. 파일명 규칙            동영상 마스크/chl/conf = frame_%06d.png (원본 프레임 인덱스),
                            셋 다 **원본 해상도**로 저장한다(chl/conf 는 bilinear 업샘플)
  8. 메모리 정책            prefetch 큐 깊이 = min(16, max(2, num_workers*8))
                            — N>=2 면 큐 깊이 16 으로 포화되어 값이 더 커져도 같다,
                            렌더 결과를 리스트로 쌓지 않고 제너레이터로 흘린다
  9. 인자 검증              validate() — resize 32배수 포함, 양쪽 동일하게 거부
 10. 종료 코드              정상 0 / UserError 2 / KeyboardInterrupt 130

고유 인자(예외로 인정되는 차이):
  이전 구현 : --model_type, --student_model
  bloomnet    : --config, --ema, --save_chl, --save_conf, --print_config

동작 요약
--------
1. 체크포인트의 ``args``(= ``cfg.to_dict()``)로 config 를 복원하고 ``build_bloomnet`` 으로
   모델을 재구성한다. ``--config`` 로 명시 지정도 가능하다.
2. **S0-RGB 고정** — ``rgb`` 만 넣고 ``avail=[1,0,0,0,0]`` 이다 (msi/bio/ir/pol 없음).
3. ``BloomNet.forward`` 는 **H/4 해상도**로만 출력한다 (04 §9.1). 이 스크립트가
   ``seg_logits_s4`` 를 bilinear 로 ``--resize`` 해상도까지 올리고, argmax 한 class-id 를
   **PIL NEAREST 로 원본 해상도**까지 되돌려 오버레이한다.
   (logits 를 곧바로 4K 로 올리지 않는 이유는 메모리다 — 3840×2160×12×4B ≈ 398 MB/프레임.)
4. ``model.deploy()`` 로 학습 전용 모듈(edge/aux/siam)을 떼고 ``eval()`` +
   ``torch.no_grad()`` 로만 추론한다. 학습 전용 출력은 애초에 나오지 않는다 (04 §9.2).

★★ Chl-a 경고 ★★
    BloomNet 은 멀티태스크(seg + Chl-a 회귀 + 불확실도)지만, **현재 S0 체크포인트는
    Chl-a 라벨 없이 학습됐다**(``data.chl.supervision == "none"``). ``--save_chl`` 로 뽑는
    Chl-a 맵은 ``ChlHead`` 의 prior bias(log1p 평균 ≈ 1.7559)에서 거의 움직이지 않은
    **의미 없는 값**이며, 절대 농도로 해석하거나 경보 단계로 변환해서는 안 된다.
    ``--save_conf`` 의 confidence 도 X-25 정의상 **단위 없는 상대 신뢰도**이지 확률이 아니다.

동영상 백엔드
-----------
``imageio-ffmpeg``(1순위, 자체 ffmpeg 번들 → HEVC 디코딩 가능) → ``cv2``(폴백).
imageio-ffmpeg 가 런타임에 실패하면 cv2 가 있을 때 자동으로 폴백한다.
둘 다 없으면 설치 방법을 담은 에러를 낸다. 시스템 ffmpeg 바이너리는 요구하지 않는다.

사용 예 (cwd = <repo_root>)
    # 도움말
    python -m bloomnet.tools.predict_media --help

    # 동영상 (10프레임마다 1장, 처음 300장만)
    CUDA_VISIBLE_DEVICES=0 python \\
        -m bloomnet.tools.predict_media \\
        --input <repo_root>/1.MP4 \\
        --output outputs/viz/1_bloomnet.mp4 \\
        --checkpoint outputs/bloomnet_s0_asis_cmp/best.pt \\
        --stride 10 --max_frames 300 --mode side_by_side

    # 클래스 이름으로 골라 보기 (이전 구현 쪽에서도 같은 명령줄이 그대로 동작한다)
    # 범례 표시는 영문이지만, 기존 한글 이름도 그대로 받는다.
    ... --classes Floating_Debris Pond

레벨 L7 (최상위 진입점). 무거운 import(모델/체크포인트)는 함수 안에서 한다 —
``--help`` 는 GPU 를 잡지 않는다 (torch 는 ``bloomnet.constants`` 경유로 로드된다).
numpy/PIL 은 짝 스크립트와 공유 코드를 바이트 단위로 같게 유지하기 위해 모듈 최상단에서 읽는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

# 파일 경로로 직접 실행(``python bloomnet/tools/predict_media.py``)해도 되도록
# 저장소 루트를 import 경로에 넣는다 (이전 구현 쪽과 같은 패턴).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import copy  # noqa: E402
import os  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

__all__ = ["main", "build_parser"]


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ▼▼▼  SHARED CONTRACT BLOCK — BEGIN  ▼▼▼                                ║
# ║                                                                         ║
# ║  이 마커와 END 마커 사이의 코드는 아래 두 파일에서 **바이트 단위로 동일**  ║
# ║  해야 한다. 한쪽만 고치면 두 모델의 결과 차이가 "모델 차이"인지 "전처리    ║
# ║  차이"인지 구분할 수 없게 되어 비교 실험 자체가 무효가 된다.               ║
# ║                                                                         ║
# ║    A: tools/predict_media.py          ║
# ║    B: <repo_root>/bloomnet/tools/predict_media.py            ║
# ║                                                                         ║
# ║  대조 방법 — diff 출력이 비어야 정상(아래 마커는 파일 안에서 유일하다):     ║
# ║    sed -n '/[▼][▼][▼]/,/[▲][▲][▲]/p' A > /tmp/a                          ║
# ║    sed -n '/[▼][▼][▼]/,/[▲][▲][▲]/p' B > /tmp/b                          ║
# ║    diff /tmp/a /tmp/b                                                    ║
# ╚═════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────
# 팔레트 / 클래스 이름
#   12클래스 팔레트 정본: 이전 구현/tools/visualize_predictions.py 의 FINE_PALETTE
#   12클래스 이름 정본  : 이전 구현/eval_tta_segmentation.py 의 FINE_CLASS_NAMES
#                        (= bloomnet.constants.FINE_CLASS_NAMES 와 동일 문자열)
#   4클래스(대분류) 정본: tools/visualize_predictions.py 의 COARSE_PALETTE / COARSE_CLASS_NAMES
# ─────────────────────────────────────────────────────────────────────────
PALETTE_12: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),        # 0  background            black
    (240, 220, 0),    # 1  Cropland              yellow
    (204, 80, 160),   # 2  Residue               magenta
    (0, 158, 115),    # 3  Drainage              teal
    (145, 70, 255),   # 4  Greenhouse            violet
    (110, 210, 0),    # 5  Orchard               lime
    (230, 120, 0),    # 6  Livestock_Shed        orange
    (135, 70, 20),    # 7  Manure_Pile           brown
    (255, 105, 180),  # 8  Pasture               pink
    (210, 40, 40),    # 9  Manure_Facility       red
    (0, 90, 220),     # 10 Floating_Debris       blue
    (0, 200, 230),    # 11 Pond                  cyan
)
# 범례에 그려지는 표시 이름은 **영문 고정**이다. 한글 폰트가 없는 환경에서
# 두부 글자(□□□)가 되는 것을 막고, 두 스크립트 출력이 서로 같아 보이게 한다.
# 원본 한글 이름(이전 구현 eval_tta_segmentation.FINE_CLASS_NAMES /
# bloomnet.constants.FINE_CLASS_NAMES)은 아래 CLASS_ALIASES_12 에 보존되어
# `--classes 부유쓰레기` 같은 기존 명령줄이 그대로 동작한다.
CLASS_NAMES_12: Tuple[str, ...] = (
    "background",
    "Cropland",
    "Residue",
    "Drainage",
    "Greenhouse",
    "Orchard",
    "Livestock_Shed",
    "Manure_Pile",
    "Pasture",
    "Manure_Facility",
    "Floating_Debris",
    "Pond",
)
CLASS_ALIASES_12: Tuple[str, ...] = (
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
PALETTE_4: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),        # 0 background
    (0, 180, 0),      # 1 Agriculture
    (230, 70, 40),    # 2 Livestock
    (0, 110, 255),    # 3 Water_Riparian
)
CLASS_NAMES_4: Tuple[str, ...] = ("background", "Agriculture", "Livestock", "Water_Riparian")
CLASS_ALIASES_4: Tuple[str, ...] = ("background", "농업계", "축산계", "하천수면/수변")

#: 표시 이름 → 원본 한글 이름 (--classes 입력 호환). 길이가 맞지 않으면 무시된다.
CLASS_ALIAS_TABLE: Dict[int, Tuple[str, ...]] = {
    len(CLASS_NAMES_12): CLASS_ALIASES_12,
    len(CLASS_NAMES_4): CLASS_ALIASES_4,
}

# aihub092_semseg/data/dataset.py / bloomnet.constants 의 IMAGENET_MEAN·STD 와 동일한 값.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VIDEO_EXTS: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")

# 모델 입력 H,W 제약. bloomnet 은 SIZE_DIVISOR(=32, 04 §1.2), 이전 구현 교사는
# DINOv3 어댑터가 H/8·H/16·H/32 특징맵을 만들기 때문에 둘 다 32 의 배수를 요구한다.
# → 공용 명령줄이 한쪽에서만 죽지 않도록 **양쪽 모두 하드 에러**로 통일한다.
SIZE_DIVISOR: int = 32

# 범례 폰트 후보. 이 머신에는 nanum/noto-cjk 가 없고 DroidSansFallbackFull 만
# 한글을 그릴 수 있어서 목록에 포함한다(없으면 legend 가 전부 두부 글자가 된다).
# 범례 표시 이름이 ASCII(영문)이므로 라틴 폰트를 먼저 본다.
# ★ 이 머신의 DroidSansFallbackFull.ttf 는 ASCII 조차 두부(□)로 그리는 손상된 폰트라
#   후순위로 내렸다. 순서만으로는 불충분해서 _load_font() 가 실제 렌더 검사도 한다.
FONT_CANDIDATES: Tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)
FONT_ENV_VAR: str = "PREDICT_MEDIA_LEGEND_FONT"

# 인코더 설정 — 두 스크립트가 같은 코덱/CRF 여야 압축 아티팩트가 비교를 방해하지 않는다.
# macro_block_size=1 + 자체 짝수 크롭(even_size/crop_to) 조합이라 imageio 가
# 프레임을 몰래 리사이즈하지 않는다.
VIDEO_CODEC: str = "libx264"
VIDEO_QUALITY: int = 6
VIDEO_MACRO_BLOCK_SIZE: int = 1
VIDEO_PIX_FMT_OUT: str = "yuv420p"
CV2_FOURCC: str = "mp4v"

# imageio-ffmpeg 버전차로 TypeError 가 날 때 **이 순서로 하나씩만** 뺄 수 있는 인자.
# codec / quality / pix_fmt_out / macro_block_size 는 인코딩 계약 그 자체라 절대 빼지
# 않는다(빠지면 두 스크립트의 CRF·기하가 달라져 비교 실험이 무효가 된다).
VIDEO_OPTIONAL_KWARGS: Tuple[str, ...] = ("ffmpeg_log_level",)

# 완주하지 못한 mp4 가 최종 파일명을 차지하지 않도록 여기에 먼저 쓴다(성공 시 os.replace).
VIDEO_PART_SUFFIX: str = ".part"

# 디코딩 프리페치 큐 상한(프레임 수). 4K RGB 한 장이 ≈24.9MB 라 16장 ≈ 400MB.
PREFETCH_MAX_QUEUE: int = 16

# --save_mask_dir / --save_chl / --save_conf 대량 생성 경고 임계값(총 파일 수)과
# 픽셀당 바이트 어림값. 마스크는 class-id 라 PNG 가 잘 눌리고(4K 실측 0.007~0.05 B/px),
# chl/conf 는 16bit 연속값이라 훨씬 크다. 경고용이므로 보수적으로 큰 쪽을 쓴다.
SIDE_OUTPUT_WARN_FILES: int = 1000
SIDE_BYTES_PER_PIXEL: Dict[str, float] = {"mask": 0.05, "chl": 0.9, "conf": 0.9}


class UserError(RuntimeError):
    """사용자가 바로 조치할 수 있는 한국어 에러. main() 이 잡아서 exit code 2."""


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter,
                    argparse.RawDescriptionHelpFormatter):
    """기본값은 자동으로 붙이고, epilog 의 줄바꿈은 그대로 둔다.

    required 인자에는 ``(default: None)`` 을 붙이지 않는다 — argparse 가 이걸 붙일지는
    파이썬 패치 버전마다 달라서, 그대로 두면 두 환경(이전 구현 3.13.9 /
    bloomnet 3.13.14)의 ``--help`` 출력이 갈린다. 여기서 고정해 둔다.
    """

    def _get_help_string(self, action: argparse.Action) -> Optional[str]:
        if getattr(action, "required", False):
            return action.help
        return super()._get_help_string(action)


# ═══════════════════════════════════════════════════════════════════════════
# 팔레트 / 범례
# ═══════════════════════════════════════════════════════════════════════════
def _generated_palette(num_classes: int) -> "np.ndarray":
    """12/4 클래스가 아닐 때만 쓰는 폴백 팔레트(HSV 등간격)."""
    import colorsys

    colors: List[Tuple[int, int, int]] = [(0, 0, 0)]
    for idx in range(1, num_classes):
        hue = (idx - 1) / max(1, num_classes - 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(colors, dtype=np.uint8)


def resolve_palette(num_classes: int) -> Tuple["np.ndarray", List[str]]:
    """클래스 수 → (팔레트 uint8 (K,3), 클래스 이름 리스트)."""
    if num_classes == len(CLASS_NAMES_12):
        return np.array(PALETTE_12, dtype=np.uint8), list(CLASS_NAMES_12)
    if num_classes == len(CLASS_NAMES_4):
        return np.array(PALETTE_4, dtype=np.uint8), list(CLASS_NAMES_4)
    print(
        f"[경고] {num_classes} 클래스용 정본 팔레트가 없어 자동 생성 팔레트를 씁니다. "
        "(두 스크립트 비교 시 색이 다를 수 있음)"
    )
    return _generated_palette(num_classes), [f"class_{i}" for i in range(num_classes)]


def _font_renders_ascii(font: Any) -> bool:
    """실제로 글리프를 그리는지 확인한다.

    설치돼 있어도 ASCII 글리프가 없는 폰트가 있다(이 머신의 DroidSansFallbackFull).
    그런 폰트를 고르면 범례가 통째로 두부(□)가 되므로 여기서 걸러낸다.
    """
    try:
        probe = Image.new("L", (24, 24), 0)
        ImageDraw.Draw(probe).text((1, 1), "A", fill=255, font=font)
        return bool(np.asarray(probe).any())
    except Exception:  # noqa: BLE001 - 폰트가 이상하면 그냥 다음 후보로
        return False


def _load_font(size: int) -> Any:
    override = os.environ.get(FONT_ENV_VAR)
    fallback = None
    for path in ((override,) if override else ()) + FONT_CANDIDATES:
        try:
            if not (path and Path(path).exists()):
                continue
            font = ImageFont.truetype(path, size)
        except OSError:
            continue
        if _font_renders_ascii(font):
            return font
        if fallback is None:
            fallback = font  # 전부 실패하면 최소한 무언가는 돌려준다
    default = ImageFont.load_default()
    if _font_renders_ascii(default):
        return default
    return fallback if fallback is not None else default


def _text_width(draw: "ImageDraw.ImageDraw", text: str, font: Any) -> int:
    try:
        return int(draw.textlength(text, font=font))
    except Exception:  # noqa: BLE001 - 아주 오래된 Pillow 폴백
        try:
            return int(font.getlength(text))
        except Exception:  # noqa: BLE001
            return int(len(text) * 8)


def draw_legend(
    image: "Image.Image",
    class_ids: Sequence[int],
    class_names: Sequence[str],
    palette: "np.ndarray",
) -> "Image.Image":
    """좌상단 (0,0) 에 흰 박스 + 검정 글씨 범례를 그린다. 폰트가 없어도 죽지 않는다."""
    if not class_ids:
        return image
    if image.mode != "RGB":
        image = image.convert("RGB")

    font_size = int(max(13, min(34, round(image.height * 0.016))))
    font = _load_font(font_size)
    swatch = font_size + 2
    pad = max(6, font_size // 2)
    line_h = swatch + max(4, font_size // 3)

    draw = ImageDraw.Draw(image, "RGBA")
    labels = [
        f"{cid} {class_names[cid] if cid < len(class_names) else f'class_{cid}'}"
        for cid in class_ids
    ]
    text_w = max((_text_width(draw, t, font) for t in labels), default=40)
    box_w = pad * 2 + swatch + pad // 2 + text_w
    box_h = pad * 2 + line_h * len(class_ids) - (line_h - swatch)

    draw.rectangle([0, 0, box_w, box_h], fill=(255, 255, 255, 190))
    draw.rectangle([0, 0, box_w, box_h], outline=(0, 0, 0, 220), width=1)

    for row, cid in enumerate(class_ids):
        y = pad + row * line_h
        x = pad
        color = tuple(int(v) for v in palette[min(cid, len(palette) - 1)])
        draw.rectangle([x, y, x + swatch, y + swatch], fill=color + (255,), outline=(0, 0, 0, 255))
        draw.text((x + swatch + pad // 2, y), labels[row], fill=(0, 0, 0, 255), font=font)
    return image


# ═══════════════════════════════════════════════════════════════════════════
# 전처리 / 업샘플 / 렌더링
# ═══════════════════════════════════════════════════════════════════════════
def resize_frames(frames: Sequence["np.ndarray"], resize_wh: Tuple[int, int]) -> "np.ndarray":
    """모델 입력 리사이즈 **정본**: PIL BILINEAR, uint8, CPU.

    - GPU 에 4K fp32 텐서를 올리지 않으므로 4K/batch8 기준 ~1.2GB VRAM 스파이크가 없다.
    - torch bilinear(antialias=True) 와 PIL BILINEAR 는 결과가 다르다. 두 모델이
      **같은 입력 텐서**를 보게 하려면 여기 한 군데로 고정해야 한다.
    - aihub092_semseg/data/dataset.py 의 학습 전처리와도 동일하다.
    """
    out: List["np.ndarray"] = []
    for frame in frames:
        arr = np.ascontiguousarray(frame)
        if (arr.shape[1], arr.shape[0]) != resize_wh:
            arr = np.asarray(
                Image.fromarray(arr).resize(resize_wh, Image.BILINEAR), dtype=np.uint8
            )
        out.append(arr)
    return np.stack(out, axis=0)


def to_input_tensor(small: "np.ndarray", torch_module: Any, device: Any, mean: Any, std: Any) -> Any:
    """uint8 (B,H,W,3) → 정규화된 float32 (B,3,H,W) 텐서. 두 스크립트 공통 경로."""
    x = torch_module.from_numpy(np.ascontiguousarray(small)).to(device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - mean) / std


def upscale_ids(class_ids_small: "np.ndarray", size_wh: Tuple[int, int]) -> "np.ndarray":
    """마스크 업샘플 **정본**: PIL NEAREST (출력 픽셀 중심 기준 매핑).

    torch 의 mode="nearest" 는 floor(dst*scale) 레거시 규칙이라 PIL 대비 최대 반
    출력픽셀(4K 기준 ~2px) 계통적으로 밀린다. 픽셀 단위 IoU/오버레이 정렬 비교를
    위해 두 스크립트 모두 이 함수만 쓴다.
    """
    ids = np.ascontiguousarray(class_ids_small).astype(np.uint8)
    if (ids.shape[1], ids.shape[0]) == size_wh:
        return ids
    return np.asarray(Image.fromarray(ids, mode="L").resize(size_wh, Image.NEAREST))


def render_frame(
    frame_rgb: "np.ndarray",
    class_ids: "np.ndarray",
    palette: "np.ndarray",
    mode: str,
    alpha: float,
    paint_flags: "np.ndarray",
) -> "np.ndarray":
    """frame_rgb(H,W,3 uint8) 와 원본 해상도 class_ids(H,W) 로 출력 프레임을 만든다.

    paint_flags[c] 가 True 인 클래스만 색칠한다. overlay/side_by_side 에서는
    배경(0)을 칠하지 않는다 — 배경색이 검정이라 화면 전체가 어두워지기만 한다.

    메모리: 알파 블렌딩을 **가로 스트라이프(≈1M 픽셀)** 로 나눠 계산한다. 4K
    (3840x2160 = 8.29M px) / mode=side_by_side / 기본 --classes(배경 제외 전부) 기준
    한 번에 계산하면 nonzero 좌표 int64 x2 = 132.7MB, src/dst float32 = 199MB,
    clip 중간 배열 4개 = 398MB → 프레임당 피크 ≈ 850MB 였다. 스트라이프로 나누면
    이 부분이 ≈1/8(약 90MB)로 떨어진다. 상시 배열(color 24.9MB + overlay 24.9MB +
    concatenate 49.8MB ≈ 100MB)은 그대로이므로 프레임당 피크는 850MB → 약 200MB.
    출력 픽셀값은 한 번에 계산할 때와 **비트 단위로 동일**하다(같은 식·같은 dtype).
    """
    idx = np.clip(class_ids, 0, len(palette) - 1)
    color = palette[idx]
    paint = paint_flags[np.clip(class_ids, 0, len(paint_flags) - 1)]

    if mode == "mask":
        out = np.zeros_like(frame_rgb)
        np.copyto(out, color, where=paint[..., None])
        return out

    overlay = np.ascontiguousarray(frame_rgb).copy()
    rows = max(1, (1 << 20) // max(1, overlay.shape[1]))  # 스트라이프당 ≈1M 픽셀
    for y0 in range(0, overlay.shape[0], rows):
        y1 = min(y0 + rows, overlay.shape[0])
        sel = np.nonzero(paint[y0:y1])
        if not sel[0].size:
            continue
        view = overlay[y0:y1]  # basic slicing → view, 대입이 overlay 원본에 그대로 쓰인다
        src = view[sel].astype(np.float32)
        dst = color[y0:y1][sel].astype(np.float32)
        view[sel] = np.clip(src * (1.0 - alpha) + dst * alpha, 0, 255).astype(np.uint8)

    if mode == "side_by_side":
        return np.concatenate([np.ascontiguousarray(frame_rgb), overlay], axis=1)
    return overlay


def even_size(size_wh: Tuple[int, int]) -> Tuple[int, int]:
    """yuv420p 인코딩을 위해 짝수 해상도로 맞춘다(필요시 우/하단 1픽셀 크롭)."""
    return (size_wh[0] - size_wh[0] % 2, size_wh[1] - size_wh[1] % 2)


def crop_to(frame: "np.ndarray", size_wh: Tuple[int, int]) -> "np.ndarray":
    return frame[: size_wh[1], : size_wh[0]]


def save_gray16(arr: "np.ndarray", path: Path, scale: float) -> None:
    """float 맵 → 16bit PNG. PNG 저장이 안 되는 Pillow 면 .npy 로 폴백한다."""
    value = np.clip(np.asarray(arr, dtype=np.float64) * float(scale), 0, 65535).astype(np.uint16)
    try:
        Image.fromarray(value).save(path)
    except Exception:  # noqa: BLE001
        np.save(path.with_suffix(".npy"), np.asarray(arr, dtype=np.float32))


# ═══════════════════════════════════════════════════════════════════════════
# AMP
# ═══════════════════════════════════════════════════════════════════════════
_BF16_WARNED: List[bool] = []  # 배치마다 같은 경고를 찍지 않도록


def amp_context(amp: str, device: Any, torch_module: Any) -> Any:
    from contextlib import nullcontext

    if amp == "off" or device.type != "cuda":
        return nullcontext()
    if amp == "bf16":
        try:
            supported = bool(torch_module.cuda.is_bf16_supported())
        except Exception:  # noqa: BLE001
            supported = False
        if not supported:
            if not _BF16_WARNED:
                _BF16_WARNED.append(True)
                print("[경고] 이 GPU 는 bf16 을 지원하지 않아 fp16 으로 내립니다.", flush=True)
            return torch_module.autocast("cuda", dtype=torch_module.float16)
        return torch_module.autocast("cuda", dtype=torch_module.bfloat16)
    return torch_module.autocast("cuda", dtype=torch_module.float16)


# ═══════════════════════════════════════════════════════════════════════════
# 동영상 백엔드 (imageio-ffmpeg 우선, cv2 폴백)
# ═══════════════════════════════════════════════════════════════════════════
def _backend_missing_error(what: str) -> UserError:
    py = sys.executable
    return UserError(
        f"동영상 {what} 백엔드가 없습니다. 아래 중 하나를 설치하세요.\n"
        f"  1) (권장, HEVC/H.265 디코딩 가능 — 자체 ffmpeg 번들)\n"
        f"     {py} -m pip install imageio-ffmpeg\n"
        f"  2) (폴백, 빌드에 따라 HEVC 가 안 될 수 있음)\n"
        f"     {py} -m pip install opencv-python\n"
        f"  시스템 ffmpeg 바이너리는 이 환경에 없으므로 1) 을 권장합니다.\n"
        f"  (이미지 / 이미지 디렉터리 입력은 PIL 만으로 동작합니다.)"
    )


def _have(module_name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):  # pragma: no cover - 손상된 설치
        return False


def _imageio_ffmpeg_ready() -> Tuple[bool, str]:
    """imageio-ffmpeg 를 **실제로** 쓸 수 있는지 확인한다 → (가능?, 설명).

    모듈만 있고 번들 ffmpeg 바이너리가 없는 설치가 흔하다(오프라인이면 런타임
    다운로드도 실패한다). 그래서 get_ffmpeg_exe() 까지 호출해 본다.
    """
    if not _have("imageio_ffmpeg"):
        return False, "모듈이 설치돼 있지 않습니다"
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 - 번들 미포함 / 다운로드 실패 등 모두 포함
        return False, f"ffmpeg 바이너리를 얻지 못했습니다 ({type(exc).__name__}: {exc})"
    if not exe:
        return False, "ffmpeg 바이너리 경로가 비어 있습니다"
    return True, str(exe)


def ensure_video_backend(what: str) -> None:
    """동영상 백엔드를 **모델을 만들기 전에** 확인한다.

    이 검사가 없으면 4GB 교사 체크포인트를 다 읽고 학습 중인 GPU 까지 잡은 뒤에야
    '백엔드 없음' 으로 죽는다. 그래서 인자 검증 단계(check_io_paths)에서 부른다.
    """
    ok, detail = _imageio_ffmpeg_ready()
    if ok:
        return
    if _have("cv2"):
        print(
            f"[알림] imageio-ffmpeg 를 쓸 수 없어 동영상 {what} 을(를) cv2 로 시도합니다.\n"
            f"       원인: {detail}\n"
            "       HEVC/H.265 입력은 cv2 빌드에 따라 실패할 수 있습니다.",
            flush=True,
        )
        return
    raise UserError(f"{_backend_missing_error(what)}\n  (imageio-ffmpeg 확인 결과: {detail})")


def checkpoint_read_error(checkpoint: Any, exc: BaseException) -> UserError:
    """체크포인트 읽기 실패 → 두 스크립트가 **같은 문구**로 안내한다.

    학습 스크립트가 best.pt / last.pt 를 덮어쓰는 중이면 zip 헤더가 반쯤 쓰인 상태를
    읽게 되어 RuntimeError / EOFError / UnpicklingError / BadZipFile 로 죽는다.
    """
    return UserError(
        f"체크포인트를 읽을 수 없습니다: {checkpoint}\n"
        f"  원인: {type(exc).__name__}: {exc}\n"
        "  학습이 체크포인트를 덮어쓰는 중일 수 있습니다. "
        "먼저 다른 경로로 복사한 뒤 그 사본을 지정하세요:\n"
        f"    cp {checkpoint} /tmp/ckpt_snapshot.pt"
    )


def checkpoint_load_errors() -> Tuple[type, ...]:
    """체크포인트가 '쓰이는 중' 일 때 나오는 예외 집합(두 스크립트 공통).

    torch.load 는 잘린 zip 을 RuntimeError(PytorchStreamReader ...) 로, 잘린 pickle 을
    EOFError / UnpicklingError 로, 지워진 파일을 OSError 로 낸다.
    """
    import zipfile
    from pickle import UnpicklingError

    return (RuntimeError, EOFError, OSError, UnpicklingError, zipfile.BadZipFile)


def is_out_of_memory_error(exc: BaseException) -> bool:
    """torch.OutOfMemoryError / 'out of memory' RuntimeError / MemoryError 판별.

    torch 를 import 하지 않고 판별한다(--help 경로에서 torch 를 끌어오지 않기 위해).
    """
    if isinstance(exc, MemoryError):
        return True
    if "OutOfMemory" in type(exc).__name__:
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def out_of_memory_message(exc: BaseException) -> str:
    """OOM 안내 문구(두 스크립트 동일)."""
    if isinstance(exc, MemoryError) and "OutOfMemory" not in type(exc).__name__:
        where = "호스트 RAM 부족"
        extra = "    --num_workers 1   (프리페치 큐가 4K 프레임 16장 ≈ 400MB 를 잡습니다)\n"
    else:
        where = "GPU 메모리 부족"
        extra = "    CUDA_VISIBLE_DEVICES 로 여유 있는 GPU 를 지정하거나 --device cpu\n"
    return (
        f"{where}: {type(exc).__name__}: {exc}\n"
        "  --batch_size 를 줄이거나(예: 2) --resize 를 낮추세요. "
        "현재 두 GPU 가 학습에 사용 중일 수 있습니다.\n"
        "    --batch_size 2      (기본 8)\n"
        "    --resize 384 384    (기본 512 512)\n"
        f"{extra}"
        "  파편화가 의심되면 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 로 재실행하세요."
    )


class VideoReader:
    """(width, height, fps, nframes) 메타와 RGB uint8 프레임 이터레이터를 제공."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.backend = ""
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.nframes: Optional[int] = None
        self._handle: Any = None
        self._open()

    def _open(self) -> None:
        if _have("imageio_ffmpeg"):
            try:
                self._open_imageio()
                return
            except UserError as exc:
                if not _have("cv2"):
                    raise
                print(f"[알림] imageio-ffmpeg 실패 → cv2 로 폴백합니다.\n  원인: {exc}", flush=True)
        if _have("cv2"):
            self._open_cv2()
            return
        raise _backend_missing_error("디코딩")

    def _open_imageio(self) -> None:
        import imageio_ffmpeg

        try:
            imageio_ffmpeg.get_ffmpeg_exe()  # 번들 바이너리 확인 (없으면 여기서 실패)
            gen = imageio_ffmpeg.read_frames(str(self.path), pix_fmt="rgb24")
            meta = next(gen)
        except Exception as exc:  # noqa: BLE001 - 원인을 그대로 보여준다
            raise UserError(
                f"imageio-ffmpeg 로 동영상을 열지 못했습니다: {self.path}\n"
                f"  원인: {exc}\n"
                "  파일 경로/손상 여부와 코덱(HEVC 등)을 확인하세요."
            ) from exc
        self.backend = "imageio-ffmpeg"
        self._handle = gen
        self.width, self.height = (int(v) for v in meta["size"])
        self.fps = float(meta.get("fps") or 0.0) or 30.0
        nframes = meta.get("nframes")
        if nframes in (None, float("inf")) or (isinstance(nframes, float) and nframes != nframes):
            duration = meta.get("duration")
            nframes = int(round(float(duration) * self.fps)) if duration else None
        self.nframes = int(nframes) if nframes else None

    def _open_cv2(self) -> None:
        import cv2

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise UserError(
                f"cv2 로 동영상을 열지 못했습니다: {self.path}\n"
                "  이 영상이 HEVC/H.265 라면 opencv-python 빌드가 못 읽는 경우가 많습니다.\n"
                f"  {sys.executable} -m pip install imageio-ffmpeg 를 설치하면 해결됩니다."
            )
        self.backend = "cv2"
        self._handle = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.nframes = count if count > 0 else None

    def frames(self) -> Iterator["np.ndarray"]:
        if self.backend == "imageio-ffmpeg":
            nbytes = self.width * self.height * 3
            for chunk in self._handle:
                if len(chunk) != nbytes:  # pragma: no cover - 방어
                    print(f"[경고] 프레임 바이트 수 불일치({len(chunk)} != {nbytes}) — 건너뜁니다.")
                    continue
                yield np.frombuffer(chunk, dtype=np.uint8).reshape(self.height, self.width, 3).copy()
        else:
            import cv2

            while True:
                ok, bgr = self._handle.read()
                if not ok:
                    break
                yield cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            if self.backend == "imageio-ffmpeg":
                self._handle.close()
            else:
                self._handle.release()
        except Exception:  # noqa: BLE001
            pass
        self._handle = None


class VideoWriter:
    """스트리밍 기록 — 프레임을 메모리에 쌓지 않는다.

    백엔드는 reader 와 **독립적으로** 고른다. 입력이 HEVC 라 reader 가 cv2 로 폴백해도
    쓰기는 imageio-ffmpeg 가 문제없이 처리하며, 7680x2160(side_by_side 4K)을 cv2/mp4v 로
    쓰면 화질이 크게 나빠지거나 아예 열리지 않는 빌드가 있다.
    """

    def __init__(self, path: Path, size_wh: Tuple[int, int], fps: float) -> None:
        self.path = path
        self.size_wh = (int(size_wh[0]), int(size_wh[1]))
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.backend = ""
        self.effective_kwargs: Dict[str, Any] = {}
        self.dropped_kwargs: List[str] = []
        self._handle: Any = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open()

    def _open(self) -> None:
        if _have("imageio_ffmpeg"):
            try:
                self._open_imageio()
                return
            except UserError as exc:
                if not _have("cv2"):
                    raise
                print(
                    f"[알림] imageio-ffmpeg writer 실패 → cv2 로 폴백합니다.\n  원인: {exc}",
                    flush=True,
                )
        if _have("cv2"):
            self._open_cv2()
            return
        raise _backend_missing_error("인코딩")

    def _open_imageio(self) -> None:
        import inspect

        import imageio_ffmpeg

        self.backend = "imageio-ffmpeg"
        full: Dict[str, Any] = dict(
            pix_fmt_in="rgb24",
            pix_fmt_out=VIDEO_PIX_FMT_OUT,
            fps=self.fps,
            codec=VIDEO_CODEC,
            quality=VIDEO_QUALITY,
            # 이미 짝수 해상도로 크롭했으므로 imageio 의 자동 리사이즈를 끈다.
            macro_block_size=VIDEO_MACRO_BLOCK_SIZE,
            ffmpeg_log_level="error",
        )
        # '전부 아니면 전무' 재시도(옛 minimal 폴백)는 codec/quality/pix_fmt_out/
        # macro_block_size 를 통째로 버려 인코딩 계약을 조용히 무효화했다. 이제는
        # (1) 시그니처에 없는 키만 골라 빼고 (2) 그래도 TypeError 면 계약과 무관한
        # VIDEO_OPTIONAL_KWARGS 만 하나씩 뺀다. 무엇이 빠졌는지는 반드시 알린다.
        kwargs = dict(full)
        dropped: List[str] = []
        try:
            params = inspect.signature(imageio_ffmpeg.write_frames).parameters
        except (TypeError, ValueError):  # pragma: no cover - 시그니처를 못 읽는 구현
            params = {}  # type: ignore[assignment]
        if params and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            for key in [k for k in kwargs if k not in params]:
                kwargs.pop(key)
                dropped.append(key)

        handle: Any = None
        for key in (None,) + VIDEO_OPTIONAL_KWARGS:
            if key is not None:
                if key not in kwargs:
                    continue
                kwargs.pop(key)
                dropped.append(key)
            try:
                handle = imageio_ffmpeg.write_frames(str(self.path), self.size_wh, **kwargs)
                handle.send(None)  # 초기화
                break
            except TypeError:  # imageio-ffmpeg 버전차로 키워드가 없을 때만 여기로 온다
                _close_quietly(handle)  # 이미 뜬 ffmpeg 프로세스를 반드시 회수한다
                handle = None
                continue
            except Exception as exc:  # noqa: BLE001
                _close_quietly(handle)
                raise UserError(
                    f"동영상 writer 를 만들 수 없습니다 (imageio-ffmpeg): {self.path}\n"
                    f"  원인: {exc}"
                ) from exc
        if handle is None:
            raise UserError(
                f"동영상 writer 를 만들 수 없습니다 (imageio-ffmpeg 인자 불일치): {self.path}\n"
                f"  {sys.executable} -m pip install -U imageio-ffmpeg 로 올리세요."
            )
        self._handle = handle
        self.effective_kwargs = kwargs
        self.dropped_kwargs = dropped
        if dropped:
            print(
                f"[경고] imageio-ffmpeg 가 {dropped} 를 받지 않아 해당 설정이 빠집니다 — "
                "짝 스크립트와 코덱/CRF/기하가 달라질 수 있습니다.\n"
                f"       {sys.executable} -m pip install -U imageio-ffmpeg 를 권장합니다.",
                flush=True,
            )

    def encoder_summary(self) -> str:
        """**실제로 적용된** 인코딩 설정. 로그가 상수를 그대로 찍어 거짓말하지 않게 한다."""
        if self.backend != "imageio-ffmpeg":
            return f"cv2 fourcc={CV2_FOURCC}"
        k = self.effective_kwargs
        text = (
            f"{k.get('codec', '(기본값)')} q={k.get('quality', '(기본값)')} "
            f"{k.get('pix_fmt_out', '(기본값)')} "
            f"macro_block_size={k.get('macro_block_size', '(기본값)')}"
        )
        if self.dropped_kwargs:
            text += f" [빠진 인자: {','.join(self.dropped_kwargs)}]"
        return text

    def _open_cv2(self) -> None:
        import cv2

        self.backend = "cv2"
        writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*CV2_FOURCC), self.fps, self.size_wh
        )
        if not writer.isOpened():
            raise UserError(
                f"cv2 로 출력 동영상을 열지 못했습니다: {self.path}\n"
                f"  코덱({CV2_FOURCC}) 또는 출력 경로 권한을 확인하세요."
            )
        self._handle = writer

    def write(self, frame_rgb: "np.ndarray") -> None:
        # ascontiguousarray 필수: [:, :, ::-1] 같은 음수 stride 뷰를 cv2 에 넘기면
        # "step[ndims-1] != elemsize" 로 죽거나 잘못된 프레임을 쓴다.
        frame = np.ascontiguousarray(frame_rgb)
        if self.backend == "imageio-ffmpeg":
            self._handle.send(frame.tobytes())
        else:
            import cv2

            self._handle.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            if self.backend == "imageio-ffmpeg":
                self._handle.close()
            else:
                self._handle.release()
        except Exception:  # noqa: BLE001
            pass
        self._handle = None


def _close_quietly(handle: Any) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except Exception:  # noqa: BLE001
        pass


def prefetch(iterable: Iterable[Any], workers: int) -> Iterator[Any]:
    """디코딩을 별도 스레드로 미리 돌린다(num_workers >= 1 일 때).

    동영상 디코딩은 순차 스트림이라 스레드 1개(+큐)만 의미가 있다. num_workers 는
    큐 깊이를 키우는 용도로도 쓰지만 PREFETCH_MAX_QUEUE 로 상한을 둔다
    (4K 프레임 1장 ≈ 24.9MB — 상한이 없으면 호스트 RAM 이 GB 단위로 튄다).

    **중단 가능**: 소비자가 예외/GeneratorExit 으로 빠져나가면 stop 이벤트를 세우고
    큐를 비운 뒤 스레드를 join 한다. 이 처리가 없으면 데몬 스레드가 가득 찬 큐의
    put() 에서 영원히 블록된 채 살아남아 4K 프레임 16장(≈400MB)과 ffmpeg 디코더
    자식 프로세스를 그대로 붙들고 있게 된다. 또한 워커가 리더를 순회하는 도중에
    다른 스레드가 reader.close() 를 부르는 상황도 이 join 으로 막는다.
    """
    if workers <= 0:
        yield from iterable
        return

    depth = min(PREFETCH_MAX_QUEUE, max(2, workers * 8))
    q: "queue.Queue[Any]" = queue.Queue(maxsize=depth)
    sentinel = object()
    error: List[BaseException] = []
    stop = threading.Event()

    def _worker() -> None:
        try:
            for item in iterable:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 - 메인 스레드로 전달
            error.append(exc)
        finally:
            try:
                q.put_nowait(sentinel)
            except queue.Full:  # 소비자가 이미 떠났다 — 어차피 큐를 비우며 회수한다
                pass

    thread = threading.Thread(target=_worker, name="frame-decoder", daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
    finally:
        stop.set()
        while True:  # 워커가 put 에서 풀려나도록 큐를 비운다
            try:
                q.get_nowait()
            except queue.Empty:
                break
        thread.join(timeout=5.0)
    if error:
        raise error[0]


def batched(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    buf: List[Any] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


# ═══════════════════════════════════════════════════════════════════════════
# 진행 표시
# ═══════════════════════════════════════════════════════════════════════════
def _fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


class Progress:
    """tqdm 이 있으면 막대, 없으면 ``every`` 프레임마다 한 줄."""

    def __init__(self, total: Optional[int], desc: str, every: int = 50) -> None:
        self.total = total
        self.desc = desc
        self.every = max(1, int(every))
        self.n = 0
        self._t0 = time.time()
        self._last = 0
        self._bar: Any = None
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(total=total, desc=desc, unit="frame", dynamic_ncols=True)
        except Exception:  # noqa: BLE001
            total_s = str(total) if total else "?"
            print(
                f"[{desc}] 시작 (총 {total_s} 프레임, tqdm 없음 — "
                f"{self.every} 프레임마다 한 줄 출력)",
                flush=True,
            )

    def update(self, k: int = 1) -> None:
        self.n += int(k)
        if self._bar is not None:
            self._bar.update(int(k))
            return
        if self.n - self._last >= self.every:
            self._last = self.n
            el = time.time() - self._t0
            rate = self.n / el if el > 0 else 0.0
            eta = ((self.total - self.n) / rate) if (self.total and rate > 0) else 0.0
            tot = f"/{self.total}" if self.total else ""
            print(
                f"[{self.desc}] {self.n}{tot} 프레임  {rate:.2f} f/s  "
                f"경과 {_fmt_hms(el)}  ETA {_fmt_hms(eta)}",
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            return
        el = time.time() - self._t0
        rate = self.n / el if el > 0 else 0.0
        print(f"[{self.desc}] 완료 — {self.n} 프레임, {_fmt_hms(el)} ({rate:.2f} f/s)", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 입출력 경로 해석 / 검증
# ═══════════════════════════════════════════════════════════════════════════
def classify_input(path: Path) -> str:
    """``"image" | "dir" | "video"``."""
    if path.is_dir():
        return "dir"
    if not path.exists():
        raise UserError(
            f"입력을 찾을 수 없습니다: {path}\n"
            "  --input 에 이미지 파일 / 이미지 디렉터리 / 동영상 파일 경로를 주세요.\n"
            "  (상대 경로는 현재 작업 디렉터리 기준입니다)"
        )
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in IMAGE_EXTS:
        return "image"
    raise UserError(
        f"지원하지 않는 확장자입니다: {path.suffix}\n"
        f"  이미지: {' '.join(IMAGE_EXTS)}\n"
        f"  동영상: {' '.join(VIDEO_EXTS)}"
    )


def list_images(directory: Path) -> List[Path]:
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise UserError(
            f"이미지 디렉터리에 읽을 파일이 없습니다: {directory}\n"
            f"  허용 확장자: {' '.join(IMAGE_EXTS)} (하위 디렉터리는 훑지 않습니다)"
        )
    return files


def same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:  # pragma: no cover - 심볼릭 링크 루프 등
        return False


def resolve_video_output(args: argparse.Namespace) -> Path:
    """동영상 출력 확장자를 **check_io_paths 보다 먼저** 보정한다.

    보정을 run_video 안에서 늦게 하면 '기존 출력을 덮어씁니다' 알림이 존재하지도 않는
    out.txt 를 보고, 정작 덮어써지는 out.mp4 에 대해서는 아무 말도 하지 않게 된다.
    """
    out = Path(args.output).expanduser()
    if out.suffix.lower() not in VIDEO_EXTS:
        out = out.with_suffix(".mp4")
        print(f"[알림] 출력 확장자가 동영상이 아니라 {out} 로 저장합니다.")
        args.output = str(out)
        if same_path(Path(args.input).expanduser(), out):
            raise UserError(
                "확장자 보정 결과 --output 이 --input 과 같아졌습니다 — 원본이 덮어써집니다.\n"
                f"  경로: {out}"
            )
    return out


def image_output_dir(args: argparse.Namespace, kind: str) -> Optional[Path]:
    """--output 이 **디렉터리로 쓰일 때** 그 디렉터리(아니면 None). dispatch() 와 같은 규칙."""
    if kind == "video":
        return None
    out_path = Path(args.output).expanduser()
    if kind == "dir" or out_path.is_dir() or out_path.suffix.lower() not in IMAGE_EXTS:
        return out_path
    return None  # 단일 이미지 파일 출력 — 디렉터리가 아니라 파일 이름으로 비교한다


def check_io_paths(args: argparse.Namespace, kind: str) -> None:
    """입력==출력 사고 방지 + 동영상 백엔드 사전 검사. 모델을 만들기 **전에** 부른다.

    동영상에서 --input 과 --output 이 같으면 ffmpeg 이 원본을 truncate 한 채
    디코딩을 이어가 원본이 파괴된다. 이미지 디렉터리에서 같으면 재실행 시
    결과물(.png)이 다시 입력으로 잡혀 오버레이 위에 오버레이가 씌워진다.
    """
    in_path = Path(args.input).expanduser()
    out_path = Path(args.output).expanduser()
    # 백엔드 검사를 여기서 한다 — 모델(교사 4GB)을 다 읽은 뒤에 죽으면 너무 늦다.
    if kind == "video" or out_path.suffix.lower() in VIDEO_EXTS:
        ensure_video_backend("디코딩/인코딩")
    if same_path(in_path, out_path):
        raise UserError(
            "--input 과 --output 이 같은 경로입니다 — 원본이 덮어써집니다.\n"
            f"  입력: {in_path}\n"
            f"  출력: {out_path}\n"
            "  출력은 반드시 다른 파일/디렉터리로 지정하세요."
        )
    if kind == "dir" and same_path(in_path, out_path.parent) and out_path.suffix == "":
        print(
            f"[알림] 출력 디렉터리가 입력 디렉터리 바로 아래입니다: {out_path}\n"
            "       재실행 시 결과물이 입력으로 다시 잡히지 않도록 주의하세요."
        )
    # 부가 출력 디렉터리는 '입력이 들어있는 디렉터리' 와도 충돌한다. 단일 이미지 입력에서
    # --save_mask_dir 를 입력 파일의 부모로 주면 마스크 파일명이 {stem}.png 라 입력 원본을
    # 그대로 덮어써 복구가 불가능해진다(같은 이유로 이미지 출력 디렉터리와도 충돌한다).
    in_dir = in_path if in_path.is_dir() else in_path.parent
    out_dir = image_output_dir(args, kind)
    sides: List[Tuple[str, Path]] = []
    for flag in ("save_mask_dir", "save_chl", "save_conf"):
        value = getattr(args, flag, None)
        if not value:
            continue
        side = Path(value).expanduser()
        sides.append((flag, side))
        if same_path(side, in_path):
            raise UserError(
                f"--{flag} 가 --input 과 같은 경로입니다 — 입력을 덮어씁니다.\n"
                f"  경로: {side}"
            )
        if kind != "video" and same_path(side, in_dir):
            # 이미지 입력에서 결과 파일명은 {stem}.png 라 입력과 정확히 같은 이름이 된다.
            # (동영상은 frame_%06d.png 라 원본과 겹치지 않으므로 알림만 한다.)
            raise UserError(
                f"--{flag} 가 입력이 들어있는 디렉터리와 같습니다 — "
                "같은 이름의 결과 PNG 가 입력 파일을 덮어씁니다.\n"
                f"  경로: {side}\n"
                f"  입력: {in_path}\n"
                "  다른 디렉터리를 지정하세요."
            )
        if kind == "video" and same_path(side, in_dir):
            print(
                f"[알림] --{flag} 가 입력 동영상이 있는 디렉터리입니다: {side}\n"
                "       frame_%06d.png 라 원본과 이름이 겹치지는 않지만 수천 장이 섞입니다."
            )
        if out_dir is not None and same_path(side, out_dir):
            raise UserError(
                f"--{flag} 가 이미지 출력 디렉터리와 같습니다 — "
                "오버레이와 마스크가 같은 파일 이름으로 서로를 덮어씁니다.\n"
                f"  경로: {side}\n"
                f"  출력: {out_path}"
            )
        if out_dir is None and kind == "image" and same_path(side / f"{in_path.stem}.png", out_path):
            raise UserError(
                f"--{flag} 의 결과 파일이 --output 과 같습니다 — 서로를 덮어씁니다.\n"
                f"  경로: {side / f'{in_path.stem}.png'}"
            )
    for i in range(len(sides)):
        for j in range(i + 1, len(sides)):
            if same_path(sides[i][1], sides[j][1]):
                raise UserError(
                    f"--{sides[i][0]} 와 --{sides[j][0]} 가 같은 디렉터리입니다 — "
                    "같은 파일 이름으로 서로를 덮어씁니다.\n"
                    f"  경로: {sides[i][1]}"
                )
    if kind == "video" and out_path.exists():
        print(f"[알림] 기존 출력 파일을 덮어씁니다: {out_path}")
        print(
            f"       (완주 전까지는 {out_path.with_name(out_path.stem + VIDEO_PART_SUFFIX + out_path.suffix).name} "
            "에 기록하므로 중단해도 기존 파일은 그대로 남습니다.)"
        )


def parse_classes(
    tokens: Optional[Sequence[str]], class_names: Sequence[str]
) -> Optional[List[int]]:
    """``--classes`` 토큰(번호 / 영문 표시명 / 한글 원본명) → 정렬 유지 int 리스트."""
    if not tokens:
        return None
    lowered = {name.lower(): idx for idx, name in enumerate(class_names)}
    # 한글 원본명도 계속 받는다 (기존 명령줄 호환). 표시명과 충돌하면 표시명 우선.
    for idx, alias in enumerate(CLASS_ALIAS_TABLE.get(len(class_names), ())):
        lowered.setdefault(alias.lower(), idx)
    selected: List[int] = []
    for token in tokens:
        text = str(token).strip()
        if text.lstrip("-").isdigit():
            cid = int(text)
        elif text.lower() in lowered:
            cid = lowered[text.lower()]
        else:
            raise UserError(
                f"--classes 값 '{token}' 을 해석할 수 없습니다.\n"
                "  클래스 번호(정수) 또는 아래 이름 중 하나를 쓰세요:\n"
                + "\n".join(f"    {i:>2} {n}" for i, n in enumerate(class_names))
            )
        if not 0 <= cid < len(class_names):
            raise UserError(
                f"--classes 값 {cid} 가 범위를 벗어났습니다 (허용 0 ~ {len(class_names) - 1})."
            )
        if cid not in selected:
            selected.append(cid)
    return selected


def build_paint_flags(
    selected: Optional[Sequence[int]], num_classes: int, mode: str
) -> "np.ndarray":
    """색칠 대상 클래스 불리언 배열.

    규칙(양쪽 동일): overlay / side_by_side 에서는 **배경(0)을 절대 칠하지 않는다**.
    사용자가 ``--classes 0`` 을 명시하면 조용히 버리지 않고 알림을 찍고,
    그 결과 칠할 것이 하나도 없으면 무음 no-op 대신 에러를 낸다.
    """
    flags = np.zeros(num_classes, dtype=bool)
    if selected is None:
        flags[:] = True
    else:
        for cid in selected:
            if 0 <= cid < num_classes:
                flags[cid] = True
    if mode != "mask":
        if selected is not None and 0 in selected:
            print(
                "[알림] overlay / side_by_side 에서는 배경(0)을 칠하지 않습니다 — "
                "--classes 0 은 무시됩니다.\n"
                "       배경까지 색으로 보려면 --mode mask 를 쓰세요."
            )
        flags[0] = False
    if not flags.any():
        raise UserError(
            "--classes 로 고른 클래스 중 실제로 칠할 것이 하나도 없습니다.\n"
            "  overlay / side_by_side 는 배경(0)을 칠하지 않으므로 0 만 고르면 아무 것도 그려지지 않습니다.\n"
            "  1 이상의 클래스를 함께 지정하거나 --mode mask 를 쓰세요."
        )
    return flags


def legend_class_ids(paint_flags: "np.ndarray") -> List[int]:
    """범례는 **항상 paint 집합과 일치**한다(칠하지도 않는 검정 스와치를 넣지 않는다)."""
    return [int(i) for i in np.nonzero(paint_flags)[0]]


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """두 스크립트가 **완전히 같은** 공통 인자 집합을 갖게 한다.

    이름·타입·기본값·metavar·help 문구까지 동일해야 한다. 여기에 없는 인자만
    각 스크립트의 고유 그룹(이전 구현: --model_type/--student_model,
    bloomnet: --config/--ema/--save_chl/--save_conf/--print_config)에 넣는다.
    """
    req = parser.add_argument_group("필수")
    req.add_argument(
        "--input",
        required=True,
        help="이미지 파일 | 이미지 디렉터리 | 동영상 파일 (확장자로 자동 판별)",
    )
    req.add_argument(
        "--output",
        required=True,
        help="출력 경로. 입력이 동영상이면 동영상, 이미지면 이미지, 디렉터리면 디렉터리",
    )
    req.add_argument("--checkpoint", required=True, help="체크포인트(.pt) 경로")

    com = parser.add_argument_group("공통 옵션")
    com.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="추론 디바이스")
    com.add_argument("--batch_size", type=int, default=8, help="프레임 배치 크기")
    com.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=[512, 512],
        help=f"모델 입력 크기(각각 {SIZE_DIVISOR} 의 배수). 출력은 항상 원본 해상도로 되돌린다",
    )
    com.add_argument(
        "--stride", type=int, default=1, help="N 프레임마다 1장 처리(이미지 디렉터리에도 적용)"
    )
    com.add_argument("--max_frames", type=int, default=0, help="처리할 최대 프레임 수 (0=무제한)")
    com.add_argument("--alpha", type=float, default=0.5, help="오버레이 투명도")
    com.add_argument(
        "--mode",
        default="overlay",
        choices=["overlay", "mask", "side_by_side"],
        help="출력 형태. side_by_side 는 [원본 | 오버레이] 를 가로로 붙인다",
    )
    com.add_argument(
        "--save_mask_dir",
        default=None,
        help="class-id 마스크 PNG(mode=L, 값 0..K-1) 저장 디렉터리. 동영상은 frame_%%06d.png(원본 프레임 인덱스). "
        "4K 전체 프레임이면 수 GB + 프레임당 100~300ms 가 더 붙는다 — --stride 와 함께 쓸 것",
    )
    com.add_argument(
        "--fps", type=float, default=None, help="출력 동영상 fps (기본: 입력 fps / stride)"
    )
    com.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help=f"동영상 디코딩 프리페치 (0=동기, 1 이상=디코드 스레드 1개). "
        f"큐 깊이 = min({PREFETCH_MAX_QUEUE}, max(2, N*8)) — N>=2 부터는 동일하다",
    )
    com.add_argument(
        "--amp",
        default="bf16",
        choices=["off", "bf16", "fp16"],
        help="자동 혼합정밀도 (cuda 일 때만 적용)",
    )
    com.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLASS",
        help="표시할 클래스만 지정(번호 또는 이름). 미지정 시 전부. "
        "overlay/side_by_side 에서는 배경(0)을 칠하지 않는다",
    )
    com.add_argument(
        "--legend",
        default="all",
        choices=["all", "first", "off"],
        help="범례 표시 (all=매 프레임, first=동영상 첫 프레임만, off=끔). "
        "마스크 PNG 에는 어느 설정에서도 범례를 그리지 않는다",
    )
    com.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="tqdm 이 없을 때 몇 프레임마다 진행률 한 줄을 찍을지",
    )
    return parser


def validate(args: argparse.Namespace) -> None:
    """공통 인자 범위 검사. 양쪽이 **같은 명령줄에서 같은 결과**를 내야 한다."""
    if args.batch_size < 1:
        raise UserError(f"--batch_size 는 1 이상이어야 합니다 (받은 값: {args.batch_size}).")
    if args.stride < 1:
        raise UserError(f"--stride 는 1 이상이어야 합니다 (받은 값: {args.stride}).")
    if args.max_frames < 0:
        raise UserError(f"--max_frames 는 0(무제한) 이상이어야 합니다 (받은 값: {args.max_frames}).")
    if not 0.0 <= args.alpha <= 1.0:
        raise UserError(f"--alpha 는 0.0 ~ 1.0 이어야 합니다 (받은 값: {args.alpha}).")
    if args.num_workers < 0:
        raise UserError(f"--num_workers 는 0 이상이어야 합니다 (받은 값: {args.num_workers}).")
    if args.progress_every < 1:
        raise UserError(f"--progress_every 는 1 이상이어야 합니다 (받은 값: {args.progress_every}).")
    if any(v < 1 for v in args.resize):
        raise UserError(
            f"--resize W H 는 각각 1 이상이어야 합니다 (받은 값: {args.resize[0]} {args.resize[1]})."
        )
    if any(v % SIZE_DIVISOR for v in args.resize):
        raise UserError(
            f"--resize W H 는 각각 {SIZE_DIVISOR} 의 배수여야 합니다 "
            f"(받은 값: {args.resize[0]} {args.resize[1]}).\n"
            f"  bloomnet 은 H·W ≡ 0 (mod {SIZE_DIVISOR}) 을 요구하고(04 §1.2),\n"
            "  이전 구현 교사는 DINOv3 어댑터가 H/8·H/16·H/32 특징맵을 만듭니다.\n"
            "  두 스크립트에 같은 명령줄을 쓸 수 있도록 양쪽 모두 같은 제약을 겁니다.\n"
            "  예: --resize 512 512 / --resize 1024 576"
        )
    if args.fps is not None and args.fps <= 0:
        raise UserError(f"--fps 는 0 보다 커야 합니다 (받은 값: {args.fps}).")
    # --classes 의 값싼 사전 검사(num_classes 없이 할 수 있는 것만). 정본 검사는
    # build_render_context 에 그대로 둔다 — 여기서 걸러야 4GB 교사를 읽기 전에 죽는다.
    if getattr(args, "classes", None):
        known = {n.lower() for n in CLASS_NAMES_12} | {n.lower() for n in CLASS_NAMES_4}
        for token in args.classes:
            text = str(token).strip()
            if (
                not text.lstrip("-").isdigit()
                and text.lower() not in known
                and not text.lower().startswith("class_")
            ):
                raise UserError(
                    f"--classes 값 '{token}' 을 해석할 수 없습니다. 번호 또는 클래스 이름을 쓰세요.\n"
                    "  12클래스: " + " ".join(CLASS_NAMES_12) + "\n"
                    "  4클래스:  " + " ".join(CLASS_NAMES_4)
                )
        if args.mode != "mask" and all(
            str(t).strip() in ("0", "background") for t in args.classes
        ):
            raise UserError(
                "overlay / side_by_side 는 배경(0)을 칠하지 않습니다 — "
                "--classes 0 만 주면 아무 것도 그려지지 않습니다.\n"
                "  1 이상의 클래스를 함께 지정하거나 --mode mask 를 쓰세요."
            )


# ═══════════════════════════════════════════════════════════════════════════
# 렌더 컨텍스트 / 부가 출력
# ═══════════════════════════════════════════════════════════════════════════
class RenderContext:
    """팔레트 · 클래스명 · paint 집합 · 범례 id 를 한 덩어리로 들고 다닌다."""

    def __init__(self, palette: "np.ndarray", class_names: Sequence[str],
                 paint_flags: "np.ndarray", legend_ids: Sequence[int]) -> None:
        self.palette = palette
        self.class_names = list(class_names)
        self.paint_flags = paint_flags
        self.legend_ids = list(legend_ids)


def build_render_context(args: argparse.Namespace, num_classes: int) -> RenderContext:
    palette, class_names = resolve_palette(num_classes)
    selected = parse_classes(args.classes, class_names)
    paint_flags = build_paint_flags(selected, num_classes, args.mode)
    return RenderContext(palette, class_names, paint_flags, legend_class_ids(paint_flags))


def open_side_output_dirs(args: argparse.Namespace) -> Dict[str, Path]:
    dirs: Dict[str, Path] = {}
    for key, flag in (("mask", "save_mask_dir"), ("chl", "save_chl"), ("conf", "save_conf")):
        value = getattr(args, flag, None)
        if not value:
            continue
        directory = Path(value).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        # 절대 지우지 않는다. 다만 이전 실행 결과가 섞인다는 사실은 알려야 한다
        # (--stride 3 으로 돌린 뒤 --stride 1 로 다시 돌리면 두 실행이 한 폴더에 섞인다).
        try:
            not_empty = any(directory.iterdir())
        except OSError:  # pragma: no cover - 권한 문제
            not_empty = False
        if not_empty:
            print(
                f"[알림] {directory} 에 이미 파일이 있습니다 — 같은 이름은 덮어쓰고, "
                "이전 실행의 나머지 파일은 그대로 남아 섞입니다.",
                flush=True,
            )
        dirs[key] = directory
    return dirs


def warn_side_output_volume(
    dirs: Dict[str, Path], count: Optional[int], size_wh: Tuple[int, int]
) -> None:
    """부가 출력이 대량으로 생길 때 **시작 전에** 알린다(디스크·시간이 함께 터진다)."""
    if not dirs or not count or count <= 0:
        return
    total_files = int(count) * len(dirs)
    if total_files <= SIDE_OUTPUT_WARN_FILES:
        return
    pixels = max(1, int(size_wh[0]) * int(size_wh[1]))
    total_bytes = sum(
        float(count) * pixels * SIDE_BYTES_PER_PIXEL.get(key, 0.5) for key in dirs
    )
    names = "/".join(sorted(dirs))
    print(
        f"[알림] {names} PNG {total_files}장(약 {total_bytes / 1e9:.1f} GB)이 생성됩니다 "
        f"— {size_wh[0]}x{size_wh[1]}, 프레임당 PNG 인코딩 100~300ms 가 추가됩니다.\n"
        f"       줄이려면 --stride / --max_frames 를 쓰세요.",
        flush=True,
    )


def save_side_outputs(
    dirs: Dict[str, Path],
    result: Dict[str, Any],
    k: int,
    filename: str,
    mask_full: "np.ndarray",
) -> None:
    """마스크/chl/conf 를 **같은 파일 이름 · 같은 해상도(원본)** 로 저장한다.

    이름으로 픽셀 페어링을 하는 계약이므로 기하가 달라서는 안 된다. mask 는 이미
    원본 해상도(upscale_ids)이고, chl/conf 는 모델 출력이라 --resize 해상도다.
    연속값이므로 BILINEAR 로 원본 해상도까지 올려 셋을 통일한다.
    """
    target_wh = (int(mask_full.shape[1]), int(mask_full.shape[0]))

    def _to_full(a: Any) -> "np.ndarray":
        arr = np.asarray(a, dtype=np.float32)
        if arr.ndim != 2 or (arr.shape[1], arr.shape[0]) == target_wh:
            return arr
        return np.asarray(
            Image.fromarray(arr, mode="F").resize(target_wh, Image.BILINEAR), dtype=np.float32
        )

    if "mask" in dirs:
        Image.fromarray(np.ascontiguousarray(mask_full).astype(np.uint8), mode="L").save(
            dirs["mask"] / filename
        )
    if "chl" in dirs and "chl" in result:
        save_gray16(_to_full(result["chl"][k]), dirs["chl"] / filename, 100.0)
    if "conf" in dirs and "conf" in result:
        save_gray16(_to_full(result["conf"][k]), dirs["conf"] / filename, 65535.0)


# ═══════════════════════════════════════════════════════════════════════════
# 실행: 동영상
# ═══════════════════════════════════════════════════════════════════════════
def run_video(args: argparse.Namespace, predictor: Any, ctx: RenderContext) -> int:
    input_path = Path(args.input).expanduser()
    # 확장자 보정은 run() 에서 check_io_paths 보다 먼저 끝났다(resolve_video_output).
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 완주하기 전까지는 임시 이름에 쓴다 → (a) 재실행이 기존 완성본을 첫 프레임 전에
    # truncate 하지 않고 (b) 중단되면 깨진 파일이 최종 이름을 차지하지 않는다.
    part_path = output_path.with_name(
        output_path.stem + VIDEO_PART_SUFFIX + output_path.suffix
    )
    if part_path.exists():
        print(f"[알림] 이전 중단으로 남은 임시 파일을 덮어씁니다: {part_path}")

    reader = VideoReader(input_path)
    print(
        f"입력 동영상: {input_path}\n"
        f"  백엔드={reader.backend}  해상도={reader.width}x{reader.height}  "
        f"fps={reader.fps:.3f}  총프레임≈{reader.nframes}"
    )
    if reader.width <= 0 or reader.height <= 0:
        reader.close()
        raise UserError(
            f"동영상 해상도를 읽지 못했습니다({reader.width}x{reader.height}): {input_path}\n"
            f"  백엔드={reader.backend}. imageio-ffmpeg 설치를 권장합니다\n"
            "  (cv2 는 HEVC 메타를 못 읽는 빌드가 있습니다)."
        )

    resize_wh = (int(args.resize[0]), int(args.resize[1]))
    frame_size = even_size((reader.width, reader.height))  # yuv420p 는 짝수 해상도만 받는다
    out_w = frame_size[0] * 2 if args.mode == "side_by_side" else frame_size[0]
    out_size = (out_w, frame_size[1])
    out_fps = float(args.fps) if args.fps else max(1e-3, reader.fps / max(1, args.stride))

    # 준비 단계에서 실패해도 리더(=ffmpeg 디코더 자식 프로세스)를 고아로 남기지 않는다.
    writer: Any = None
    try:
        writer = VideoWriter(part_path, out_size, out_fps)
        print(
            f"출력 동영상: {output_path}  "
            f"({out_size[0]}x{out_size[1]} @ {out_fps:.3f}fps, {writer.backend}, "
            f"{writer.encoder_summary()})\n"
            f"  기록 중 임시 파일: {part_path}"
        )
        dirs = open_side_output_dirs(args)
        total = None
        if reader.nframes:
            total = (reader.nframes + args.stride - 1) // args.stride
            if args.max_frames:
                total = min(total, args.max_frames)
        warn_side_output_volume(dirs, total or args.max_frames or None, frame_size)
        progress = Progress(total, f"predict {predictor.label}", args.progress_every)
    except BaseException:
        if writer is not None:
            _close_quietly(writer)
        reader.close()
        raise

    def selected_frames() -> Iterator[Tuple[int, "np.ndarray"]]:
        taken = 0
        for index, frame in enumerate(reader.frames()):
            if index % args.stride:
                continue
            # 짝수 크롭을 **여기서 한 번** 한다 → 모델 입력·마스크·영상 프레임이
            # 모두 같은 기하를 보게 되어 홀수 해상도에서도 1px 어긋남이 없다.
            yield index, crop_to(frame, frame_size)
            taken += 1
            if args.max_frames and taken >= args.max_frames:
                break

    written = 0
    first = True
    legend_mode = getattr(args, "legend", "all")
    completed = False
    # prefetch 제너레이터를 변수로 잡아 둔다 — finally 에서 reader 보다 **먼저** 닫아야
    # 워커 스레드가 순회 중인 리더를 다른 스레드에서 닫는 일이 생기지 않는다.
    frame_iter = prefetch(selected_frames(), args.num_workers)
    try:
        for chunk in batched(frame_iter, args.batch_size):
            indices = [i for i, _ in chunk]
            frames = [f for _, f in chunk]
            result = predictor.predict(frames, resize_wh)
            masks = result["mask"]
            # 렌더 결과를 리스트로 쌓지 않고 한 장씩 writer 로 흘린다(호스트 RAM 상한).
            for k, (source_index, frame) in enumerate(zip(indices, frames)):
                full = upscale_ids(masks[k], (frame.shape[1], frame.shape[0]))
                save_side_outputs(dirs, result, k, f"frame_{source_index:06d}.png", full)
                rendered = render_frame(
                    frame, full, ctx.palette, args.mode, args.alpha, ctx.paint_flags
                )
                rendered = crop_to(rendered, out_size)
                # 계약: 범례 표시 시점은 --legend (all=매 프레임 / first=첫 프레임만 / off).
                if legend_mode == "all" or (legend_mode == "first" and first):
                    rendered = np.asarray(
                        draw_legend(
                            Image.fromarray(rendered), ctx.legend_ids, ctx.class_names, ctx.palette
                        )
                    )
                first = False
                writer.write(rendered)
                written += 1
                progress.update(1)
        completed = True
    finally:
        # 순서가 중요하다. writer 를 **가장 먼저** 닫아야 ffmpeg stdin 이 닫히고 moov
        # atom 이 기록된다(progress.close() 가 BrokenPipeError 로 죽어도 영상은 산다).
        # 각각을 독립적으로 감싸 하나가 실패해도 나머지 정리를 막지 않는다.
        for _closer in (writer.close, frame_iter.close, reader.close, progress.close):
            try:
                _closer()
            except Exception:  # noqa: BLE001 - 정리 중 예외로 다른 정리를 막지 않는다
                pass
        if not completed and part_path.exists():
            print(
                f"[알림] 중단되어 부분 결과가 남았습니다: {part_path}\n"
                f"       완주하지 않았으므로 {output_path} 는 그대로입니다.",
                flush=True,
            )

    if written == 0:
        raise UserError(
            f"처리된 프레임이 0개입니다: {input_path}\n"
            "  --stride / --max_frames 값을 확인하거나 동영상이 온전한지 확인하세요.\n"
            f"  (0프레임 임시 파일이 남아 있으면 지우세요: {part_path})"
        )
    os.replace(part_path, output_path)  # 완주했을 때만 최종 이름을 차지한다
    print(f"완료: {output_path}  ({written} 프레임)")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# 실행: 이미지 / 이미지 디렉터리
# ═══════════════════════════════════════════════════════════════════════════
def run_images(
    args: argparse.Namespace,
    predictor: Any,
    ctx: RenderContext,
    image_paths: Sequence[Path],
    output_is_dir: bool,
) -> int:
    output_path = Path(args.output).expanduser()
    if output_is_dir:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    dirs = open_side_output_dirs(args)
    resize_wh = (int(args.resize[0]), int(args.resize[1]))
    legend_mode = getattr(args, "legend", "all")
    if dirs and image_paths:
        try:  # 용량 추정용 — 헤더만 읽는다. 실패해도 추론은 계속한다.
            with Image.open(image_paths[0]) as probe:
                warn_side_output_volume(dirs, len(image_paths), probe.size)
        except Exception:  # noqa: BLE001
            pass
    progress = Progress(len(image_paths), f"predict {predictor.label}", args.progress_every)

    def loaded() -> Iterator[Tuple[Path, "np.ndarray"]]:
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                raise UserError(f"이미지를 열 수 없습니다: {path}\n  원인: {exc}") from exc
            yield path, np.asarray(image, dtype=np.uint8)

    try:
        for chunk in batched(loaded(), args.batch_size):
            # 배치 안에 해상도가 섞이면 한 장씩 처리한다(이미지 입력에서는 흔하다).
            sizes = {f.shape[:2] for _, f in chunk}
            groups = [chunk] if len(sizes) == 1 else [[item] for item in chunk]
            for group in groups:
                frames = [f for _, f in group]
                result = predictor.predict(frames, resize_wh)
                masks = result["mask"]
                for k, (path, frame) in enumerate(group):
                    full = upscale_ids(masks[k], (frame.shape[1], frame.shape[0]))
                    filename = f"{path.stem}.png"
                    # 최종 방어: 부가 출력이 입력 이미지를 덮어쓰지 않게 저장 직전에 다시 본다
                    # (check_io_paths 가 이미 막지만, 심볼릭 링크/상대경로로 새어 들어올 수 있다).
                    for key, directory in dirs.items():
                        if same_path(directory / filename, path):
                            raise UserError(
                                f"부가 출력({key})이 입력 이미지를 덮어씁니다: {directory / filename}\n"
                                "  --save_mask_dir / --save_chl / --save_conf 를 다른 디렉터리로 지정하세요."
                            )
                    save_side_outputs(dirs, result, k, filename, full)
                    rendered = render_frame(
                        frame, full, ctx.palette, args.mode, args.alpha, ctx.paint_flags
                    )
                    image = Image.fromarray(rendered)
                    if legend_mode != "off":
                        image = draw_legend(image, ctx.legend_ids, ctx.class_names, ctx.palette)
                    destination = output_path / filename if output_is_dir else output_path
                    if same_path(destination, path):
                        raise UserError(
                            f"출력이 입력 이미지를 덮어씁니다: {destination}\n"
                            "  --output 을 다른 파일/디렉터리로 지정하세요."
                        )
                    image.save(destination)
                    progress.update(1)
    finally:
        try:
            progress.close()
        except Exception:  # noqa: BLE001 - 정리 중 예외로 결과를 잃지 않는다
            pass
    print(f"완료: {output_path}")
    return 0


def dispatch(args: argparse.Namespace, kind: str, predictor: Any) -> int:
    """--input 종류에 따라 실행 경로를 고른다. 두 스크립트 공통."""
    ctx = build_render_context(args, predictor.num_classes)
    if kind == "video":
        return run_video(args, predictor, ctx)
    if kind == "dir":
        paths = list_images(Path(args.input).expanduser())
        if args.stride > 1:
            paths = paths[:: args.stride]
        if args.max_frames > 0:
            paths = paths[: args.max_frames]
        return run_images(args, predictor, ctx, paths, output_is_dir=True)
    path = Path(args.input).expanduser()
    output = Path(args.output).expanduser()
    output_is_dir = output.is_dir() or output.suffix.lower() not in IMAGE_EXTS
    return run_images(args, predictor, ctx, [path], output_is_dir=output_is_dir)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ▲▲▲  SHARED CONTRACT BLOCK — END  ▲▲▲                                  ║
# ╚═════════════════════════════════════════════════════════════════════════╝


# ═══════════════════════════════════════════════════════════════════════════
# 여기서부터는 BloomNet 고유(config/모델/추론) — 짝 스크립트와 달라도 되는 부분
# ═══════════════════════════════════════════════════════════════════════════
def _die(message: str) -> None:
    """사용자가 바로 조치할 수 있는 한국어 에러.

    짝 스크립트와 종료 코드를 맞추기 위해 ``UserError`` 를 던진다 (main() 이 2 를 돌려준다).
    """
    raise UserError(message)


def _warn(message: str) -> None:
    print(f"[경고] {message}", file=sys.stderr, flush=True)


def _banner(title: str) -> str:
    return f"\n{'=' * 78}\n  {title}\n{'=' * 78}"


# ═══════════════════════════════════════════════════════════════════════════
#  config / 모델
# ═══════════════════════════════════════════════════════════════════════════
#: 추론에 무관하지만 V 규칙을 건드릴 수 있는 항목만 최소로 중화한다.
#: (경로가 사라진 teacher(V11) / 밴드순서 미확정 이식(V12·V18) / 런타임 유도 스케줄(V15))
_INFERENCE_NEUTRALIZE: Tuple[Tuple[Tuple[str, ...], Any], ...] = (
    (("model", "siam", "enabled"), False),
    (("model", "siam", "teacher"), None),
    (("spec", "transplant", "enabled"), False),
    (("data", "sampler", "class_stats"), None),
    (("schedule", "iters_per_epoch"), None),
    (("schedule", "warmup_iters"), None),
)


def _set_in(tree: Dict[str, Any], keys: Sequence[str], value: Any) -> bool:
    node: Any = tree
    for k in keys[:-1]:
        if not isinstance(node, dict) or k not in node:
            return False
        node = node[k]
    if not isinstance(node, dict) or keys[-1] not in node:
        return False
    if node[keys[-1]] == value:
        return False
    node[keys[-1]] = value
    return True


def config_from_checkpoint(payload: Dict[str, Any], *, source: str) -> Any:
    """``ckpt["args"]`` (= ``cfg.to_dict()``) → ``BloomNetConfig``.

    1차는 **원본 그대로** 시도한다. V 규칙에 걸리면 추론에 무관한 항목만 중화하고
    ``allow_contract_break=True`` 로 재시도하며, 무엇을 바꿨는지 반드시 출력한다.
    """
    from bloomnet.config import from_dict

    raw = payload.get("args")
    if not isinstance(raw, dict) or not raw:
        _die(
            "체크포인트에 config(args)가 없다 — 모델을 재구성할 수 없다.\n"
            "  --config 로 학습에 쓴 YAML(또는 프리셋 이름 s0_rgb_aihub092)을 명시한다.\n"
            "  런 디렉터리의 config.resolved.yaml 을 그대로 주는 것이 가장 안전하다."
        )
    try:
        return from_dict(copy.deepcopy(raw), source=source)
    except Exception as first:  # noqa: BLE001
        data = copy.deepcopy(raw)
        changed = [".".join(k) for k, v in _INFERENCE_NEUTRALIZE if _set_in(data, k, v)]
        try:
            cfg = from_dict(data, source=f"{source} (추론용 중화)", allow_contract_break=True)
        except Exception as second:  # noqa: BLE001
            _die(
                "체크포인트의 config 로 BloomNetConfig 를 만들 수 없다.\n"
                f"  1차 실패: {first}\n"
                f"  중화 후 실패: {second}\n"
                "  --config 로 학습에 쓴 YAML 을 직접 지정하라."
            )
        _warn(
            "체크포인트 config 가 현재 검증 규칙을 통과하지 못해 추론용으로 중화했다.\n"
            f"        1차 실패 사유: {first}\n"
            f"        중화 항목: {changed or '(없음)'} + allow_contract_break=True\n"
            "        아키텍처(channels/decoder/heads)는 손대지 않았으므로 가중치 로딩에는 영향이 없다."
        )
        return cfg


def config_from_file(name: str) -> Any:
    from bloomnet.config import load_config
    from bloomnet.tools._cli import DEFAULT_BASE, resolve_base_path, resolve_config_path

    try:
        path = resolve_config_path(name)
    except FileNotFoundError as exc:
        _die(str(exc))
    try:
        return load_config(path, base=resolve_base_path(str(DEFAULT_BASE)))
    except Exception as exc:  # noqa: BLE001
        _die(f"config 를 로드할 수 없다: {path}\n  원인: {exc}")


def _looks_like_state_dict(obj: Any) -> bool:
    import torch

    if not isinstance(obj, dict) or not obj:
        return False
    vals = list(obj.values())[:64]
    n_tensor = sum(1 for v in vals if isinstance(v, torch.Tensor))
    return n_tensor >= max(1, int(0.9 * len(vals)))


def extract_state_dict(payload: Dict[str, Any], *, ema: str) -> Tuple[Dict[str, Any], str]:
    """05 §6.3 payload 에서 가중치를 꺼낸다. ``(state_dict, 설명)``."""
    if _looks_like_state_dict(payload):
        return dict(payload), "raw state_dict"

    if ema in ("auto", "on"):
        es = payload.get("ema_state_dict")
        shadow = es.get("shadow", es) if isinstance(es, dict) else None
        if _looks_like_state_dict(shadow):
            return dict(shadow), "ema_state_dict.shadow"
        if ema == "on":
            _die(
                "--ema on 인데 체크포인트에 ema_state_dict 가 없다.\n"
                "  --ema off 로 model_state_dict 를 쓰거나 EMA 를 담은 체크포인트를 지정한다."
            )

    for key in ("model_state_dict", "state_dict", "model", "weights"):
        cand = payload.get(key)
        if _looks_like_state_dict(cand):
            return dict(cand), key

    _die(
        "체크포인트에서 가중치를 찾을 수 없다.\n"
        f"  최상위 키: {sorted(payload)[:20]}\n"
        "  05 §6.3 payload 는 'model_state_dict' / 'ema_state_dict' 를 담는다."
    )
    return {}, ""  # pragma: no cover


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    if not any(k.startswith("module.") for k in state):
        return state
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def build_model(cfg: Any, payload: Dict[str, Any], *, ema: str, device: Any) -> Any:
    """config → ``build_bloomnet`` → ``deploy()`` → shape-tolerant load → ``eval()``."""
    from bloomnet.models.bloomnet import build_bloomnet
    from bloomnet.utils.checkpoint import load_state_dict_shape_tolerant

    model = build_bloomnet(cfg)
    # 학습 전용 모듈(edge_head/aux_heads/siam_proj) 제거 + eval + requires_grad_(False).
    # best.pt 는 애초에 이 키들을 담지 않으므로(05 §6.3) 먼저 떼야 missing 보고가 깨끗하다.
    model.deploy()

    state, src = extract_state_dict(payload, ema=ema)
    state = _strip_module_prefix(state)
    report = load_state_dict_shape_tolerant(model, state, verbose=False)
    n_loaded, n_missing = len(report["loaded"]), len(report["missing"])
    print(f"  가중치      {src}: loaded={n_loaded} missing={n_missing} "
          f"unexpected={len(report['unexpected'])} shape_mismatch={len(report['shape_mismatch'])}")
    if n_loaded == 0:
        _die(
            "체크포인트에서 로드된 파라미터가 0개다 — config 와 가중치가 서로 다른 아키텍처다.\n"
            "  --config 로 학습에 쓴 YAML(런 디렉터리의 config.resolved.yaml)을 명시하라."
        )
    if report["missing"]:
        _warn(f"로드되지 않은 파라미터 {n_missing}개 (앞 10개): {report['missing'][:10]}")
    if report["shape_mismatch"]:
        _warn(f"shape 불일치로 건너뛴 키: {report['shape_mismatch'][:10]}")

    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  추론
# ═══════════════════════════════════════════════════════════════════════════
class Predictor:
    """프레임 배치 → {"mask": class-id 맵(--resize 해상도), "chl"?, "conf"?}.

    전처리·업샘플은 공유 블록(resize_frames / to_input_tensor / upscale_ids)만 쓴다.
    원본 4K 프레임을 GPU 에 올리지 않으므로 batch 8 / 4K 에서도 VRAM 스파이크가 없다.
    """

    label = "bloomnet"

    def __init__(self, model: Any, *, device: Any, amp: str, num_classes: int,
                 want_chl: bool, want_conf: bool) -> None:
        import torch

        from bloomnet import constants as _bn_constants

        self.model = model
        self.device = device
        self.amp = amp
        self.num_classes = int(num_classes)
        self.want_chl = bool(want_chl)
        self.want_conf = bool(want_conf)
        # 공유 블록의 IMAGENET_MEAN/STD 는 bloomnet.constants 와 같은 값이어야 한다.
        # (짝 스크립트와 바이트 단위로 같은 전처리를 보장하는 지점이라 런타임에 확인한다)
        for name, shared in (("IMAGENET_MEAN", IMAGENET_MEAN), ("IMAGENET_STD", IMAGENET_STD)):
            canon = tuple(float(v) for v in getattr(_bn_constants, name).flatten().tolist())
            mine = tuple(float(v) for v in shared)
            if len(canon) != len(mine) or any(abs(a - b) > 1e-6 for a, b in zip(canon, mine)):
                _warn(
                    f"공유 블록의 {name}={mine} 가 bloomnet.constants.{name}={canon} 와 다르다 "
                    "— 짝 스크립트와 전처리가 어긋난다. 두 파일을 함께 고쳐라."
                )
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(3, 1, 1)
        self._avail_cache: Dict[int, Any] = {}

    def _avail(self, b: int) -> Any:
        """S0-RGB 규약: ``avail = [1, 0, 0, 0, 0]`` (MODALITY_ORDER = rgb,msi,bio,ir,pol)."""
        import torch

        from bloomnet.constants import MODALITY_ORDER

        if b not in self._avail_cache:
            av = torch.zeros(b, len(MODALITY_ORDER), dtype=torch.float32, device=self.device)
            av[:, 0] = 1.0
            self._avail_cache[b] = av
        return self._avail_cache[b]

    def predict(
        self, frames: Sequence["np.ndarray"], resize_wh: Tuple[int, int]
    ) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        from bloomnet.constants import OUT_CHL, OUT_LOGVAR, OUT_SEG

        rw, rh = int(resize_wh[0]), int(resize_wh[1])
        small = resize_frames(frames, (rw, rh))  # PIL BILINEAR, CPU, uint8
        x = to_input_tensor(small, torch, self.device, self._mean, self._std)

        with torch.no_grad(), amp_context(self.amp, self.device, torch):
            out = self.model(rgb=x, avail=self._avail(x.shape[0]))

        # ── seg: H/4 -> --resize (bilinear) -> argmax ────────────────────────
        # 원본 해상도로 되돌리는 nearest 업샘플은 공유 블록의 upscale_ids(PIL NEAREST)가 한다.
        seg = out[OUT_SEG].float()
        seg = F.interpolate(seg, size=(rh, rw), mode="bilinear", align_corners=False)
        result: Dict[str, Any] = {"mask": seg.argmax(dim=1).to(torch.uint8).cpu().numpy()}

        # ── 부가 출력 (기본 끔). 여기서는 --resize 해상도로 돌려주고, 파일로 쓸 때
        #    공유 블록 save_side_outputs 가 마스크와 같은 원본 해상도로 올린다 ────
        if self.want_chl:
            from bloomnet.modules.heads import ChlHead  # U_MAX 정본 (리터럴 복제 금지)

            u = out[OUT_CHL].float()
            u = F.interpolate(u, size=(rh, rw), mode="bilinear", align_corners=False)
            chl = torch.exp(u.clamp(max=ChlHead.U_MAX)) - 1.0  # 정정 B-13: expm1 금지
            result["chl"] = chl.clamp_(0.0, 500.0).squeeze(1).cpu().numpy()
        if self.want_conf:
            from bloomnet.deploy.postprocess import confidence_map

            s = out[OUT_LOGVAR].float()
            s = F.interpolate(s, size=(rh, rw), mode="bilinear", align_corners=False)
            result["conf"] = confidence_map(s).squeeze(1).cpu().numpy()
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bloomnet.tools.predict_media",
        description=(
            "BloomNet(S0-RGB) 12클래스 하천오염 세그멘테이션 추론 "
            "(이미지 / 이미지 디렉터리 / 동영상 → 오버레이·마스크·좌우비교). "
            "동영상 백엔드는 imageio-ffmpeg(권장, HEVC 가능) → cv2 순으로 자동 선택."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "출력 규약 (이전 구현 쪽 predict_media.py 와 동일)\n"
            "  - 오버레이/마스크는 항상 원본 해상도로 되돌린다(모델은 --resize 로 추론).\n"
            "  - 동영상은 스트리밍 기록이다(전 프레임을 메모리에 쌓지 않는다).\n"
            "  - 팔레트/전처리/업샘플/범례/인코더/CLI 는 두 스크립트가 동일하다.\n"
            "  - 동영상 마스크 파일명은 frame_%06d.png (원본 프레임 인덱스).\n"
            "\n"
            "★ Chl-a 주의: 현재 S0 체크포인트는 Chl-a 라벨 없이 학습됐다.\n"
            "  --save_chl 결과는 prior 근처에서 거의 움직이지 않는 무의미한 값이며\n"
            "  절대 농도/경보 단계로 해석하면 안 된다. --save_conf 도 확률이 아니다.\n"
        ),
    )
    add_common_arguments(parser)

    bn = parser.add_argument_group("BloomNet 전용")
    bn.add_argument(
        "--config", default=None,
        help=("config YAML 경로 또는 동봉 프리셋 이름(예: s0_rgb_aihub092). "
              "미지정 시 체크포인트에 저장된 args 로 모델을 재구성한다"),
    )
    bn.add_argument("--ema", default="auto", choices=["auto", "on", "off"],
                    help="EMA 가중치 사용 (auto = 체크포인트에 있으면 사용)")
    bn.add_argument(
        "--save_chl", default=None, metavar="DIR",
        help=("Chl-a 맵을 16bit PNG 로 저장. 값 = round(mg/m³ × 100). "
              "★ 현재 S0 체크포인트는 Chl-a 라벨 없이 학습되어 이 출력은 무의미하다"),
    )
    bn.add_argument(
        "--save_conf", default=None, metavar="DIR",
        help=("confidence 맵을 16bit PNG 로 저장. 값 = round(conf × 65535), "
              "conf = 1/(1+exp(0.5·s)) ∈ [0.0293, 0.9705]. 확률이 아니라 상대 신뢰도다"),
    )
    bn.add_argument("--print_config", action="store_true",
                    help="복원된 config 를 stdout 에 덤프하고 종료 (추론하지 않는다)")
    return parser


def run(args: argparse.Namespace) -> int:
    # 검사 순서는 짝 스크립트와 같다: 인자 범위 → 입력/출력(+동영상 백엔드) → 체크포인트 → 디바이스.
    validate(args)

    # --print_config 는 --input/--output 을 보지 않는다(존재하지 않는 입력이어도 덤프는 나온다).
    kind = ""
    if not args.print_config:
        kind = classify_input(Path(args.input).expanduser())
        if kind == "video":
            resolve_video_output(args)  # 확장자 보정을 덮어쓰기 알림보다 먼저 끝낸다
        check_io_paths(args, kind)

    # --config 가 있으면 체크포인트도 CUDA 도 필요 없다 — 두 GPU 가 학습 중이어도 덤프는 나온다.
    if args.print_config and args.config:
        from bloomnet.config import to_yaml

        sys.stdout.write(to_yaml(config_from_file(args.config)))
        return 0

    ck_path = Path(args.checkpoint).expanduser()
    if not ck_path.is_file():
        _die(
            f"체크포인트가 없다: {ck_path}\n"
            "  --checkpoint 에 .pt 파일 경로를 준다 (예: outputs/<run>/best.pt).\n"
            "  학습이 진행 중이면 best.pt 가 아직 없을 수 있다 — last.pt 도 확인하라."
        )

    import torch

    if not args.print_config and args.device == "cuda" and not torch.cuda.is_available():
        _die(
            "--device cuda 인데 CUDA 를 쓸 수 없다.\n"
            f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}\n"
            "  GPU 가 다른 작업에 물려 있으면 --device cpu 를 명시하라 (매우 느리다)."
        )
    device = torch.device(args.device)
    if args.amp != "off" and device.type != "cuda":
        print("[알림] --amp 는 cuda 에서만 적용됩니다. CPU 이므로 fp32 로 실행합니다.")

    # ── config + 모델 ────────────────────────────────────────────────────
    print(_banner("BloomNet 추론 — 모델 준비"))
    from bloomnet.utils.checkpoint import load_checkpoint

    print(f"  체크포인트  {ck_path}")
    try:
        payload = load_checkpoint(ck_path, map_location="cpu")  # 헌법 C-5.2: GPU 를 잡지 않는다
    except checkpoint_load_errors() as exc:
        # RuntimeError(PytorchStreamReader) / EOFError / OSError / UnpicklingError /
        # BadZipFile — 문구는 짝 스크립트와 동일하게 공유 블록에서 만든다.
        raise checkpoint_read_error(ck_path, exc) from exc
    except Exception as exc:  # noqa: BLE001
        raise checkpoint_read_error(ck_path, exc) from exc
    if isinstance(payload, dict):
        ep, bm = payload.get("epoch"), payload.get("best_metric")
        if ep is not None:
            print(f"  학습 상태   epoch={ep} best_{payload.get('metric_name', '?')}={bm}")

    if args.config:
        cfg = config_from_file(args.config)
        print(f"  config      --config {args.config}")
    else:
        cfg = config_from_checkpoint(payload, source=str(ck_path))
        print("  config      체크포인트의 args 로 복원")

    if args.print_config:
        from bloomnet.config import to_yaml

        sys.stdout.write(to_yaml(cfg))
        return 0

    num_classes = int(cfg.data.num_classes)
    if tuple(cfg.data.modalities) != ("rgb",):
        _warn(
            f"data.modalities={list(cfg.data.modalities)} — 이 스크립트는 S0-RGB 전용이라 "
            "rgb 만 넣고 avail=[1,0,0,0,0] 로 추론한다. 다른 모달은 결측으로 처리된다."
        )
    if getattr(cfg.data.chl, "supervision", "none") == "none":
        print("  Chl-a       supervision=none — chl 출력은 학습되지 않았다 (아래 경고 참조)")

    model = build_model(cfg, payload, ema=args.ema, device=device)
    del payload
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  모델        BloomNet {cfg.mode} · {num_classes}클래스 · "
          f"{n_param / 1e6:.2f}M params · active_paths={list(model.active_paths)}")
    print(f"  디바이스    {device} · amp={args.amp if device.type == 'cuda' else 'off (cpu)'} "
          f"· resize={args.resize[0]}x{args.resize[1]} · batch={args.batch_size} · mode={args.mode}")

    if args.save_chl:
        _warn(
            "--save_chl 이 켜졌다. 현재 S0 체크포인트는 Chl-a 라벨 없이 학습됐다"
            "(data.chl.supervision=none).\n"
            "        출력 Chl-a 맵은 ChlHead prior 근처에서 거의 움직이지 않는 **무의미한 값**이며\n"
            "        절대 농도나 경보 단계(정상/관심/경계/대발생)로 해석해서는 안 된다."
        )
    if args.save_conf:
        _warn(
            "--save_conf 의 confidence 는 X-25 정의(1/(1+exp(0.5·s)))의 **단위 없는 상대 신뢰도**다.\n"
            "        캘리브레이션된 확률이 아니므로 '신뢰도 90% 이상' 같은 문구를 만들면 안 된다."
        )

    predictor = Predictor(
        model, device=device, amp=args.amp, num_classes=num_classes,
        want_chl=bool(args.save_chl), want_conf=bool(args.save_conf),
    )
    return dispatch(args, kind, predictor)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except UserError as exc:
        print(f"\n[오류] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 중지했습니다. 여기까지 기록된 결과는 그대로 남습니다.", file=sys.stderr)
        return 130
    except MemoryError as exc:
        print(f"\n[오류] {out_of_memory_message(exc)}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - torch.OutOfMemoryError 만 걸러 내고 나머지는 그대로
        if not is_out_of_memory_error(exc):
            raise
        print(f"\n[오류] {out_of_memory_message(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
