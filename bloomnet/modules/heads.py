"""Multi-task 헤드 7종 — 04 §8 / 05 §2.3.2 / 06 §3.3.8 (레벨 L2).

추론 유지: :class:`SegHead`, :class:`RegTrunk`, :class:`ChlHead`, :class:`UncHead`.
학습 전용: :class:`EdgeHead`, :class:`AuxSegHead`, :class:`SiamProjection`
(`deploy()` 가 제거한다 — `constants.TRAIN_ONLY_MODULES`).

동결 사항
    * **헤드 activation = GELU(approximate='tanh')** (04 §1.2/§1.3, 사업문서 s42/43).
      ``'tanh'`` 근사로 고정해 학습과 ONNX export 가 같은 수를 낸다(04 §9.3-G).
    * **``torch.exp(u) - 1.0`` 만 쓴다** (정정 B-13, 06 §10 규칙 13). ``exp``+``m1`` 로 붙여 쓰는
      단항 연산자는 ONNX 표준 opset 에 **존재하지 않아** export 가 확정 실패한다
      (venv 실측: ``UnsupportedOperatorError``, ``symbolic_opset9`` 에 심볼릭 부재).
      교체 오차는 최대 3.05e-05 (상대 4.10e-05) 로 Chl-a 라벨 정밀도보다 3자릿수 작다.
      **이 파일에 그 이름이 문자열로도 등장하지 않는 것 자체가 계약**이다(04 §13.7 회귀 테스트).
    * **``log_var`` clamp 는 :class:`UncHead` 의 ``[-7, 7]`` 하나뿐이다** (정정 B-17/A-20, X-26).
      배포 경로에는 criterion 이 없으므로 헤드 clamp 만이 학습·추론 양쪽에서 동일 작동한다.
      파생: ``σ ∈ [0.0302, 33.115]``, ``conf = 1/(1+σ) ∈ [0.0293, 0.9705]`` — **1 에 도달하지
      못한다**(X-25).
    * **``num_classes`` 의존 텐서는 ``cls.weight``/``cls.bias`` 둘뿐이다** (헌법 C-6 R7).
      S0(12클래스) → S1(2클래스) 전이는 shape-filtered partial load 로 처리한다.

배포 정밀도 정책 (04 §9.3-L, `deploy/trt_policy.py` 소관)
    :class:`ChlHead` · :class:`UncHead` 는 **INT8 엔진에서 fp16 고정**(``FP16_LOCKED``) 대상이다 —
    ``Softplus``/``exp`` 는 동적 범위가 커서 양자화 스케일 추정이 실패한다.
    ``fp32_forced`` 는 ``bmef.fusion`` + ``encoder.*.litemla.attention`` 뿐이며(정정 A-22/B-6)
    헤드에는 해당 사항이 없다. FP16 안전성은 ``Z_MAX=11`` clamp(§9.3-D)와
    ``S_MAX=7`` clamp(§9.3-F)가 담보한다.

파라미터 회귀 상수 (06 §10)
    SegHead 147,970(K=2) / 149,260(K=12) · RegTrunk 9,728 · ChlHead 65 · UncHead 65(66) ·
    EdgeHead 9,313 · AuxSegHead 19,020 / 93,388 / 371,084 (K=12, enc taps) + 74,892 (p8) ·
    SiamProjection 51,840 / 164,864 / 33,280.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bloomnet.constants import BD_CH, DEC_CH, K235_LOG1P_MEAN
from bloomnet.modules.common import build_act, build_norm, init_encoder

__all__ = [
    "SegHead",
    "RegTrunk",
    "ChlHead",
    "UncHead",
    "EdgeHead",
    "AuxSegHead",
    "SiamProjection",
]

_HEAD_NORM = "bn"  # 04 §1.2 — 헤드도 BatchNorm 고정 (TRT conv+BN fusion)
_REG_MID = 64


def _softplus_inv(y: float) -> float:
    """``softplus(x) = y`` 의 해. ``x = log(exp(y) - 1)``.

    ``y > 0`` 에서만 정의된다. y 가 1e-3 미만이면 ``exp(y)-1`` 의 상쇄로 유효숫자를 잃으므로
    막는다 — 실제 사용값은 ``log1p`` 공간 평균 1.7559 (X-10) 다.
    """
    if not (y > 1e-3):
        raise ValueError(f"softplus_inv 는 y > 1e-3 에서만 쓴다 (got {y!r})")
    return math.log(math.exp(y) - 1.0)


class SegHead(nn.Module):
    """주 세그멘테이션 헤드 — 04 §8.1. ``Conv3x3 → BN → GELU → Conv1x1``.

    입력 ``F_dec`` (B,128,H/4,W/4) → ``seg_logits_s4`` (B,K,H/4,W/4).
    **업샘플하지 않는다** — 학습은 criterion, 배포는 ``export_forward`` 가 올린다(04 §9.1).

    params 147,970 (K=2) / 149,260 (K=12). MAC 9.6805 GMAC @1024² = 디코더+헤드의 41 %.
    지연시간이 부족하면 ``mid=64`` 가 첫 번째 레버다(−4.84 GMAC) — 다만 헌법 C-4 가 128 을
    명시하므로 기본값은 바꾸지 않는다.
    """

    def __init__(self, in_ch: int = DEC_CH, mid: int = DEC_CH, num_classes: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, mid, 3, stride=1, padding=1, bias=False)
        self.bn = build_norm(_HEAD_NORM, mid)
        self.act = build_act("gelu")
        self.cls = nn.Conv2d(mid, num_classes, 1, bias=True)
        self.apply(init_encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.cls(self.act(self.bn(self.conv(x))))


class RegTrunk(nn.Module):
    """Chl-a / 불확실성 공유 trunk — 04 §8.2. ``DW3x3+BN → PW1x1+BN → GELU``.

    (B,128,H/4,W/4) → (B,64,H/4,W/4). params 9,728.

    depthwise-separable 인 근거: Chl-a 필드는 IDW 보간 GT 가 115 m footprint 안에서 사실상
    이미지당 상수라 고주파 표현력이 필요 없다([분석] I-17). full ``Conv3x3(128→64)`` 대비
    0.612 vs 4.83 GMAC @1024² 로 8배 싸다.
    평균과 분산을 같은 표현에서 뽑는 것은 Kendall–Gal 2017 heteroscedastic head 의 표준형이다.
    """

    def __init__(self, in_ch: int = DEC_CH, mid: int = _REG_MID) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1, groups=in_ch, bias=False)
        self.bn1 = build_norm(_HEAD_NORM, in_ch)
        self.pw = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn2 = build_norm(_HEAD_NORM, mid)
        self.act = build_act("gelu")
        self.apply(init_encoder)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn2(self.pw(self.bn1(self.dw(x)))))


class ChlHead(nn.Module):
    """Chl-a 회귀 헤드 — 04 §8.2.1. 네트워크는 ``u = log1p(Chl)`` 를 예측한다.

    ``z = clamp(Conv1x1(64→1)(f), Z_MIN, Z_MAX)`` → ``u = softplus(z) ≥ 0``.
    ``return_mgm3=True`` 면 ``exp(clamp(u, max=U_MAX)) - 1 ∈ [0, 500]`` 로 되돌린다.

    log 공간을 쓰는 근거: 235 실측 분포가 우편향 로그정규형(median 3.75, max 90.34)이라
    원공간 Huber 는 대발생 표본이 손실을 지배하고, 분광 지수와의 상관도 원공간 r=0.638 대
    log 공간 r=0.697 로 로그 쪽이 더 선형적이다.

    Args:
        prior_log1p_mean: 초기 예측값(log1p 공간). bias 를 ``softplus⁻¹`` 로 역산해
            **첫 스텝부터 전역 평균을 예측**하게 한다. 기본 1.7559 = 8-site 필터후 log1p 평균
            (★X-10 — 04 초판의 2.00553 은 3-site 부분집합 median 7.43 기반이라 폐기).
            weight = 0 이므로 초기 출력은 입력과 무관하게 정확히 이 값이다.

    Note:
        ``Z_MAX = 11.0`` 은 fp16 안전 장치다 — ONNX ``Softplus`` 에는 PyTorch 의 threshold=20
        선형 fallback 이 없어 ``exp(z)`` 를 직접 계산하고 z > 11.09 에서 65504 를 넘긴다.
        학습에서는 안 터지고 **배포에서만** 터지는 종류의 버그다(04 §9.3-D).
    """

    U_MAX: float = 6.2166061  # log1p(500) — chl <= 500 mg/m³
    Z_MIN: float = -20.0
    Z_MAX: float = 11.0

    def __init__(self, in_ch: int = _REG_MID, *, prior_log1p_mean: float = K235_LOG1P_MEAN) -> None:
        super().__init__()
        self.prior_log1p_mean = float(prior_log1p_mean)
        self.out = nn.Conv2d(in_ch, 1, 1, bias=True)
        self.apply(init_encoder)
        # ★ init_encoder 뒤에 덮어쓴다 (X-10): weight=0, bias=softplus⁻¹(prior)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, _softplus_inv(self.prior_log1p_mean))

    def forward(self, f: Tensor, *, return_mgm3: bool = False) -> Tensor:
        z = self.out(f).clamp(min=self.Z_MIN, max=self.Z_MAX)
        u = F.softplus(z)
        if not return_mgm3:
            return u  # 학습 손실은 이 공간에서 (★X-14)
        # ★ (정정 B-13) 여기서 exp(·)-1 대신 대체 함수를 쓰면 ONNX export 가 실패한다
        return torch.exp(u.clamp(max=self.U_MAX)) - 1.0


class UncHead(nn.Module):
    """aleatoric log-variance 헤드 — 04 §8.2.2. ``s = log σ²`` 를 ``u`` 공간 기준으로 낸다.

    ``s = clamp(Conv1x1(→1)(f'), -7, 7)`` → ``σ ∈ [0.0302, 33.115]``.
    회귀에만 붙인다 — 분류용 aleatoric 은 로짓 MC 샘플링(T≈10–50)이 필요해 H/4 에서 비용 과다이고,
    세그멘테이션 불확실성은 softmax 엔트로피로 무료로 얻는다.

    hard clamp 를 쓰는 이유: 범위 밖은 이미 발산 상태라 gradient 를 끊는 편이 안전하다.
    soft clamp 는 정상 영역의 gradient 까지 압축한다. **σ 를 직접 예측하지 않는다.**

    Args:
        use_vacuity: ``True`` 면 입력 채널이 ``in_ch+1`` 이 되고 BMEF ``vacuity`` 를 concat
            한다(★X-12, 기본 False). ``vacuity`` 는 감독 신호가 전혀 없고 03 자신이 U-5 로
            미결정 처리했으므로 기본 경로에서는 **진단 전용**이다.

    Note:
        ``enabled=False`` 는 05 소관의 warm-up 훅이다 — ``σ²=1`` 이 되어 손실이 순수
        Huber 로 환원된다. degenerate 해(σ 를 부풀려 gradient 를 지우는 붕괴)가 저데이터에서
        알려진 위험이다.
    """

    S_MIN: float = -7.0
    S_MAX: float = 7.0

    def __init__(self, in_ch: int = _REG_MID, *, use_vacuity: bool = False) -> None:
        super().__init__()
        self.use_vacuity = bool(use_vacuity)
        self.out = nn.Conv2d(in_ch + (1 if use_vacuity else 0), 1, 1, bias=True)
        self.apply(init_encoder)
        # ★ init_encoder 뒤에 덮어쓴다: s ≡ 0 → σ² = 1 → 초기 손실이 정확히 ½·Huber
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self, f: Tensor, *, vacuity: Optional[Tensor] = None, enabled: bool = True
    ) -> Tensor:
        if self.use_vacuity:
            if vacuity is None:
                raise ValueError("UncHead(use_vacuity=True) 는 vacuity (B,1,h,w) 를 요구한다")
            f = torch.cat([f, vacuity], dim=1)
        # use_vacuity=False 이면 vacuity 는 조용히 무시된다 (호출부가 항상 넘겨도 되게).
        s = self.out(f).clamp(min=self.S_MIN, max=self.S_MAX)
        if not enabled:
            s = torch.zeros_like(s)
        return s


class EdgeHead(nn.Module):
    """경계 헤드 (학습 전용) — 04 §8.3. ``d32`` 에서 분기한다.

    구조는 :class:`SegHead` 규약(``Conv3x3 → BN → GELU → Conv1x1``)을 재사용하되 채널만 32.
    ``F_dec`` 이 아니라 D 분기에서 뽑는 이유는 경계 분기를 둔 목적 자체가 그 분기로 경계를
    학습시키는 것이기 때문이다([분석] D-DEC-10 이 Phase E 다이어그램을 오류로 판정).
    params 9,313. 추론 시 제거된다 — **D 분기 자체(``d128``)는 항상 활성**이다.

    Args:
        prior_pos: 경계 픽셀 기대 비율. ``out.bias = log(p/(1−p))`` (RetinaNet prior init)
            로 극단 불균형에서 초기 손실 폭주를 막는다. 기본 0.03 은 01 [M4] 의 H/4 양성비
            실측 3.112 % 를 반영한 값이다(04 초판의 가정치 0.02 대신).
    """

    def __init__(self, in_ch: int = BD_CH, mid: int = BD_CH, *, prior_pos: float = 0.03) -> None:
        super().__init__()
        if not (0.0 < prior_pos < 1.0):
            raise ValueError(f"prior_pos must be in (0,1), got {prior_pos!r}")
        self.prior_pos = float(prior_pos)
        self.conv = nn.Conv2d(in_ch, mid, 3, stride=1, padding=1, bias=False)
        self.bn = build_norm(_HEAD_NORM, mid)
        self.act = build_act("gelu")
        self.out = nn.Conv2d(mid, 1, 1, bias=True)
        self.apply(init_encoder)
        nn.init.constant_(self.out.bias, math.log(self.prior_pos / (1.0 - self.prior_pos)))

    def forward(self, d32: Tensor) -> Tensor:
        return self.out(self.act(self.bn(self.conv(d32))))


class AuxSegHead(nn.Module):
    """deep supervision 헤드 (학습 전용) — 05 §2.3.2 (★X-06).

    PIDNet ``segmenthead`` 와 동일한 **pre-activation** 형태:
    ``BN → ReLU → Conv3x3(bias=False) → BN → ReLU → Conv1x1``.
    04 §8.4 의 post-activation 형태(``Conv3x3→BN→GELU→1x1``)가 아니다 — X-06 이 05 구조를
    채택했고 T16 의 파라미터 회귀도 그 값이다.

    ``c_mid`` 는 규칙이 아니라 **리터럴 표**다 (정정 B-21): ``constants.AUX_TAP_MID`` =
    ``{enc_s8: 32, enc_s16: 64, enc_s32: 128, p8: 64}``. ``C_in//2`` 규칙으로 구현하면
    enc_s16 93,388 → 116,652, enc_s32 371,084 → 463,692 로 T16 이 즉시 깨진다.

    params ``2·c_in + 9·c_in·c_mid + 2·c_mid + c_mid·K + K``.
    손실은 OHEM 없는 평범한 CE 다(PIDNet 공식 구현과 동일) — 05 소관.
    """

    def __init__(self, c_in: int, c_mid: int, num_classes: int) -> None:
        super().__init__()
        self.bn1 = build_norm(_HEAD_NORM, c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(c_in, c_mid, 3, stride=1, padding=1, bias=False)
        self.bn2 = build_norm(_HEAD_NORM, c_mid)
        self.cls = nn.Conv2d(c_mid, num_classes, 1, bias=True)
        self.apply(init_encoder)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(self.relu(self.bn1(x)))
        return self.cls(self.relu(self.bn2(y)))


class SiamProjection(nn.Sequential):
    """SIAM(SCTNet) student→teacher 채널 사영 (학습 전용, 기본 비활성 λ_siam=0).

    ``Conv1x1(bias=False) + BN``. 160→320 = 51,840 / 320→512 = 164,864 / 128→256 = 33,280.

    1차년도에는 쓰지 않는다 — SegFormer-B2 teacher 가 로컬 캐시에 없어 헌법 C-5.3(인터넷
    의존 금지)과 충돌하고, RGB-only teacher 는 다중모달이 이기는 바로 그 영역(glint·탁수)에서
    틀린 답을 강요한다([분석] D-SIAM-01). 코드 경로만 만들고 승격 게이트를 둔다.
    """

    def __init__(self, c_student: int, c_teacher: int) -> None:
        super().__init__(
            nn.Conv2d(c_student, c_teacher, 1, bias=False),
            build_norm(_HEAD_NORM, c_teacher),
        )
        self.apply(init_encoder)
