"""배포 후처리 (L1) — 06 §3.6 / 04 §8.2.3 · §8.5.

헌법 C-1: 경보 단계 결정은 **후처리**이며 end-to-end 학습 대상이 아니다.
학습 그래프에 이 모듈의 어떤 함수도 들어가지 않는다.

단일 출처 규약
--------------
* 경보 임계는 :data:`bloomnet.constants.ALARM_THRESHOLDS_MGM3` 만을 쓴다.
* 버킷 경계 규약(``x >= 15`` → 관심)은 :func:`bloomnet.utils.metrics_reg.alarm_level`
  이 유일 구현이며 여기서는 **위임**한다. 리터럴을 복제하면 평가 경로(metrics_reg)와
  배포 경로가 조용히 갈라진다.
* ``conf`` 는 X-25(정정 A-20)가 동결한 ``1/(1+exp(0.5·s))`` 가 **유일 구현**이며
  T25 가 강제한다.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from bloomnet.constants import ALARM_THRESHOLDS_MGM3
from bloomnet.utils.metrics_reg import alarm_level

__all__ = ["chl_to_alert_level", "confidence_map", "sigma_chl", "CONF_MIN", "CONF_MAX"]

#: ``s ∈ [-7, 7]`` hard clamp(04 §8.2.2, 정정 B-17)가 유도하는 conf 의 실제 상·하한.
#: ``conf`` 는 1 에도 0 에도 도달하지 않는다 — T25 가 이 두 값을 assert 한다.
CONF_MIN: float = 0.029312
CONF_MAX: float = 0.970688


def chl_to_alert_level(
    chl_mgm3: Tensor,
    *,
    levels: Tuple[float, float, float] = ALARM_THRESHOLDS_MGM3,  # type: ignore[assignment]
    seg_id: Optional[Tensor] = None,
    algae_id: int = 1,
) -> Tensor:
    """Chl-a(mg/m³) → 경보 단계 ``{0 정상, 1 관심, 2 경계, 3 대발생}``.

    Args:
        chl_mgm3: ``(B,1,H,W)`` 또는 ``(B,H,W)``. **mg/m³ 원단위** (log1p 공간 아님).
        levels: 임계 3개. 기본 ``(15, 25, 100)`` = 사업문서 s5.
        seg_id: 주면 ``seg_id == algae_id`` 가 아닌 픽셀의 단계를 **0 으로 마스킹**한다.
            녹조가 아닌 영역(하늘·식생)의 회귀 출력이 경보로 새는 것을 막는다.
        algae_id: 녹조 클래스 id.

    Returns:
        ``chl_mgm3`` 와 같은 shape 의 ``int64`` 텐서.

    Note:
        경계값은 **이상**이 상위 단계다(``x == 15`` → 1). 이 규약은
        ``utils.metrics_reg.alarm_level`` 의 ``bucketize(..., right=True)`` 가 담보한다.
    """
    lv = alarm_level(chl_mgm3, thresholds=tuple(levels))
    if seg_id is not None:
        sid = seg_id
        if sid.dim() == lv.dim() - 1:
            sid = sid.unsqueeze(1)
        if sid.shape != lv.shape:
            raise ValueError(f"seg_id shape {tuple(sid.shape)} != chl shape {tuple(lv.shape)}")
        lv = torch.where(sid == int(algae_id), lv, torch.zeros_like(lv))
    return lv


def confidence_map(log_var: Tensor) -> Tensor:
    """``s = log σ_u²`` → 상대 신뢰도 ``conf = 1/(1 + exp(0.5·s)) = 1/(1 + σ_u)``.

    **단위 없는 상대 신뢰도이며 절대 확률이 아니다.** 순위 비교(어느 픽셀이 더
    믿을 만한가)에만 쓴다. 캘리브레이션된 확률로 해석하거나 임계값을 씌워
    "신뢰도 90% 이상" 같은 문구를 만들면 안 된다.

    ``UncHead`` 가 ``s`` 를 ``[-7, 7]`` 로 hard clamp 하므로 출력은 항상
    ``[CONF_MIN, CONF_MAX] = [0.0293, 0.9705]`` 안에 있다 — 1 에 도달하지 않는다.
    05 §2.5.4 의 ``exp(-0.5·s)`` 정의는 s=-7 에서 33.1 이 되어 유계 계약을
    원리적으로 만족할 수 없으므로 폐기됐다 (X-25, 정정 A-20/B-23).
    """
    s = log_var.to(torch.promote_types(log_var.dtype, torch.float32))
    return 1.0 / (1.0 + torch.exp(0.5 * s))


def sigma_chl(log_var: Tensor, chl_mgm3: Tensor) -> Tensor:
    """mg/m³ 원단위의 표준편차 추정 (delta method, 04 §8.5).

    모델은 log1p 공간에서 ``σ_u = exp(0.5·s)`` 를 낸다. ``chl = exp(u) - 1`` 이므로
    ``d chl / d u = exp(u) = 1 + chl`` → ``σ_chl ≈ σ_u · (1 + chl)``.

    1차 근사이므로 ``σ_u`` 가 크면(> 0.5 정도) 과소평가한다. UI 의 오차막대 용도이며
    통계적 신뢰구간이 아니다.
    """
    s = log_var.to(torch.promote_types(log_var.dtype, torch.float32))
    c = chl_mgm3.to(s.dtype)
    return torch.exp(0.5 * s) * (1.0 + c)
