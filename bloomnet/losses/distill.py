"""Distillation 손실 프리미티브 — 05 §2.6 / API 동결 06 §3.5 (레벨 L1).

1차년도 기본값은 ``λ_siam = 0.0`` 이다(05 §2.6.2: 지정된 teacher 가 로컬 캐시에 없고
헌법 C-5.3 이 인터넷 의존을 금지한다). 그럼에도 코드 경로는 구현·CPU 테스트한다(C-5.1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["cwd_loss"]


def cwd_loss(student: Tensor, teacher: Tensor, *, T: float = 4.0) -> Tensor:
    """Channel-Wise Distillation (SCTNet/CWD, Shu 2021) — 05 §2.6.1.

    ``φ_c(y)_i = softmax_i(y_{c,i} / T)``  (채널별로 **공간축**에 softmax)
    ``L = (T² / (C·N)) · Σ_{b,c,i} φ_c(t)_i · [log φ_c(t)_i − log φ_c(s)_i]``

    정규화 상수의 ``N`` 은 **배치 크기**다(공식 구현 관례: 행 = ``N*C`` 로 flatten 후
    합계를 ``C*N`` 으로 나눔). teacher 는 항상 detach 되므로 gradient 가 student 로만 흐른다
    (05 §2.6.3 teacher 위생 규칙: ``eval()`` / ``requires_grad_(False)`` / ``no_grad``).

    Args:
        student: ``(B, C, h, w)``. 공간 크기가 다르면 teacher 격자로 bilinear 업샘플한다.
        teacher: ``(B, C, H, W)``.
        T: 온도. SCTNet 공식값 4.0.

    Returns:
        스칼라. KL 이므로 항상 >= 0 이고, 두 입력이 같으면 정확히 0 이다.
    """
    if tuple(student.shape[-2:]) != tuple(teacher.shape[-2:]):
        student = F.interpolate(
            student,
            size=(int(teacher.shape[-2]), int(teacher.shape[-1])),
            mode="bilinear",
            align_corners=False,
        )

    b, c = student.shape[0], student.shape[1]
    s = student.float().reshape(b, c, -1)  # (B,C,N_spatial)
    t = teacher.float().reshape(b, c, -1).detach()

    log_pt = F.log_softmax(t / T, dim=-1)
    log_ps = F.log_softmax(s / T, dim=-1)
    pt = log_pt.exp()

    # Σ p_t (log p_t − log p_s) — p_t 는 detach 되어 있으므로 gradient 는 log_ps 로만 간다.
    kl = (pt * (log_pt - log_ps)).sum()
    return kl * (T * T) / float(c * b)
