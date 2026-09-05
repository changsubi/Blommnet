"""옵티마이저 · 파라미터 그룹 (05 §5.1.2 / 02 §11 / 03 §13.1, 06 §3.6 동결).

4 그룹 (`PARAM_GROUPS`, 이름 동결):

===========  ===================================  ==============  ===============
그룹          내용                                  lr              weight_decay
===========  ===================================  ==============  ===============
``physics``  ``model.physics_params()``            lr × mult(50)   0.0
``no_decay`` ``model.no_weight_decay()``           lr              0.0
``encoder``  사전학습된 인코더 파라미터              lr_encoder      wd
``main``     나머지(디코더·헤드 + fresh encoder)     lr              wd
===========  ===================================  ==============  ===============

우선순위는 ``physics > no_decay > encoder > main`` 이며, 각 파라미터는 **정확히 한
그룹**에만 들어간다. physics 가 no_decay 보다 앞서는 이유는 두 집합이 겹치는데
(physics scalar 는 전부 no_decay 이기도 하다) lr×50 이 적용되어야 하기 때문이다.

**정정 B-28 (fresh_encoder)** — "encoder 그룹 = 인코더 전체" 로 두면 S1 Phase B 에서
`lr_encoder = 6e-5` 가 **사전학습된 적이 없는 6 M 파라미터**(SPS/TPS·spec/phys path·
BMEF 정밀도 계열)에도 적용되어 심각한 underfit 이 된다. 06 §3.6 이 그룹 **이름 4개**를
동결했으므로 `fresh_encoder` 라는 다섯 번째 이름을 만들지 않고, 대신
:func:`build_optimizer` 의 ``pretrained_keys`` 로 **encoder 그룹의 멤버십을 좁힌다** —
사전학습 체크포인트에 실제로 존재했던 키만 ``encoder``(보호 대상)로 가고, 나머지
인코더 파라미터는 ``main``(= from-scratch lr)로 떨어진다. ``pretrained_keys=None``
(기본, S0)이면 인코더 전체가 ``encoder`` 그룹이며 06 문자 그대로의 동작이다.
S0 에서는 ``lr_encoder`` 가 null → ``lr`` 과 같으므로 어느 쪽이든 값이 동일하다.

**빈 그룹도 유지한다.** 05 §5.1.2 가 "S0 에서는 비어 있다. 그룹만 존재" 라고 명시했고,
그래야 모드 전환 시 ``optimizer_state_dict`` 의 그룹 수가 달라지지 않아 재개가 깨지지
않는다.

레벨 L2 — `bloomnet.config`(L1)는 타입 힌트로만 참조한다(런타임 duck-typing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Final, Iterable, List, Optional, Sequence, Set

import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover - 런타임 의존 없음
    from bloomnet.config import BloomNetConfig, OptimConfig

__all__ = [
    "PARAM_GROUPS",
    "ENCODER_PREFIXES",
    "collect_no_decay",
    "collect_physics",
    "assign_param_groups",
    "build_optimizer",
]

#: 06 §3.6 동결. 순서도 계약이다 — ``optimizer.param_groups[i]`` 인덱스가 이 순서다.
PARAM_GROUPS: Final = ("main", "encoder", "no_decay", "physics")

#: 인코더 소속 판정용 dotted 경로 **세그먼트 접두**. `models/bloomnet.py` 의 등록 이름과
#: 맞물린다 (`backbone.encoder.…`, `backbone.bmef…`). 부분 문자열이 아니라 세그먼트
#: 단위로 비교하므로 `main_encoder_gate` 같은 무관한 이름을 삼키지 않는다
#: (`utils/checkpoint.prune_train_only_keys` 와 동일 규약).
ENCODER_PREFIXES: Final[tuple] = ("backbone", "encoder", "stems", "ppn", "sps", "tps", "bmef")


# ═══════════════════════════════════════════════════════════════════════
#  파라미터 이름 집합
# ═══════════════════════════════════════════════════════════════════════
def collect_no_decay(model: nn.Module) -> Set[str]:
    """weight decay 를 걸지 않을 파라미터 **이름** 집합.

    정본은 ``model.no_weight_decay()`` 위임이다 (06 §3.4.3 이 그 목록을 동결했다:
    norm affine, 모든 bias, LayerScale γ, ``ppn.a_hat``/``ppn.b``, ``ppn.inpainter.*``,
    ``sps/tps.c_abs``, ``biogate.beta_hat``, ``bmef.p_raw_*``/``kappa_raw``/``log_tau0``/
    ``fb_gamma_*``).

    모델이 그 메서드를 제공하지 않으면(더미 모델·단위 테스트) ``ndim <= 1`` 휴리스틱으로
    떨어진다 — norm affine·bias·1D 스칼라를 정확히 잡는 timm 관례다. 폴백을 썼는지
    여부는 반환값만 봐서는 알 수 없으므로, 학습 경로는 반드시 정본을 쓰는 모델을 넘겨라.
    """
    fn = getattr(model, "no_weight_decay", None)
    if callable(fn):
        names = set(fn())
        known = {n for n, _ in model.named_parameters()}
        unknown = sorted(names - known)
        if unknown:
            raise KeyError(
                f"no_weight_decay() 가 존재하지 않는 파라미터 이름을 반환했다: {unknown[:5]}"
            )
        return names
    return {name for name, p in model.named_parameters() if p.ndim <= 1}


def collect_physics(model: nn.Module) -> Set[str]:
    """lr×50 물리 스칼라 이름 집합. ``model.physics_params()`` 위임, 없으면 빈 집합.

    06 §3.4.3: ``ppn.a_hat``, ``ppn.b``, ``biogate.beta_hat``, ``bmef.kappa_raw``,
    ``bmef.log_tau0``.
    """
    fn = getattr(model, "physics_params", None)
    if not callable(fn):
        return set()
    names = set(fn())
    known = {n for n, _ in model.named_parameters()}
    unknown = sorted(names - known)
    if unknown:
        raise KeyError(f"physics_params() 가 존재하지 않는 이름을 반환했다: {unknown[:5]}")
    return names


def _is_encoder(name: str, prefixes: Sequence[str]) -> bool:
    return any(seg.startswith(p) for seg in name.split(".") for p in prefixes)


def assign_param_groups(
    model: nn.Module,
    *,
    pretrained_keys: Optional[Iterable[str]] = None,
    encoder_prefixes: Sequence[str] = ENCODER_PREFIXES,
) -> Dict[str, List[str]]:
    """파라미터 이름을 4 그룹으로 **분할**한다 (교집합 없음, 합집합 = 전체).

    ``requires_grad=False`` 인 파라미터는 어느 그룹에도 넣지 않는다 (S1 Phase A 동결).
    """
    no_decay = collect_no_decay(model)
    physics = collect_physics(model)
    pre: Optional[Set[str]] = None if pretrained_keys is None else set(pretrained_keys)

    groups: Dict[str, List[str]] = {g: [] for g in PARAM_GROUPS}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in physics:
            groups["physics"].append(name)
        elif name in no_decay:
            groups["no_decay"].append(name)
        elif _is_encoder(name, encoder_prefixes) and (pre is None or name in pre):
            groups["encoder"].append(name)
        else:
            groups["main"].append(name)
    return groups


# ═══════════════════════════════════════════════════════════════════════
#  옵티마이저
# ═══════════════════════════════════════════════════════════════════════
def _optim_cfg(cfg: Any) -> Any:
    """``BloomNetConfig`` 를 받아도 ``OptimConfig`` 를 받아도 동작하게 한다."""
    return getattr(cfg, "optim", cfg)


def build_optimizer(
    model: nn.Module,
    cfg: "OptimConfig | BloomNetConfig",
    *,
    pretrained_keys: Optional[Iterable[str]] = None,
    encoder_prefixes: Sequence[str] = ENCODER_PREFIXES,
) -> torch.optim.AdamW:
    """05 §5.1.2 의 4 그룹 AdamW.

    Args:
        cfg: ``OptimConfig`` (또는 ``.optim`` 을 가진 ``BloomNetConfig``).
        pretrained_keys: 정정 B-28 — ``encoder`` 그룹에 넣을 파라미터 이름 화이트리스트.
            보통 ``train.init_from`` 체크포인트에서 **실제로 로드된 키**
            (``load_state_dict_shape_tolerant(...)["loaded"]``) 를 넘긴다.
            None 이면 인코더 전체가 ``encoder`` 그룹(06 문자 그대로).

    Note:
        ``no_decay`` 그룹은 05 §5.1.2 대로 **단일 lr(=`optim.lr`)** 을 쓴다. 따라서 S1
        Phase B 에서 사전학습된 인코더의 norm affine·bias 는 `lr_encoder` 가 아니라
        `lr` 을 받는다. 그룹 이름 4개가 06 에서 동결되어 있어 이를 나눌 자리가 없다 —
        S1 레시피 확정 시 06 §3.6 의 `PARAM_GROUPS` 개정이 필요하다(인계 사항).
    """
    o = _optim_cfg(cfg)
    name = str(getattr(o, "name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"optim.name 은 'adamw' 만 지원한다 (받은 값 {name!r}) — 05 §5.1.2")

    lr = float(o.lr)
    lr_encoder = float(o.lr_encoder) if getattr(o, "lr_encoder", None) is not None else lr
    wd = float(o.weight_decay)
    physics_lr = lr * float(getattr(o, "physics_lr_mult", 50.0))
    betas = tuple(float(b) for b in getattr(o, "betas", (0.9, 0.999)))
    eps = float(getattr(o, "eps", 1e-8))

    named = dict(model.named_parameters())
    assign = assign_param_groups(
        model, pretrained_keys=pretrained_keys, encoder_prefixes=encoder_prefixes
    )
    spec = {
        "main": (lr, wd),
        "encoder": (lr_encoder, wd),
        "no_decay": (lr, 0.0),
        "physics": (physics_lr, 0.0),
    }
    param_groups: List[Dict[str, Any]] = []
    for g in PARAM_GROUPS:
        g_lr, g_wd = spec[g]
        param_groups.append(
            {
                "name": g,
                "params": [named[n] for n in assign[g]],
                "lr": g_lr,
                "weight_decay": g_wd,
            }
        )
    if not any(pg["params"] for pg in param_groups):
        raise ValueError("학습 가능한 파라미터가 하나도 없다 (requires_grad=True 인 것이 없음)")
    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=wd)
