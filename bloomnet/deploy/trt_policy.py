"""TensorRT 정밀도 정책 (L1) — 04 §9.3 위험표 + 03 §13.1.

이 모듈은 **선언만** 한다. 실제 layer precision 설정은 TRT 빌더 스크립트가
:data:`FP32_FORCED_MODULES` / :data:`FP16_LOCKED_MODULES` 를 읽어 수행한다.
tensorrt 는 Jetson 이미지에 포함되므로 requirements 어디에도 넣지 않는다(06 §5.0).

정책의 정본은 ``config.deploy.fp32_forced`` / ``fp16_locked`` 이며 V13 이
"``train.amp=='fp16'`` 또는 ``deploy.precision ∈ {fp16,int8}`` ⟹
``fp32_forced ⊇ {bmef.fusion, encoder.*.litemla.attention}``" 를 강제한다.
여기의 상수는 그 **하한**(반드시 포함되어야 하는 최소 집합)이다.
"""

from __future__ import annotations

import fnmatch
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "FP32_FORCED_MODULES",
    "FP16_LOCKED_MODULES",
    "RISK_TABLE",
    "MODULE_TOKEN_NOTES",
    "matches_module_pattern",
    "assert_precision_policy",
]

#: fp16/int8 엔진에서도 **fp32 로 실행해야 하는** 모듈 (누락 시 NaN/발산).
#:
#: * ``bmef.fusion`` — 03 §5.3: ``exp(A - amax)`` 후 정밀도 가중 평균. 최악 가중치
#:   ``w_min = 3.072e-6`` 이 fp16 에서 상대오차 8.9e-3 로 무너진다(test_bmef 실측).
#: * ``encoder.*.litemla.attention`` — 02 §3.3 / 04 §9.3-N (정정 B-6/B-15):
#:   먼저 터지는 것은 ``kv`` 가 아니라 ``out = q @ kv`` 다. N=4,096(stage3 @1024²)에서
#:   입력 std 2배면 상대오차 0.89, 4배면 NaN. N=16,384 는 단위분산에서도 0.92.
FP32_FORCED_MODULES: Tuple[str, ...] = (
    "bmef.fusion",
    "encoder.*.litemla.attention",
)

#: int8 엔진에서 **fp16 이하로 내리지 않는** 모듈 (동적 범위가 커서 스케일 추정 실패).
#:
#: 04 §9.3-L: PagFM 채널 내적, Softplus/exp 계열은 양자화 스케일 추정이 쉽게 발산한다.
FP16_LOCKED_MODULES: Tuple[str, ...] = (
    "chl_head",
    "unc_head",
    "pagfm",
    "encoder.*.litemla.attention",
)

#: 04 §9.3 위험표 A~O. 각 행 = (id, 위험 연산, 회피책, 본 설계 반영 위치).
#: 정보용이며 코드 경로에서 분기하지 않는다 — 리포트/문서 생성기가 소비한다.
RISK_TABLE: Tuple[Dict[str, str], ...] = (
    {
        "id": "A",
        "op": "nn.AdaptiveAvgPool2d(1) (PAPPM global branch)",
        "why": "동적 shape 에서 GlobalAveragePool 로 접히지 않는 경우가 있다",
        "fix": "x.mean(dim=(2,3), keepdim=True) → ReduceMean(axes 정적)",
        "where": "modules/decoder_blocks.py::PAPPM",
    },
    {
        "id": "B",
        "op": "F.interpolate(..., size=t.shape[-2:])",
        "why": "Shape→Gather→Concat→Resize 서브그래프 → fusion 실패, FP32 fallback",
        "fix": "학습은 size= 가 정본. export 시 ExportWrapper 가 out_hw 로 int 리터럴 주입",
        "where": "PagFM.out_hw / PAPPM.out_hw, models/bloomnet.py::ExportWrapper (정정 A-21/B-14)",
    },
    {
        "id": "C",
        "op": "동적 H/W",
        "why": "프로파일이 넓으면 커널 선택 열화·재빌드",
        "fix": "H,W 정적 고정(deploy.input_hw), batch 만 동적. 512 가 필요하면 엔진 2개",
        "where": "deploy/export_onnx.py::export(dynamic_batch=True)",
    },
    {
        "id": "D",
        "op": "Softplus fp16",
        "why": "ONNX Softplus 는 threshold 가 없어 z>11.09 에서 fp16 overflow(65504)",
        "fix": "ChlHead 에서 z.clamp(-20, 11) 후 softplus",
        "where": "modules/heads.py::ChlHead",
    },
    {
        "id": "E",
        "op": "torch.expm1",
        "why": "ONNX 에 Expm1 연산자가 없다 (venv 실측: UnsupportedOperatorError)",
        "fix": "torch.exp(u.clamp(max=6.2166)) - 1.0 — 06 §10 금지 목록 13",
        "where": "modules/heads.py::ChlHead, utils/metrics_reg.py (정정 B-13)",
    },
    {
        "id": "F",
        "op": "torch.exp(0.5*s) in conf",
        "why": "s 가 무제한이면 overflow",
        "fix": "UncHead 가 s ∈ [-7,7] hard clamp → exp(3.5)=33.1 로 안전",
        "where": "modules/heads.py::UncHead, deploy/postprocess.py::confidence_map",
    },
    {
        "id": "G",
        "op": "GELU",
        "why": "opset<20 은 Erf 분해. erf 정확형과 tanh 근사형의 수치가 다르다",
        "fix": "학습·export 모두 approximate='tanh' 로 통일",
        "where": "modules/common.py::build_act",
    },
    {
        "id": "H",
        "op": "PagFM 채널 내적 (k*q).sum(1)",
        "why": "fp16 누산기 오버플로 우려",
        "fix": "dot_scale=1/8 로 합의 표준편차 ≈1 (실사용 ±40)",
        "where": "modules/decoder_blocks.py::PagFM",
    },
    {
        "id": "I",
        "op": "AvgPool2d(17, 8, padding=8)",
        "why": "ONNX count_include_pad 와 TRT 기본값 불일치 이력",
        "fix": "count_include_pad=False 고정 + polygraphy layer-wise 대조 필수",
        "where": "modules/decoder_blocks.py::PAPPM",
    },
    {
        "id": "J",
        "op": "BatchNorm",
        "why": "eval 모드가 아니면 conv 로 접히지 않는다",
        "fix": "deploy() 가 .eval() 강제, export 전 assert not model.training",
        "where": "models/bloomnet.py::BloomNet.deploy / ExportWrapper.__init__",
    },
    {
        "id": "K",
        "op": "dict 반환",
        "why": "ONNX 트레이싱에서 키 순서·이름 손실",
        "fix": "ExportWrapper.forward 가 3-tuple 반환",
        "where": "models/bloomnet.py::ExportWrapper",
    },
    {
        "id": "L",
        "op": "INT8 양자화",
        "why": "PagFM 내적·Softplus·linear attention 은 동적 범위가 커서 스케일 추정 실패",
        "fix": "FP16_LOCKED_MODULES 를 float16 으로 고정 (mixed precision)",
        "where": "deploy/trt_policy.py::FP16_LOCKED_MODULES (정정 B-15)",
    },
    {
        "id": "M",
        "op": "최종 Resize ×4",
        "why": "align_corners=False → pytorch_half_pixel coordinate transform",
        "fix": "TRT 8.x+ 지원. 지연이 문제면 argmax@H/4 → nearest ×4 (정확도 영향 미측정, M6)",
        "where": "models/bloomnet.py::ExportWrapper.forward",
    },
    {
        "id": "N",
        "op": "LiteMLA ReLU-linear attention 의 fp16 누산",
        "why": "out = q @ kv 가 먼저 터진다. N=4096·std2 → 상대오차 0.89, std4 → NaN",
        "fix": "fp32_forced 에 encoder.*.litemla.attention 추가 + V13 강화",
        "where": "modules/blocks_ela.py::relu_linear_attention (_fp32_ctx) (정정 B-6/B-15)",
    },
    {
        "id": "O",
        "op": "tensor-shape 유래 adaptive pooling",
        "why": "출력 크기가 feature shape 에서 와 항목 B 와 같은 Shape→Gather 의존 생성",
        "fix": "F.avg_pool2d(t, k, k) 로 대체 — CPU 실측 maxdiff 0.000e+00",
        "where": "modules/common.py::build_bio_pyramid, modules/bmef.py::_reduce_gpol (B-7/B-12)",
    },
)


#: ★ 정책 토큰은 **PyTorch 모듈 경로가 아니다.** 어디를 가리키는지 기록해 둔다 —
#: 이 표가 없으면 TRT 빌더가 토큰을 잘못 해석해도 아무도 모른다.
#:
#: * ``bmef.fusion`` — ``BMEF.forward`` 안의 §5.3~5.4 블록. ``nn.Module`` 이 아니다
#:   (실제 모듈 경로는 ``backbone.bmef.<i>``). ONNX 노드 기준으로는 그 stage 의
#:   ``Exp``/``ReduceSum``/``Div`` 체인이다.
#: * ``encoder.*.litemla.attention`` — ``blocks_ela.relu_linear_attention`` **함수**.
#:   모듈이 아니므로 이름 기반 layer 매칭으로 못 잡는다. `LiteMLA.proj` 입력을 별도
#:   출력으로 마킹해 정밀도 경계를 확인해야 한다 (04 §9.4 step 4).
#: * ``chl_head`` / ``unc_head`` / ``pagfm`` — 실제 모듈이며 경로는
#:   ``chl_head`` / ``unc_head`` / ``decoder.pag4|pag8`` 계열.
MODULE_TOKEN_NOTES: Dict[str, str] = {
    "bmef.fusion": "BMEF.forward §5.3~5.4 (모듈 아님; backbone.bmef.<i> 안의 계산)",
    "encoder.*.litemla.attention": "blocks_ela.relu_linear_attention (함수, 모듈 아님)",
    "chl_head": "modules/heads.py::ChlHead",
    "unc_head": "modules/heads.py::UncHead",
    "pagfm": "modules/decoder_blocks.py::PagFM (decoder.pag4 / pag8)",
}


def matches_module_pattern(name: str, patterns: Sequence[str]) -> bool:
    """``name`` 이 ``patterns`` 중 하나에 걸리는가.

    ``name`` 은 PyTorch 모듈 경로일 수도, ONNX/TRT layer 이름일 수도 있다. 둘 다
    ``.``(또는 ``/``)로 구분된 계층 이름이므로 **구간(segment) 순서 부분열**로 맞춘다:
    패턴 ``bmef.fusion`` 은 ``backbone.bmef.0.fusion`` 에 걸리고
    (중간의 ``0`` 은 건너뛴다) ``fusion.bmef`` 에는 걸리지 않는다.
    각 구간은 ``fnmatch`` 또는 부분 문자열로 비교한다 (``pagfm`` ↔ ``pagfm_4``).
    패턴의 ``*`` 구간은 임의 구간 수를 소비한다.
    """
    segs = [s for s in name.replace("/", ".").split(".") if s]
    for pat in patterns:
        parts = [p for p in pat.replace("/", ".").split(".") if p and p != "*"]
        if not parts:
            continue
        i = 0
        for part in parts:
            while i < len(segs) and not (
                fnmatch.fnmatch(segs[i], part) or part in segs[i]
            ):
                i += 1
            if i == len(segs):
                break
            i += 1
        else:
            return True
    return False


def assert_precision_policy(
    fp32_forced: Iterable[str],
    fp16_locked: Iterable[str],
    *,
    precision: str,
    amp: str = "none",
) -> None:
    """V13 (정정 A-22) 하한 검사. config 검증과 **같은 규칙**을 배포 스크립트에서도 쓴다.

    Raises:
        ValueError: fp16/int8 경로인데 필수 fp32 모듈이 빠졌을 때.
    """
    if str(precision) not in {"fp16", "int8"} and str(amp) != "fp16":
        return
    have32 = set(map(str, fp32_forced))
    missing32: List[str] = [m for m in FP32_FORCED_MODULES if m not in have32]
    if missing32:
        raise ValueError(
            f"V13: precision={precision} amp={amp} 인데 deploy.fp32_forced 에 {missing32} 가 없다 "
            "(03 §5.3 / 02 §3.3 — 선택이 아니라 필수)"
        )
    if str(precision) == "int8":
        have16 = set(map(str, fp16_locked))
        missing16 = [m for m in FP16_LOCKED_MODULES if m not in have16]
        if missing16:
            raise ValueError(
                f"04 §9.3-L: precision=int8 인데 deploy.fp16_locked 에 {missing16} 가 없다"
            )
