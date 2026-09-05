"""T-utils — `bloomnet/utils/` (L0/L1) 계약 테스트.

검증 범위
* `metrics_seg`  : 손계산 혼동행렬과 mIoU 일치, ignore_index 처리, **absent 클래스 제외**(X-20),
                   `mIoU_C`(정정 A-4), 군 단위 부트스트랩 CI
* `metrics_reg`  : RMSE/MAE/R²/Spearman 정확성, log1p↔mg/m³ 왕복,
                   경보 F1 과 **support=0 자동 제외**(정정 A-30)
* `metrics_boundary` : BF-score / Boundary IoU / edge PRF 의 shape·값
* `seed` / `logging_csv` / `checkpoint` / `flops` / `distributed`

헌법 C-5: CPU 전용, 소형 텐서.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import bloomnet.constants as C
from bloomnet.utils import checkpoint as ckpt_mod
from bloomnet.utils import distributed as dist_mod
from bloomnet.utils import flops as flops_mod
from bloomnet.utils import logging_csv as log_mod
from bloomnet.utils import metrics_boundary as mb
from bloomnet.utils import metrics_reg as mr
from bloomnet.utils import metrics_seg as ms
from bloomnet.utils import seed as seed_mod


@pytest.fixture(autouse=True)
def _no_gpu() -> None:
    """헌법 C-5.2 — 어떤 테스트도 GPU 를 잡지 않는다."""
    assert not torch.cuda.is_available(), "CUDA_VISIBLE_DEVICES='' 로 실행해야 한다"
    torch.manual_seed(0)
    torch.set_num_threads(4)


# ════════════════════════════════════════════════════════════ metrics_seg


def _known_confusion() -> torch.Tensor:
    """손계산 대상 혼동행렬 (행 = GT, 열 = pred).

        [[5, 1, 0],
         [2, 3, 0],
         [0, 0, 0]]     ← class 2 는 GT·pred 모두 0 = absent
    """
    return torch.tensor([[5, 1, 0], [2, 3, 0], [0, 0, 0]], dtype=torch.int64)


def test_miou_matches_hand_computation() -> None:
    cm = _known_confusion()
    # class0: tp=5, gt=6, pred=7 -> union 8 -> 0.625
    # class1: tp=3, gt=5, pred=4 -> union 6 -> 0.500
    # class2: union 0            -> 제외
    assert ms.compute_mean_iou(cm) == pytest.approx((0.625 + 0.5) / 2)


def test_miou_excludes_absent_class_not_counts_it_as_zero() -> None:
    """★X-20 회귀 고정: absent 클래스를 IoU 0 으로 세는 두 번째 규약을 쓰지 않는다."""
    cm = _known_confusion()
    excluded = ms.compute_mean_iou(cm)
    as_zero = (0.625 + 0.5 + 0.0) / 3
    assert excluded == pytest.approx(0.5625)
    assert as_zero == pytest.approx(0.375)
    assert excluded != pytest.approx(as_zero)


def test_per_class_iou_is_nan_for_absent_class() -> None:
    ious = ms.per_class_iou(_known_confusion())
    assert ious[0] == pytest.approx(0.625)
    assert ious[1] == pytest.approx(0.5)
    assert math.isnan(float(ious[2]))


def test_present_classes_matches_miou_criterion() -> None:
    assert ms.present_classes(_known_confusion()) == [0, 1]
    assert ms.present_classes(torch.zeros((4, 4), dtype=torch.int64)) == []


def test_compute_mean_iou_all_absent_returns_zero() -> None:
    assert ms.compute_mean_iou(torch.zeros((3, 3), dtype=torch.int64)) == 0.0


def test_update_confusion_matrix_reproduces_known_matrix_and_ignores_255() -> None:
    target = torch.tensor([0] * 6 + [1] * 5 + [255, 255], dtype=torch.int64)
    pred = torch.tensor([0, 0, 0, 0, 0, 1] + [0, 0, 1, 1, 1] + [2, 0], dtype=torch.int64)
    cm = ms.new_confusion_matrix(3)
    ms.update_confusion_matrix(cm, pred, target, num_classes=3, ignore_index=255)
    assert torch.equal(cm, _known_confusion())
    assert int(cm.sum()) == 11  # ignore 2픽셀은 어디에도 세지 않는다


def test_update_confusion_matrix_drops_out_of_range_targets() -> None:
    target = torch.tensor([0, 1, 7, -1], dtype=torch.int64)
    pred = torch.tensor([0, 1, 0, 0], dtype=torch.int64)
    cm = ms.new_confusion_matrix(2)
    ms.update_confusion_matrix(cm, pred, target, num_classes=2, ignore_index=255)
    assert int(cm.sum()) == 2


def test_update_confusion_matrix_clamps_predictions() -> None:
    target = torch.tensor([0, 0], dtype=torch.int64)
    pred = torch.tensor([9, 0], dtype=torch.int64)
    cm = ms.new_confusion_matrix(3)
    ms.update_confusion_matrix(cm, pred, target, num_classes=3)
    assert int(cm[0, 2]) == 1 and int(cm[0, 0]) == 1


def test_update_confusion_matrix_all_ignore_is_noop() -> None:
    cm = ms.new_confusion_matrix(3)
    ms.update_confusion_matrix(
        cm, torch.zeros(4, dtype=torch.int64), torch.full((4,), 255, dtype=torch.int64), 3
    )
    assert int(cm.sum()) == 0


def test_confusion_from_logits_upsamples_logits_not_labels() -> None:
    logits = torch.zeros(1, 2, 2, 2)
    logits[:, 1] = 1.0                       # 전부 class1 예측
    target = torch.ones(1, 8, 8, dtype=torch.int64)
    cm = ms.confusion_from_logits(logits, target, num_classes=2)
    assert cm.shape == (2, 2)
    assert int(cm[1, 1]) == 64               # 라벨 해상도(8×8)에서 집계됐다
    assert ms.compute_mean_iou(cm) == pytest.approx(1.0)


def test_mean_iou_on_classes_is_c_common_average() -> None:
    cm = _known_confusion()
    assert ms.mean_iou_on_classes(cm, [0]) == pytest.approx(0.625)
    assert ms.mean_iou_on_classes(cm, [0, 1]) == pytest.approx(0.5625)
    # absent 는 지정해도 제외된다
    assert ms.mean_iou_on_classes(cm, [0, 1, 2]) == pytest.approx(0.5625)
    # 남는 클래스가 없으면 0.0 이 아니라 nan (K2 판정 오염 방지)
    assert math.isnan(ms.mean_iou_on_classes(cm, [2]))


def test_pixel_accuracy() -> None:
    assert ms.pixel_accuracy(_known_confusion()) == pytest.approx(8 / 11)
    assert math.isnan(ms.pixel_accuracy(torch.zeros((3, 3), dtype=torch.int64)))


def test_per_class_metric_rows_schema_and_names() -> None:
    rows = ms.per_class_metric_rows(_known_confusion(), epoch=0, class_names=("bg", "algae", "x"))
    assert len(rows) == 3
    assert set(rows[0]) == set(log_mod.PER_CLASS_FIELDS)
    assert rows[0]["epoch"] == 1                      # 1-based
    assert rows[0]["class_name"] == "bg"
    assert rows[0]["iou"] == pytest.approx(0.625)
    assert rows[2]["gt_present"] is False
    assert math.isnan(rows[2]["iou"])
    generic = ms.per_class_metric_rows(_known_confusion(), epoch=3)
    assert generic[2]["class_name"] == "class_2"
    assert generic[0]["epoch"] == 4


def test_confusion_matrix_accumulator() -> None:
    acc = ms.ConfusionMatrix(num_classes=3)
    target = torch.tensor([0] * 6 + [1] * 5 + [255, 255], dtype=torch.int64)
    pred = torch.tensor([0, 0, 0, 0, 0, 1] + [0, 0, 1, 1, 1] + [2, 0], dtype=torch.int64)
    acc.update(pred[:6], target[:6])
    acc.update(pred[6:], target[6:])
    assert torch.equal(acc.matrix, _known_confusion())
    out = acc.compute()
    assert out["miou"] == pytest.approx(0.5625)
    assert out["present_classes"] == [0, 1]
    acc.reset()
    assert int(acc.matrix.sum()) == 0


def test_bootstrap_ci_is_deterministic_and_brackets_point() -> None:
    groups = [_known_confusion() for _ in range(5)]
    groups.append(torch.tensor([[1, 4, 0], [4, 1, 0], [0, 0, 0]], dtype=torch.int64))
    a = ms.bootstrap_ci(groups, ms.compute_mean_iou, n_boot=200, seed=7)
    b = ms.bootstrap_ci(groups, ms.compute_mean_iou, n_boot=200, seed=7)
    assert a == b                                        # 동일 시드 → bit-identical
    point, lo, hi = a
    assert lo <= point <= hi
    c = ms.bootstrap_ci(groups, ms.compute_mean_iou, n_boot=200, seed=8)
    assert c[0] == pytest.approx(point)                  # 점추정은 시드와 무관
    assert ms.bootstrap_ci([], ms.compute_mean_iou) == pytest.approx(
        (float("nan"),) * 3, nan_ok=True
    )


# ════════════════════════════════════════════════════════════ metrics_reg


def test_log1p_mgm3_roundtrip_and_no_expm1() -> None:
    y = torch.tensor([0.0, 0.19, 3.75, 90.34, 500.0])
    assert torch.allclose(mr.log1p_to_mgm3(mr.mgm3_to_log1p(y)), y, atol=1e-4)
    # 정정 B-13: ONNX 에 Expm1 이 없으므로 코드베이스 전체가 exp(x)-1 을 쓴다.
    src = Path(mr.__file__).read_text(encoding="utf-8")
    assert "expm1" not in src.replace("``torch.expm1``", "")


def test_mgm3_to_log1p_clamps_negative_labels() -> None:
    assert float(mr.mgm3_to_log1p(torch.tensor([-5.0]))) == pytest.approx(0.0)


def test_alarm_level_buckets() -> None:
    lv = mr.alarm_level(torch.tensor([0.0, 14.9, 15.0, 24.9, 25.0, 99.9, 100.0, 500.0]))
    assert lv.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_spearman_corr_known_values() -> None:
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert mr.spearman_corr(a, a * 3.0 + 1.0) == pytest.approx(1.0)
    assert mr.spearman_corr(a, -a) == pytest.approx(-1.0)
    assert math.isnan(mr.spearman_corr(a, torch.ones(4)))       # 순위 분산 0
    assert math.isnan(mr.spearman_corr(a[:1], a[:1]))           # 표본 < 2
    # 동점 평균 순위: [1,2,2,3] vs [1,2,3,4] 는 완전 단조지만 1.0 미만이어야 한다
    rho = mr.spearman_corr(torch.tensor([1.0, 2.0, 2.0, 3.0]), a)
    assert 0.9 < rho < 1.0


def test_regression_accumulator_exact_values() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    true = torch.tensor([1.5, 1.5, 3.5, 3.0])
    acc = mr.RegressionAccumulator()
    acc.update(pred[:2], true[:2])                 # 스트리밍(2배치) 이 1배치와 같아야 한다
    acc.update(pred[2:], true[2:])
    out = acc.compute()

    err = (pred - true)
    n = 4
    ss_res = float((err**2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    assert out["n_valid"] == n
    assert out["mae_log"] == pytest.approx(float(err.abs().sum()) / n)
    assert out["rmse_log"] == pytest.approx(math.sqrt(ss_res / n))
    assert out["bias_log"] == pytest.approx(float(err.sum()) / n)
    assert out["r2_log"] == pytest.approx(1.0 - ss_res / ss_tot)
    assert out["spearman"] == pytest.approx(mr.spearman_corr(pred, true))

    pm, tm = mr.log1p_to_mgm3(pred), mr.log1p_to_mgm3(true)
    assert out["rmse_mgm3"] == pytest.approx(math.sqrt(float(((pm - tm) ** 2).sum()) / n), rel=1e-6)
    assert out["mae_mgm3"] == pytest.approx(float((pm - tm).abs().sum()) / n, rel=1e-6)


def test_regression_accumulator_streaming_equals_single_shot() -> None:
    torch.manual_seed(3)
    pred = torch.rand(64) * 4
    true = torch.rand(64) * 4
    one = mr.RegressionAccumulator()
    one.update(pred, true)
    many = mr.RegressionAccumulator()
    for i in range(0, 64, 8):
        many.update(pred[i : i + 8], true[i : i + 8])
    a, b = one.compute(), many.compute()
    for k in ("n_valid", "mae_log", "rmse_log", "r2_log", "mae_mgm3", "rmse_mgm3", "alarm_acc"):
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9), k


def test_regression_accumulator_mask_and_empty() -> None:
    acc = mr.RegressionAccumulator()
    pred = torch.zeros(1, 1, 4, 4)
    true = torch.ones(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    acc.update(pred, true, mask)                      # 유효 픽셀 0 — NaN 없이 빈 상태
    out = acc.compute()
    assert out["n_valid"] == 0
    assert math.isnan(out["rmse_log"])
    assert out["f1_at_15"] is None

    mask[0, 0, 0, :] = True
    acc.update(pred, true, mask)
    assert acc.compute()["n_valid"] == 4


def test_regression_accumulator_ignores_nonfinite() -> None:
    acc = mr.RegressionAccumulator()
    acc.update(
        torch.tensor([1.0, float("nan"), 2.0, float("inf")]),
        torch.tensor([1.0, 1.0, 2.0, 1.0]),
    )
    out = acc.compute()
    assert out["n_valid"] == 2
    assert out["rmse_log"] == pytest.approx(0.0)


def test_regression_accumulator_broadcast_error_message() -> None:
    acc = mr.RegressionAccumulator()
    with pytest.raises(ValueError, match="broadcast"):
        acc.update(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 8, 8))


def test_x14_debug_assert_rejects_raw_mgm3_target() -> None:
    acc = mr.RegressionAccumulator(debug_assert=True)
    with pytest.raises(AssertionError, match="log1p"):
        acc.update(torch.tensor([1.0]), torch.tensor([90.34]))     # 원단위를 넣었다
    ok = mr.RegressionAccumulator(debug_assert=True)
    ok.update(torch.tensor([1.0]), mr.mgm3_to_log1p(torch.tensor([90.34])))
    assert ok.compute()["n_valid"] == 1


def test_alarm_metrics_and_support_zero_exclusion() -> None:
    """정정 A-30 / B-30 — 표본이 0인 경보 구간은 macro-F1 에서 자동 제외한다."""
    mgm3 = torch.tensor([1.0, 20.0, 30.0, 5.0])       # levels 0,1,2,0 — level 3 표본 0개
    y = mr.mgm3_to_log1p(mgm3)
    acc = mr.RegressionAccumulator()
    acc.update(y, y)                                   # 완전 예측
    out = acc.compute()

    assert out["alarm_acc"] == pytest.approx(1.0)
    assert out["alarm_macro_f1"] == pytest.approx(1.0)   # 3구간 평균. level3 포함이면 0.75 가 된다
    assert out["alarm_excluded_levels"] == [3]
    assert "100" in out["alarm_exclusion_reasons"]["3"]
    assert out["f1_at_15"] == pytest.approx(1.0)
    assert out["f1_at_25"] == pytest.approx(1.0)
    assert out["f1_at_100"] is None
    assert "no samples >= 100 mg/m3" in out["f1_at_100_reason"]


def test_alarm_f1_penalises_wrong_predictions() -> None:
    true = mr.mgm3_to_log1p(torch.tensor([1.0, 20.0, 20.0, 20.0]))
    pred = mr.mgm3_to_log1p(torch.tensor([1.0, 1.0, 20.0, 20.0]))
    acc = mr.RegressionAccumulator()
    acc.update(pred, true)
    out = acc.compute()
    # 15 임계: GT 양성 3, 예측 양성 2, tp 2 -> P=1.0 R=2/3 -> F1=0.8
    assert out["f1_at_15"] == pytest.approx(0.8)
    assert out["alarm_acc"] == pytest.approx(0.75)


def test_regression_accumulator_nll_and_sigma() -> None:
    pred = torch.tensor([1.0, 2.0])
    true = torch.tensor([1.0, 2.0])
    s = torch.tensor([0.0, 0.0])
    acc = mr.RegressionAccumulator()
    acc.update(pred, true, None, s)
    out = acc.compute()
    # 잔차 0, s=0 -> NLL = 0.5*log(2pi)
    assert out["nll"] == pytest.approx(0.5 * math.log(2 * math.pi))
    assert out["sigma_mean"] == pytest.approx(1.0)


def test_regression_accumulator_sigma_err_spearman() -> None:
    pred = torch.tensor([0.0, 0.0, 0.0, 0.0])
    true = torch.tensor([0.1, 0.2, 0.3, 0.4])          # |err| 단조 증가
    s = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]) ** 2)   # sigma 도 단조 증가
    acc = mr.RegressionAccumulator()
    acc.update(pred, true, None, s)
    assert acc.compute()["sigma_err_spearman"] == pytest.approx(1.0)


def test_reservoir_is_uniform_over_the_whole_stream() -> None:
    """저수지가 앞 배치에 편향되면 Spearman 이 val 앞부분만 반영한다."""
    acc = mr.RegressionAccumulator(max_rank_samples=1000, seed=1)
    acc.update(torch.zeros(50_000), torch.zeros(50_000))
    acc.update(torch.full((50_000,), 5.0), torch.full((50_000,), 5.0))
    out = acc.compute()
    assert out["n_rank_samples"] == 1000
    late = float((acc._res[:, 0] > 1.0).double().mean())
    assert 0.45 < late < 0.55                      # 두 블록이 같은 크기 -> 대략 반반


def test_reservoir_is_deterministic_and_capped() -> None:
    a = mr.RegressionAccumulator(max_rank_samples=64, seed=5)
    b = mr.RegressionAccumulator(max_rank_samples=64, seed=5)
    torch.manual_seed(0)
    chunks = [torch.rand(500) for _ in range(4)]
    for c in chunks:
        a.update(c, c * 2)
    for c in chunks:
        b.update(c, c * 2)
    assert a.compute()["spearman"] == b.compute()["spearman"]
    assert a.compute()["n_rank_samples"] == 64
    assert a.compute()["n_valid"] == 2000          # 저수지 상한은 지표 표본 수를 줄이지 않는다


def test_max_rank_samples_zero_disables_spearman() -> None:
    acc = mr.RegressionAccumulator(max_rank_samples=0)
    acc.update(torch.rand(100), torch.rand(100))
    out = acc.compute()
    assert out["n_valid"] == 100
    assert math.isnan(out["spearman"])


def test_regression_accumulator_reset() -> None:
    acc = mr.RegressionAccumulator()
    acc.update(torch.zeros(4), torch.ones(4))
    acc.reset()
    assert acc.compute()["n_valid"] == 0


# ═══════════════════════════════════════════════════════ metrics_boundary


def _square_mask(h: int = 12, w: int = 12) -> torch.Tensor:
    m = torch.zeros(1, h, w, dtype=torch.int64)
    m[0, 3:9, 3:9] = 1
    return m


def test_bf_score_perfect_match() -> None:
    edge = torch.zeros(1, 1, 8, 8)
    edge[0, 0, 3, 2:6] = 1.0
    out = mb.bf_score(edge, edge.clone())
    assert set(out) == {
        "bf_1px", "bf_3px", "bf_5px",
        "bf_precision_1px", "bf_precision_3px", "bf_precision_5px",
        "bf_recall_1px", "bf_recall_3px", "bf_recall_5px",
    }
    for t in (1, 3, 5):
        assert out[f"bf_{t}px"] == pytest.approx(1.0)


def test_bf_score_tolerance_is_monotone_in_tau() -> None:
    gt = torch.zeros(1, 1, 12, 12)
    gt[0, 0, 5, 2:10] = 1.0
    pred = torch.zeros_like(gt)
    pred[0, 0, 8, 2:10] = 1.0                 # 3 px 어긋난 경계
    out = mb.bf_score(pred, gt)
    assert out["bf_1px"] == pytest.approx(0.0)
    assert out["bf_3px"] == pytest.approx(1.0)
    assert out["bf_5px"] == pytest.approx(1.0)


def test_bf_score_empty_cases() -> None:
    zero = torch.zeros(1, 1, 6, 6)
    nonzero = torch.zeros(1, 1, 6, 6)
    nonzero[0, 0, 2, 2] = 1.0
    assert mb.bf_score(zero, zero)["bf_1px"] == pytest.approx(1.0)     # 둘 다 경계 없음
    assert mb.bf_score(zero, nonzero)["bf_1px"] == pytest.approx(0.0)
    assert mb.bf_score(nonzero, zero)["bf_1px"] == pytest.approx(0.0)


def test_bf_score_accepts_hw_bhw_b1hw_shapes() -> None:
    hw = torch.zeros(6, 6)
    hw[2, 2:5] = 1.0
    ref = mb.bf_score(hw, hw)["bf_1px"]
    assert mb.bf_score(hw[None], hw[None])["bf_1px"] == pytest.approx(ref)
    assert mb.bf_score(hw[None, None], hw[None, None])["bf_1px"] == pytest.approx(ref)
    with pytest.raises(ValueError):
        mb.bf_score(torch.zeros(2, 3, 6, 6), torch.zeros(2, 3, 6, 6))


def test_bf_score_valid_mask_removes_pixels() -> None:
    gt = torch.zeros(1, 1, 6, 6)
    gt[0, 0, 1, 1] = 1.0
    pred = torch.zeros_like(gt)
    pred[0, 0, 5, 5] = 1.0
    valid = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    valid[0, 0, 5, 5] = False
    out = mb.bf_score(pred, gt, valid=valid)
    assert out["bf_1px"] == pytest.approx(0.0)          # pred 가 통째로 제거 -> 한쪽만 비었다


def test_boundary_iou_identical_masks() -> None:
    gt = _square_mask()
    out = mb.boundary_iou(gt, gt.clone(), num_classes=2)
    assert set(out) == {"biou_1px", "biou_3px", "biou_5px"}
    for k, v in out.items():
        assert v == pytest.approx(1.0), k


def test_boundary_iou_disjoint_masks_is_zero() -> None:
    gt = torch.zeros(1, 10, 10, dtype=torch.int64)
    gt[0, 0:4, 0:4] = 1
    pred = torch.zeros(1, 10, 10, dtype=torch.int64)
    pred[0, 6:10, 6:10] = 1
    out = mb.boundary_iou(pred, gt, num_classes=2, tolerances=(1,))
    assert 0.0 <= out["biou_1px"] < 1.0


def test_boundary_iou_absent_class_excluded_and_all_absent_is_nan() -> None:
    gt = torch.zeros(1, 8, 8, dtype=torch.int64)
    out = mb.boundary_iou(gt, gt.clone(), num_classes=3, tolerances=(1,))
    assert out["biou_1px"] == pytest.approx(1.0)    # class0 만 present
    counts = mb.boundary_iou_counts(gt, gt, num_classes=3, tolerances=(1,))
    assert int(counts[1][1, 1]) == 0 and int(counts[1][1, 2]) == 0


def test_boundary_iou_respects_ignore_index() -> None:
    gt = _square_mask()
    gt[0, 0, :] = 255
    pred = _square_mask()
    out = mb.boundary_iou(pred, gt, num_classes=2, tolerances=(1,))
    assert out["biou_1px"] == pytest.approx(1.0)


def test_edge_prf_shapes_and_values() -> None:
    y = torch.zeros(2, 1, 8, 8)
    y[:, :, 3, :] = 1.0
    logits = torch.where(y > 0.5, torch.full_like(y, 6.0), torch.full_like(y, -6.0))
    out = mb.edge_prf(logits, y)
    assert set(out) == {"edge_precision", "edge_recall", "edge_f1", "edge_pos_ratio_pred"}
    assert out["edge_precision"] == pytest.approx(1.0)
    assert out["edge_recall"] == pytest.approx(1.0)
    assert out["edge_f1"] == pytest.approx(1.0)
    assert out["edge_pos_ratio_pred"] == pytest.approx(1 / 8)


def test_edge_prf_upsamples_logits_to_target() -> None:
    y = torch.ones(1, 1, 8, 8)
    logits = torch.full((1, 1, 2, 2), 6.0)
    out = mb.edge_prf(logits, y)
    assert out["edge_recall"] == pytest.approx(1.0)


def test_boundary_accumulator_streams_bf_counts() -> None:
    edge = torch.zeros(1, 1, 8, 8)
    edge[0, 0, 4, 1:7] = 1.0
    acc = mb.BoundaryAccumulator(num_classes=2)
    acc.update_from_edges(edge, edge.clone())
    acc.update_from_edges(edge, edge.clone())
    out = acc.compute()
    assert out["bf_1px"] == pytest.approx(1.0)
    assert math.isnan(out["biou_1px"])          # update() 를 안 썼으므로 미측정
    acc.update_edge_head(torch.full((1, 1, 8, 8), 6.0), torch.ones(1, 1, 8, 8))
    out2 = acc.compute()
    assert out2["edge_f1"] == pytest.approx(1.0)
    assert out2["edge_pos_ratio_pred"] == pytest.approx(1.0)
    acc.reset()
    assert acc.compute()["bf_1px"] == pytest.approx(1.0)   # 0/0 -> 완전 일치 규약


def test_boundary_accumulator_with_injected_extractor() -> None:
    """`bloomnet.data.boundary` 없이도 누적 경로가 도는지 확인한다(주입 지점)."""

    def fake(mask, *, ignore_index=255, radius=0):
        m = mask if mask.dim() == 3 else mask[None]
        edge = torch.zeros(m.shape[0], 1, *m.shape[-2:])
        edge[:, 0] = (m != 0).float()
        return edge, torch.ones_like(edge, dtype=torch.bool)

    acc = mb.BoundaryAccumulator(num_classes=2, boundary_fn=fake)
    gt = _square_mask()
    acc.update(gt.clone(), gt)
    out = acc.compute()
    assert out["bf_1px"] == pytest.approx(1.0)
    assert out["biou_1px"] == pytest.approx(1.0)


def test_boundary_from_mask_uses_single_operator() -> None:
    gt = _square_mask()
    edge, valid = mb.boundary_from_mask(gt)
    assert edge.shape[-2:] == gt.shape[-2:]           # radius=0, out_stride=1 -> 원해상도
    assert set(torch.unique(edge).tolist()) <= {0.0, 1.0}
    assert valid.dtype == torch.bool
    acc = mb.BoundaryAccumulator(num_classes=2)
    acc.update(gt.clone(), gt)
    out = acc.compute()
    assert out["bf_1px"] == pytest.approx(1.0)
    assert out["biou_1px"] == pytest.approx(1.0)


# ═════════════════════════════════════════════════════════════════ seed


def test_seed_everything_strict_reproduces_rng() -> None:
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        seed_mod.seed_everything_strict(4321)
        a = torch.rand(5)
        seed_mod.seed_everything_strict(4321)
        b = torch.rand(5)
        assert torch.equal(a, b)
        assert seed_mod.get_base_seed() == 4321
        assert torch.are_deterministic_algorithms_enabled()
        # warn_only 는 필수다 (05 §7.2 R1: bilinear interpolate backward 가 비결정적)
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(det, warn_only=warn)


def test_seed_everything_strict_can_skip_deterministic_flags() -> None:
    """deterministic=False 는 속도 우선 모드: 결정론 플래그를 끄고 cudnn benchmark 를 켠다 (이전 상태와 무관)."""
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        seed_mod.seed_everything_strict(1, deterministic=False)
        assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.backends.cudnn.benchmark is True
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before[1], before[2]


def test_make_generator_is_deterministic() -> None:
    g1, g2 = seed_mod.make_generator(99), seed_mod.make_generator(99)
    assert torch.equal(torch.rand(4, generator=g1), torch.rand(4, generator=g2))
    g3 = seed_mod.make_generator(100)
    assert not torch.equal(torch.rand(4, generator=g1), torch.rand(4, generator=g3))


def test_worker_init_fn_is_deterministic_per_worker() -> None:
    seed_mod.worker_init_fn(0, base_seed=7)
    a = torch.rand(3)
    seed_mod.worker_init_fn(0, base_seed=7)
    b = torch.rand(3)
    seed_mod.worker_init_fn(1, base_seed=7)
    c = torch.rand(3)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# ═══════════════════════════════════════════════════════════ logging_csv


def test_csv_logger_writes_header_once(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "epoch_metrics.csv"
    lg = log_mod.CsvLogger(p, log_mod.EPOCH_FIELDS)
    lg.append({"epoch": 1, "train_loss": 1.0})
    lg.append({"epoch": 2, "train_loss": 0.5, "val_algae_iou": float("nan")})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].split(",")[0] == "epoch"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    assert rows[0]["val_loss"] == ""            # 미기재 컬럼은 빈 칸
    assert rows[1]["val_algae_iou"] == "nan"


def test_csv_logger_rejects_unknown_column(tmp_path: Path) -> None:
    lg = log_mod.CsvLogger(tmp_path / "a.csv", ("epoch", "x"))
    with pytest.raises(ValueError, match="미지 컬럼"):
        lg.append({"epoch": 1, "typo": 3})
    assert not (tmp_path / "a.csv").exists()


def test_csv_logger_preamble_records_derived_schedule(tmp_path: Path) -> None:
    """정정 A-18 — 런타임 유도되는 iters_per_epoch 등을 CSV 자체에 남긴다."""
    p = tmp_path / "epoch_metrics.csv"
    lg = log_mod.CsvLogger(
        p, log_mod.EPOCH_FIELDS,
        preamble={"iters_per_epoch": 2108, "warmup_iters": 4216, "total_iters": 122264},
    )
    lg.append({"epoch": 1})
    lg.append({"epoch": 2})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# iters_per_epoch=2108")
    assert lines[1].startswith("epoch,")
    assert len(lines) == 4                      # preamble + header + 2 rows


def test_append_per_class_and_presence(tmp_path: Path) -> None:
    cm = _known_confusion()
    pc = tmp_path / "val_per_class_metrics.csv"
    log_mod.append_per_class_metrics(pc, 0, cm, class_names=("bg", "algae", "ghost"))
    rows = list(csv.DictReader(pc.open(encoding="utf-8")))
    assert len(rows) == 3 and rows[1]["class_name"] == "algae"

    ps = tmp_path / "val_presence_summary.csv"
    cm2 = torch.tensor([[5, 1, 0], [2, 3, 0], [1, 0, 0]], dtype=torch.int64)
    log_mod.append_presence_summary(ps, 0, cm2)
    row = list(csv.DictReader(ps.open(encoding="utf-8")))[0]
    assert row["gt_present_pred_missing_class_ids"] == "2"     # GT 있으나 한 번도 예측 안 됨
    assert row["low_recall_lt_0_1_class_ids"] == "2"


def test_regression_row_maps_accumulator_output(tmp_path: Path) -> None:
    y = mr.mgm3_to_log1p(torch.tensor([1.0, 20.0, 30.0, 5.0]))
    acc = mr.RegressionAccumulator()
    acc.update(y, y)
    row = log_mod.regression_row(0, acc.compute(), ignore_ratio=0.12)
    assert set(row) <= set(log_mod.REGRESSION_FIELDS)
    assert row["epoch"] == 1
    assert row["n_valid_px"] == 4
    assert row["f1_at_100"] is None
    assert "no samples >= 100 mg/m3" in row["alarm_exclusion_reason"]

    p = tmp_path / "val_regression_metrics.csv"
    log_mod.append_regression_metrics(p, row)
    out = list(csv.DictReader(p.open(encoding="utf-8")))[0]
    assert out["f1_at_100"] == ""               # null 로 남는다 (0.0 으로 위장하지 않는다)
    assert out["alarm_excluded_levels"] == "3"
    assert out["ignore_ratio"] == "0.12"


def test_append_boundary_and_stability(tmp_path: Path) -> None:
    bp = tmp_path / "val_boundary_metrics.csv"
    log_mod.append_boundary_metrics(bp, {"epoch": 1, "bf_1px": 0.4, "biou_3px": 0.2})
    assert list(csv.DictReader(bp.open(encoding="utf-8")))[0]["bf_1px"] == "0.4"

    sp = tmp_path / "stability.csv"
    log_mod.append_stability(sp, {"epoch": 1, "bio_beta_s1": 0.0, "glint_a": 20.0})
    assert "bio_beta_s4" in sp.read_text(encoding="utf-8").splitlines()[0]


# ════════════════════════════════════════════════════════════ checkpoint


class _TinyNet(nn.Module):
    def __init__(self, num_classes: int = 12) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.seg_head = nn.ModuleDict({"cls": nn.Conv2d(4, num_classes, 1)})
        self.edge_head = nn.Conv2d(4, 1, 1)
        self.aux_heads = nn.ModuleList([nn.Conv2d(4, num_classes, 1)])
        self.siam_proj = nn.Conv2d(4, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head["cls"](self.stem(x))


def test_atomic_save_leaves_no_tmp(tmp_path: Path) -> None:
    p = ckpt_mod.atomic_save({"a": 1}, tmp_path / "runs" / "last.pt")
    assert p.exists()
    assert list(tmp_path.rglob("*.tmp")) == []
    assert ckpt_mod.load_checkpoint(p)["a"] == 1


def test_save_checkpoint_requires_payload_keys(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="필수 키 누락"):
        ckpt_mod.save_checkpoint(tmp_path / "best.pt", {"epoch": 1})
    payload = {
        "epoch": 3, "global_step": 100, "best_metric": 0.42, "metric_name": "val_miou",
        "model_state_dict": _TinyNet().state_dict(),
        "rng_state": ckpt_mod.collect_rng_state(),
        "git_commit": ckpt_mod.git_commit(),
    }
    p = ckpt_mod.save_checkpoint(tmp_path / "best.pt", payload)
    loaded = ckpt_mod.load_checkpoint(p)
    assert loaded["metric_name"] == "val_miou"
    ckpt_mod.restore_rng_state(loaded["rng_state"])


def test_shape_tolerant_load_skips_only_class_head(capsys: pytest.CaptureFixture[str]) -> None:
    """K=12 체크포인트를 K=2 모델에 로드하면 걸러지는 키는 cls weight/bias 뿐이다 (04 §8.1)."""
    src = _TinyNet(num_classes=12)
    dst = _TinyNet(num_classes=2)
    report = ckpt_mod.load_state_dict_shape_tolerant(dst, src.state_dict())
    assert set(report["missing"]) == {
        "seg_head.cls.weight", "seg_head.cls.bias",
        "aux_heads.0.weight", "aux_heads.0.bias",
    }
    assert set(report["shape_mismatch"]) == set(report["missing"])
    assert report["unexpected"] == []
    assert torch.equal(dst.stem.weight, src.stem.weight)      # 나머지는 전부 로드됐다
    assert "missing" in capsys.readouterr().out               # 로그로 반드시 출력 (04 §8.1)


def test_shape_tolerant_load_reports_unexpected() -> None:
    dst = _TinyNet(num_classes=2)
    sd = dict(dst.state_dict())
    sd["ghost.weight"] = torch.zeros(1)
    report = ckpt_mod.load_state_dict_shape_tolerant(dst, sd, verbose=False)
    assert report["unexpected"] == ["ghost.weight"]
    assert report["missing"] == []


def test_prune_train_only_keys() -> None:
    sd = _TinyNet().state_dict()
    pruned = ckpt_mod.prune_train_only_keys(sd)
    assert any(k.startswith("stem") for k in pruned)
    assert any(k.startswith("seg_head") for k in pruned)
    assert not [k for k in pruned if "edge_head" in k or "aux_" in k or "siam" in k]
    # 계약은 constants.TRAIN_ONLY_MODULES 의 **이름**이다. 목록에 없는 이름은 남는다.
    other = {"aux_head_s8.conv.weight": torch.zeros(1)}
    assert ckpt_mod.prune_train_only_keys(other) == other
    assert ckpt_mod.prune_train_only_keys(other, extra_prefixes=("aux_head",)) == {}
    # 무관한 모듈이 부분 문자열 때문에 조용히 지워지면 안 된다
    keep = {"main_edge_head_gate.weight": torch.zeros(1)}
    assert ckpt_mod.prune_train_only_keys(keep) == keep


def test_collect_and_restore_rng_state() -> None:
    torch.manual_seed(11)
    state = ckpt_mod.collect_rng_state()
    a = torch.rand(4)
    ckpt_mod.restore_rng_state(state)
    assert torch.equal(torch.rand(4), a)


def test_git_commit_returns_string() -> None:
    rev = ckpt_mod.git_commit()
    assert isinstance(rev, str) and rev


# ═════════════════════════════════════════════════════════════════ flops


def _conv_stack() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 4, 1, bias=True),
    )


def test_count_macs_hooks_matches_hand_arithmetic() -> None:
    model = _conv_stack()
    x = torch.randn(1, 3, 16, 16)
    rep = flops_mod.count_macs_hooks(model, x)
    expected = 9 * 3 * 8 * 16 * 16 + 1 * 8 * 4 * 16 * 16      # 55,296 + 8,192
    assert rep.total == expected == 63488
    assert rep.by_module["0"] == 55296 and rep.by_module["3"] == 8192
    assert rep.gflop == pytest.approx(2 * rep.gmac)
    assert rep.input_hw == (16, 16)
    assert rep.top(1)[0][0] == "0"


def test_count_macs_hooks_counts_groups_and_batch() -> None:
    dw = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
    rep = flops_mod.count_macs_hooks(dw, torch.randn(2, 16, 8, 8))
    assert rep.total == 2 * (9 * 1 * 16 * 8 * 8)


def test_count_macs_hooks_linear() -> None:
    rep = flops_mod.count_macs_hooks(nn.Linear(10, 5), torch.randn(4, 10))
    assert rep.total == 10 * 5 * 4


def test_count_macs_flop_counter_agrees_with_hooks() -> None:
    model = _conv_stack()
    x = torch.randn(1, 3, 16, 16)
    a = flops_mod.count_macs_hooks(model, x)
    b = flops_mod.count_macs_flop_counter(model, x)
    assert b.method == "flop_counter"
    assert b.total == a.total
    assert flops_mod.count_macs(model, x, method="hooks").total == a.total
    with pytest.raises(ValueError, match="unknown method"):
        flops_mod.count_macs(model, x, method="nope")


def test_count_macs_does_not_flip_training_mode() -> None:
    model = _conv_stack()
    model.train()
    flops_mod.count_macs_hooks(model, torch.randn(1, 3, 8, 8))
    assert model.training
    flops_mod.count_macs_flop_counter(model, torch.randn(1, 3, 8, 8))
    assert model.training


def test_count_macs_accepts_tuple_and_dict_inputs() -> None:
    class Two(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c = nn.Conv2d(3, 4, 1, bias=False)

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return self.c(a) + self.c(b)

    m = Two()
    x = torch.randn(1, 3, 8, 8)
    per_call = 3 * 4 * 8 * 8
    assert flops_mod.count_macs_hooks(m, (x, x)).total == 2 * per_call
    assert flops_mod.count_macs_hooks(m, {"a": x, "b": x}).total == 2 * per_call


def test_scale_macs_matches_a24_ratios() -> None:
    """정정 A-24 — 256² 에서 1회 재고 ×4(512²) / ×16(1024²) 로 스케일한다."""
    base = 1_000_000
    assert flops_mod.scale_macs(base, from_hw=(256, 256), to_hw=(512, 512)) == pytest.approx(4e6)
    at1024 = flops_mod.scale_macs(base, from_hw=(256, 256), to_hw=(1024, 1024))
    assert at1024 == pytest.approx(1.6e7)
    with pytest.raises(ValueError):
        flops_mod.scale_macs(base, from_hw=(0, 256), to_hw=(512, 512))


def test_scale_macs_is_exact_for_conv_stack() -> None:
    model = _conv_stack()
    at64 = flops_mod.count_macs_hooks(model, torch.randn(1, 3, 64, 64)).total
    at128 = flops_mod.count_macs_hooks(model, torch.randn(1, 3, 128, 128)).total
    assert flops_mod.scale_macs(at64, from_hw=(64, 64), to_hw=(128, 128)) == pytest.approx(at128)


def test_count_parameters() -> None:
    m = _conv_stack()
    assert flops_mod.count_parameters(m) == sum(p.numel() for p in m.parameters())
    for p in m[0].parameters():
        p.requires_grad_(False)
    assert flops_mod.count_parameters(m, trainable_only=True) < flops_mod.count_parameters(m)


# ═══════════════════════════════════════════════════════════ distributed


def test_distributed_helpers_without_process_group() -> None:
    assert dist_mod.is_dist_available() is False
    assert dist_mod.get_rank() == 0
    assert dist_mod.get_world_size() == 1
    assert dist_mod.is_main_process() is True
    dist_mod.barrier()                                   # 무연산
    t = torch.tensor([2.0])
    assert torch.equal(dist_mod.all_reduce_sum(t), torch.tensor([2.0]))
    assert torch.equal(dist_mod.all_reduce_mean(t), torch.tensor([2.0]))


def test_maybe_convert_syncbn_reports_reason_when_skipped() -> None:
    m = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    out, info = dist_mod.maybe_convert_syncbn(m, effective_batch_size=2)
    assert info["converted"] is False
    assert info["reason"]                                 # 조용한 skip 금지
    assert isinstance(out[1], nn.BatchNorm2d) and not isinstance(out[1], nn.SyncBatchNorm)


def test_convert_syncbn_replaces_batchnorm() -> None:
    m = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    conv = dist_mod.convert_syncbn(m)
    assert isinstance(conv[1], nn.SyncBatchNorm)
    forced, info = dist_mod.maybe_convert_syncbn(
        nn.Sequential(nn.BatchNorm2d(4)), effective_batch_size=32, force=True
    )
    assert info["converted"] is True and isinstance(forced[0], nn.SyncBatchNorm)


def test_syncbn_threshold_matches_spec() -> None:
    assert dist_mod.SYNCBN_BATCH_THRESHOLD == 8          # 04 §1.2


# ══════════════════════════════════════════════ constants 단일 출처 확인


def test_utils_take_literals_from_constants_only() -> None:
    """utils 는 리터럴을 복제하지 않고 ``constants.py`` 를 그대로 재export 한다 (06 §2.1.1)."""
    assert ms.IGNORE_INDEX is C.IGNORE_INDEX == 255
    assert mr.ALARM_THRESHOLDS_MGM3 is C.ALARM_THRESHOLDS_MGM3
    assert mr.ALARM_THRESHOLDS_LOG1P is C.ALARM_THRESHOLDS_LOG1P
    assert mr.ALARM_LEVEL_NAMES is C.ALARM_LEVEL_NAMES
    assert ckpt_mod.TRAIN_ONLY_MODULES is C.TRAIN_ONLY_MODULES
    assert mb.IGNORE_INDEX is C.IGNORE_INDEX


def test_default_alarm_thresholds_are_the_business_document_values() -> None:
    """기본 임계값이 사업문서 s5 의 15/25/100 이고 log1p 표와 정합한다."""
    acc = mr.RegressionAccumulator()
    assert acc.thresholds_mgm3 == (15.0, 25.0, 100.0)
    assert acc.n_levels == len(C.ALARM_LEVEL_NAMES) == 4
    for t, lg in zip(C.ALARM_THRESHOLDS_MGM3, C.ALARM_THRESHOLDS_LOG1P):
        assert math.log1p(t) == pytest.approx(lg, abs=1e-6)
