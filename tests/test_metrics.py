"""Metrics against scikit-learn and hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from neutroclf.metrics import (
    abnormal_auroc,
    bootstrap_ci,
    brier_score,
    classification_metrics,
    expected_calibration_error,
    macro_f1,
    net_benefit,
    per_category_scores,
    predict_labels,
    reliability_bins,
    subset_accuracy,
    weighted_f1,
)


@pytest.fixture()
def toy():
    rng = np.random.default_rng(0)
    labels = (rng.random((300, 7)) < 0.3).astype(int)
    labels[:, 0] = (labels[:, 1:].sum(1) == 0).astype(int)
    probs = np.clip(labels * 0.6 + rng.random(labels.shape) * 0.5, 0, 1)
    return probs, labels


def test_f1_family_matches_sklearn(toy):
    probs, labels = toy
    pred = predict_labels(probs, np.full(7, 0.5))
    s = per_category_scores(labels, pred)
    np.testing.assert_allclose(s["f1"], f1_score(labels, pred, average=None))
    np.testing.assert_allclose(s["sensitivity"], recall_score(labels, pred, average=None))
    np.testing.assert_allclose(s["ppv"], precision_score(labels, pred, average=None))
    assert macro_f1(labels, pred) == pytest.approx(f1_score(labels, pred, average="macro"))
    assert weighted_f1(labels, pred) == pytest.approx(f1_score(labels, pred, average="weighted"))
    assert subset_accuracy(labels, pred) == pytest.approx((labels == pred).all(1).mean())


def test_abnormal_auroc_uses_one_minus_normal(toy):
    probs, labels = toy
    expected = roc_auc_score(1 - labels[:, 0], 1 - probs[:, 0])
    assert abnormal_auroc(labels, probs, 0) == pytest.approx(expected)


def test_ece_and_brier_hand_computed():
    probs = np.array([0.05, 0.15, 0.95, 0.85])
    labels = np.array([0, 1, 1, 1])
    # bins: [0,0.1): p=.05,y=0 ; [0.1,0.2): p=.15,y=1 ; [0.8,0.9): .85,1 ; [0.9,1]: .95,1
    expected = (0.05 + 0.85 + 0.15 + 0.05) / 4
    assert expected_calibration_error(probs, labels, 10) == pytest.approx(expected)
    assert brier_score(probs, labels) == pytest.approx(np.mean((probs - labels) ** 2))
    bins = reliability_bins(probs, labels, 10)
    assert sum(b["count"] for b in bins) == 4 and bins[-1]["count"] == 1


def test_ece_small_for_calibrated_probabilities():
    rng = np.random.default_rng(1)
    p = rng.random(200000)
    y = (rng.random(p.size) < p).astype(int)
    assert expected_calibration_error(p, y, 10) < 0.01


def test_net_benefit_formula():
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.6, 0.1, 0.2])
    row = net_benefit(y, score, [0.5])[0]
    # at 0.5: TP = 1, FP = 1, n = 5, odds = 1
    assert row["net_benefit_model"] == pytest.approx(1 / 5 - 1 / 5)
    assert row["net_benefit_all"] == pytest.approx(0.4 - 0.6)


def test_classification_metrics_keys_and_bootstrap(toy):
    probs, labels = toy
    cats = [f"c{k}" for k in range(7)]
    m = classification_metrics(probs, labels, np.full(7, 0.5), cats, 0)
    for key in (
        "abnormal_auroc",
        "macro_f1",
        "weighted_f1",
        "mean_ap",
        "subset_accuracy",
        "specificity",
        "ece",
        "brier",
        "f1/c3",
        "sensitivity/c3",
        "ppv/c3",
    ):
        assert key in m and np.isfinite(m[key])
    ci = bootstrap_ci(
        lambda idx: classification_metrics(probs[idx], labels[idx], np.full(7, 0.5), cats, 0),
        len(labels),
        n_boot=50,
        seed=0,
    )
    lo, hi = ci["macro_f1"]
    assert lo <= m["macro_f1"] <= hi
