"""Temperature scaling, cut-off tuning, conformal shortlist and confidence-based review."""

from __future__ import annotations

import numpy as np
import pytest

from neutroclf.calibration import (
    apply_temperatures,
    conformal_coverage,
    conformal_qhat,
    conformal_sets,
    fit_temperature,
    fit_temperatures,
    joint_confidence,
    referral_curve,
    sigmoid,
    threshold_grid,
    tune_thresholds,
)


def simulated_logits(n: int, t_true: float, rng: np.random.Generator):
    """True log-odds z; labels ~ Bernoulli(sigmoid(z)); the model reports z * t_true."""
    z = rng.normal(0, 2.0, n)
    y = (rng.random(n) < sigmoid(z)).astype(float)
    return z * t_true, y


@pytest.mark.parametrize("t_true", [0.5, 1.0, 2.5])
def test_temperature_recovers_known_scale(t_true):
    rng = np.random.default_rng(0)
    logits, y = simulated_logits(20000, t_true, rng)
    assert fit_temperature(logits, y) == pytest.approx(t_true, rel=0.06)


def test_temperature_preserves_ranking_and_degenerate_columns():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 3))
    labels = np.stack([rng.random(200) < 0.3, rng.random(200) < 0.5, np.zeros(200)], 1).astype(int)
    temps = fit_temperatures(logits, labels)
    assert temps[2] == 1.0  # no positives -> not identifiable
    probs = apply_temperatures(logits, temps)
    for k in range(3):
        assert (np.argsort(probs[:, k]) == np.argsort(logits[:, k])).all()


def test_threshold_grid_and_tuning():
    grid = threshold_grid(0.05, 0.95, 0.05)
    assert len(grid) == 19 and grid[0] == 0.05 and grid[-1] == 0.95
    probs = np.array([[0.1], [0.2], [0.62], [0.7], [0.9]])
    labels = np.array([[0], [0], [1], [1], [1]])
    thr = tune_thresholds(probs, labels, grid)
    assert 0.2 < thr[0] <= 0.62  # lowest maximiser of F1 (= 1.0)
    assert tune_thresholds(probs, np.zeros_like(labels), grid)[0] == 0.5


def test_conformal_qhat_definition():
    probs = np.array([[0.9, 0.1], [0.8, 0.7], [0.6, 0.2], [0.3, 0.95]])
    labels = np.array([[1, 0], [1, 1], [1, 0], [0, 1]])
    # positive scores 1 - p: [0.1, 0.2, 0.3, 0.4, 0.05], n = 5
    qhat, n = conformal_qhat(probs, labels, alpha=0.5)
    level = np.ceil((n + 1) * 0.5) / n  # 0.6
    assert n == 5
    assert qhat == pytest.approx(np.quantile([0.1, 0.2, 0.3, 0.4, 0.05], level, method="higher"))
    qhat_all, _ = conformal_qhat(probs, labels, alpha=0.01)  # level > 1 -> include everything
    assert qhat_all == 1.0
    assert conformal_sets(probs, qhat_all).all()


def test_conformal_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(3)
    coverages = []
    for _ in range(30):
        z = rng.normal(0, 2.0, size=(1200, 7))
        probs = sigmoid(z)
        labels = (rng.random(z.shape) < probs).astype(int)
        cal, test = slice(0, 600), slice(600, None)
        qhat, _ = conformal_qhat(probs[cal], labels[cal], alpha=0.05)
        cov = conformal_coverage(conformal_sets(probs[test], qhat), labels[test])
        coverages.append(cov["pair_coverage"])
    assert np.mean(coverages) >= 0.95 - 0.005
    assert 0.0 <= cov["cell_coverage"] <= 1.0
    assert sum(cov["set_size_counts"].values()) == 600


def test_coverage_statistics_on_toy_example():
    sets = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
    labels = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0]])
    cov = conformal_coverage(sets, labels)
    assert cov["pair_coverage"] == pytest.approx(2 / 4)
    assert cov["cell_coverage"] == pytest.approx(1 / 3)
    assert cov["mean_set_size"] == pytest.approx(4 / 3)


def test_joint_confidence_formula():
    p = np.array([[0.9, 0.2, 0.5], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(joint_confidence(p), [0.9 * 0.8 * 0.5, 1.0])


def test_referral_curve_refers_least_confident_first():
    conf = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
    correct = np.array([1, 0, 1, 0, 1])
    rows = referral_curve(conf, correct, [0.0, 0.2, 0.4])
    assert [r["n_referred"] for r in rows] == [0, 1, 2]
    np.testing.assert_allclose([r["accuracy_retained"] for r in rows], [0.6, 0.75, 1.0])
