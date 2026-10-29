"""Post-hoc calibration, cut-off tuning, split-conformal shortlist and confidence-based review.

* Temperature scaling (per category): ``p~_k = sigmoid(z_k / T_k)`` with ``T_k > 0``
  minimising the binary cross-entropy on the validation split. Monotone, so it never
  changes the ranking of cells within a category.
* Cut-offs: per category, the value of the grid 0.05, 0.10, ..., 0.95 maximising F1 on
  the (calibrated) validation probabilities.
* Split conformal shortlist: nonconformity ``s = 1 - p~_y`` over all positive
  (cell, label) pairs of the calibration (validation) split; ``qhat`` is the empirical
  quantile of level ``ceil((n + 1)(1 - alpha)) / n``; the shortlist of a cell is
  ``C(x) = {k : p~_k >= 1 - qhat}``.
* Joint confidence for review: ``pi(x) = prod_k max(p~_k, 1 - p~_k)``. The least
  confident cells are referred; accuracy (all labels correct) is reported on the rest.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from neutroclf.metrics import per_category_scores, predict_labels
from neutroclf.utils import get_logger

T_BOUNDS = (0.05, 20.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -x))


def _bce_at_temperature(log_t: float, z: np.ndarray, y: np.ndarray) -> float:
    s = z / math.exp(log_t)
    return float(np.mean(np.logaddexp(0.0, s) - y * s))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature ``T`` minimising the BCE of ``sigmoid(z / T)`` (bounded 1-D search on log T)."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if y.min() == y.max():
        return 1.0  # no positives or no negatives: temperature is not identifiable
    res = minimize_scalar(
        _bce_at_temperature,
        bounds=(math.log(T_BOUNDS[0]), math.log(T_BOUNDS[1])),
        args=(z, y),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(math.exp(res.x))


def fit_temperatures(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One temperature per category (columns)."""
    temps = np.array([fit_temperature(logits[:, k], labels[:, k]) for k in range(logits.shape[1])])
    degenerate = [k for k in range(labels.shape[1]) if labels[:, k].min() == labels[:, k].max()]
    if degenerate:
        get_logger().warning(
            "temperature fixed to 1 for categories without both classes: %s", degenerate
        )
    return temps


def apply_temperatures(logits: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    """Calibrated probabilities ``sigmoid(z / T)``."""
    return sigmoid(np.asarray(logits, dtype=np.float64) / np.asarray(temperatures)[None, :])


def threshold_grid(start: float = 0.05, stop: float = 0.95, step: float = 0.05) -> np.ndarray:
    n = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(n), 10)


def tune_thresholds(probs: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Per-category cut-off maximising F1 on ``(probs, labels)``; the lowest maximiser wins.

    Categories without a positive cell keep 0.5.
    """
    thresholds = np.full(probs.shape[1], 0.5)
    for k in range(probs.shape[1]):
        y = labels[:, k : k + 1]
        if y.sum() == 0:
            continue
        f1s = [
            per_category_scores(y, predict_labels(probs[:, k : k + 1], t))["f1"][0] for t in grid
        ]
        thresholds[k] = float(grid[int(np.nanargmax(f1s))])
    return thresholds


# ----------------------------------------------------------------------------- conformal
def conformal_qhat(probs: np.ndarray, labels: np.ndarray, alpha: float = 0.05) -> tuple[float, int]:
    """Split-conformal quantile over positive (cell, label) pairs of the calibration set.

    Returns ``(qhat, n)`` where ``n`` is the number of calibration pairs. When the
    quantile level exceeds 1 (too few pairs), ``qhat = 1`` and every category is shortlisted.
    """
    scores = 1.0 - np.asarray(probs, dtype=np.float64)[np.asarray(labels) == 1]
    n = scores.size
    if n == 0:
        raise ValueError("No positive labels in the calibration split")
    level = math.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return 1.0, n
    return float(np.quantile(scores, level, method="higher")), n


def conformal_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean shortlist matrix ``C(x) = {k : p~_k >= 1 - qhat}``."""
    return np.asarray(probs) >= (1.0 - qhat) - 1e-12


def conformal_coverage(sets: np.ndarray, labels: np.ndarray) -> dict:
    """Pair coverage (positive labels in the shortlist), per-cell coverage (all positives in
    the shortlist), per-category coverage and shortlist size distribution."""
    labels = np.asarray(labels).astype(bool)
    sets = np.asarray(sets).astype(bool)
    covered = sets & labels
    sizes = sets.sum(axis=1)
    per_cat = [
        float(covered[:, k].sum() / labels[:, k].sum()) if labels[:, k].any() else None
        for k in range(labels.shape[1])
    ]
    return {
        "pair_coverage": float(covered.sum() / labels.sum()),
        "cell_coverage": float((covered.sum(axis=1) == labels.sum(axis=1)).mean()),
        "per_category_coverage": per_cat,
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "set_size_counts": {int(s): int((sizes == s).sum()) for s in np.unique(sizes)},
    }


# ----------------------------------------------------------------------------- review
def joint_confidence(probs: np.ndarray) -> np.ndarray:
    """``pi(x) = prod_k max(p~_k, 1 - p~_k)``: confidence that every label state is right."""
    p = np.asarray(probs, dtype=np.float64)
    return np.prod(np.maximum(p, 1.0 - p), axis=1)


def referral_curve(
    confidence: np.ndarray, correct: np.ndarray, fractions: Sequence[float]
) -> list[dict[str, float]]:
    """Accuracy on retained cells after referring the ``floor(f * n)`` least confident cells."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(confidence, kind="stable")  # ascending: least confident first
    n = len(confidence)
    rows = []
    for f in fractions:
        k = int(math.floor(f * n + 1e-9))
        kept = order[k:]
        rows.append(
            {
                "fraction_referred": float(f),
                "n_referred": k,
                "n_retained": int(len(kept)),
                "accuracy_retained": float(correct[kept].mean()) if len(kept) else float("nan"),
            }
        )
    return rows
