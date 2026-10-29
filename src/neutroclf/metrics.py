"""Evaluation metrics (NumPy), bootstrap confidence intervals and decision-curve analysis.

Conventions
-----------
* ``probs`` (N, C) are per-category probabilities, ``labels`` (N, C) multi-hot.
* Predictions use per-category cut-offs: ``pred_k = probs_k >= threshold_k``.
* Abnormal vs normal: a cell is truly abnormal when it is not labelled normal; the
  abnormality score is ``1 - p_normal`` and a cell is called abnormal when
  ``p_normal < threshold_normal``. Specificity is the proportion of normal cells called
  normal; abnormal sensitivity the proportion of abnormal cells called abnormal.
* Macro F1 averages the categories that have at least one positive cell; weighted F1
  weights each category by its number of positive cells.
* ECE uses equal-width bins over all (cell, category) probability/label pairs pooled.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def predict_labels(probs: np.ndarray, thresholds: np.ndarray | float) -> np.ndarray:
    """Multi-hot predictions at per-category cut-offs (broadcast over cells)."""
    return (np.asarray(probs) >= np.asarray(thresholds, dtype=float)).astype(np.int64)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full_like(a, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


def per_category_scores(labels: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per-category support, sensitivity (recall), PPV (precision), specificity and F1."""
    y = np.asarray(labels).astype(bool)
    p = np.asarray(pred).astype(bool)
    tp = (y & p).sum(0)
    fp = (~y & p).sum(0)
    fn = (y & ~p).sum(0)
    tn = (~y & ~p).sum(0)
    sens = _safe_div(tp, tp + fn)
    ppv = _safe_div(tp, tp + fp)
    ppv = np.where((tp + fp == 0) & (tp + fn > 0), 0.0, ppv)  # nothing flagged -> PPV 0
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "support": tp + fn,
        "sensitivity": sens,
        "ppv": ppv,
        "specificity": _safe_div(tn, tn + fp),
        "f1": f1,
    }


def macro_f1(labels: np.ndarray, pred: np.ndarray) -> float:
    s = per_category_scores(labels, pred)
    keep = s["support"] > 0
    return float(np.mean(s["f1"][keep])) if keep.any() else float("nan")


def weighted_f1(labels: np.ndarray, pred: np.ndarray) -> float:
    s = per_category_scores(labels, pred)
    keep = s["support"] > 0
    if not keep.any():
        return float("nan")
    return float(np.sum(s["f1"][keep] * s["support"][keep]) / np.sum(s["support"][keep]))


def subset_accuracy(labels: np.ndarray, pred: np.ndarray) -> float:
    """Proportion of cells with all labels correct."""
    return float((np.asarray(labels) == np.asarray(pred)).all(axis=1).mean())


def average_precisions(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Per-category average precision (NaN where a category has no positive or no negative)."""
    out = np.full(labels.shape[1], np.nan)
    for k in range(labels.shape[1]):
        y = labels[:, k]
        if 0 < y.sum() < len(y):
            out[k] = average_precision_score(y, probs[:, k])
    return out


def abnormal_auroc(labels: np.ndarray, probs: np.ndarray, normal_index: int) -> float:
    """AUROC of ``1 - p_normal`` for abnormal (not normal) versus normal cells."""
    y_abn = 1 - np.asarray(labels)[:, normal_index]
    if y_abn.min() == y_abn.max():
        return float("nan")
    return float(roc_auc_score(y_abn, 1.0 - np.asarray(probs)[:, normal_index]))


def binary_auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, score))


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[dict[str, float | int | None]]:
    """Equal-width reliability-diagram bins over the flattened probability/label pairs."""
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(labels, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        c = int(m.sum())
        bins.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": c,
                "mean_pred": float(p[m].mean()) if c else None,
                "obs_freq": float(y[m].mean()) if c else None,
            }
        )
    return bins


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / n) |obs_freq_b - mean_pred_b| with equal-width bins, pooled pairs."""
    bins = reliability_bins(probs, labels, n_bins)
    n = np.asarray(probs).size
    return float(
        sum(b["count"] / n * abs(b["obs_freq"] - b["mean_pred"]) for b in bins if b["count"])
    )


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between probabilities and 0/1 labels over all pairs."""
    return float(np.mean((np.asarray(probs, float) - np.asarray(labels, float)) ** 2))


def net_benefit(
    y: np.ndarray, score: np.ndarray, thresholds: Sequence[float]
) -> list[dict[str, float]]:
    """Decision-curve analysis (Vickers & Elkin, 2006).

    At threshold probability ``pt``: ``NB = TP/n - FP/n * pt / (1 - pt)``; the
    treat-all reference is ``prev - (1 - prev) * pt / (1 - pt)``; treat-none is 0.
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype=float)
    n = len(y)
    prev = y.mean() if n else float("nan")
    rows = []
    for pt in thresholds:
        odds = pt / (1.0 - pt)
        flag = score >= pt
        tp = np.sum(flag & y)
        fp = np.sum(flag & ~y)
        rows.append(
            {
                "threshold": float(pt),
                "net_benefit_model": float(tp / n - fp / n * odds),
                "net_benefit_all": float(prev - (1 - prev) * odds),
                "net_benefit_none": 0.0,
            }
        )
    return rows


def classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    categories: Sequence[str],
    normal_index: int,
    ece_bins: int = 10,
) -> dict[str, float]:
    """Flat dictionary of headline and per-category metrics (keys ``metric`` / ``metric/category``)."""
    labels = np.asarray(labels).astype(np.int64)
    pred = predict_labels(probs, thresholds)
    scores = per_category_scores(labels, pred)
    aps = average_precisions(labels, probs)
    y_abn = 1 - labels[:, normal_index]
    called_abn = 1 - pred[:, normal_index]
    out: dict[str, float] = {
        "abnormal_auroc": abnormal_auroc(labels, probs, normal_index),
        "abnormal_sensitivity": float(called_abn[y_abn == 1].mean()) if y_abn.any() else np.nan,
        "specificity": float((called_abn[y_abn == 0] == 0).mean())
        if (y_abn == 0).any()
        else np.nan,
        "macro_f1": macro_f1(labels, pred),
        "weighted_f1": weighted_f1(labels, pred),
        "macro_sensitivity": float(np.nanmean(scores["sensitivity"])),
        "mean_ap": float(np.nanmean(aps)) if np.isfinite(aps).any() else float("nan"),
        "subset_accuracy": subset_accuracy(labels, pred),
        "ece": expected_calibration_error(probs, labels, ece_bins),
        "brier": brier_score(probs, labels),
    }
    supp = scores["support"]
    out["weighted_sensitivity"] = (
        float(np.nansum(scores["sensitivity"] * supp) / supp.sum()) if supp.sum() else np.nan
    )
    for k, cat in enumerate(categories):
        out[f"support/{cat}"] = float(supp[k])
        out[f"f1/{cat}"] = float(scores["f1"][k])
        out[f"sensitivity/{cat}"] = float(scores["sensitivity"][k])
        out[f"ppv/{cat}"] = float(scores["ppv"][k])
        out[f"ap/{cat}"] = float(aps[k])
        out[f"ece/{cat}"] = expected_calibration_error(probs[:, k], labels[:, k], ece_bins)
        out[f"brier/{cat}"] = brier_score(probs[:, k], labels[:, k])
    return out


def bootstrap_ci(
    fn: Callable[[np.ndarray], dict[str, float]],
    n: int,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs: resample ``n`` cells with replacement ``n_boot`` times.

    ``fn`` maps an index array to a flat dict of metrics. Resamples where a metric is
    undefined (NaN) are ignored for that metric.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for key, val in fn(idx).items():
            draws.setdefault(key, []).append(val)
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out = {}
    for key, vals in draws.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[key] = (
            (float(np.percentile(arr, lo_q)), float(np.percentile(arr, hi_q)))
            if arr.size
            else (float("nan"), float("nan"))
        )
    return out
