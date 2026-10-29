"""``calibrate``, ``evaluate`` and ``summarize`` commands operating on a run directory."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from neutroclf.calibration import (
    apply_temperatures,
    conformal_coverage,
    conformal_qhat,
    conformal_sets,
    fit_temperatures,
    joint_confidence,
    referral_curve,
    sigmoid,
    threshold_grid,
    tune_thresholds,
)
from neutroclf.metrics import (
    bootstrap_ci,
    brier_score,
    classification_metrics,
    expected_calibration_error,
    net_benefit,
    predict_labels,
    reliability_bins,
)
from neutroclf.predict import load_predictions
from neutroclf.utils import get_logger, read_json, setup_logging, write_json

HEADLINE = (
    "abnormal_auroc",
    "macro_f1",
    "weighted_f1",
    "macro_sensitivity",
    "weighted_sensitivity",
    "mean_ap",
    "subset_accuracy",
    "specificity",
    "abnormal_sensitivity",
    "ece",
    "brier",
)


def load_run_config(run_dir: str | Path) -> dict:
    with open(Path(run_dir) / "config.yaml") as fh:
        return yaml.safe_load(fh)


# ----------------------------------------------------------------------------- calibrate
def calibrate_run(run_dir: str | Path) -> dict:
    """Fit temperatures, cut-offs and the conformal quantile on the validation predictions."""
    run_dir = Path(run_dir)
    log = setup_logging()
    cfg = load_run_config(run_dir)
    ccfg = cfg["calibration"]
    val = load_predictions(run_dir / "predictions_val.npz")
    logits, labels = val["category_logits"], val["labels"]
    n_bins = int(cfg["evaluation"]["ece_bins"])

    temps = fit_temperatures(logits, labels)
    probs = apply_temperatures(logits, temps)
    g = ccfg["threshold_grid"]
    thresholds = tune_thresholds(probs, labels, threshold_grid(g["start"], g["stop"], g["step"]))
    alpha = float(ccfg["alpha"])
    qhat, n_pairs = conformal_qhat(probs, labels, alpha)
    raw = sigmoid(logits)
    result = {
        "categories": cfg["categories"],
        "temperatures": temps,
        "thresholds": thresholds,
        "conformal": {
            "alpha": alpha,
            "qhat": qhat,
            "probability_threshold": 1.0 - qhat,
            "n_calibration_pairs": n_pairs,
        },
        "validation": {
            "n_cells": int(len(labels)),
            "ece_uncalibrated": expected_calibration_error(raw, labels, n_bins),
            "ece_calibrated": expected_calibration_error(probs, labels, n_bins),
            "brier_uncalibrated": brier_score(raw, labels),
            "brier_calibrated": brier_score(probs, labels),
        },
    }
    write_json(result, run_dir / "calibration.json")
    log.info(
        "calibration: T=%s thresholds=%s qhat=%.4f (n=%d) | val ECE %.4f -> %.4f",
        np.round(temps, 3).tolist(),
        thresholds.tolist(),
        qhat,
        n_pairs,
        result["validation"]["ece_uncalibrated"],
        result["validation"]["ece_calibrated"],
    )
    return result


# ----------------------------------------------------------------------------- evaluate
def evaluate_predictions(
    logits: np.ndarray,
    labels: np.ndarray,
    calibration: dict,
    cfg: dict,
    n_bootstrap: int | None = None,
) -> dict:
    """All internal-test metrics for one run (calibrated probabilities, validation cut-offs)."""
    ecfg = cfg["evaluation"]
    categories = list(cfg["categories"])
    normal_index = categories.index(cfg.get("normal_category", "normal"))
    temps = np.asarray(calibration["temperatures"])
    thresholds = np.asarray(calibration["thresholds"])
    n_bins = int(ecfg["ece_bins"])
    probs = apply_temperatures(logits, temps)
    labels = np.asarray(labels).astype(np.int64)
    n = len(labels)

    def core(idx: np.ndarray) -> dict[str, float]:
        return classification_metrics(
            probs[idx], labels[idx], thresholds, categories, normal_index, n_bins
        )

    point = core(np.arange(n))
    n_boot = int(ecfg["n_bootstrap"] if n_bootstrap is None else n_bootstrap)
    ci = bootstrap_ci(core, n, n_boot, int(ecfg.get("bootstrap_seed", 0))) if n_boot > 0 else {}

    pred = predict_labels(probs, thresholds)
    correct = (pred == labels).all(axis=1)
    conf = joint_confidence(probs)
    step = float(ecfg["referral_step"])
    fractions = np.round(np.arange(0.0, float(ecfg["referral_max_fraction"]) + step / 2, step), 6)
    sets = conformal_sets(probs, float(calibration["conformal"]["qhat"]))
    d = ecfg["decision_thresholds"]
    pts = threshold_grid(d["start"], d["stop"], d["step"])
    raw = sigmoid(logits)
    return {
        "n_cells": n,
        "categories": categories,
        "thresholds": thresholds,
        "temperatures": temps,
        "headline": {k: point[k] for k in HEADLINE},
        "headline_ci95": {k: ci.get(k) for k in HEADLINE} if ci else {},
        "per_category": {
            cat: {
                m: point[f"{m}/{cat}"]
                for m in ("support", "f1", "sensitivity", "ppv", "ap", "ece", "brier")
            }
            | {f"{m}_ci95": ci.get(f"{m}/{cat}") for m in ("f1", "sensitivity", "ppv") if ci}
            | {"threshold": float(thresholds[k])}
            for k, cat in enumerate(categories)
        },
        "calibration": {
            "reliability_bins": reliability_bins(probs, labels, n_bins),
            "ece_uncalibrated": expected_calibration_error(raw, labels, n_bins),
            "brier_uncalibrated": brier_score(raw, labels),
        },
        "conformal": {"alpha": calibration["conformal"]["alpha"]}
        | conformal_coverage(sets, labels),
        "referral": referral_curve(conf, correct, fractions),
        "decision_curve": net_benefit(
            1 - labels[:, normal_index], 1.0 - probs[:, normal_index], pts
        ),
        "n_bootstrap": n_boot,
    }


def per_cell_table(
    cell_ids: Sequence[str],
    logits: np.ndarray,
    feature_scores: np.ndarray,
    labels: np.ndarray,
    calibration: dict,
    cfg: dict,
) -> pd.DataFrame:
    """One row per cell: labels, calibrated probabilities, calls, shortlist, features, confidence."""
    categories = list(cfg["categories"])
    features = list(cfg["feature_map"]["features"])
    probs = apply_temperatures(logits, np.asarray(calibration["temperatures"]))
    pred = predict_labels(probs, np.asarray(calibration["thresholds"]))
    sets = conformal_sets(probs, float(calibration["conformal"]["qhat"]))
    df = pd.DataFrame({"cell_id": list(cell_ids)})
    df["true"] = [";".join(c for c, v in zip(categories, row, strict=True) if v) for row in labels]
    df["predicted"] = [
        ";".join(c for c, v in zip(categories, row, strict=True) if v) for row in pred
    ]
    df["shortlist"] = [
        ";".join(categories[k] for k in np.argsort(-p) if s[k])
        for p, s in zip(probs, sets, strict=True)
    ]
    df["all_correct"] = (pred == labels).all(axis=1).astype(int)
    df["joint_confidence"] = joint_confidence(probs)
    for k, c in enumerate(categories):
        df[f"p_{c}"] = probs[:, k]
    for j, f in enumerate(features):
        df[f"feature_{f}"] = feature_scores[:, j]
    return df


def evaluate_run(
    run_dir: str | Path, n_bootstrap: int | None = None, figures: bool = False
) -> dict:
    """Evaluate ``predictions_test.npz`` with ``calibration.json``; writes metrics and tables."""
    run_dir = Path(run_dir)
    log = setup_logging()
    cfg = load_run_config(run_dir)
    cal_path = run_dir / "calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(f"{cal_path} not found: run `neutroclf calibrate` first")
    calibration = read_json(cal_path)
    test = load_predictions(run_dir / "predictions_test.npz")
    res = evaluate_predictions(
        test["category_logits"], test["labels"], calibration, cfg, n_bootstrap
    )
    feat_mae = np.abs(test["feature_scores"] - test["feature_targets"]).mean(axis=0)
    res["feature_mae_vs_mapped_targets"] = dict(
        zip(cfg["feature_map"]["features"], feat_mae.tolist(), strict=True)
    )
    write_json(res, run_dir / "metrics_test.json")
    per_cell_table(
        test["cell_ids"],
        test["category_logits"],
        test["feature_scores"],
        test["labels"],
        calibration,
        cfg,
    ).to_csv(run_dir / "per_cell_test.csv", index=False)
    h = res["headline"]
    log.info(
        "test (n=%d): AUROC %.3f | macro-F1 %.3f | weighted-F1 %.3f | mAP %.3f | accuracy %.3f | "
        "specificity %.3f | ECE %.4f | Brier %.4f | conformal pair cov %.3f, cell cov %.3f",
        res["n_cells"],
        h["abnormal_auroc"],
        h["macro_f1"],
        h["weighted_f1"],
        h["mean_ap"],
        h["subset_accuracy"],
        h["specificity"],
        h["ece"],
        h["brier"],
        res["conformal"]["pair_coverage"],
        res["conformal"]["cell_coverage"],
    )
    if figures:
        from neutroclf.figures import plot_run_figures

        plot_run_figures(res, run_dir / "figures")
    return res


# ----------------------------------------------------------------------------- summarize
def summarize_runs(run_dirs: Sequence[str | Path], out: str | Path | None = None) -> pd.DataFrame:
    """Mean and SD across runs (e.g. five seeds) of headline and per-category test metrics."""
    rows = []
    for rd in run_dirs:
        m = read_json(Path(rd) / "metrics_test.json")
        row = {"run": str(rd)} | {k: m["headline"][k] for k in HEADLINE}
        for cat, vals in m["per_category"].items():
            for k in ("support", "f1", "sensitivity", "ppv"):
                row[f"{k}/{cat}"] = vals[k]
        for r in m["referral"]:
            row[f"accuracy_retained@{r['fraction_referred']:.2f}"] = r["accuracy_retained"]
        row["conformal_pair_coverage"] = m["conformal"]["pair_coverage"]
        row["conformal_cell_coverage"] = m["conformal"]["cell_coverage"]
        ext_path = Path(rd) / "external_metrics.json"
        if ext_path.exists():  # Table 3: external evaluation, if run
            for k, v in read_json(ext_path)["metrics"].items():
                row[f"external/{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows).set_index("run")
    summary = pd.DataFrame(
        {
            "mean": df.mean(numeric_only=True),
            "sd": df.std(numeric_only=True, ddof=1),
            "n_runs": df.count(numeric_only=True),
        }
    )
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "per_run.csv")
        summary.to_csv(out / "summary.csv")
        _write_markdown_tables(summary, out / "tables.md", [str(c) for c in _categories(run_dirs)])
        get_logger().info("wrote %s", out)
    return summary


def _categories(run_dirs: Sequence[str | Path]) -> list[str]:
    return list(read_json(Path(run_dirs[0]) / "metrics_test.json")["categories"])


def _fmt(summary: pd.DataFrame, key: str, digits: int = 3) -> str:
    if key not in summary.index or pd.isna(summary.loc[key, "mean"]):
        return "n/a"
    m, s = summary.loc[key, "mean"], summary.loc[key, "sd"]
    return f"{m:.{digits}f}" if pd.isna(s) else f"{m:.{digits}f} ± {s:.{digits}f}"


def _write_markdown_tables(summary: pd.DataFrame, path: Path, categories: list[str]) -> None:
    lines = ["# Test-set summary across runs (mean ± SD)", "", "| Metric | Value |", "|---|---|"]
    for k in HEADLINE:
        lines.append(f"| {k} | {_fmt(summary, k)} |")
    lines += [
        "",
        "## Per-category detection (Table 2 layout)",
        "",
        "| Category | Positive test cells (n) | F1 | Sensitivity | PPV |",
        "|---|---|---|---|---|",
    ]
    for c in categories:
        lines.append(
            f"| {c} | {_fmt(summary, f'support/{c}', 1)} | {_fmt(summary, f'f1/{c}')} | "
            f"{_fmt(summary, f'sensitivity/{c}')} | {_fmt(summary, f'ppv/{c}')} |"
        )
    lines.append(
        f"| Macro-average | | {_fmt(summary, 'macro_f1')} | {_fmt(summary, 'macro_sensitivity')} | |"
    )
    lines.append(
        f"| Weighted average | | {_fmt(summary, 'weighted_f1')} | {_fmt(summary, 'weighted_sensitivity')} | |"
    )
    lines += [
        "",
        "## Confidence-based review",
        "",
        "| Fraction referred | Accuracy on retained |",
        "|---|---|",
    ]
    for key in summary.index:
        if str(key).startswith("accuracy_retained@"):
            lines.append(f"| {str(key).split('@')[1]} | {_fmt(summary, key)} |")
    external = [k for k in summary.index if str(k).startswith("external/")]
    if external:
        lines += [
            "",
            "## External evaluation (Table 3 layout)",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        lines += [f"| {str(k).split('/', 1)[1]} | {_fmt(summary, k)} |" for k in external]
    path.write_text("\n".join(lines) + "\n")
