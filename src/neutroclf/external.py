"""Evaluation on an external single-label dataset without retraining (e.g. NeuNN).

The source labels are mapped to this model's categories through a YAML ``label_map``;
unmapped labels are excluded. Probabilities are calibrated with the run's validation
temperatures and every cut-off is carried over unchanged from the internal validation
split. Reported: abnormal-vs-normal AUROC (score ``1 - p_normal``), sensitivity and
specificity at the carried-over normal cut-off, and per-category recall
(``p_k >= cut-off_k``) for each mapped abnormal category.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from neutroclf.calibration import apply_temperatures
from neutroclf.data import read_external_manifest
from neutroclf.infer import load_run
from neutroclf.metrics import binary_auroc, bootstrap_ci
from neutroclf.predict import predict_paths
from neutroclf.utils import get_logger, setup_logging, write_json


def normalise_label(label: str) -> str:
    """Case-fold and drop spaces, underscores and hyphens (``"Dohle bodies"`` -> ``"dohlebodies"``)."""
    return re.sub(r"[\s_\-]+", "", str(label).strip().casefold())


def load_label_map(path: str | Path, categories: list[str]) -> dict[str, str]:
    with open(path) as fh:
        spec = yaml.safe_load(fh) or {}
    raw = spec.get("label_map", spec)
    out = {}
    for src, dst in raw.items():
        if dst is None:
            continue
        if dst not in categories:
            raise ValueError(f"label_map maps {src!r} to unknown category {dst!r}")
        out[normalise_label(src)] = dst
    return out


def external_metrics(
    probs: np.ndarray,
    mapped: list[str],
    thresholds: np.ndarray,
    categories: list[str],
    normal_category: str,
) -> dict[str, float]:
    """Flat metrics for mapped external cells."""
    k_n = categories.index(normal_category)
    y_abn = np.array([m != normal_category for m in mapped], dtype=int)
    called_abn = (probs[:, k_n] < thresholds[k_n]).astype(int)
    out = {
        "abnormal_auroc": binary_auroc(y_abn, 1.0 - probs[:, k_n]),
        "sensitivity": float(called_abn[y_abn == 1].mean()) if y_abn.any() else float("nan"),
        "specificity": float((called_abn[y_abn == 0] == 0).mean())
        if (y_abn == 0).any()
        else float("nan"),
    }
    arr = np.asarray(mapped)
    for cat in sorted(set(mapped) - {normal_category}):
        k = categories.index(cat)
        rows = arr == cat
        out[f"recall/{cat}"] = float((probs[rows, k] >= thresholds[k]).mean())
    return out


def run_external_eval(
    run_dir: str | Path,
    manifest: str | Path,
    label_map_path: str | Path,
    data_root: str | Path | None = None,
    out: str | Path | None = None,
    device: str = "auto",
    n_bootstrap: int | None = None,
    batch_size: int = 64,
) -> dict:
    """Apply a trained, calibrated run to an external manifest and write ``external_metrics.json``."""
    log = setup_logging()
    model, cfg, calibration = load_run(run_dir, device)
    categories = list(cfg["categories"])
    normal = cfg.get("normal_category", "normal")
    label_map = load_label_map(label_map_path, categories)
    ids, paths, labels = read_external_manifest(manifest, data_root)
    mapped = [label_map.get(normalise_label(lab)) for lab in labels]
    keep = [i for i, m in enumerate(mapped) if m is not None]
    excluded = Counter(labels[i] for i, m in enumerate(mapped) if m is None)
    if not keep:
        raise ValueError("No external label could be mapped; check the label_map")
    log.info(
        "external: %d cells mapped, %d excluded %s",
        len(keep),
        sum(excluded.values()),
        dict(excluded),
    )
    device_t = next(model.parameters()).device
    pred = predict_paths(
        model, [paths[i] for i in keep], int(cfg["data"]["image_size"]), device_t, batch_size
    )
    probs = apply_temperatures(pred["category_logits"], np.asarray(calibration["temperatures"]))
    thresholds = np.asarray(calibration["thresholds"])
    kept_labels = [mapped[i] for i in keep]
    point = external_metrics(probs, kept_labels, thresholds, categories, normal)
    n_boot = int(cfg["evaluation"]["n_bootstrap"] if n_bootstrap is None else n_bootstrap)
    ci = {}
    if n_boot > 0:
        lab_arr = np.asarray(kept_labels)
        ci = bootstrap_ci(
            lambda idx: external_metrics(
                probs[idx], list(lab_arr[idx]), thresholds, categories, normal
            ),
            len(keep),
            n_boot,
            int(cfg["evaluation"].get("bootstrap_seed", 0)),
        )
    res = {
        "n_mapped": len(keep),
        "n_excluded": int(sum(excluded.values())),
        "excluded_labels": dict(excluded),
        "mapped_counts": dict(Counter(kept_labels)),
        "cut_offs": dict(zip(categories, thresholds.tolist(), strict=True)),
        "metrics": point,
        "ci95": ci,
        "mean_feature_scores": {
            cat: dict(
                zip(
                    cfg["feature_map"]["features"],
                    pred["feature_scores"][np.asarray(kept_labels) == cat].mean(axis=0).tolist(),
                    strict=True,
                )
            )
            for cat in sorted(set(kept_labels))
        },
    }
    out = Path(out) if out is not None else Path(run_dir) / "external_metrics.json"
    write_json(res, out)
    get_logger().info(
        "external: AUROC %.3f | sensitivity %.3f | specificity %.3f | %s",
        point["abnormal_auroc"],
        point["sensitivity"],
        point["specificity"],
        {k: round(v, 3) for k, v in point.items() if k.startswith("recall/")},
    )
    return res
