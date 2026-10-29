"""Inference: images in -> JSON with calibrated category probabilities, calls, shortlist, features."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from neutroclf.calibration import apply_temperatures, conformal_sets, joint_confidence
from neutroclf.data import collect_images
from neutroclf.model import NeutrophilClassifier, load_checkpoint
from neutroclf.predict import predict_paths
from neutroclf.utils import read_json, resolve_device, setup_logging, write_json


def load_run(run_dir: str | Path, device: str = "auto") -> tuple[NeutrophilClassifier, dict, dict]:
    """Load ``checkpoint.pt`` and ``calibration.json`` from a run directory."""
    run_dir = Path(run_dir)
    cal_path = run_dir / "calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(f"{cal_path} not found: run `neutroclf calibrate` first")
    model, cfg = load_checkpoint(run_dir / "checkpoint.pt", resolve_device(device))
    return model, cfg, read_json(cal_path)


def format_predictions(
    paths: Sequence[Path],
    logits: np.ndarray,
    feature_scores: np.ndarray,
    cfg: dict,
    calibration: dict,
) -> list[dict]:
    """Per-image result dictionaries."""
    categories = list(cfg["categories"])
    features = list(cfg["feature_map"]["features"])
    thresholds = np.asarray(calibration["thresholds"])
    probs = apply_temperatures(logits, np.asarray(calibration["temperatures"]))
    sets = conformal_sets(probs, float(calibration["conformal"]["qhat"]))
    conf = joint_confidence(probs)
    results = []
    for i, path in enumerate(paths):
        order = np.argsort(-probs[i])
        results.append(
            {
                "image": str(path),
                "categories": {c: round(float(probs[i, k]), 4) for k, c in enumerate(categories)},
                "predicted": [categories[k] for k in order if probs[i, k] >= thresholds[k]],
                "shortlist": [categories[k] for k in order if sets[i, k]],
                "features": {
                    f: round(float(feature_scores[i, j]), 4) for j, f in enumerate(features)
                },
                "joint_confidence": round(float(conf[i]), 4),
            }
        )
    return results


def run_inference(
    run_dir: str | Path,
    inputs: Sequence[str | Path],
    out: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 32,
) -> list[dict]:
    """Predict on image files / directories; write JSON to ``out`` or stdout."""
    setup_logging()
    model, cfg, calibration = load_run(run_dir, device)
    paths = collect_images(inputs)
    if not paths:
        raise ValueError("No images found")
    device_t = next(model.parameters()).device
    pred = predict_paths(model, paths, int(cfg["data"]["image_size"]), device_t, batch_size)
    results = format_predictions(
        paths, pred["category_logits"], pred["feature_scores"], cfg, calibration
    )
    meta = {
        "categories_note": "calibrated probabilities (per-category temperature scaling)",
        "predicted_note": "categories at or above their validation-tuned cut-off",
        "shortlist_note": (
            f"split-conformal shortlist, target coverage {1 - calibration['conformal']['alpha']:.2f}"
        ),
        "features_note": "graded feature scores in [0, 1]",
    }
    payload = {
        "meta": meta,
        "cut_offs": dict(zip(cfg["categories"], calibration["thresholds"], strict=True)),
        "results": results,
    }
    if out is None:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        write_json(payload, out)
    return results
