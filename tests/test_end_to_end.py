"""Tiny offline end-to-end run: toy data -> train -> calibrate -> evaluate -> infer ...

Uses a small randomly initialised ViT (tests/toy_config.yaml with fewer epochs), so it
runs on CPU without any download.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from neutroclf.cli import main
from neutroclf.utils import read_json


@pytest.fixture(scope="module")
def toy(tmp_path_factory, repo_root):
    out = tmp_path_factory.mktemp("toy")
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "tests" / "toy_data.py"),
            "--out",
            str(out),
            "--n",
            "90",
            "--n-external",
            "16",
            "--size",
            "32",
            "--seed",
            "0",
        ],
        check=True,
    )
    return out


@pytest.fixture(scope="module")
def run_dir(toy, tmp_path_factory, repo_root):
    run = tmp_path_factory.mktemp("runs") / "e2e"
    main(
        [
            "train",
            "--config",
            str(repo_root / "tests" / "toy_config.yaml"),
            "--manifest",
            str(toy / "manifest.csv"),
            "--out",
            str(run),
            "--seed",
            "0",
            "--set",
            "train.epochs=2",
            "--set",
            "data.image_size=32",
            "--set",
            "data.augmentation=stain",
            "--set",
            "evaluation.n_bootstrap=20",
        ]
    )
    return run


def test_train_outputs(run_dir):
    for name in (
        "checkpoint.pt",
        "config.yaml",
        "split.csv",
        "train_log.csv",
        "predictions_val.npz",
        "predictions_test.npz",
    ):
        assert (run_dir / name).exists(), name
    split = pd.read_csv(run_dir / "split.csv")
    assert set(split["split"]) == {"train", "val", "test"}
    with np.load(run_dir / "predictions_test.npz") as d:
        assert d["category_logits"].shape[1] == 7 and d["feature_scores"].shape[1] == 4
        assert np.all((d["feature_scores"] >= 0) & (d["feature_scores"] <= 1))


def test_calibrate_evaluate(run_dir):
    main(["calibrate", "--run", str(run_dir)])
    cal = read_json(run_dir / "calibration.json")
    assert len(cal["temperatures"]) == 7 and all(t > 0 for t in cal["temperatures"])
    assert all(0.05 <= t <= 0.95 for t in cal["thresholds"])
    assert 0 <= cal["conformal"]["qhat"] <= 1
    main(["evaluate", "--run", str(run_dir), "--figures"])
    m = read_json(run_dir / "metrics_test.json")
    for key in ("macro_f1", "weighted_f1", "mean_ap", "subset_accuracy", "ece", "brier"):
        assert key in m["headline"]
    assert m["referral"][0]["fraction_referred"] == 0.0
    assert m["referral"][-1]["fraction_referred"] == pytest.approx(0.2)
    assert (run_dir / "figures" / "reliability.png").exists()
    assert (run_dir / "per_cell_test.csv").exists()


def test_infer_attention_external(run_dir, toy, repo_root, capsys):
    out = run_dir / "infer.json"
    main(
        [
            "infer",
            "--run",
            str(run_dir),
            str(toy / "images" / "cell_0000.png"),
            str(toy / "images" / "cell_0001.png"),
            "--out",
            str(out),
        ]
    )
    res = json.loads(out.read_text())["results"]
    assert len(res) == 2
    r = res[0]
    assert set(r["categories"]) == {
        "normal",
        "hypogranulation",
        "hypergranulation",
        "hyposegmentation",
        "hypersegmentation",
        "chromatin_hypercondensation",
        "dohle_bodies",
    }
    assert set(r["features"]) == {
        "nuclear_segmentation",
        "chromatin_hypercondensation",
        "granularity",
        "dohle_bodies",
    }
    assert isinstance(r["shortlist"], list) and 0 < r["joint_confidence"] <= 1

    png = run_dir / "attention.png"
    main(
        [
            "attention",
            "--run",
            str(run_dir),
            "--out",
            str(png),
            str(toy / "images"),
            "--max-images",
            "2",
        ]
    )
    assert png.exists() and png.stat().st_size > 0

    main(
        [
            "external-eval",
            "--run",
            str(run_dir),
            "--manifest",
            str(toy / "external_manifest.csv"),
            "--label-map",
            str(repo_root / "configs" / "external_neunn.yaml"),
            "--n-bootstrap",
            "10",
        ]
    )
    ext = read_json(run_dir / "external_metrics.json")
    assert ext["n_excluded"] > 0 and "abnormal_auroc" in ext["metrics"]
    assert "recall/hypogranulation" in ext["metrics"]


def test_reader_study_summarize_manifest(run_dir, toy, tmp_path):
    main(
        [
            "reader-study",
            "--ratings",
            str(toy / "reader_study_toy.csv"),
            "--out",
            str(tmp_path / "rs"),
            "--n-bootstrap",
            "20",
        ]
    )
    assert (tmp_path / "rs" / "reader_study.json").exists()
    main(["summarize", str(run_dir), str(run_dir), "--out", str(tmp_path / "summary")])
    assert "macro_f1" in (tmp_path / "summary" / "tables.md").read_text()
    main(
        [
            "make-manifest",
            "--folders",
            str(toy / "external"),
            "--external",
            "--out",
            str(tmp_path / "ext.csv"),
        ]
    )
    assert list(pd.read_csv(tmp_path / "ext.csv").columns) == ["image_path", "label"]


def test_module_entry_point():
    out = subprocess.run(
        [sys.executable, "-m", "neutroclf", "--version"], capture_output=True, text=True, check=True
    )
    assert "neutroclf" in out.stdout
