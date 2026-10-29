"""Reader-study statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from neutroclf.constants import DEFAULT_CATEGORIES, DEFAULT_FEATURES
from neutroclf.reader_study import (
    category_indicators,
    partial_spearman,
    quadratic_kappa,
    score_reader_study,
    write_template,
)


def test_partial_spearman_removes_category_effect():
    rng = np.random.default_rng(0)
    cat = rng.integers(0, 2, 400)
    x = 2.0 * cat + rng.normal(0, 0.1, 400)  # driven by the category only
    y = 2.0 * cat + rng.normal(0, 0.1, 400)
    assert spearmanr(x, y).statistic > 0.6
    assert abs(partial_spearman(x, y, cat[:, None])) < 0.15


def test_partial_spearman_keeps_within_category_signal():
    rng = np.random.default_rng(1)
    cat = rng.integers(0, 2, 400)
    shared = rng.normal(0, 1, 400)
    x = 3.0 * cat + shared + rng.normal(0, 0.3, 400)
    y = 3.0 * cat + shared + rng.normal(0, 0.3, 400)
    assert partial_spearman(x, y, cat[:, None]) > 0.7


def test_partial_spearman_without_covariates_equals_spearman():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=50), rng.normal(size=50)
    assert partial_spearman(x, y, np.zeros((50, 0))) == pytest.approx(spearmanr(x, y).statistic)


def test_quadratic_kappa_matches_sklearn():
    a = np.array([0, 1, 2, 2, 1, 0, 2, 1])
    b = np.array([0, 2, 2, 1, 1, 0, 2, 0])
    assert quadratic_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b, weights="quadratic"))


def test_category_indicators_parsing():
    m = category_indicators(["normal", "hypogranulation;hyposegmentation", ""], DEFAULT_CATEGORIES)
    assert m.sum(1).tolist() == [1, 2, 0]
    with pytest.raises(ValueError):
        category_indicators(["unknown"], DEFAULT_CATEGORIES)


def test_template_roundtrip_and_scoring(tmp_path):
    path = tmp_path / "template.csv"
    write_template(path, DEFAULT_FEATURES, n_rows=3)
    df = pd.read_csv(path)
    assert list(df.columns[:2]) == ["cell_id", "categories"] and len(df) == 3
    rng = np.random.default_rng(0)
    n = 40
    cats = rng.choice(["normal", "hypogranulation", "hypersegmentation"], n)
    filled = pd.DataFrame({"cell_id": [f"c{i}" for i in range(n)], "categories": cats})
    for f in DEFAULT_FEATURES:
        truth = rng.normal(size=n)
        filled[f"model_{f}"] = 1 / (1 + np.exp(-truth))
        filled[f"rater1_{f}"] = np.clip(np.round(truth + 1 + rng.normal(0, 0.5, n)), 0, 2)
        filled[f"rater2_{f}"] = np.clip(np.round(truth + 1 + rng.normal(0, 0.5, n)), 0, 2)
    filled.loc[0, f"rater2_{DEFAULT_FEATURES[0]}"] = np.nan  # missing grade tolerated
    filled.to_csv(path, index=False)
    res = score_reader_study(path, DEFAULT_FEATURES, DEFAULT_CATEGORIES, n_bootstrap=20)
    for f in DEFAULT_FEATURES:
        s = res["features"][f]
        assert s["rho_model_consensus"] > 0.3
        assert len(s["rho_model_consensus_ci95"]) == 2
