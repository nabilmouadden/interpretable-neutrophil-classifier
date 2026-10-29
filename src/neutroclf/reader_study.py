"""Reader-study scoring: model graded scores versus two raters.

Input CSV (one row per cell; see ``templates/reader_study_template.csv``)::

    cell_id, categories, model_<feature>..., rater1_<feature>..., rater2_<feature>...

* ``categories``: the cell's categories, separated by ``;`` (used for the partial
  correlation), e.g. ``hypogranulation;hyposegmentation``.
* ``model_<feature>``: the model's graded score in [0, 1] (``neutroclf infer`` output).
* ``rater<r>_<feature>``: ordinal grade of rater ``r`` (e.g. 0, 1, 2); empty = not graded.

Per feature the script reports:

* Spearman rho between the model score and the two-rater consensus (mean of the
  available grades), and between the two raters (benchmark);
* the partial Spearman correlation: both variables are rank-transformed, each ranked
  variable is regressed (OLS with intercept) on indicator variables of the cell's
  categories, and the Pearson correlation of the two residual vectors is reported;
* quadratic-weighted Cohen's kappa between the two raters;
* percentile bootstrap 95 % intervals over cells.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import cohen_kappa_score

from neutroclf.utils import get_logger, setup_logging, write_json


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> float:
    """Spearman correlation of ``x`` and ``y`` after removing the linear effect of ``covariates``
    from their ranks (OLS with intercept), i.e. Pearson correlation of the rank residuals."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    z = np.column_stack([np.ones(len(x)), np.asarray(covariates, float).reshape(len(x), -1)])
    res_x = rx - z @ np.linalg.lstsq(z, rx, rcond=None)[0]
    res_y = ry - z @ np.linalg.lstsq(z, ry, rcond=None)[0]
    if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(res_x, res_y)[0, 1])


def quadratic_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Quadratic-weighted Cohen's kappa between two raters' ordinal grades."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(np.union1d(a, b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def category_indicators(categories_col: Sequence[str], categories: Sequence[str]) -> np.ndarray:
    """Multi-hot matrix from ``;``-separated category strings."""
    out = np.zeros((len(categories_col), len(categories)))
    for i, entry in enumerate(categories_col):
        for name in str(entry).split(";"):
            name = name.strip()
            if not name or name.lower() == "nan":
                continue
            if name not in categories:
                raise ValueError(f"Unknown category {name!r} in row {i}")
            out[i, list(categories).index(name)] = 1.0
    return out


def _feature_stats(m, r1, r2, cats) -> dict[str, float]:
    cons = np.nanmean(np.column_stack([r1, r2]), axis=1)
    both = np.isfinite(r1) & np.isfinite(r2)
    return {
        "n_cells": int(len(m)),
        "n_both_raters": int(both.sum()),
        "rho_model_consensus": spearman(m, cons),
        "partial_rho_model_consensus": partial_spearman(m, cons, cats),
        "rho_rater1_rater2": spearman(r1[both], r2[both]),
        "kappa_rater1_rater2": quadratic_kappa(r1[both], r2[both]),
        "rho_model_rater1": spearman(m[np.isfinite(r1)], r1[np.isfinite(r1)]),
        "rho_model_rater2": spearman(m[np.isfinite(r2)], r2[np.isfinite(r2)]),
    }


def score_reader_study(
    ratings_csv: str | Path,
    features: Sequence[str],
    categories: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    """Compute per-feature agreement statistics (see module docstring)."""
    df = pd.read_csv(ratings_csv)
    if "cell_id" not in df or "categories" not in df:
        raise ValueError("Reader-study CSV needs `cell_id` and `categories` columns")
    cats_all = category_indicators(df["categories"].fillna("").astype(str), categories)
    rng = np.random.default_rng(seed)
    results: dict[str, dict] = {}
    for feat in features:
        cols = [f"model_{feat}", f"rater1_{feat}", f"rater2_{feat}"]
        missing = [c for c in cols if c not in df]
        if missing:
            raise ValueError(f"Reader-study CSV lacks columns {missing}")
        m = pd.to_numeric(df[cols[0]], errors="coerce").to_numpy(float)
        r1 = pd.to_numeric(df[cols[1]], errors="coerce").to_numpy(float)
        r2 = pd.to_numeric(df[cols[2]], errors="coerce").to_numpy(float)
        keep = np.isfinite(m) & (np.isfinite(r1) | np.isfinite(r2))
        m, r1, r2, cats = m[keep], r1[keep], r2[keep], cats_all[keep]
        stats = _feature_stats(m, r1, r2, cats)
        if n_bootstrap > 0 and len(m) >= 3:
            draws: dict[str, list[float]] = {}
            for _ in range(n_bootstrap):
                idx = rng.integers(0, len(m), len(m))
                s = _feature_stats(m[idx], r1[idx], r2[idx], cats[idx])
                for k in (
                    "rho_model_consensus",
                    "partial_rho_model_consensus",
                    "rho_rater1_rater2",
                    "kappa_rater1_rater2",
                ):
                    draws.setdefault(k, []).append(s[k])
            for k, vals in draws.items():
                arr = np.asarray(vals, float)
                arr = arr[np.isfinite(arr)]
                stats[f"{k}_ci95"] = (
                    [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
                    if arr.size
                    else [float("nan"), float("nan")]
                )
        results[feat] = stats
    return {"n_rows": int(len(df)), "n_bootstrap": n_bootstrap, "features": results}


def run_reader_study(
    ratings_csv: str | Path,
    out_dir: str | Path,
    features: Sequence[str],
    categories: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    """Score a reader-study CSV and write ``reader_study.json`` / ``reader_study.csv``."""
    setup_logging()
    res = score_reader_study(ratings_csv, features, categories, n_bootstrap, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(res, out_dir / "reader_study.json")
    rows = [
        {"feature": f} | {k: v for k, v in s.items() if not k.endswith("_ci95")}
        for f, s in res["features"].items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "reader_study.csv", index=False)
    for f, s in res["features"].items():
        get_logger().info(
            "%-28s rho(model, consensus) %.2f | partial rho %.2f | rho(r1, r2) %.2f | kappa(r1, r2) %.2f",
            f,
            s["rho_model_consensus"],
            s["partial_rho_model_consensus"],
            s["rho_rater1_rater2"],
            s["kappa_rater1_rater2"],
        )
    return res


def write_template(path: str | Path, features: Sequence[str], n_rows: int = 0) -> None:
    """Write an empty reader-study template with ``n_rows`` blank cells."""
    cols = ["cell_id", "categories"]
    cols += [f"model_{f}" for f in features]
    cols += [f"rater{r}_{f}" for r in (1, 2) for f in features]
    cols += ["notes"]
    df = pd.DataFrame({c: [""] * n_rows for c in cols})
    if n_rows:
        df["cell_id"] = [f"cell_{i + 1:02d}" for i in range(n_rows)]
    df.to_csv(path, index=False)
