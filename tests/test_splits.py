"""Iterative stratification."""

from __future__ import annotations

import numpy as np

from neutroclf.splits import stratified_split


def make_labels(n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = np.array([0.47, 0.30, 0.05, 0.18, 0.04, 0.08, 0.05])
    y = (rng.random((n, 7)) < probs).astype(int)
    y[:, 0] = (y[:, 1:].sum(1) == 0).astype(int)
    return y


def test_split_is_a_partition_with_target_sizes():
    y = make_labels()
    s = stratified_split(y, (0.8, 0.1, 0.1), seed=0)
    idx = np.concatenate([s["train"], s["val"], s["test"]])
    assert sorted(idx.tolist()) == list(range(len(y)))
    for name, frac in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
        assert abs(len(s[name]) / len(y) - frac) < 0.01


def test_split_preserves_every_category_proportion():
    y = make_labels()
    s = stratified_split(y, (0.8, 0.1, 0.1), seed=1)
    overall = y.mean(0)
    for name in ("train", "val", "test"):
        assert np.all(np.abs(y[s[name]].mean(0) - overall) < 0.02)


def test_split_is_deterministic_per_seed():
    y = make_labels()
    a = stratified_split(y, seed=7)
    b = stratified_split(y, seed=7)
    c = stratified_split(y, seed=8)
    assert all((a[k] == b[k]).all() for k in a)
    assert any(len(a[k]) != len(c[k]) or (a[k] != c[k]).any() for k in a)


def test_tiny_dataset_has_no_empty_subset():
    y = np.repeat(np.eye(7, dtype=int), 3, axis=0)  # 3 cells per category
    s = stratified_split(y, seed=0)
    assert all(len(v) > 0 for v in s.values())
