"""Cell-level stratified multi-label split by iterative stratification.

Implements the algorithm of Sechidis, Tsoumakas and Vlahavas (ECML-PKDD 2011,
"On the stratification of multi-label data"): labels are processed from the rarest to
the most common, and every cell carrying the current label is assigned to the subset
that still needs the most examples of that label. This keeps every category's
proportion close to the target in each subset, including the rare ones.

The split is at the *cell* level. Cells from the same slide or patient may end up in
different subsets; if slide identifiers are available, supply your own split through the
manifest ``split`` column instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SPLIT_NAMES = ("train", "val", "test")


def iterative_stratification(
    labels: np.ndarray, fractions: Sequence[float], seed: int = 0
) -> np.ndarray:
    """Assign each row of a multi-hot ``labels`` matrix (N, L) to one of ``len(fractions)`` subsets.

    Returns an integer array of subset indices of length N.
    """
    labels = np.asarray(labels).astype(bool)
    n, n_labels = labels.shape
    fractions = np.asarray(fractions, dtype=float)
    if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("fractions must be non-negative and sum to 1")
    rng = np.random.default_rng(seed)
    n_subsets = len(fractions)

    desired_rows = fractions * n  # remaining capacity per subset
    desired_label = np.outer(fractions, labels.sum(axis=0)).astype(float)  # (S, L)
    assignment = np.full(n, -1, dtype=int)
    unassigned = np.ones(n, dtype=bool)

    def pick_subset(candidates: np.ndarray) -> int:
        best = candidates[desired_rows[candidates] == desired_rows[candidates].max()]
        return int(rng.choice(best))

    while True:
        remaining = labels & unassigned[:, None]
        counts = remaining.sum(axis=0)
        if counts.sum() == 0:
            break
        # rarest label that still has unassigned examples (ties broken at random)
        positive = np.flatnonzero(counts > 0)
        rarest = positive[counts[positive] == counts[positive].min()]
        lab = int(rng.choice(rarest))
        rows = np.flatnonzero(remaining[:, lab])
        rng.shuffle(rows)
        for row in rows:
            need = desired_label[:, lab]
            cand = np.flatnonzero(need == need.max())
            subset = pick_subset(cand) if len(cand) > 1 else int(cand[0])
            assignment[row] = subset
            unassigned[row] = False
            desired_label[subset] -= labels[row]
            desired_rows[subset] -= 1

    # rows without any label: fill the subsets that still have capacity
    for row in np.flatnonzero(unassigned):
        subset = pick_subset(np.arange(n_subsets))
        assignment[row] = subset
        desired_rows[subset] -= 1

    # Tiny-dataset safeguard: with very few examples per label the greedy rule can leave
    # a subset with a positive fraction empty; move one random row from the largest subset.
    for subset in range(n_subsets):
        if fractions[subset] > 0 and not np.any(assignment == subset) and n >= n_subsets:
            sizes = np.bincount(assignment, minlength=n_subsets)
            donor = int(np.argmax(sizes))
            assignment[int(rng.choice(np.flatnonzero(assignment == donor)))] = subset
    return assignment


def stratified_split(
    labels: np.ndarray,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Train/val/test indices from an iterative-stratification assignment."""
    assignment = iterative_stratification(labels, fractions, seed=seed)
    return {name: np.flatnonzero(assignment == i) for i, name in enumerate(SPLIT_NAMES)}
