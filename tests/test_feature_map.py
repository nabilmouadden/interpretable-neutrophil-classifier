"""Category-to-feature map: matrix, max-deviation targets, implied profile."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neutroclf.feature_map import FeatureMap

# rows follow configs/default.yaml category order; columns: seg, chromatin, granularity, dohle
EXPECTED_M = np.array(
    [
        [0.5, 0.0, 0.5, 0.0],  # normal
        [0.5, 0.0, 0.0, 0.0],  # hypogranulation
        [0.5, 0.0, 1.0, 0.0],  # hypergranulation
        [0.0, 0.0, 0.5, 0.0],  # hyposegmentation
        [1.0, 0.0, 0.5, 0.0],  # hypersegmentation
        [0.5, 1.0, 0.5, 0.0],  # chromatin hypercondensation
        [0.5, 0.0, 0.5, 1.0],  # dohle bodies
    ],
    dtype=np.float32,
)


def onehot(fmap: FeatureMap, *names: str) -> torch.Tensor:
    y = torch.zeros(1, fmap.num_categories)
    for n in names:
        y[0, fmap.categories.index(n)] = 1.0
    return y


def test_matrix_matches_documented_map(fmap):
    assert fmap.features == (
        "nuclear_segmentation",
        "chromatin_hypercondensation",
        "granularity",
        "dohle_bodies",
    )
    np.testing.assert_allclose(fmap.matrix, EXPECTED_M)
    np.testing.assert_allclose(fmap.normal_profile, [0.5, 0.0, 0.5, 0.0])
    assert np.all(fmap.deviations[fmap.normal_index] == 0)


def test_single_label_targets_equal_profile_rows(fmap):
    for k, cat in enumerate(fmap.categories):
        np.testing.assert_allclose(fmap.targets(onehot(fmap, cat))[0].numpy(), EXPECTED_M[k])


def test_multilabel_takes_max_deviation_per_feature(fmap):
    t = fmap.targets(onehot(fmap, "hypogranulation", "hyposegmentation"))[0].numpy()
    np.testing.assert_allclose(t, [0.0, 0.0, 0.0, 0.0])
    t = fmap.targets(
        onehot(fmap, "hypergranulation", "dohle_bodies", "chromatin_hypercondensation")
    )
    np.testing.assert_allclose(t[0].numpy(), [0.5, 1.0, 1.0, 1.0])


def test_tie_with_different_values_falls_back_to_normal(fmap):
    t = fmap.targets(onehot(fmap, "hyposegmentation", "hypersegmentation", "hypogranulation"))
    np.testing.assert_allclose(t[0].numpy(), [0.5, 0.0, 0.0, 0.0])


def test_empty_label_vector_gives_normal_profile(fmap):
    t = fmap.targets(torch.zeros(2, fmap.num_categories))
    np.testing.assert_allclose(t.numpy(), np.tile(fmap.normal_profile, (2, 1)))


def test_implied_profile_matches_targets_on_single_labels(fmap):
    eye = torch.eye(fmap.num_categories)
    np.testing.assert_allclose(fmap.implied_profile(eye).numpy(), fmap.targets(eye).numpy())


def test_implied_profile_formula_and_clipping(fmap):
    p = torch.zeros(1, fmap.num_categories)
    p[0, fmap.categories.index("hypogranulation")] = 0.6
    p[0, fmap.categories.index("hypergranulation")] = 0.2
    p[0, fmap.categories.index("dohle_bodies")] = 0.3
    phi = fmap.implied_profile(p)[0].numpy()
    # granularity: 0.5 + 0.6 * (0 - 0.5) + 0.2 * (1 - 0.5) = 0.3 ; dohle: 0 + 0.3 * 1
    np.testing.assert_allclose(phi, [0.5, 0.0, 0.3, 0.3], atol=1e-6)
    p = torch.ones(1, fmap.num_categories)
    phi = fmap.implied_profile(p)[0].numpy()
    assert phi.min() >= 0 and phi.max() <= 1


def test_implied_profile_is_differentiable(fmap):
    p = torch.full((3, fmap.num_categories), 0.3, requires_grad=True)
    fmap.implied_profile(p).sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_invalid_specs_raise(default_cfg):
    spec = dict(default_cfg["feature_map"])
    bad = dict(spec, profiles={"not_a_category": {"granularity": 0.0}})
    with pytest.raises(ValueError):
        FeatureMap.from_config(default_cfg["categories"], bad)
    bad = dict(spec, profiles={"hypogranulation": {"granularity": 1.5}})
    with pytest.raises(ValueError):
        FeatureMap.from_config(default_cfg["categories"], bad)
