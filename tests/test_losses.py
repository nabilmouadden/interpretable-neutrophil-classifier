"""Focal loss, positive weights, feature and consistency terms."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from neutroclf.losses import (
    FocalBCEWithLogits,
    JointLoss,
    consistency_loss,
    feature_loss,
    positive_weights,
)


def test_positive_weights_ratio_and_cap():
    y = np.zeros((100, 3))
    y[:50, 0] = 1  # 50/50 -> 1
    y[:10, 1] = 1  # 90/10 -> 9
    y[:1, 2] = 1  # 99/1 -> capped at 50
    np.testing.assert_allclose(positive_weights(y, cap=50).numpy(), [1.0, 9.0, 50.0])


def test_focal_reduces_to_weighted_bce_when_gamma_zero():
    torch.manual_seed(0)
    z = torch.randn(8, 7)
    y = (torch.rand(8, 7) > 0.7).float()
    focal = FocalBCEWithLogits(gamma=0.0, alpha=0.5)(z, y)
    bce = 0.5 * F.binary_cross_entropy_with_logits(z, y, reduction="none").sum(1).mean()
    torch.testing.assert_close(focal, bce)


def test_focal_downweights_easy_examples():
    y = torch.tensor([[1.0]])
    easy = FocalBCEWithLogits(gamma=2.0, alpha=0.25)(torch.tensor([[4.0]]), y)
    hard = FocalBCEWithLogits(gamma=2.0, alpha=0.25)(torch.tensor([[-4.0]]), y)
    bce_easy = 0.25 * F.binary_cross_entropy_with_logits(torch.tensor([[4.0]]), y)
    assert easy < bce_easy * 0.01
    assert hard > easy * 100


def test_focal_pos_weight_scales_positive_term():
    z, y = torch.tensor([[0.3]]), torch.tensor([[1.0]])
    a = FocalBCEWithLogits(pos_weight=torch.tensor([1.0]))(z, y)
    b = FocalBCEWithLogits(pos_weight=torch.tensor([5.0]))(z, y)
    torch.testing.assert_close(b, 5 * a)


def test_feature_loss_minimised_at_target():
    t = torch.tensor([[0.5, 0.0, 1.0, 0.0]])
    at_target = feature_loss(torch.logit(t.clamp(1e-6, 1 - 1e-6)), t)
    off = feature_loss(torch.zeros_like(t) + 2.0, t)
    assert at_target < off


def test_consistency_zero_when_features_match_implied_profile(fmap):
    p = torch.rand(5, fmap.num_categories)
    c = fmap.implied_profile(p)
    assert consistency_loss(c, p, fmap).item() < 1e-12
    expected = ((torch.zeros_like(c) - c) ** 2).mean()
    torch.testing.assert_close(consistency_loss(torch.zeros_like(c), p, fmap), expected)


def test_joint_loss_combines_terms_and_backpropagates(fmap):
    torch.manual_seed(0)
    zc = torch.randn(6, fmap.num_categories, requires_grad=True)
    zf = torch.randn(6, fmap.num_features, requires_grad=True)
    y = torch.zeros(6, fmap.num_categories)
    y[torch.arange(6), torch.tensor([0, 1, 2, 3, 4, 5])] = 1
    crit = JointLoss(fmap, None, lambda_feature=2.0, lambda_consistency=0.1)
    total, parts = crit(zc, zf, y)
    assert (
        abs(parts["loss"] - (parts["focal"] + 2.0 * parts["feature"] + 0.1 * parts["consistency"]))
        < 1e-5
    )
    total.backward()
    assert zc.grad.abs().sum() > 0 and zf.grad.abs().sum() > 0
