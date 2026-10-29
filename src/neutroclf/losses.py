"""Training objective.

    L = L_focal(p, y) + lambda_f * BCE(c, t) + lambda_c * L_cons

* ``L_focal``: multi-label focal loss (gamma = 2, alpha = 0.25) on the category logits,
  with per-category positive weights ``w_k = min(n_neg,k / n_pos,k, cap)`` (cap 50)
  computed on the training split; summed over categories and averaged over the batch.
* ``BCE(c, t)``: binary cross-entropy of the feature scores ``c = sigmoid(feature
  logits)`` against the soft targets ``t`` produced by the category-to-feature map,
  averaged over the four features and the batch.
* ``L_cons = (1/F) sum_j (c_j - phi_j(p))^2`` averaged over the batch, where
  ``p = sigmoid(category logits)`` and ``phi`` is the feature profile implied by the
  category probabilities through the same map (see :mod:`neutroclf.feature_map`).
  Gradients flow into both heads.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neutroclf.feature_map import FeatureMap


def positive_weights(labels: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    """Per-category ``min(n_neg / n_pos, cap)`` from a multi-hot training label matrix."""
    labels = np.asarray(labels, dtype=np.float64)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    weights = np.minimum(neg / np.maximum(pos, 1.0), cap)
    return torch.tensor(weights, dtype=torch.float32)


class FocalBCEWithLogits(nn.Module):
    """Multi-label focal loss (Lin et al., 2017) with optional per-category positive weights.

    ``loss = alpha_t * (1 - p_t)^gamma * BCE_w(z, y)`` where ``BCE_w`` is the
    ``pos_weight``-weighted binary cross-entropy, ``p_t`` the probability of the true
    state and ``alpha_t = alpha * y + (1 - alpha) * (1 - y)``.
    """

    def __init__(
        self, gamma: float = 2.0, alpha: float = 0.25, pos_weight: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).sum(dim=1).mean()


def feature_loss(feature_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE between feature scores and soft feature targets, averaged over features and cells."""
    return F.binary_cross_entropy_with_logits(feature_logits, targets.float())


def consistency_loss(
    feature_scores: torch.Tensor, category_probs: torch.Tensor, fmap: FeatureMap
) -> torch.Tensor:
    """``mean_b (1/F) sum_j (c_j - phi_j(p))^2``."""
    implied = fmap.implied_profile(category_probs)
    return ((feature_scores - implied) ** 2).mean()


class JointLoss(nn.Module):
    """Focal category loss + weighted feature BCE + weighted consistency term."""

    def __init__(
        self,
        fmap: FeatureMap,
        pos_weight: torch.Tensor | None,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        lambda_feature: float = 2.0,
        lambda_consistency: float = 0.1,
    ) -> None:
        super().__init__()
        self.fmap = fmap
        self.focal = FocalBCEWithLogits(focal_gamma, focal_alpha, pos_weight)
        self.lambda_feature = lambda_feature
        self.lambda_consistency = lambda_consistency

    def forward(
        self,
        category_logits: torch.Tensor,
        feature_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        category_logits = category_logits.float()
        feature_logits = feature_logits.float()
        targets = self.fmap.targets(labels)
        l_cat = self.focal(category_logits, labels)
        l_feat = feature_loss(feature_logits, targets)
        l_cons = consistency_loss(
            torch.sigmoid(feature_logits), torch.sigmoid(category_logits), self.fmap
        )
        total = l_cat + self.lambda_feature * l_feat + self.lambda_consistency * l_cons
        parts = {
            "loss": float(total.detach()),
            "focal": float(l_cat.detach()),
            "feature": float(l_feat.detach()),
            "consistency": float(l_cons.detach()),
        }
        return total, parts
