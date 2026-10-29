"""Fixed category-to-feature map: per-cell feature targets and the implied feature profile.

Notation (documented identically in the README):

* ``M`` in [0, 1]^{C x F}: one row per category, one column per graded feature.
  ``m_N`` is the row of the normal category.
* Feature target of a cell with active category set ``A``::

      t_j = m_{k*_j, j},   k*_j = argmax_{k in A} |m_kj - m_Nj|

  If the maximum deviation is reached by categories with *different* values (a tie
  such as a cell labelled both hypo- and hypersegmented), or if no active category
  deviates on feature ``j``, the target is the normal value ``m_Nj``.
* Implied feature profile of category probabilities ``p`` (soft, differentiable)::

      phi_j(p) = clip( m_Nj + sum_{k != N} p_k (m_kj - m_Nj), 0, 1 )
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

_TIE_TOL = 1e-6


@dataclass(frozen=True)
class FeatureMap:
    """Category-to-feature map ``M`` with helpers to build targets and implied profiles."""

    categories: tuple[str, ...]
    features: tuple[str, ...]
    matrix: np.ndarray  # (C, F) float32, row k = profile of category k
    normal_index: int

    @classmethod
    def from_config(
        cls,
        categories: Sequence[str],
        spec: Mapping,
        normal_category: str = "normal",
    ) -> FeatureMap:
        """Build the map from the ``feature_map`` section of a config.

        ``spec`` has keys ``features`` (ordered list), ``normal`` (value of every feature
        for the normal category) and ``profiles`` (per abnormal category, the features it
        changes; unspecified features keep the normal value).
        """
        features = tuple(spec["features"])
        normal = spec["normal"]
        missing = [f for f in features if f not in normal]
        if missing:
            raise ValueError(f"feature_map.normal lacks values for {missing}")
        profiles = spec.get("profiles", {}) or {}
        unknown = [c for c in profiles if c not in categories]
        if unknown:
            raise ValueError(f"feature_map.profiles names unknown categories {unknown}")
        if normal_category in profiles:
            raise ValueError("The normal category is defined by feature_map.normal, not profiles")
        base = np.array([float(normal[f]) for f in features], dtype=np.float32)
        rows = []
        for cat in categories:
            row = base.copy()
            for feat, value in (profiles.get(cat) or {}).items():
                if feat not in features:
                    raise ValueError(f"Profile of {cat!r} uses unknown feature {feat!r}")
                row[features.index(feat)] = float(value)
            rows.append(row)
        matrix = np.stack(rows).astype(np.float32)
        if matrix.min() < 0.0 or matrix.max() > 1.0:
            raise ValueError("Feature-map values must lie in [0, 1]")
        return cls(tuple(categories), features, matrix, list(categories).index(normal_category))

    # ------------------------------------------------------------------ properties
    @property
    def num_categories(self) -> int:
        return len(self.categories)

    @property
    def num_features(self) -> int:
        return len(self.features)

    @property
    def normal_profile(self) -> np.ndarray:
        """``m_N``, shape (F,)."""
        return self.matrix[self.normal_index]

    @property
    def deviations(self) -> np.ndarray:
        """Signed deviations ``M - m_N``, shape (C, F); the normal row is zero."""
        return self.matrix - self.normal_profile[None, :]

    def _tensors(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normal = torch.as_tensor(self.normal_profile, dtype=ref.dtype, device=ref.device)
        dev = torch.as_tensor(self.deviations, dtype=ref.dtype, device=ref.device)
        return normal, dev

    # ------------------------------------------------------------------ targets
    def targets(self, labels: torch.Tensor) -> torch.Tensor:
        """Per-cell feature targets from multi-hot labels (B, C) -> (B, F) (max deviation)."""
        labels = labels.float()
        normal, dev = self._tensors(labels)
        active = (labels > 0.5).unsqueeze(-1)  # (B, C, 1)
        absdev = dev.abs().unsqueeze(0).expand(labels.shape[0], -1, -1)  # (B, C, F)
        masked = torch.where(active, absdev, torch.full_like(absdev, -1.0))
        max_abs = masked.max(dim=1).values  # (B, F)
        is_max = active & ((masked - max_abs.unsqueeze(1)).abs() <= _TIE_TOL)
        is_max = is_max & (max_abs.unsqueeze(1) > _TIE_TOL)
        dev_b = dev.unsqueeze(0).expand_as(absdev)
        hi = torch.where(is_max, dev_b, torch.full_like(dev_b, float("-inf"))).max(dim=1).values
        lo = torch.where(is_max, dev_b, torch.full_like(dev_b, float("inf"))).min(dim=1).values
        unique = is_max.any(dim=1) & ((hi - lo).abs() <= _TIE_TOL)
        chosen = torch.where(unique, hi, torch.zeros_like(hi))
        return (normal.unsqueeze(0) + chosen).clamp(0.0, 1.0)

    def targets_numpy(self, labels: np.ndarray) -> np.ndarray:
        """NumPy convenience wrapper around :meth:`targets`."""
        return self.targets(torch.as_tensor(np.asarray(labels), dtype=torch.float32)).numpy()

    # ------------------------------------------------------------------ implied profile
    def implied_profile(self, probs: torch.Tensor) -> torch.Tensor:
        """``phi(p)``: feature profile implied by category probabilities (B, C) -> (B, F).

        The normal row of ``M - m_N`` is zero, so summing over all categories equals the
        sum over ``k != N``.
        """
        normal, dev = self._tensors(probs)
        return (normal.unsqueeze(0) + probs @ dev).clamp(0.0, 1.0)

    def as_table(self) -> list[dict[str, float | str]]:
        """Rows of ``M`` as dictionaries (for logging / README tables)."""
        return [
            {"category": c, **{f: float(self.matrix[i, j]) for j, f in enumerate(self.features)}}
            for i, c in enumerate(self.categories)
        ]
