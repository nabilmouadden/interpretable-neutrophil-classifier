"""Default label space (the configuration file remains the source of truth for training)."""

from __future__ import annotations

DEFAULT_CATEGORIES: tuple[str, ...] = (
    "normal",
    "hypogranulation",
    "hypergranulation",
    "hyposegmentation",
    "hypersegmentation",
    "chromatin_hypercondensation",
    "dohle_bodies",
)

DEFAULT_FEATURES: tuple[str, ...] = (
    "nuclear_segmentation",
    "chromatin_hypercondensation",
    "granularity",
    "dohle_bodies",
)
