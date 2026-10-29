"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from neutroclf.config import load_config
from neutroclf.feature_map import FeatureMap

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def default_cfg() -> dict:
    return load_config(REPO / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def fmap(default_cfg) -> FeatureMap:
    return FeatureMap.from_config(
        default_cfg["categories"], default_cfg["feature_map"], default_cfg["normal_category"]
    )
