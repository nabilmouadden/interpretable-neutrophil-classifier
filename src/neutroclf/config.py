"""YAML configuration loading with ``base:`` inheritance and ``key.path=value`` overrides."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]

REQUIRED_SECTIONS = (
    "categories",
    "feature_map",
    "data",
    "model",
    "loss",
    "train",
    "calibration",
    "evaluation",
)


def deep_merge(base: Config, override: Config) -> Config:
    """Recursively merge ``override`` into a copy of ``base`` (dicts merge, other values replace)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_with_bases(path: Path, _seen: tuple[Path, ...] = ()) -> Config:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular `base:` reference involving {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a YAML mapping")
    base_ref = cfg.pop("base", None)
    if base_ref is not None:
        base_cfg = _load_with_bases(path.parent / base_ref, _seen + (path,))
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply ``section.key=value`` overrides; values are parsed as YAML scalars/collections."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def validate_config(cfg: Config) -> None:
    """Fail early on missing sections or an inconsistent label space."""
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"Config is missing sections: {missing}")
    cats = cfg["categories"]
    if len(set(cats)) != len(cats):
        raise ValueError("Duplicate category names in config")
    normal = cfg.get("normal_category", "normal")
    if normal not in cats:
        raise ValueError(f"normal_category {normal!r} is not among categories {cats}")
    frac = cfg["data"]["split_fractions"]
    total = float(frac["train"]) + float(frac["val"]) + float(frac["test"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split_fractions must sum to 1, got {total}")


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML config (resolving ``base:``), apply overrides and validate it."""
    cfg = _load_with_bases(Path(path))
    cfg = apply_overrides(cfg, overrides)
    cfg.setdefault("normal_category", "normal")
    validate_config(cfg)
    return cfg


def dump_config(cfg: Config, path: str | Path) -> None:
    """Write a resolved config to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
