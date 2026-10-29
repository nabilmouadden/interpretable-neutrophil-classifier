"""Manifest reading, image datasets and train/eval transforms (incl. stain augmentation).

Multi-label manifest (CSV, one row per cell)::

    image_path,normal,hypogranulation,hypergranulation,hyposegmentation,hypersegmentation,chromatin_hypercondensation,dohle_bodies
    images/cell_0001.png,1,0,0,0,0,0,0
    images/cell_0002.png,0,1,0,1,0,0,0

Optional columns: ``cell_id`` (defaults to ``image_path``) and ``split`` (``train``/
``val``/``test``; overrides the automatic stratified split). ``image_path`` is resolved
relative to ``--data-root`` (default: the manifest's directory) unless it is absolute.

External single-label manifest: columns ``image_path`` and ``label`` (+ optional
``cell_id``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ----------------------------------------------------------------------------- manifests
@dataclass
class Manifest:
    """Parsed multi-label manifest."""

    cell_ids: list[str]
    paths: list[Path]
    labels: np.ndarray  # (N, C) int64 multi-hot, columns in config category order
    splits: list[str] | None  # per-row split from the manifest, if provided


def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else root / p


def _check_exist(paths: Sequence[Path]) -> None:
    missing = [p for p in paths if not p.is_file()]
    if missing:
        preview = ", ".join(str(p) for p in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} image(s) listed in the manifest not found: {preview}"
        )


def read_manifest(
    manifest: str | Path,
    categories: Sequence[str],
    data_root: str | Path | None = None,
    normal_category: str = "normal",
    check_files: bool = True,
) -> Manifest:
    """Read and validate a multi-label manifest."""
    manifest = Path(manifest)
    root = Path(data_root) if data_root is not None else manifest.parent
    df = pd.read_csv(manifest, dtype={"image_path": str})
    missing_cols = [c for c in ["image_path", *categories] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest {manifest} lacks columns {missing_cols}")
    labels = df[list(categories)].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Category columns must contain only 0/1")
    labels = labels.astype(np.int64)
    empty = np.flatnonzero(labels.sum(axis=1) == 0)
    if len(empty):
        raise ValueError(f"{len(empty)} manifest row(s) carry no category, e.g. row {empty[0]}")
    n_idx = list(categories).index(normal_category)
    conflict = np.flatnonzero((labels[:, n_idx] == 1) & (labels.sum(axis=1) > 1))
    if len(conflict):
        raise ValueError(
            f"{len(conflict)} row(s) are labelled normal together with an abnormality "
            f"(normal is mutually exclusive), e.g. row {conflict[0]}"
        )
    paths = [_resolve(p, root) for p in df["image_path"].astype(str)]
    if check_files:
        _check_exist(paths)
    ids = df["cell_id"].astype(str).tolist() if "cell_id" in df else df["image_path"].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("cell_id / image_path values must be unique")
    splits = None
    if "split" in df.columns:
        splits = df["split"].astype(str).str.strip().str.lower().tolist()
        bad = sorted(set(splits) - {"train", "val", "test"})
        if bad:
            raise ValueError(f"Unknown split values in manifest: {bad}")
    return Manifest(ids, paths, labels, splits)


def read_external_manifest(
    manifest: str | Path, data_root: str | Path | None = None, check_files: bool = True
) -> tuple[list[str], list[Path], list[str]]:
    """Read a single-label manifest (``image_path``, ``label``)."""
    manifest = Path(manifest)
    root = Path(data_root) if data_root is not None else manifest.parent
    df = pd.read_csv(manifest, dtype={"image_path": str, "label": str})
    for col in ("image_path", "label"):
        if col not in df.columns:
            raise ValueError(f"External manifest {manifest} lacks column {col!r}")
    paths = [_resolve(p, root) for p in df["image_path"].astype(str)]
    if check_files:
        _check_exist(paths)
    ids = df["cell_id"].astype(str).tolist() if "cell_id" in df else df["image_path"].tolist()
    return ids, paths, df["label"].astype(str).tolist()


def collect_images(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories (recursively) into a sorted list of image paths."""
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in IMAGE_EXTENSIONS))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


# ----------------------------------------------------------------------------- transforms
class RandomGamma:
    """Gamma change with exponent drawn uniformly from ``[low, high]`` (exposure differences)."""

    def __init__(self, low: float = 0.6, high: float = 1.6) -> None:
        self.low, self.high = low, high

    def __call__(self, img: Image.Image) -> Image.Image:
        return TF.adjust_gamma(img, random.uniform(self.low, self.high))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(low={self.low}, high={self.high})"


def _resize_size(size: int) -> int:
    return int(round(size * 256 / 224))


def _geometric(size: int) -> list:
    return [
        transforms.Resize(
            (_resize_size(size), _resize_size(size)),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
    ]


def _normalise() -> list:
    return [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def eval_transform(size: int) -> transforms.Compose:
    """Deterministic resize + ImageNet normalisation (DinoBloom pre-processing)."""
    return transforms.Compose(
        [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)]
        + _normalise()
    )


def stain_augmentation(size: int) -> list:
    """Stain and colour augmentation.

    Hue +-0.25, saturation +-0.5, brightness +-0.4; Gaussian blur of 0.5-2.5 px; gamma
    0.6-1.6; ~10 % greyscale; small elastic distortion.
    """
    kernel = min(17, max(3, (size // 2) * 2 - 1))
    kernel = kernel if kernel % 2 == 1 else kernel - 1
    return [
        transforms.ColorJitter(brightness=0.4, saturation=0.5, hue=0.25),
        transforms.GaussianBlur(kernel_size=kernel, sigma=(0.5, 2.5)),
        RandomGamma(0.6, 1.6),
        transforms.RandomGrayscale(p=0.1),
        transforms.ElasticTransform(alpha=15.0 * size / 224, sigma=5.0 * size / 224 + 1.0),
    ]


def train_transform(size: int, mode: str = "standard") -> transforms.Compose:
    """Training transform.

    ``standard``: geometric (random resized crop, flips, rotation) + mild colour jitter,
    occasional blur and random erasing. ``stain``: the same geometric part followed by the
    stain/colour augmentation of :func:`stain_augmentation`. ``none``: evaluation
    transform.
    """
    if mode == "none":
        return eval_transform(size)
    if mode == "standard":
        return transforms.Compose(
            _geometric(size)
            + [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.3),
            ]
            + _normalise()
            + [transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))]
        )
    if mode == "stain":
        return transforms.Compose(_geometric(size) + stain_augmentation(size) + _normalise())
    raise ValueError(f"Unknown augmentation mode {mode!r} (standard | stain | none)")


# ----------------------------------------------------------------------------- dataset
class CellDataset(Dataset):
    """Single-cell images with optional multi-hot labels.

    Items are ``(image_tensor, label_tensor, index)``; the label tensor is empty when the
    dataset has no labels.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        labels: np.ndarray | None,
        transform: transforms.Compose,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        if self.labels is not None and len(self.labels) != len(self.paths):
            raise ValueError("paths and labels have different lengths")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        x = self.transform(img)
        y = torch.zeros(0) if self.labels is None else torch.from_numpy(self.labels[i])
        return x, y, i


def load_display_image(path: str | Path, size: int) -> np.ndarray:
    """RGB image resized like the evaluation transform, as float array in [0, 1] (H, W, 3)."""
    with Image.open(path) as im:
        img = im.convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def grid_shape(num_patches: int) -> tuple[int, int]:
    """Square patch grid for ``num_patches`` tokens."""
    side = int(math.isqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patch tokens into a square grid")
    return side, side
