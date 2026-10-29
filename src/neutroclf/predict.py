"""Batched forward passes and the on-disk prediction format shared by all commands."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neutroclf.data import CellDataset, eval_transform
from neutroclf.model import NeutrophilClassifier


@torch.no_grad()
def predict_loader(
    model: NeutrophilClassifier,
    loader: DataLoader,
    device: torch.device,
    keep_attention: bool = False,
) -> dict[str, np.ndarray]:
    """Run the model over a loader; returns category logits, feature scores (and attention)."""
    model.eval()
    cat, feat, att, idx = [], [], [], []
    for x, _, i in loader:
        out = model(x.to(device))
        cat.append(out.category_logits.float().cpu())
        feat.append(torch.sigmoid(out.feature_logits.float()).cpu())
        if keep_attention:
            att.append(out.feature_attention.float().cpu())
        idx.append(i)
    res = {
        "category_logits": torch.cat(cat).numpy(),
        "feature_scores": torch.cat(feat).numpy(),
        "index": torch.cat(idx).numpy(),
    }
    if keep_attention:
        res["feature_attention"] = torch.cat(att).numpy()
    return res


def predict_paths(
    model: NeutrophilClassifier,
    paths: Sequence[Path],
    image_size: int,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    keep_attention: bool = False,
) -> dict[str, np.ndarray]:
    """Predict on a list of image files with the evaluation transform."""
    ds = CellDataset(paths, None, eval_transform(image_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return predict_loader(model, loader, device, keep_attention=keep_attention)


def save_predictions(
    path: str | Path,
    cell_ids: Sequence[str],
    category_logits: np.ndarray,
    feature_scores: np.ndarray,
    labels: np.ndarray,
    feature_targets: np.ndarray,
) -> None:
    """Write a split's predictions (``predictions_<split>.npz``)."""
    np.savez_compressed(
        path,
        cell_ids=np.asarray(list(cell_ids), dtype=str),
        category_logits=category_logits.astype(np.float32),
        feature_scores=feature_scores.astype(np.float32),
        labels=labels.astype(np.int64),
        feature_targets=feature_targets.astype(np.float32),
    )


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
