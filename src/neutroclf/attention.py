"""Per-feature attention overlays.

For each graded feature, the attention weights of its learned query over the encoder
patch tokens (averaged over attention heads) are reshaped to the patch grid, upsampled
bilinearly to the image size, min-max normalised per map and overlaid on the cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neutroclf.calibration import apply_temperatures
from neutroclf.data import collect_images, grid_shape, load_display_image
from neutroclf.figures import plot_attention_grid
from neutroclf.infer import load_run
from neutroclf.predict import predict_paths
from neutroclf.utils import get_logger, setup_logging


def upsample_attention(attention: np.ndarray, size: int) -> np.ndarray:
    """(N, F, P) patch attention -> (N, F, size, size), each map scaled to [0, 1]."""
    n, n_feat, n_patch = attention.shape
    gh, gw = grid_shape(n_patch)
    t = torch.from_numpy(attention).float().reshape(n * n_feat, 1, gh, gw)
    up = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    up = up.reshape(n, n_feat, size, size).numpy()
    lo = up.min(axis=(2, 3), keepdims=True)
    hi = up.max(axis=(2, 3), keepdims=True)
    return (up - lo) / np.maximum(hi - lo, 1e-12)


def run_attention(
    run_dir: str | Path,
    inputs: Sequence[str | Path],
    out: str | Path,
    device: str = "auto",
    max_images: int = 8,
) -> Path:
    """Write a grid of per-feature attention overlays for up to ``max_images`` images."""
    setup_logging()
    model, cfg, calibration = load_run(run_dir, device)
    paths = collect_images(inputs)[:max_images]
    if not paths:
        raise ValueError("No images found")
    size = int(cfg["data"]["image_size"])
    device_t = next(model.parameters()).device
    pred = predict_paths(model, paths, size, device_t, keep_attention=True)
    maps = upsample_attention(pred["feature_attention"], size)
    probs = apply_temperatures(pred["category_logits"], np.asarray(calibration["temperatures"]))
    categories = list(cfg["categories"])
    titles = [
        f"{Path(p).name}\n{categories[int(np.argmax(pr))].replace('_', ' ')} ({pr.max():.2f})"
        for p, pr in zip(paths, probs, strict=True)
    ]
    images = [load_display_image(p, size) for p in paths]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_attention_grid(
        images, maps, pred["feature_scores"], cfg["feature_map"]["features"], titles, out
    )
    get_logger().info("wrote %s", out)
    return out
