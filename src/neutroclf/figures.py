"""Static matplotlib figures: reliability diagram, referral curve, decision curve, attention."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3 (validated all-pairs)
OVERLAY = LinearSegmentedColormap.from_list("attn", ["#eb683400", "#eb6834ff"])


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)


def _new(figsize=(4.6, 4.0)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    _style(ax)
    return fig, ax


def plot_reliability(bins: Sequence[dict], ece: float, brier: float, path: Path) -> None:
    """Reliability diagram over pooled (cell, category) pairs; marker area grows with bin count."""
    fig, ax = _new()
    ax.plot([0, 1], [0, 1], ls="--", color=TEXT_2, lw=1, label="Perfect calibration")
    pts = [b for b in bins if b["count"]]
    x = [b["mean_pred"] for b in pts]
    y = [b["obs_freq"] for b in pts]
    counts = np.array([b["count"] for b in pts], dtype=float)
    sizes = 20 + 180 * np.sqrt(counts / counts.max()) if len(counts) else []
    ax.plot(x, y, color=SERIES[0], lw=2, label="Model")
    ax.scatter(x, y, s=sizes, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Stated probability", ylabel="Observed frequency")
    ax.set_title(f"Calibration (ECE {ece:.3f}, Brier {brier:.3f})", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_referral(rows: Sequence[dict], path: Path) -> None:
    """Accuracy (all labels correct) on retained cells versus fraction referred for review."""
    fig, ax = _new()
    f = [100 * r["fraction_referred"] for r in rows]
    acc = [r["accuracy_retained"] for r in rows]
    ax.plot(f, acc, color=SERIES[0], lw=2, marker="o", ms=4)
    ax.annotate(
        f"{acc[0]:.2f}",
        (f[0], acc[0]),
        textcoords="offset points",
        xytext=(4, -12),
        fontsize=8,
        color=TEXT,
    )
    ax.annotate(
        f"{acc[-1]:.2f}",
        (f[-1], acc[-1]),
        textcoords="offset points",
        xytext=(-10, 6),
        fontsize=8,
        color=TEXT,
    )
    ax.set(xlabel="Least confident cells referred (%)", ylabel="Accuracy on retained cells")
    ax.set_title("Confidence-based review", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_decision_curve(rows: Sequence[dict], path: Path) -> None:
    """Net benefit of the abnormal-vs-normal call versus reviewing all or no cells."""
    fig, ax = _new()
    pt = [r["threshold"] for r in rows]
    model = np.array([r["net_benefit_model"] for r in rows])
    allc = np.array([r["net_benefit_all"] for r in rows])
    ax.plot(pt, model, color=SERIES[0], lw=2, label="Model")
    ax.plot(pt, allc, color=SERIES[1], lw=2, label="Review all cells")
    ax.plot(pt, np.zeros_like(model), color=SERIES[2], lw=2, label="Review none")
    lo = min(-0.05, float(np.nanmin(model)) - 0.02)
    ax.set(
        ylim=(lo, max(float(np.nanmax(model)), float(np.nanmax(allc))) + 0.05),
        xlabel="Threshold probability",
        ylabel="Net benefit",
    )
    ax.set_title("Decision curve (abnormal vs normal)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_run_figures(metrics: dict, out_dir: Path) -> None:
    """Write the three run-level figures for an ``evaluate`` result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h = metrics["headline"]
    plot_reliability(
        metrics["calibration"]["reliability_bins"],
        h["ece"],
        h["brier"],
        out_dir / "reliability.png",
    )
    plot_referral(metrics["referral"], out_dir / "referral.png")
    plot_decision_curve(metrics["decision_curve"], out_dir / "decision_curve.png")


def plot_attention_grid(
    images: Sequence[np.ndarray],
    maps: np.ndarray,
    scores: np.ndarray,
    features: Sequence[str],
    titles: Sequence[str],
    path: Path,
) -> None:
    """Rows = cells, columns = input + one overlay per graded feature.

    ``maps`` (N, F, H, W) are upsampled attention maps normalised to [0, 1] per map;
    the overlay opacity follows the attention.
    """
    n, n_feat = len(images), len(features)
    fig, axes = plt.subplots(
        n, n_feat + 1, figsize=(2.0 * (n_feat + 1), 2.1 * n), facecolor=SURFACE, squeeze=False
    )
    for i in range(n):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(titles[i], fontsize=7, color=TEXT)
        for j in range(n_feat):
            ax = axes[i, j + 1]
            ax.imshow(images[i])
            ax.imshow(maps[i, j], cmap=OVERLAY, vmin=0, vmax=1, alpha=0.85)
            label = features[j].replace("_", " ")
            ax.set_title(f"{label}\nscore {scores[i, j]:.2f}", fontsize=7, color=TEXT)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
