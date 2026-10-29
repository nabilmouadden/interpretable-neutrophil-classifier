#!/usr/bin/env python
"""Generate a tiny toy dataset in the manifest layout, used by the end-to-end test.

The "cells" are cartoons: a round cell with a purple lobed nucleus, cytoplasmic
granules and, for Doehle bodies, a pale-blue peripheral inclusion. Lobe count,
granule density, nuclear darkness and the inclusion follow the cell's categories,
so the quick-start model has something learnable. The images are NOT real
neutrophils and any metric obtained on them is meaningless.

Writes, under --out:

    images/cell_0000.png ...           multi-label training cells
    manifest.csv                       image_path + 7 category columns
    external/<Label>/ext_0000.png ...  single-label "external" cells (other stain)
    external_manifest.csv              image_path,label
    reader_study_toy.csv         filled reader-study sheet (toy grades)

Usage:
    python tests/toy_data.py --out /tmp/toy --n 400 --seed 0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

CATEGORIES = [
    "normal",
    "hypogranulation",
    "hypergranulation",
    "hyposegmentation",
    "hypersegmentation",
    "chromatin_hypercondensation",
    "dohle_bodies",
]
FEATURES = ["nuclear_segmentation", "chromatin_hypercondensation", "granularity", "dohle_bodies"]
ABNORMAL = CATEGORIES[1:]
# pairs that cannot co-occur (opposite ends of the same feature)
EXCLUSIVE = {
    frozenset({"hypogranulation", "hypergranulation"}),
    frozenset({"hyposegmentation", "hypersegmentation"}),
}


def sample_labels(rng: np.random.Generator) -> set[str]:
    if rng.random() < 0.4:
        return {"normal"}
    first = str(rng.choice(ABNORMAL))
    labels = {first}
    if rng.random() < 0.25:
        options = [c for c in ABNORMAL if c != first and frozenset({c, first}) not in EXCLUSIVE]
        labels.add(str(rng.choice(options)))
    return labels


def draw_cell(
    labels: set[str], size: int, rng: np.random.Generator, tint: tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    s = 4 * size  # draw at 4x and downsample (anti-aliasing)
    bg = np.clip(np.array([236, 214, 218]) + np.array(tint), 0, 255)
    img = Image.new("RGB", (s, s), tuple(int(v) for v in bg))
    d = ImageDraw.Draw(img)
    cx, cy = s / 2 + rng.normal(0, s * 0.02), s / 2 + rng.normal(0, s * 0.02)
    r = s * rng.uniform(0.34, 0.40)
    cyto = np.clip(np.array([214, 184, 206]) + np.array(tint), 0, 255)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(v) for v in cyto))

    # granules
    n_gran = 12 if "hypogranulation" in labels else 160 if "hypergranulation" in labels else 60
    for _ in range(n_gran):
        a, rr = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.uniform(0, 0.92))
        gx, gy = cx + rr * math.cos(a), cy + rr * math.sin(a)
        g = rng.uniform(3.0, 4.5) * s / 256
        d.ellipse([gx - g, gy - g, gx + g, gy + g], fill=(150, 90, 150))

    # Doehle body: pale-blue peripheral inclusion
    if "dohle_bodies" in labels:
        a = rng.uniform(0, 2 * math.pi)
        bx, by = cx + 0.78 * r * math.cos(a), cy + 0.78 * r * math.sin(a)
        br = r * 0.16
        d.ellipse([bx - br, by - br * 0.7, bx + br, by + br * 0.7], fill=(120, 150, 215))

    # nucleus: lobes on an arc joined by thin bridges
    n_lobes = 1 if "hyposegmentation" in labels else 6 if "hypersegmentation" in labels else 3
    dark = "chromatin_hypercondensation" in labels
    colour = (55, 15, 75) if dark else (120, 60, 150)
    base = rng.uniform(0, 2 * math.pi)
    lobe_r = r * (0.42 if n_lobes == 1 else 0.24 if n_lobes == 3 else 0.17)
    span = 0.0 if n_lobes == 1 else 1.6 * math.pi if n_lobes == 6 else 1.1 * math.pi
    arc_r = 0.0 if n_lobes == 1 else r * 0.45
    centres = []
    for i in range(n_lobes):
        t = base + (span * i / max(n_lobes - 1, 1))
        centres.append((cx + arc_r * math.cos(t), cy + arc_r * math.sin(t)))
    for (x0, y0), (x1, y1) in zip(centres, centres[1:], strict=False):
        d.line([x0, y0, x1, y1], fill=colour, width=max(2, int(s / 90)))
    for x, y in centres:
        d.ellipse([x - lobe_r, y - lobe_r * 0.85, x + lobe_r, y + lobe_r * 0.85], fill=colour)

    img = img.filter(ImageFilter.GaussianBlur(radius=s / 220))
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 4, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def feature_profile(labels: set[str]) -> dict[str, float]:
    return {
        "nuclear_segmentation": 0.0
        if "hyposegmentation" in labels
        else 1.0
        if "hypersegmentation" in labels
        else 0.5,
        "chromatin_hypercondensation": 1.0 if "chromatin_hypercondensation" in labels else 0.0,
        "granularity": 0.0
        if "hypogranulation" in labels
        else 1.0
        if "hypergranulation" in labels
        else 0.5,
        "dohle_bodies": 1.0 if "dohle_bodies" in labels else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=400, help="number of multi-label cells")
    ap.add_argument("--n-external", type=int, default=40, help="number of external cells")
    ap.add_argument("--size", type=int, default=64, help="image side in pixels")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows, label_sets = [], []
    for i in range(args.n):
        labels = sample_labels(rng)
        label_sets.append(labels)
        rel = f"images/cell_{i:04d}.png"
        draw_cell(labels, args.size, rng).save(out / rel)
        rows.append({"image_path": rel} | {c: int(c in labels) for c in CATEGORIES})
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)

    # "external" single-label cells with a stain tint; "Other" is an unmapped class
    ext_rows = []
    ext_labels = {
        "Normal": {"normal"},
        "Hypogranulated": {"hypogranulation"},
        "Dohle bodies": {"dohle_bodies"},
        "Other": {"normal"},
    }
    names = list(ext_labels)
    for i in range(args.n_external):
        name = (
            names[i % len(names)]
            if i < len(names)
            else str(rng.choice(names, p=[0.55, 0.2, 0.15, 0.1]))
        )
        folder = out / "external" / name
        folder.mkdir(parents=True, exist_ok=True)
        rel = f"external/{name}/ext_{i:04d}.png"
        draw_cell(ext_labels[name], args.size, rng, tint=(-12, 6, 10)).save(out / rel)
        ext_rows.append({"image_path": rel, "label": name})
    pd.DataFrame(ext_rows).to_csv(out / "external_manifest.csv", index=False)

    # toy reader-study sheet (grades 0/1/2 on the feature scale, with noise)
    rs = []
    for i in range(min(30, args.n)):
        prof = feature_profile(label_sets[i])
        row = {"cell_id": f"cell_{i:04d}", "categories": ";".join(sorted(label_sets[i]))}
        for f in FEATURES:
            row[f"model_{f}"] = float(np.clip(prof[f] + rng.normal(0, 0.15), 0, 1))
        for rater in (1, 2):
            for f in FEATURES:
                row[f"rater{rater}_{f}"] = int(
                    np.clip(round(2 * prof[f] + rng.normal(0, 0.5)), 0, 2)
                )
        row["notes"] = "toy"
        rs.append(row)
    pd.DataFrame(rs).to_csv(out / "reader_study_toy.csv", index=False)

    counts = pd.DataFrame(rows)[CATEGORIES].sum().to_dict()
    print(f"wrote {args.n} cells to {out} | positives per category: {counts}")
    print(f"wrote {len(ext_rows)} external cells and a toy reader-study sheet")


if __name__ == "__main__":
    main()
