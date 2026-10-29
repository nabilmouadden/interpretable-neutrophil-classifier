"""Joint training of the category and feature heads on a partially fine-tuned encoder.

Outputs written to the run directory:

* ``checkpoint.pt``        best-validation weights (EMA weights when EMA is on) + config
* ``config.yaml``          resolved configuration (incl. the timm encoder architecture)
* ``split.csv``            cell_id -> train/val/test
* ``train_log.csv``        per-epoch losses and validation metrics
* ``predictions_val.npz``  / ``predictions_test.npz``  logits, feature scores, labels, targets
"""

from __future__ import annotations

import contextlib
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from neutroclf.calibration import sigmoid
from neutroclf.config import Config, dump_config
from neutroclf.data import CellDataset, Manifest, eval_transform, read_manifest, train_transform
from neutroclf.ema import EMA
from neutroclf.feature_map import FeatureMap
from neutroclf.losses import JointLoss, positive_weights
from neutroclf.metrics import (
    average_precisions,
    macro_f1,
    predict_labels,
    subset_accuracy,
    weighted_f1,
)
from neutroclf.model import build_model, save_checkpoint
from neutroclf.predict import predict_loader, save_predictions
from neutroclf.splits import SPLIT_NAMES, stratified_split
from neutroclf.utils import get_logger, resolve_device, seed_worker, set_seed, setup_logging


def assign_splits(manifest: Manifest, cfg: Config, seed: int) -> dict[str, np.ndarray]:
    """Train/val/test indices: the manifest ``split`` column if present, else iterative
    stratification with the configured fractions."""
    if manifest.splits is not None:
        arr = np.asarray(manifest.splits)
        return {name: np.flatnonzero(arr == name) for name in SPLIT_NAMES}
    frac = cfg["data"]["split_fractions"]
    split_seed = cfg["data"].get("split_seed")
    split_seed = seed if split_seed is None else int(split_seed)
    return stratified_split(
        manifest.labels, (frac["train"], frac["val"], frac["test"]), seed=split_seed
    )


def _selection_score(name: str, probs: np.ndarray, labels: np.ndarray) -> float:
    pred = predict_labels(probs, 0.5)
    if name == "macro_f1":
        return macro_f1(labels, pred)
    if name == "weighted_f1":
        return weighted_f1(labels, pred)
    if name == "subset_accuracy":
        return subset_accuracy(labels, pred)
    if name == "mean_ap":
        return float(np.nanmean(average_precisions(labels, probs)))
    raise ValueError(f"Unknown selection_metric {name!r}")


def run_training(
    cfg: Config,
    manifest_path: str | Path,
    out_dir: str | Path,
    data_root: str | Path | None = None,
    seed: int | None = None,
) -> Path:
    """Train one model and write the run directory. Returns the run directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "train.log")
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    cfg = dict(cfg, seed=seed)
    set_seed(seed)
    tcfg, dcfg, lcfg = cfg["train"], cfg["data"], cfg["loss"]
    device = resolve_device(tcfg.get("device", "auto"))
    categories = list(cfg["categories"])
    fmap = FeatureMap.from_config(categories, cfg["feature_map"], cfg["normal_category"])
    log.info("run %s | seed %d | device %s", out_dir, seed, device)

    # ------------------------------------------------------------------ data
    man = read_manifest(manifest_path, categories, data_root, cfg["normal_category"])
    splits = assign_splits(man, cfg, seed)
    for name in SPLIT_NAMES:
        if len(splits[name]) == 0:
            raise ValueError(f"The {name} split is empty; provide more cells or a split column")
        counts = man.labels[splits[name]].sum(axis=0)
        log.info(
            "%-5s n=%-5d %s",
            name,
            len(splits[name]),
            " ".join(f"{c}={int(n)}" for c, n in zip(categories, counts, strict=True)),
        )
    split_col = np.empty(len(man.cell_ids), dtype=object)
    for name, idx in splits.items():
        split_col[idx] = name
    pd.DataFrame({"cell_id": man.cell_ids, "split": split_col}).to_csv(
        out_dir / "split.csv", index=False
    )

    size = int(dcfg["image_size"])
    tr_idx = splits["train"]
    train_ds = CellDataset(
        [man.paths[i] for i in tr_idx],
        man.labels[tr_idx],
        train_transform(size, dcfg["augmentation"]),
    )
    gen = torch.Generator().manual_seed(seed)
    workers = int(dcfg.get("num_workers", 0))
    bs = int(tcfg["batch_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=gen,
        pin_memory=device.type == "cuda",
    )

    def eval_loader(name: str) -> DataLoader:
        idx = splits[name]
        ds = CellDataset([man.paths[i] for i in idx], man.labels[idx], eval_transform(size))
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)

    val_loader = eval_loader("val")
    val_labels = man.labels[splits["val"]]

    # ------------------------------------------------------------------ model / loss
    model = build_model(cfg).to(device)
    pos_weight = positive_weights(man.labels[tr_idx], float(lcfg["pos_weight_cap"]))
    log.info("pos_weight %s", np.round(pos_weight.numpy(), 2).tolist())
    criterion = JointLoss(
        fmap,
        pos_weight,
        float(lcfg["focal_gamma"]),
        float(lcfg["focal_alpha"]),
        float(lcfg["lambda_feature"]),
        float(lcfg["lambda_consistency"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.param_groups(
            float(tcfg["lr_encoder"]),
            float(tcfg["lr_category_head"]),
            float(tcfg["lr_feature_head"]),
        ),
        weight_decay=float(tcfg["weight_decay"]),
    )
    epochs = int(tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    ema_decay = float(tcfg.get("ema_decay", 0.0))
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    max_steps = tcfg.get("max_steps_per_epoch")
    selection = str(tcfg.get("selection_metric", "macro_f1"))

    # ------------------------------------------------------------------ loop
    log_path = out_dir / "train_log.csv"
    fields = [
        "epoch",
        "loss",
        "focal",
        "feature",
        "consistency",
        f"val_{selection}",
        "val_feature_mae",
        "lr_encoder",
        "seconds",
    ]
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(fields)
    best_score, best_epoch, best_state = -math.inf, -1, None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        sums: dict[str, float] = defaultdict(float)
        n_batches = 0
        for step, (x, y, _) in enumerate(train_loader):
            if max_steps is not None and step >= int(max_steps):
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
            loss, parts = criterion(out.category_logits, out.feature_logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            for k, v in parts.items():
                sums[k] += v
            n_batches += 1
        scheduler.step()

        ctx = ema.applied(model) if ema is not None else contextlib.nullcontext()
        with ctx:
            val = predict_loader(model, val_loader, device)
            score = _selection_score(selection, sigmoid(val["category_logits"]), val_labels)
            val_mae = float(np.abs(val["feature_scores"] - fmap.targets_numpy(val_labels)).mean())
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = [epoch] + [
            sums[k] / max(n_batches, 1) for k in ("loss", "focal", "feature", "consistency")
        ]
        row += [score, val_mae, optimizer.param_groups[0]["lr"], time.time() - t0]
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
        log.info(
            "epoch %3d | loss %.4f (focal %.4f feat %.4f cons %.4f) | val %s %.4f | feat MAE %.3f | %.0fs",
            epoch,
            row[1],
            row[2],
            row[3],
            row[4],
            selection,
            score,
            val_mae,
            time.time() - t0,
        )

    if best_state is None:
        raise RuntimeError("No epoch was run (train.epochs must be >= 1)")
    log.info("best epoch %d (val %s %.4f)", best_epoch, selection, best_score)
    model.load_state_dict(best_state)
    save_checkpoint(
        out_dir / "checkpoint.pt",
        model,
        cfg,
        extra={"best_epoch": best_epoch, "best_val_score": best_score},
    )
    resolved = dict(cfg)
    resolved["model"] = dict(cfg["model"], encoder_architecture=model.encoder.architecture)
    dump_config(resolved, out_dir / "config.yaml")

    for name in ("val", "test"):
        idx = splits[name]
        pred = predict_loader(model, eval_loader(name), device)
        save_predictions(
            out_dir / f"predictions_{name}.npz",
            [man.cell_ids[i] for i in idx],
            pred["category_logits"],
            pred["feature_scores"],
            man.labels[idx],
            fmap.targets_numpy(man.labels[idx]),
        )
    log.info("wrote %s", out_dir)
    return out_dir


def train_command(args) -> None:
    """CLI entry: ``neutroclf train``."""
    from neutroclf.config import load_config

    cfg = load_config(args.config, args.set or [])
    run_training(cfg, args.manifest, args.out, data_root=args.data_root, seed=args.seed)
    get_logger().info("done")
