"""Command-line interface: ``neutroclf <command>`` or ``python -m neutroclf <command>``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from neutroclf import __version__


def _cmd_train(a: argparse.Namespace) -> None:
    from neutroclf.train import train_command

    train_command(a)


def _cmd_calibrate(a: argparse.Namespace) -> None:
    from neutroclf.evaluate import calibrate_run

    calibrate_run(a.run)


def _cmd_evaluate(a: argparse.Namespace) -> None:
    from neutroclf.evaluate import evaluate_run

    evaluate_run(a.run, n_bootstrap=a.n_bootstrap, figures=a.figures)


def _cmd_summarize(a: argparse.Namespace) -> None:
    from neutroclf.evaluate import summarize_runs

    summary = summarize_runs(a.runs, a.out)
    if a.out is None:
        print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


def _cmd_infer(a: argparse.Namespace) -> None:
    from neutroclf.infer import run_inference

    run_inference(a.run, a.images, a.out, a.device, a.batch_size)


def _cmd_attention(a: argparse.Namespace) -> None:
    from neutroclf.attention import run_attention

    run_attention(a.run, a.images, a.out, a.device, a.max_images)


def _cmd_external(a: argparse.Namespace) -> None:
    from neutroclf.external import run_external_eval

    run_external_eval(
        a.run, a.manifest, a.label_map, a.data_root, a.out, a.device, a.n_bootstrap, a.batch_size
    )


def _cmd_reader(a: argparse.Namespace) -> None:
    from neutroclf.constants import DEFAULT_CATEGORIES, DEFAULT_FEATURES
    from neutroclf.reader_study import run_reader_study, write_template

    if a.config:
        from neutroclf.config import load_config

        cfg = load_config(a.config)
        categories, features = cfg["categories"], cfg["feature_map"]["features"]
    else:
        categories, features = list(DEFAULT_CATEGORIES), list(DEFAULT_FEATURES)
    if a.write_template:
        write_template(a.write_template, features, a.template_rows)
        return
    if not a.ratings or not a.out:
        raise SystemExit("--ratings and --out are required (or use --write-template)")
    run_reader_study(a.ratings, a.out, features, categories, a.n_bootstrap, a.seed)


def _cmd_manifest(a: argparse.Namespace) -> None:
    from neutroclf.manifest import build_manifest_command

    build_manifest_command(a)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neutroclf",
        description="Interpretable and calibrated classifier for co-occurring neutrophil abnormalities.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("train", help="train one model (one seed) and write a run directory")
    s.add_argument("--config", required=True, help="YAML config (e.g. configs/default.yaml)")
    s.add_argument("--manifest", required=True, help="multi-label manifest CSV")
    s.add_argument(
        "--data-root",
        default=None,
        help="root for relative image paths (default: the manifest's directory)",
    )
    s.add_argument("--out", required=True, help="run directory to create")
    s.add_argument("--seed", type=int, default=None, help="override the config seed")
    s.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="config override, e.g. --set train.epochs=10 (repeatable)",
    )
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("calibrate", help="fit temperatures, cut-offs and conformal quantile on val")
    s.add_argument("--run", required=True)
    s.set_defaults(func=_cmd_calibrate)

    s = sub.add_parser("evaluate", help="test-set metrics with bootstrap CIs")
    s.add_argument("--run", required=True)
    s.add_argument("--n-bootstrap", type=int, default=None, help="override evaluation.n_bootstrap")
    s.add_argument("--figures", action="store_true", help="also write PNG figures")
    s.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser("summarize", help="mean ± SD across evaluated runs (e.g. 5 seeds)")
    s.add_argument("runs", nargs="+")
    s.add_argument("--out", default=None, help="directory for per_run.csv, summary.csv, tables.md")
    s.set_defaults(func=_cmd_summarize)

    s = sub.add_parser("infer", help="predict on images -> JSON")
    s.add_argument("--run", required=True)
    s.add_argument("images", nargs="+", help="image files and/or directories")
    s.add_argument("--out", default=None, help="JSON output path (default: stdout)")
    s.add_argument("--device", default="auto")
    s.add_argument("--batch-size", type=int, default=32)
    s.set_defaults(func=_cmd_infer)

    s = sub.add_parser("attention", help="per-feature attention overlays -> PNG")
    s.add_argument("--run", required=True)
    s.add_argument("images", nargs="+")
    s.add_argument("--out", required=True, help="PNG path")
    s.add_argument("--device", default="auto")
    s.add_argument("--max-images", type=int, default=8)
    s.set_defaults(func=_cmd_attention)

    s = sub.add_parser("external-eval", help="evaluate on an external single-label dataset")
    s.add_argument("--run", required=True)
    s.add_argument("--manifest", required=True, help="CSV with image_path,label")
    s.add_argument(
        "--label-map", required=True, help="YAML label map (configs/external_neunn.yaml)"
    )
    s.add_argument("--data-root", default=None)
    s.add_argument("--out", default=None, help="JSON path (default: <run>/external_metrics.json)")
    s.add_argument("--device", default="auto")
    s.add_argument("--n-bootstrap", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=64)
    s.set_defaults(func=_cmd_external)

    s = sub.add_parser("reader-study", help="score model vs two raters (Spearman, partial, kappa)")
    s.add_argument("--ratings", default=None, help="filled reader-study CSV")
    s.add_argument("--out", default=None, help="output directory")
    s.add_argument("--config", default=None, help="config naming categories/features (optional)")
    s.add_argument("--n-bootstrap", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument(
        "--write-template",
        default=None,
        metavar="CSV",
        help="write an empty template instead of scoring",
    )
    s.add_argument("--template-rows", type=int, default=50)
    s.set_defaults(func=_cmd_reader)

    s = sub.add_parser("make-manifest", help="build a manifest from folders or a wide CSV")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("--folders", default=None, help="folder-per-class image root")
    src.add_argument(
        "--wide-csv", default=None, help="CSV with filename + one 0/1 column per class"
    )
    s.add_argument("--data-root", default=None, help="image root for --wide-csv")
    s.add_argument("--filename-column", default="filename")
    s.add_argument(
        "--alias",
        action="append",
        metavar="NAME=CATEGORY",
        help="extra folder/column alias (repeatable)",
    )
    s.add_argument("--external", action="store_true", help="write image_path,label (folder names)")
    s.add_argument("--config", default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_manifest)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
