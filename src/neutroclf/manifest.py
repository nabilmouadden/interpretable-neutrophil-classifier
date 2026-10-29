"""Build manifests from a folder-per-class image tree or from a legacy wide CSV.

* ``--folders ROOT``: every image under ``ROOT/<class>/`` becomes one row; the class folder
  name is mapped to a category through :data:`FOLDER_ALIASES` (or ``--alias``). The
  result is single-label; multi-label cells need a wide CSV.
* ``--wide-csv FILE``: a CSV with a filename column and one 0/1 column per class (e.g.
  ``filename,path,Normal,Chromatin,...``); column names are mapped
  through the same aliases and files are located by basename under ``--data-root``.
* ``--external``: write ``image_path,label`` (folder name kept verbatim) for
  ``neutroclf external-eval``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from neutroclf.constants import DEFAULT_CATEGORIES
from neutroclf.data import IMAGE_EXTENSIONS
from neutroclf.utils import get_logger, setup_logging

# common folder / column names -> category names of this package
FOLDER_ALIASES: dict[str, str] = {
    "normal": "normal",
    "hypogranulation": "hypogranulation",
    "hypergranulation": "hypergranulation",
    "hyposegmentation": "hyposegmentation",
    "hypersegmentation": "hypersegmentation",
    "chromatin": "chromatin_hypercondensation",
    "chromatin_hypercondensation": "chromatin_hypercondensation",
    "dohle": "dohle_bodies",
    "döhle": "dohle_bodies",
    "dohle_bodies": "dohle_bodies",
}


def parse_aliases(items: Sequence[str] | None) -> dict[str, str]:
    aliases = dict(FOLDER_ALIASES)
    for item in items or []:
        src, dst = item.split("=", 1)
        aliases[src.strip().casefold()] = dst.strip()
    return aliases


def _images_under(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def from_folders(
    root: str | Path,
    categories: Sequence[str],
    aliases: dict[str, str],
    external: bool = False,
) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        images = _images_under(folder)
        if external:
            rows += [{"image_path": str(p.relative_to(root)), "label": folder.name} for p in images]
            continue
        cat = aliases.get(folder.name.casefold())
        if cat is None or cat not in categories:
            raise ValueError(
                f"Folder {folder.name!r} has no category alias; pass --alias {folder.name}=<category>"
            )
        for p in images:
            row = {"image_path": str(p.relative_to(root))} | {c: int(c == cat) for c in categories}
            rows.append(row)
    if not rows:
        raise ValueError(f"No images found under {root}")
    return pd.DataFrame(rows)


def from_wide_csv(
    csv_path: str | Path,
    data_root: str | Path,
    categories: Sequence[str],
    aliases: dict[str, str],
    filename_column: str = "filename",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    data_root = Path(data_root)
    index = {p.name: p for p in _images_under(data_root)}
    col_map = {c: aliases[c.casefold()] for c in df.columns if c.casefold() in aliases}
    missing = [c for c in categories if c not in col_map.values()]
    if missing:
        raise ValueError(f"CSV has no column for categories {missing}")
    rows, not_found = [], 0
    for _, r in df.iterrows():
        p = index.get(str(r[filename_column]))
        if p is None:
            not_found += 1
            continue
        row = {"image_path": str(p.relative_to(data_root))}
        for src, dst in col_map.items():
            row[dst] = max(row.get(dst, 0), int(round(float(r[src]))))
        rows.append(row)
    if not_found:
        raise FileNotFoundError(f"{not_found} file(s) of {csv_path} not found under {data_root}")
    return pd.DataFrame(rows)[["image_path", *categories]]


def build_manifest_command(args) -> None:
    """CLI entry: ``neutroclf make-manifest``."""
    setup_logging()
    from neutroclf.config import load_config

    categories = load_config(args.config)["categories"] if args.config else list(DEFAULT_CATEGORIES)
    aliases = parse_aliases(args.alias)
    if args.folders:
        df = from_folders(args.folders, categories, aliases, external=args.external)
    elif args.wide_csv:
        if not args.data_root:
            raise SystemExit("--wide-csv requires --data-root")
        df = from_wide_csv(args.wide_csv, args.data_root, categories, aliases, args.filename_column)
    else:
        raise SystemExit("Pass --folders or --wide-csv")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    get_logger().info("wrote %d rows -> %s", len(df), out)
