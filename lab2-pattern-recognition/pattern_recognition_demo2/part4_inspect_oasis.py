"""COMP3710 Lab 2 Part 4: inspect the OASIS files before training.

This script is deliberately read-only.  It helps us confirm the exact file
layout, array shapes and value ranges on Rangpur instead of guessing them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SUPPORTED_SUFFIXES = (".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def describe_numpy_file(path: Path) -> str:
    """Return a short description without modifying the dataset."""
    try:
        loaded = np.load(path, mmap_mode="r", allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            descriptions = []
            for key in loaded.files:
                array = loaded[key]
                descriptions.append(
                    f"{key}: shape={array.shape}, dtype={array.dtype}, "
                    f"range=[{array.min():.3g}, {array.max():.3g}]"
                )
            loaded.close()
            return "; ".join(descriptions)
        return (
            f"shape={loaded.shape}, dtype={loaded.dtype}, "
            f"range=[{loaded.min():.3g}, {loaded.max():.3g}]"
        )
    except Exception as error:  # inspection should continue after one unusual file
        return f"could not inspect ({type(error).__name__}: {error})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect preprocessed OASIS data")
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("/home/groups/comp3710"))
    parser.add_argument("--limit", type=int, default=40,
                        help="maximum number of matching files to print")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"OASIS directory does not exist: {args.data_dir}")

    print(f"OASIS inspection root: {args.data_dir.resolve()}")
    children = sorted(args.data_dir.iterdir())
    print("Top-level entries:")
    for child in children[:80]:
        print("  [DIR] " if child.is_dir() else "  [FILE]", child.name)

    files = sorted(
        path for path in args.data_dir.rglob("*")
        if path.is_file() and any(str(path).lower().endswith(s) for s in SUPPORTED_SUFFIXES)
    )
    print(f"\nSupported image/array files found: {len(files)}")
    for path in files[: args.limit]:
        relative = path.relative_to(args.data_dir)
        if path.suffix.lower() in {".npy", ".npz"}:
            print(f"  {relative} -> {describe_numpy_file(path)}")
        else:
            print(f"  {relative}")

    if not files:
        print("\nNo supported files found. The dataset may use another format such as NIfTI or pickle.")


if __name__ == "__main__":
    main()
