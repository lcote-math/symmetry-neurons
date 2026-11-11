"""
Builds the datasets used by main.py. Each dataset has a name, a builder function and the csv path it produces.

Usage (from the repository root):
    python -m src.make_datasets                      # list datasets and which sizes are built
    python -m src.make_datasets gsm8k race           # build some, 20 samples each (skips ones that exist)
    python -m src.make_datasets all --n 100          # build everything with 100 samples
    python -m src.make_datasets regular_c4 --n 50 --seed 0

Each (n, seed) is stored in its own folder, data/<name>_n<n>_seed<seed>/, so different sizes don't overwrite each other.

To add a dataset: write a function `builder(out_path, n, seed)` that writes a csv to out_path,
then add one line to DATASETS below.
"""

import argparse
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable

import pandas as pd

from src.polygons import make_polygon_dataset


def _sample_and_save(df, out_path, n, seed):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.sample(n, random_state=seed).reset_index(drop=True).to_csv(out_path, index=False)
    print(f"Saved {n} samples to {out_path}")
    return out_path


def get_gsm8k(out_path, n, seed):
    url = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet"
    return _sample_and_save(pd.read_parquet(url), out_path, n, seed)


def get_race(out_path, n, seed):
    url = "https://huggingface.co/datasets/race/resolve/refs%2Fconvert%2Fparquet/high/train/0000.parquet"
    df = pd.read_parquet(url)
    # Fold the multiple-choice options into the question so the csv is self-contained.
    df["question"] = [
        q + "\n" + "\n".join(f"{letter}. {opt}" for letter, opt in zip("ABCD", opts))
        for q, opts in zip(df["question"], df["options"])
    ]
    df = df[["article", "question", "answer"]]
    return _sample_and_save(df, out_path, n, seed)


def get_linear_algebra(out_path, n, seed):
    df = pd.read_csv("data/custom_datasets/LA_2026.csv")
    return _sample_and_save(df, out_path, n, seed)


DEFAULT_N = 20
DEFAULT_SEED = 42


@dataclass
class Dataset:
    builder: Callable
    description: str


DATASETS = {
    "gsm8k": Dataset(get_gsm8k, "grade-school math word problems (Hugging Face: openai/gsm8k)"),
    "race": Dataset(get_race, "reading comprehension (Hugging Face: race, high-school split)"),
    "linear_algebra": Dataset(get_linear_algebra, "true/false linear algebra questions (data/custom_datasets/LA_2026.csv)"),
    "regular_c4": Dataset(partial(make_polygon_dataset, base="c4", n_sides=4),
                          "spot the intruder: three squares and one irregular quadrilateral"),
    "irregular_c4": Dataset(partial(make_polygon_dataset, base="irregular", n_sides=4),
                            "spot the intruder: three copies of an irregular quadrilateral and one different one"),
}


def dataset_path(name, n=DEFAULT_N, seed=DEFAULT_SEED):
    """Where dataset `name` with n samples and this seed is stored. Each (n, seed) gets its own folder."""
    return os.path.join("data", f"{name}_n{n}_seed{seed}", f"{name}.csv")


def build(name, n=DEFAULT_N, seed=DEFAULT_SEED, force=False):
    """Build dataset `name` unless it already exists (or force=True). Returns its csv path."""
    path = dataset_path(name, n, seed)
    if os.path.exists(path) and not force:
        print(f"{name}: already built at {path}, skipping")
        return path
    return DATASETS[name].builder(path, n=n, seed=seed)


def resolve(name_or_path, n=DEFAULT_N, seed=DEFAULT_SEED):
    """
    Turn a dataset name (building it with n samples if needed) or a csv path into (label, csv path).
    The label is used to name output files.
    """
    if name_or_path in DATASETS:
        return name_or_path, build(name_or_path, n, seed)
    if os.path.exists(name_or_path):
        return os.path.splitext(os.path.basename(name_or_path))[0], name_or_path
    raise FileNotFoundError(
        f"{name_or_path!r} is neither a dataset name ({', '.join(DATASETS)}) nor an existing csv file"
    )


def main():
    parser = argparse.ArgumentParser(description="Build the datasets used by main.py.")
    parser.add_argument("names", nargs="*", help=f"datasets to build, or 'all'. Available: {', '.join(DATASETS)}")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help=f"number of samples (default: {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--force", action="store_true", help="rebuild even if the csv already exists")
    args = parser.parse_args()

    if not args.names:
        for name, ds in DATASETS.items():
            prefix = f"{name}_n"
            built = sorted(d for d in (os.listdir("data") if os.path.isdir("data") else [])
                           if d.startswith(prefix) and os.path.exists(os.path.join("data", d, f"{name}.csv")))
            print(f"{name:15} {ds.description}")
            print(f"{'':15} built: {', '.join(built) if built else 'none'}")
        return

    names = list(DATASETS) if args.names == ["all"] else args.names
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(unknown)}. Available: {', '.join(DATASETS)}")

    for name in names:
        build(name, n=args.n, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
