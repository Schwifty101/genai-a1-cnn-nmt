"""Stratified train/val/test splitting for the Q1 chest X-ray manifest.

The split is produced once from `results/q1_manifest.csv` and persisted to
`results/q1_split_manifest.csv`. Every downstream Q1 module (augmentation
figure, datasets/loaders, training, evaluation) reads that file — nothing
downstream re-splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.seed import SEED
from src.q1_cnn.data_prep import CLASS_NAMES


def make_splits(
    manifest: pd.DataFrame,
    seed: int = SEED,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """Return `manifest` with an added `split` column, stratified by `label`.

    Peels off `1 - ratios[0]` as a holdout via `train_test_split`, then
    splits that holdout in half (again stratified) into val and test.
    Assignment is done by index so the returned frame preserves the input
    row order.
    """
    train_frac, val_frac, test_frac = ratios
    holdout_frac = val_frac + test_frac

    idx = manifest.index
    train_idx, holdout_idx = train_test_split(
        idx,
        test_size=holdout_frac,
        stratify=manifest["label"],
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=test_frac / holdout_frac,
        stratify=manifest.loc[holdout_idx, "label"],
        random_state=seed,
    )

    out = manifest.copy()
    out.loc[train_idx, "split"] = "train"
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise `AssertionError` if any `path` or `md5` spans more than one split."""
    if len(df) == 0:
        return

    path_counts = df.groupby("path")["split"].nunique()
    assert path_counts.max() == 1, (
        f"paths present in more than one split: "
        f"{path_counts[path_counts > 1].index.tolist()}"
    )

    if "md5" in df.columns:
        md5_counts = df.groupby("md5")["split"].nunique()
        assert md5_counts.max() == 1, (
            f"md5 values present in more than one split: "
            f"{md5_counts[md5_counts > 1].index.tolist()}"
        )


def split_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class x per-split count table (columns train/val/test) with a TOTAL row."""
    table = pd.crosstab(df["label"], df["split"]).reindex(CLASS_NAMES)
    table = table.fillna(0).astype(int)
    table.loc["TOTAL"] = table.sum(axis=0)
    return table


def subsample_train(df: pd.DataFrame, fraction: float, seed: int = SEED) -> pd.DataFrame:
    """Stratified subsample of the training split only; val/test pass through untouched.

    Iterates `train.groupby("label")` directly (rather than `.groupby().apply()`,
    which — as of pandas 3.0's `include_groups=False` default — drops the
    grouping column from the frame handed to the callable and reintroduces it
    as NaN on concat) so every sampled row keeps its `label`.
    """
    train = df[df["split"] == "train"]
    rest = df[df["split"] != "train"]

    sampled_parts = [
        group.sample(frac=fraction, random_state=seed)
        for _, group in train.groupby("label", sort=False)
    ]
    sampled = pd.concat(sampled_parts) if sampled_parts else train

    return pd.concat([sampled, rest]).sort_index()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build the Q1 stratified train/val/test split.")
    parser.add_argument("--manifest", type=Path, default=Path("results/q1_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/q1_split_manifest.csv"))
    parser.add_argument("--counts", type=Path, default=Path("results/q1_split_counts.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    split_df = make_splits(manifest, seed=args.seed)

    assert_no_leakage(split_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.out, index=False)

    counts = split_counts(split_df)
    args.counts.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.counts)

    print(counts.to_string())
    print("no leakage: OK")


if __name__ == "__main__":
    _main()
