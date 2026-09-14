"""Train/val/test splitting for the Q2 clean English-Urdu dataset.

The split is produced once from `results/q2_clean.csv` and persisted to
`results/q2_split.csv`. Every downstream Q2 module (tokenizer, dataset/
loaders, training) reads that file — nothing downstream re-splits.

There is no stratification here (unlike Q1): there are no labels to
stratify by, just parallel sentence pairs.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.common.seed import SEED


def make_splits(
    df: pd.DataFrame,
    seed: int = SEED,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """Return `df` with an added `split` column, shuffled by a seeded RNG.

    Shuffles the index with `np.random.default_rng(seed).permutation`, then
    slices at `floor(ratios[0] * n)` and `floor((ratios[0] + ratios[1]) * n)`
    so the resulting counts are exact (no row is lost or duplicated).
    """
    n = len(df)
    perm = np.random.default_rng(seed).permutation(n)

    n_train = int(np.floor(ratios[0] * n))
    n_trainval = int(np.floor((ratios[0] + ratios[1]) * n))

    split = np.empty(n, dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train:n_trainval]] = "val"
    split[perm[n_trainval:]] = "test"

    out = df.reset_index(drop=True).copy()
    out["split"] = split
    return out


def _pair_hash(en: str, ur: str) -> str:
    return hashlib.sha256(f"{en}\x00{ur}".encode("utf-8")).hexdigest()


def assert_no_overlap(df: pd.DataFrame) -> None:
    """Raise `AssertionError` if the same `(en, ur)` pair hash appears in two splits."""
    hashes = [
        _pair_hash(str(en), str(ur)) for en, ur in zip(df["en"], df["ur"])
    ]
    tmp = pd.DataFrame({"hash": hashes, "split": df["split"].to_numpy()})
    per_hash_splits = tmp.groupby("hash")["split"].nunique()
    bad = per_hash_splits[per_hash_splits > 1]
    assert len(bad) == 0, (
        f"{len(bad)} (en, ur) pair(s) appear in more than one split"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build the Q2 train/val/test split.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    data_cfg = cfg.get("data", {})

    clean_csv = Path(data_cfg.get("clean_csv", "results/q2_clean.csv"))
    split_csv = Path(data_cfg.get("split_csv", "results/q2_split.csv"))
    seed = int(cfg.get("seed", SEED))

    df = pd.read_csv(clean_csv, dtype={"en": str, "ur": str})
    split_df = make_splits(df, seed=seed)
    assert_no_overlap(split_df)

    split_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_csv, index=False)

    counts = split_df["split"].value_counts()
    print(counts.to_string())
    print("no overlap: OK")


if __name__ == "__main__":
    _main()
