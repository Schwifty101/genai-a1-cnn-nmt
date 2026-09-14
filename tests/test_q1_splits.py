import numpy as np
import pandas as pd
import pytest

from src.q1_cnn.data_prep import CLASS_NAMES
from src.q1_cnn.splits import (
    assert_no_leakage,
    make_splits,
    split_counts,
    subsample_train,
)


def _manifest(n_per_class=200):
    rows = []
    for idx, cls in enumerate(CLASS_NAMES):
        for i in range(n_per_class):
            rows.append(
                {
                    "path": f"data/q1/{cls}/{i}.png",
                    "label": cls,
                    "label_idx": idx,
                    "md5": f"{cls}-{i}",
                }
            )
    return pd.DataFrame(rows)


def test_split_ratios_are_80_10_10():
    df = make_splits(_manifest(), seed=42)
    counts = df["split"].value_counts(normalize=True)
    assert counts["train"] == pytest.approx(0.8, abs=0.01)
    assert counts["val"] == pytest.approx(0.1, abs=0.01)
    assert counts["test"] == pytest.approx(0.1, abs=0.01)


def test_split_is_stratified_per_class():
    df = make_splits(_manifest(), seed=42)
    for cls in CLASS_NAMES:
        sub = df[df["label"] == cls]["split"].value_counts(normalize=True)
        assert sub["train"] == pytest.approx(0.8, abs=0.02)
        assert sub["val"] == pytest.approx(0.1, abs=0.02)


def test_split_is_reproducible_with_the_same_seed():
    a = make_splits(_manifest(), seed=42)
    b = make_splits(_manifest(), seed=42)
    assert a["split"].tolist() == b["split"].tolist()


def test_different_seed_gives_a_different_assignment():
    a = make_splits(_manifest(), seed=42)
    b = make_splits(_manifest(), seed=7)
    assert a["split"].tolist() != b["split"].tolist()


def test_every_row_is_assigned_exactly_one_split():
    df = make_splits(_manifest(), seed=42)
    assert df["split"].isin(["train", "val", "test"]).all()
    assert len(df) == 600


def test_assert_no_leakage_passes_on_a_clean_split():
    df = make_splits(_manifest(), seed=42)
    assert_no_leakage(df)


def test_assert_no_leakage_raises_when_a_path_straddles_splits():
    df = make_splits(_manifest(), seed=42)
    train_row = df[df["split"] == "train"].iloc[0].copy()
    train_row["split"] = "test"
    broken = pd.concat([df, train_row.to_frame().T], ignore_index=True)
    with pytest.raises(AssertionError):
        assert_no_leakage(broken)


def test_assert_no_leakage_raises_on_duplicate_md5_across_splits():
    df = make_splits(_manifest(), seed=42)
    dup = df[df["split"] == "train"].iloc[0].copy()
    dup["path"] = "data/q1/other/copy.png"
    dup["split"] = "val"
    broken = pd.concat([df, dup.to_frame().T], ignore_index=True)
    with pytest.raises(AssertionError):
        assert_no_leakage(broken)


def test_split_counts_shape_and_totals():
    df = make_splits(_manifest(), seed=42)
    table = split_counts(df)
    assert {"train", "val", "test"}.issubset(table.columns)
    assert table.loc["TOTAL"].sum() == 600


def test_subsample_train_only_shrinks_train():
    df = make_splits(_manifest(), seed=42)
    half = subsample_train(df, 0.5, seed=42)
    assert len(half[half["split"] == "train"]) == pytest.approx(
        0.5 * len(df[df["split"] == "train"]), abs=3
    )
    assert len(half[half["split"] == "val"]) == len(df[df["split"] == "val"])
    assert len(half[half["split"] == "test"]) == len(df[df["split"] == "test"])


def test_subsample_train_stays_stratified():
    df = make_splits(_manifest(), seed=42)
    quarter = subsample_train(df, 0.25, seed=42)
    props = quarter[quarter["split"] == "train"]["label"].value_counts(normalize=True)
    for cls in CLASS_NAMES:
        assert props[cls] == pytest.approx(1 / 3, abs=0.05)


def test_subsample_full_fraction_is_identity():
    df = make_splits(_manifest(), seed=42)
    full = subsample_train(df, 1.0, seed=42)
    assert len(full) == len(df)
