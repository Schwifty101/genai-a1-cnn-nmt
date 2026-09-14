import pandas as pd
import pytest

from src.q2_rnn.splits import assert_no_overlap, make_splits


def _pairs(n=1000):
    return pd.DataFrame({"en": [f"sentence {i}" for i in range(n)],
                         "ur": [f"جملہ {i}" for i in range(n)]})


def test_ratios_are_80_10_10():
    df = make_splits(_pairs(), seed=42)
    counts = df["split"].value_counts()
    assert counts["train"] == 800
    assert counts["val"] == 100
    assert counts["test"] == 100


def test_split_is_reproducible():
    a = make_splits(_pairs(), seed=42)["split"].tolist()
    b = make_splits(_pairs(), seed=42)["split"].tolist()
    assert a == b


def test_different_seed_changes_assignment():
    a = make_splits(_pairs(), seed=42)["split"].tolist()
    b = make_splits(_pairs(), seed=1)["split"].tolist()
    assert a != b


def test_no_row_is_lost():
    df = make_splits(_pairs(777), seed=42)
    assert len(df) == 777
    assert df["split"].isin(["train", "val", "test"]).all()


def test_assert_no_overlap_passes_on_clean_split():
    assert_no_overlap(make_splits(_pairs(), seed=42))


def test_assert_no_overlap_raises_on_a_duplicated_pair():
    df = make_splits(_pairs(), seed=42)
    dup = df[df["split"] == "train"].iloc[0].copy()
    dup["split"] = "test"
    broken = pd.concat([df, dup.to_frame().T], ignore_index=True)
    with pytest.raises(AssertionError):
        assert_no_overlap(broken)
