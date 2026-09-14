import pandas as pd
import pytest

from src.q1_cnn.gridsearch import STAGES, expand_stage, summarize_search
from src.q1_cnn.train import RunConfig


def _cfg(**kw):
    base = dict(
        model="pneumonet", batch_size=32, lr=3e-4, epochs=12, patience=5,
        dropout=0.3, l2_lambda=0.0, l1_lambda=0.0, normalization="imagenet",
        augmentation="light", class_weighting="weighted_loss", image_size=128,
        num_workers=0, seed=42, run_name="x",
    )
    base.update(kw)
    return RunConfig(**base)


def test_total_config_count_is_thirty():
    assert sum(len(expand_stage(s["grid"])) for s in STAGES) == 30


def test_stage_names_are_unique_and_ordered():
    names = [s["name"] for s in STAGES]
    assert len(names) == len(set(names))
    assert names[0].startswith("A_")
    assert names[-1].startswith("D_")


def test_expand_stage_is_a_full_cartesian_product():
    out = expand_stage({"lr": [1, 2, 3], "batch_size": [10, 20]})
    assert len(out) == 6
    assert {"lr": 1, "batch_size": 10} in out
    assert {"lr": 3, "batch_size": 20} in out


def test_expand_stage_is_deterministic():
    grid = {"a": [1, 2], "b": ["x", "y"]}
    assert expand_stage(grid) == expand_stage(grid)


def test_expand_single_key_stage():
    assert expand_stage({"dropout": [0.2, 0.3, 0.5]}) == [
        {"dropout": 0.2}, {"dropout": 0.3}, {"dropout": 0.5}
    ]


def test_every_required_hyperparameter_is_swept():
    swept = {k for s in STAGES for k in s["grid"]}
    required = {
        "batch_size", "lr", "epochs", "dropout", "patience",
        "l1_lambda", "l2_lambda", "normalization", "augmentation",
    }
    assert required <= swept


def test_summarize_search_has_one_row_per_swept_hyperparameter():
    rows = pd.DataFrame(
        [
            {"stage": "C1_dropout", "dropout": 0.2, "val_macro_f1": 0.70},
            {"stage": "C1_dropout", "dropout": 0.3, "val_macro_f1": 0.80},
            {"stage": "C1_dropout", "dropout": 0.5, "val_macro_f1": 0.75},
        ]
    )
    table = summarize_search(rows, _cfg(dropout=0.3))
    assert set(table.columns) == {"hyperparameter", "range", "optimal"}
    assert "dropout" in table["hyperparameter"].tolist()
    row = table[table["hyperparameter"] == "dropout"].iloc[0]
    assert "0.2" in str(row["range"]) and "0.5" in str(row["range"])
    assert str(row["optimal"]) == "0.3"
