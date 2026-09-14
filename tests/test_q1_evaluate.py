import numpy as np
import pandas as pd
import pytest

from src.q1_cnn.error_analysis import collect_misclassified, error_statistics
from src.q1_cnn.evaluate import build_comparison_table

CLASSES = ["COVID19", "NORMAL", "PNEUMONIA"]


def _result():
    return {
        "y_true": [0, 0, 1, 1, 2, 2],
        "y_pred": [0, 2, 1, 0, 2, 1],
        "y_prob": [
            [0.90, 0.05, 0.05],
            [0.05, 0.05, 0.90],
            [0.10, 0.80, 0.10],
            [0.60, 0.30, 0.10],
            [0.05, 0.15, 0.80],
            [0.10, 0.70, 0.20],
        ],
        "paths": [f"img{i}.png" for i in range(6)],
    }


def test_collect_misclassified_returns_only_errors():
    errors = collect_misclassified(_result(), CLASSES, top_n=10)
    assert len(errors) == 3
    assert (errors["true"] != errors["pred"]).all()


def test_misclassified_are_ranked_by_descending_confidence():
    errors = collect_misclassified(_result(), CLASSES, top_n=10)
    conf = errors["confidence"].tolist()
    assert conf == sorted(conf, reverse=True)
    assert errors.iloc[0]["confidence"] == pytest.approx(0.90)


def test_collect_misclassified_respects_top_n():
    assert len(collect_misclassified(_result(), CLASSES, top_n=2)) == 2


def test_misclassified_labels_are_class_names_not_indices():
    errors = collect_misclassified(_result(), CLASSES, top_n=10)
    assert set(errors["true"]) <= set(CLASSES)
    assert set(errors["pred"]) <= set(CLASSES)


def test_error_statistics_counts_and_pairs():
    stats = error_statistics(_result(), CLASSES)
    assert stats["n_errors"] == 3
    assert stats["by_confusion_pair"]["COVID19->PNEUMONIA"] == 1
    assert stats["by_confusion_pair"]["NORMAL->COVID19"] == 1
    assert stats["by_true_class"]["COVID19"] == 1


def test_error_statistics_confidence_fields():
    stats = error_statistics(_result(), CLASSES)
    assert 0.0 <= stats["mean_error_confidence"] <= 1.0
    assert 0.0 <= stats["mean_correct_confidence"] <= 1.0


def test_no_errors_case_is_handled():
    clean = {
        "y_true": [0, 1, 2],
        "y_pred": [0, 1, 2],
        "y_prob": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "paths": ["a.png", "b.png", "c.png"],
    }
    assert len(collect_misclassified(clean, CLASSES, top_n=5)) == 0
    assert error_statistics(clean, CLASSES)["n_errors"] == 0


def test_build_comparison_table_columns(tmp_path):
    import json

    run_dirs = {}
    for i, name in enumerate(["pneumonet", "vgg16_frozen"]):
        d = tmp_path / name
        d.mkdir()
        (d / "result.json").write_text(
            json.dumps(
                {
                    "config": {"model": name},
                    "params_total": 1000 + i,
                    "params_trainable": 500 + i,
                    "history": {"epoch_time_s": [1.0, 2.0]},
                    "test_metrics": {
                        "accuracy": 0.9 - i * 0.1,
                        "macro": {"f1": 0.88 - i * 0.1},
                        "auc_macro": 0.95 - i * 0.1,
                    },
                }
            )
        )
        run_dirs[name] = d
    table = build_comparison_table(run_dirs)
    assert list(table.columns) == [
        "model",
        "params_total",
        "params_trainable",
        "epoch_time_s",
        "test_accuracy",
        "test_macro_f1",
        "test_macro_auc",
    ]
    assert len(table) == 2
    assert table.iloc[0]["model"] == "pneumonet"
