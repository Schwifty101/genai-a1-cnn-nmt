import numpy as np
import pytest

from src.common.metrics import compute_classification_metrics

CLASSES = ["COVID19", "NORMAL", "PNEUMONIA"]


def _perfect():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = y_true.copy()
    y_prob = np.eye(3)[y_true].astype(float)
    return y_true, y_pred, y_prob


def test_perfect_predictions_score_one():
    y_true, y_pred, y_prob = _perfect()
    m = compute_classification_metrics(y_true, y_pred, y_prob, CLASSES)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro"]["f1"] == pytest.approx(1.0)
    assert m["auc_macro"] == pytest.approx(1.0)


def test_confusion_matrix_orientation_is_true_by_predicted():
    # one COVID19 sample misread as PNEUMONIA
    y_true = np.array([0, 1, 2])
    y_pred = np.array([2, 1, 2])
    y_prob = np.eye(3)[y_pred].astype(float)
    m = compute_classification_metrics(y_true, y_pred, y_prob, CLASSES)
    cm = m["confusion_matrix"]
    assert cm[0][2] == 1, "row must be the true class, column the predicted class"
    assert cm[2][0] == 0


def test_all_required_keys_present():
    y_true, y_pred, y_prob = _perfect()
    m = compute_classification_metrics(y_true, y_pred, y_prob, CLASSES)
    assert set(m) == {
        "accuracy",
        "per_class",
        "macro",
        "confusion_matrix",
        "auc_ovr",
        "auc_macro",
    }
    assert set(m["per_class"]) == set(CLASSES)
    assert set(m["per_class"]["NORMAL"]) == {"precision", "recall", "f1", "support"}


def test_macro_f1_is_unweighted_mean_of_per_class_f1():
    y_true = np.array([0, 0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 2])
    y_prob = np.eye(3)[y_pred].astype(float)
    m = compute_classification_metrics(y_true, y_pred, y_prob, CLASSES)
    per_class_f1 = [m["per_class"][c]["f1"] for c in CLASSES]
    assert m["macro"]["f1"] == pytest.approx(sum(per_class_f1) / 3)


def test_support_counts_true_labels():
    y_true = np.array([0, 0, 0, 1, 2, 2])
    y_pred = y_true.copy()
    y_prob = np.eye(3)[y_true].astype(float)
    m = compute_classification_metrics(y_true, y_pred, y_prob, CLASSES)
    assert m["per_class"]["COVID19"]["support"] == 3
    assert m["per_class"]["NORMAL"]["support"] == 1
    assert m["per_class"]["PNEUMONIA"]["support"] == 2


def test_binary_case_two_classes():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])
    m = compute_classification_metrics(y_true, y_pred, y_prob, ["NORMAL", "PNEUMONIA"])
    assert 0.0 <= m["auc_macro"] <= 1.0
    assert len(m["confusion_matrix"]) == 2
