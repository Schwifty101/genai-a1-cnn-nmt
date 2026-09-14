"""Classification metrics shared by both studies."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute a JSON-serializable bundle of classification metrics.

    Returns a dict with keys "accuracy", "per_class", "macro",
    "confusion_matrix", "auc_ovr", "auc_macro".
    """
    n_classes = len(class_names)
    labels = list(range(n_classes))

    accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    macro = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    confusion = cm.astype(int).tolist()

    auc_ovr = {}
    for i, name in enumerate(class_names):
        indicator = (np.asarray(y_true) == i).astype(int)
        if len(np.unique(indicator)) < 2:
            auc_ovr[name] = float("nan")
        else:
            auc_ovr[name] = float(roc_auc_score(indicator, y_prob[:, i]))

    non_nan = [v for v in auc_ovr.values() if not np.isnan(v)]
    auc_macro = float(np.mean(non_nan)) if non_nan else float("nan")

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro": macro,
        "confusion_matrix": confusion,
        "auc_ovr": auc_ovr,
        "auc_macro": auc_macro,
    }


def metrics_to_markdown(metrics: dict, class_names: list[str]) -> str:
    """Render a GitHub-flavored markdown table of the metrics."""
    lines = [
        "| Class | Precision | Recall | F1 | Support | AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name in class_names:
        pc = metrics["per_class"][name]
        auc = metrics["auc_ovr"][name]
        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "nan"
        lines.append(
            f"| {name} | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"{pc['f1']:.4f} | {pc['support']} | {auc_str} |"
        )
    macro = metrics["macro"]
    lines.append(
        f"| **Macro** | {macro['precision']:.4f} | {macro['recall']:.4f} | "
        f"{macro['f1']:.4f} | - | - |"
    )
    lines.append("")
    lines.append(
        f"Accuracy: {metrics['accuracy']:.4f}, Macro AUC: {metrics['auc_macro']:.4f}"
    )
    return "\n".join(lines)
