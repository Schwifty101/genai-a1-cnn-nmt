"""Shared plotting utilities. Headless (Agg backend) for batch training runs."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def plot_training_curves(history: dict, out_path: Path, title: str) -> None:
    """Two side-by-side axes: losses on the left, accuracy/F1 on the right."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))
    best_epoch = history.get("best_epoch")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], label="train_loss", marker="o")
    ax_loss.plot(epochs, history["val_loss"], label="val_loss", marker="o")
    if best_epoch is not None:
        ax_loss.axvline(best_epoch, color="gray", linestyle="--", label="best_epoch")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(epochs, history["train_acc"], label="train_acc", marker="o")
    ax_acc.plot(epochs, history["val_acc"], label="val_acc", marker="o")
    ax_acc.plot(epochs, history["val_macro_f1"], label="val_macro_f1", marker="o")
    if best_epoch is not None:
        ax_acc.axvline(best_epoch, color="gray", linestyle="--", label="best_epoch")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("score")
    ax_acc.set_title("Accuracy / F1")
    ax_acc.legend()
    ax_acc.grid(True)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: list[list[int]],
    class_names: list[str],
    out_path: Path,
    normalize: bool = False,
) -> None:
    """Seaborn heatmap of the confusion matrix."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_arr = cm_arr / row_sums
        fmt = ".2f"
    else:
        fmt = "d"
        cm_arr = cm_arr.astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    """One ROC curve per class (one-vs-rest), plus macro-average and chance line."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(7, 6))

    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)
    n_valid = 0

    for i, name in enumerate(class_names):
        indicator = (y_true == i).astype(int)
        if len(np.unique(indicator)) < 2:
            continue
        fpr, tpr, _ = roc_curve(indicator, y_prob[:, i])
        class_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={class_auc:.3f})")
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        n_valid += 1

    if n_valid > 0:
        mean_tpr /= n_valid
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(
            all_fpr,
            mean_tpr,
            color="black",
            linestyle="--",
            linewidth=2.5,
            label=f"macro-average (AUC={macro_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_image_grid(
    images: list,
    titles: list[str],
    out_path: Path,
    ncols: int = 4,
    suptitle: str | None = None,
) -> None:
    """Lay images out into a grid, one title per cell, axes off."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(images)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i])
            ax.set_title(titles[i] if i < len(titles) else "")
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_length_hist(
    panels: list[dict],
    out_path: Path,
    suptitle: str = "",
) -> None:
    """Multi-panel overlaid histogram figure.

    `panels` is a list of `{"title": str, "series": {label: values}}` dicts,
    one per panel (e.g. one for English token lengths, one for Urdu), each
    overlaying one histogram per series entry (e.g. before/after filtering).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = max(1, len(panels))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        for label, values in panel["series"].items():
            ax.hist(values, bins=30, alpha=0.5, label=label)
        ax.set_xlabel("token length")
        ax.set_ylabel("count")
        ax.set_title(panel.get("title", ""))
        ax.legend()
        ax.grid(True)

    if suptitle:
        fig.suptitle(suptitle)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_xy(
    x: list,
    series: dict[str, list],
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Line plot with a marker per point, one line per series entry."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, ys in series.items():
        ax.plot(x, ys, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
