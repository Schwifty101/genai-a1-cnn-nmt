"""Q1 test-set evaluation, per-model artifacts, and the cross-model comparison table.

This module is the *only* place Q1 test data is read. `run_training`
(`src.q1_cnn.train`) deliberately only ever evaluates on the validation
split, so hyperparameter tuning never sees test data; `evaluate_run` below
is what finally scores a trained checkpoint against the held-out test split
and folds the result back into that run's `result.json` as a `test_metrics`
key, alongside (never replacing) `val_metrics`.

PneumoNet is the project's primary model; the pre-trained networks
(VGG16/ResNet50, frozen and fine-tuned) are comparison baselines only.
`build_comparison_table` preserves the order of its `run_dirs` argument
rather than re-sorting, so a caller that lists `pneumonet` first (as
`src.q1_cnn.models.MODEL_NAMES` does) gets a table with `pneumonet` first —
the primary result is never buried under the baselines.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.common.device import pick_device
from src.common.metrics import compute_classification_metrics
from src.common.plots import plot_confusion_matrix, plot_roc_curves, save_image_grid
from src.common.trainer import evaluate_model
from src.q1_cnn.data_prep import CLASS_NAMES
from src.q1_cnn.datasets import build_loaders
from src.q1_cnn.models import MODEL_NAMES, build_model
from src.q1_cnn.train import RunConfig

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "&": r"\&",
    "{": r"\{",
    "}": r"\}",
    "^": r"\^{}",
    "~": r"\~{}",
}


def evaluate_checkpoint(
    ckpt_path: Path,
    model_name: str,
    split_df: pd.DataFrame,
    cfg: RunConfig,
    device=None,
) -> dict:
    """Load `ckpt_path` into a fresh `model_name` model and score it on the test split.

    Predictions are paired back to file paths by re-deriving the test split
    frame the same way `build_loaders` does internally (filter to
    `split == "test"`, then `reset_index`) and reading its `path` column in
    that order. This is safe because `build_loaders` never shuffles the
    test loader, so batches are yielded in exactly that row order.
    """
    ckpt_path = Path(ckpt_path)
    resolved_device = pick_device() if device is None else device

    _train_loader, _val_loader, test_loader, _class_weights = build_loaders(
        split_df,
        image_size=cfg.image_size,
        normalization=cfg.normalization,
        policy=cfg.augmentation,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        class_weighting=cfg.class_weighting,
        seed=cfg.seed,
        root=Path("."),
    )

    model = build_model(model_name, num_classes=len(CLASS_NAMES), dropout=cfg.dropout)
    state = torch.load(ckpt_path, map_location=resolved_device, weights_only=True)
    model.load_state_dict(state)
    model.to(resolved_device)

    y_true, y_pred, y_prob, _mean_loss = evaluate_model(model, test_loader, resolved_device)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob, CLASS_NAMES)

    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)
    paths = test_df["path"].tolist()

    return {
        "metrics": metrics,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "paths": paths,
    }


def _sample_prediction_grid(
    result: dict, class_names: list[str], out_path: Path, n: int = 12
) -> None:
    """Grid of the first `n` test images (loader order) with true/pred labels."""
    paths = result["paths"]
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    n = min(n, len(paths))
    if n == 0:
        return

    images, titles = [], []
    for i in range(n):
        with Image.open(Path(".") / paths[i]) as im:
            images.append(np.array(im.convert("RGB")))
        true_name = class_names[int(y_true[i])]
        pred_name = class_names[int(y_pred[i])]
        mark = "correct" if y_true[i] == y_pred[i] else "WRONG"
        titles.append(f"{mark}: true={true_name} pred={pred_name}")

    save_image_grid(images, titles, out_path, ncols=4, suptitle="Sample test predictions")


def write_evaluation_artifacts(
    result: dict, class_names: list[str], out_dir: Path, tag: str
) -> None:
    """Write the confusion matrix (raw + normalized), ROC curves, metrics JSON,
    and a sample-prediction grid for one `evaluate_checkpoint` result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics"]
    cm = metrics["confusion_matrix"]

    plot_confusion_matrix(
        cm, class_names, out_dir / f"{tag}_confusion_matrix.png", normalize=False
    )
    plot_confusion_matrix(
        cm, class_names, out_dir / f"{tag}_confusion_matrix_norm.png", normalize=True
    )

    y_true = np.array(result["y_true"])
    y_prob = np.array(result["y_prob"])
    plot_roc_curves(y_true, y_prob, class_names, out_dir / f"{tag}_roc.png")

    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))

    _sample_prediction_grid(result, class_names, out_dir / f"{tag}_sample_predictions.png")


def evaluate_run(run_dir: Path, split_df: pd.DataFrame, device=None) -> dict:
    """Evaluate one training run's checkpoint on the test split.

    Reads `run_dir/result.json` for the run's `config` and `ckpt_path`,
    evaluates that checkpoint on the test split, and rewrites
    `run_dir/result.json` in place with a `test_metrics` key added (the
    existing `val_metrics` key, and everything else already in the file,
    is left untouched). Returns the full `evaluate_checkpoint` result
    (including per-example `y_true`/`y_pred`/`y_prob`/`paths`), which is
    NOT persisted to `result.json` — only the aggregated `metrics` dict is.
    """
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    stored = json.loads(result_path.read_text())

    cfg = RunConfig(**stored["config"])
    ckpt_path = Path(stored["ckpt_path"])

    eval_result = evaluate_checkpoint(ckpt_path, cfg.model, split_df, cfg, device=device)

    stored["test_metrics"] = eval_result["metrics"]
    result_path.write_text(json.dumps(stored, indent=2))

    return eval_result


def build_comparison_table(run_dirs: dict[str, Path]) -> pd.DataFrame:
    """One row per model, reading each `run_dirs[name]/result.json`.

    Preserves the iteration order of `run_dirs` (never re-sorted), so a
    caller listing `pneumonet` first keeps it first in the output table.
    """
    columns = [
        "model",
        "params_total",
        "params_trainable",
        "epoch_time_s",
        "test_accuracy",
        "test_macro_f1",
        "test_macro_auc",
    ]

    rows = []
    for name, run_dir in run_dirs.items():
        result = json.loads((Path(run_dir) / "result.json").read_text())

        epoch_times = result.get("history", {}).get("epoch_time_s", [])
        epoch_time_s = float(np.mean(epoch_times)) if epoch_times else float("nan")

        test_metrics = result.get("test_metrics", {}) or {}

        rows.append(
            {
                "model": name,
                "params_total": result.get("params_total"),
                "params_trainable": result.get("params_trainable"),
                "epoch_time_s": epoch_time_s,
                "test_accuracy": test_metrics.get("accuracy", float("nan")),
                "test_macro_f1": test_metrics.get("macro", {}).get("f1", float("nan")),
                "test_macro_auc": test_metrics.get("auc_macro", float("nan")),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _latex_escape(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in str(text))


def _format_cell(value) -> str:
    if isinstance(value, bool):
        text = str(value)
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        text = "nan" if np.isnan(value) else f"{value:.4f}"
    else:
        text = str(value)
    return _latex_escape(text)


def dataframe_to_latex(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    """Write `df` as a LaTeX `table`/`tabular` environment, escaping underscores."""
    out_path = Path(out_path)
    columns = list(df.columns)
    col_spec = "l" * len(columns)

    header = " & ".join(_latex_escape(c) for c in columns) + " \\\\"

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        header,
        "\\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_format_cell(row[c]) for c in columns) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q1 test-set evaluation")
    parser.add_argument(
        "--all", action="store_true", help="Evaluate every full run under --runs-root"
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs/q1"))
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("results/q1_split_manifest.csv")
    )
    parser.add_argument("--out-csv", type=Path, default=Path("results/q1_comparison.csv"))
    parser.add_argument("--figs-dir", type=Path, default=Path("results/figs"))
    parser.add_argument("--tex", type=Path, default=Path("report/tables/q1_comparison.tex"))
    return parser


def main(argv: list[str] | None = None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.all:
        parser.error("nothing to do: pass --all to evaluate every full run")

    split_df = pd.read_csv(args.split_manifest)
    device = pick_device()

    candidate_dirs = {name: Path(args.runs_root) / f"{name}_full" for name in MODEL_NAMES}
    run_dirs = {
        name: run_dir
        for name, run_dir in candidate_dirs.items()
        if (run_dir / "result.json").exists()
    }
    if not run_dirs:
        parser.error(f"no run directories with a result.json found under {args.runs_root}")

    eval_results = {}
    for name, run_dir in run_dirs.items():
        print(f"evaluating {name} ({run_dir}) on the test split...")
        eval_results[name] = evaluate_run(run_dir, split_df, device=device)

    table = build_comparison_table(run_dirs)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)

    dataframe_to_latex(
        table,
        args.tex,
        caption="Cross-model comparison on the held-out Q1 test split.",
        label="tab:q1comparison",
    )

    primary = MODEL_NAMES[0]
    if primary in eval_results:
        figs_dir = Path(args.figs_dir)
        write_evaluation_artifacts(eval_results[primary], CLASS_NAMES, figs_dir, tag="q1")
        curves_src = run_dirs[primary] / "curves.png"
        if curves_src.exists():
            figs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(curves_src, figs_dir / f"q1_curves_{primary}.png")

    print(f"wrote {args.out_csv} ({len(table)} rows)")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
