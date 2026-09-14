"""Q1 error analysis: which test-set predictions were wrong, and how wrong.

Operates on the dict shape produced by `src.q1_cnn.evaluate.evaluate_checkpoint`
(`{"y_true", "y_pred", "y_prob", "paths", ...}`) — pure array/dataframe logic,
no model or I/O dependency of its own beyond loading the misclassified
images for the montage.

The test split is imbalanced 7.6:1 (COVID19 56 / NORMAL 158 / PNEUMONIA
425), so per-class error *counts* are dominated by PNEUMONIA in absolute
terms simply because there are far more PNEUMONIA examples to get wrong.
That is expected and reported as-is here, not normalized away — it is why
the project's headline metrics are macro-averaged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.common.plots import save_image_grid
from src.q1_cnn.data_prep import CLASS_NAMES
from src.q1_cnn.evaluate import evaluate_checkpoint
from src.q1_cnn.train import RunConfig


def collect_misclassified(result: dict, class_names: list[str], top_n: int = 12) -> pd.DataFrame:
    """The `top_n` misclassified test examples, ranked by descending confidence
    in the (wrong) predicted class — the model's most confidently wrong calls.

    Returns a dataframe with columns `path`, `true`, `pred`, `confidence`.
    """
    y_true = np.asarray(result["y_true"])
    y_pred = np.asarray(result["y_pred"])
    y_prob = np.asarray(result["y_prob"])
    paths = list(result["paths"])

    error_idx = np.where(y_true != y_pred)[0]

    rows = []
    for i in error_idx:
        rows.append(
            {
                "path": paths[i],
                "true": class_names[int(y_true[i])],
                "pred": class_names[int(y_pred[i])],
                "confidence": float(y_prob[i][y_pred[i]]),
            }
        )

    df = pd.DataFrame(rows, columns=["path", "true", "pred", "confidence"])
    df = df.sort_values("confidence", ascending=False, kind="mergesort").reset_index(drop=True)
    return df.head(top_n).reset_index(drop=True)


def make_error_montage(errors: pd.DataFrame, root: Path, out_path: Path) -> None:
    """Image grid of `errors`, one cell per row, titled with true/pred/confidence."""
    if len(errors) == 0:
        return

    root = Path(root)
    images, titles = [], []
    for _, row in errors.iterrows():
        with Image.open(root / row["path"]) as im:
            images.append(np.array(im.convert("RGB")))
        titles.append(f"true={row['true']} pred={row['pred']} p={row['confidence']:.2f}")

    save_image_grid(
        images, titles, out_path, ncols=4, suptitle="Misclassified examples (ranked by confidence)"
    )


def error_statistics(result: dict, class_names: list[str]) -> dict:
    """Summary statistics over every misclassified test example.

    Returns `{"n_errors", "by_true_class", "by_confusion_pair",
    "mean_error_confidence", "mean_correct_confidence"}`. Confusion-pair
    keys are `f"{true_name}->{pred_name}"`. The two mean-confidence fields
    are `float("nan")` (never 0.0, never a crash) when their underlying set
    is empty.
    """
    y_true = np.asarray(result["y_true"])
    y_pred = np.asarray(result["y_pred"])
    y_prob = np.asarray(result["y_prob"])

    mismatch = y_true != y_pred
    error_idx = np.where(mismatch)[0]
    correct_idx = np.where(~mismatch)[0]

    by_true_class = {name: 0 for name in class_names}
    by_confusion_pair: dict[str, int] = {}
    for i in error_idx:
        true_name = class_names[int(y_true[i])]
        pred_name = class_names[int(y_pred[i])]
        by_true_class[true_name] += 1
        key = f"{true_name}->{pred_name}"
        by_confusion_pair[key] = by_confusion_pair.get(key, 0) + 1

    error_confidences = [float(y_prob[i][y_pred[i]]) for i in error_idx]
    correct_confidences = [float(y_prob[i][y_pred[i]]) for i in correct_idx]

    mean_error_confidence = (
        float(np.mean(error_confidences)) if error_confidences else float("nan")
    )
    mean_correct_confidence = (
        float(np.mean(correct_confidences)) if correct_confidences else float("nan")
    )

    return {
        "n_errors": int(len(error_idx)),
        "by_true_class": by_true_class,
        "by_confusion_pair": by_confusion_pair,
        "mean_error_confidence": mean_error_confidence,
        "mean_correct_confidence": mean_correct_confidence,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q1 error analysis")
    parser.add_argument("--run", type=Path, required=True, help="Run directory with result.json")
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("results/q1_split_manifest.csv")
    )
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: list[str] | None = None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    stored = json.loads((run_dir / "result.json").read_text())
    cfg = RunConfig(**stored["config"])
    ckpt_path = Path(stored["ckpt_path"])

    split_df = pd.read_csv(args.split_manifest)
    eval_result = evaluate_checkpoint(ckpt_path, cfg.model, split_df, cfg, device=None)

    errors = collect_misclassified(eval_result, CLASS_NAMES, top_n=args.top_n)
    stats = error_statistics(eval_result, CLASS_NAMES)

    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figs"
    make_error_montage(errors, root=args.root, out_path=figs_dir / "q1_errors.png")

    stats_path = out_dir / "q1_error_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"wrote {figs_dir / 'q1_errors.png'} ({len(errors)} examples)")
    print(f"wrote {stats_path}")
    print(json.dumps(stats, indent=2))
    return errors, stats


if __name__ == "__main__":
    main()
