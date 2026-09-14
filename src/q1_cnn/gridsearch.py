"""Q1 staged grid search and data-volume study.

The full hyperparameter space is too large to search jointly at 224px, so
this module runs a *staged coordinate search*: a fixed sequence of small
grids (`STAGES`), each swept independently at a cheap proxy resolution
(`image_size=128`, `epochs` capped at 12), with every stage starting from
the previous stage's winner. Selection is always by validation macro-F1
(never accuracy — the classes are imbalanced 7.6:1), ties broken by fewer
epochs run. Only the validation split is ever read during the search; the
test split is untouched until `run_data_volume_study`, which is the one
place it is used, and only after the winning config is already fixed.

`run_stage` is resumable: before training a config it checks whether that
config's `result.json` already exists on disk and, if so, reads it instead
of re-running — so a sweep that crashes at config 27 of 30 (this runs
unattended for hours) can simply be re-invoked rather than restarted from
scratch. It also rewrites its partial CSV after every config for the same
reason: inspectability mid-run.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.common.device import pick_device
from src.common.metrics import compute_classification_metrics
from src.common.plots import plot_xy
from src.common.seed import SEED
from src.common.trainer import evaluate_model
from src.q1_cnn.data_prep import CLASS_NAMES
from src.q1_cnn.datasets import build_loaders
from src.q1_cnn.models import build_model
from src.q1_cnn.splits import subsample_train
from src.q1_cnn.train import RunConfig, load_config, run_training

# The ordered stage definitions. Each stage's winner (by validation
# macro-F1, ties broken by fewer epochs run) becomes the base config for
# the next stage. Values are exactly as specified by the assignment — not
# to be retuned.
STAGES = [
    {"name": "A_preprocessing", "grid": {
        "normalization": ["imagenet", "zero_one", "dataset"],
        "augmentation": ["none", "light", "heavy"],
    }},                                                      # 9
    {"name": "B_optimization", "grid": {
        "lr": [1e-4, 3e-4, 1e-3],
        "batch_size": [16, 32, 64],
    }},                                                      # 9
    {"name": "C1_dropout", "grid": {"dropout": [0.2, 0.3, 0.5]}},        # 3
    {"name": "C2_l2", "grid": {"l2_lambda": [0.0, 1e-5, 1e-4]}},         # 3
    {"name": "C3_l1", "grid": {"l1_lambda": [0.0, 1e-5]}},               # 2
    {"name": "D_schedule", "grid": {
        "epochs": [30, 60],
        "patience": [5, 10],
    }},                                                      # 4
]

# Proxy-run constants (stages A through C3; D_schedule keeps image_size=128
# too but supplies its own `epochs`, since epochs is what it measures).
_PROXY_IMAGE_SIZE = 128
_PROXY_MAX_EPOCHS = 12

_NON_PARAM_COLS = {"stage", "val_macro_f1", "val_accuracy", "epochs_run", "seconds"}

_BEST_CONFIG_PATH = Path("results/q1_best_config.json")
_DATA_VOLUME_FIG_PATH = Path("results/figs/q1_datavolume.png")


def expand_stage(grid: dict[str, list]) -> list[dict]:
    """Full cartesian product of one stage's grid, in deterministic order.

    Iterates `sorted(grid)` keys so the ordering of the resulting configs
    never depends on dict insertion order.
    """
    keys = sorted(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _select_best(rows: list[dict]) -> dict:
    """Pick the winning row: max validation macro-F1, ties broken by fewer epochs run."""
    return max(rows, key=lambda r: (r["val_macro_f1"], -r["epochs_run"]))


def run_stage(
    stage: dict,
    base: RunConfig,
    split_df: pd.DataFrame,
    out_root: Path,
    device=None,
) -> tuple[dict, list[dict]]:
    """Run every config in `stage`, returning `(best_params, rows)`.

    Each config is trained into `out_root / stage_name / cfg_index/` via
    `run_training`. A row per config is appended to a partial CSV under
    that same stage directory after every single config completes, so a
    crashed sweep leaves an inspectable trail. If a config's `result.json`
    already exists (e.g. from a prior, interrupted run), it is read
    instead of re-training — this is what makes the sweep resumable rather
    than restart-from-scratch.
    """
    out_root = Path(out_root)
    stage_name = stage["name"]
    resolved_device = pick_device() if device is None else device

    param_dicts = expand_stage(stage["grid"])
    swept_keys = sorted(stage["grid"])

    stage_dir = out_root / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    partial_csv = stage_dir / "partial_results.csv"

    rows: list[dict] = []
    for i, params in enumerate(param_dicts):
        run_dir = stage_dir / str(i)
        result_path = run_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        else:
            cfg = replace(base, **params, run_name=f"{stage_name}_{i}")
            result = run_training(
                cfg, split_df, run_dir, device=resolved_device, root=Path(".")
            )

        epochs_run = len(result["history"]["train_loss"])
        row = {
            "stage": stage_name,
            **params,
            "val_macro_f1": float(result["val_metrics"]["macro"]["f1"]),
            "val_accuracy": float(result["val_metrics"]["accuracy"]),
            "epochs_run": int(epochs_run),
            "seconds": float(result["train_seconds"]),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(partial_csv, index=False)

    best_row = _select_best(rows)
    best_params = {k: best_row[k] for k in swept_keys}
    return best_params, rows


def run_search(
    base: RunConfig,
    split_df: pd.DataFrame,
    out_root: Path,
    results_csv: Path,
    device=None,
) -> tuple[RunConfig, pd.DataFrame]:
    """Run every stage in `STAGES` in order, each starting from the previous winner.

    Every stage forces `image_size=128` (the proxy resolution). Every
    stage except `D_schedule` additionally caps `epochs` at 12; D_schedule
    is left alone because `epochs` is the parameter it sweeps. Writes the
    winning config to `results/q1_best_config.json` and the full row table
    to `results_csv`.
    """
    out_root = Path(out_root)
    results_csv = Path(results_csv)
    resolved_device = pick_device() if device is None else device

    cfg = replace(base)
    all_rows: list[dict] = []

    for stage in STAGES:
        cfg = replace(cfg, image_size=_PROXY_IMAGE_SIZE)
        if stage["name"] != "D_schedule":
            cfg = replace(cfg, epochs=min(cfg.epochs, _PROXY_MAX_EPOCHS))

        best_params, rows = run_stage(stage, cfg, split_df, out_root, device=resolved_device)
        cfg = replace(cfg, **best_params)
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_csv, index=False)

    _BEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BEST_CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2))

    return cfg, results_df


def _sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _fmt_value(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def summarize_search(rows: pd.DataFrame, best: RunConfig) -> pd.DataFrame:
    """Hyperparameter / range / optimal-value table, one row per swept hyperparameter.

    Groups `rows` by `stage` (preserving first-appearance order, which for
    a `run_search` output matches `STAGES` order) and, within each stage,
    treats every column that isn't a metric/bookkeeping column and has at
    least one non-null value in that group as a swept hyperparameter for
    that stage. `range` is the sorted, comma-joined list of tried values;
    `optimal` is read off the winning `RunConfig` by the same field name.
    """
    best_dict = asdict(best)
    records: list[dict] = []
    seen: set[str] = set()

    for _, group in rows.groupby("stage", sort=False):
        param_cols = [
            c for c in group.columns
            if c not in _NON_PARAM_COLS and group[c].notna().any()
        ]
        for col in sorted(param_cols):
            if col in seen or col not in best_dict:
                continue
            seen.add(col)
            values = sorted(group[col].dropna().unique().tolist(), key=_sort_key)
            records.append({
                "hyperparameter": col,
                "range": ", ".join(_fmt_value(v) for v in values),
                "optimal": _fmt_value(best_dict[col]),
            })

    return pd.DataFrame(records, columns=["hyperparameter", "range", "optimal"])


def run_data_volume_study(
    best: RunConfig,
    split_df: pd.DataFrame,
    fractions: list[float],
    out_root: Path,
    results_csv: Path,
    device=None,
) -> pd.DataFrame:
    """Train the winning config at full size on stratified train subsamples.

    For each fraction, `subsample_train` takes a stratified slice of the
    *training* rows only (val/test pass through untouched); the model is
    trained at `image_size=224` and evaluated on the untouched test split
    — the one place in the whole search this module touches test data, and
    only after the winning config is already fixed. Returns a frame with
    columns `fraction`, `n_train`, `test_accuracy`, `test_macro_f1`, and
    also writes it incrementally to `results_csv` and a summary figure via
    `plot_xy`.
    """
    out_root = Path(out_root)
    results_csv = Path(results_csv)
    resolved_device = pick_device() if device is None else device

    rows: list[dict] = []
    for fraction in fractions:
        sub_df = subsample_train(split_df, fraction, seed=SEED)
        n_train = int((sub_df["split"] == "train").sum())

        run_cfg = replace(best, image_size=224, run_name=f"datavolume_frac_{fraction}")
        run_dir = out_root / f"frac_{fraction}"
        result = run_training(
            run_cfg, sub_df, run_dir, device=resolved_device, root=Path(".")
        )

        _, _, test_loader, _ = build_loaders(
            sub_df,
            image_size=run_cfg.image_size,
            normalization=run_cfg.normalization,
            policy=run_cfg.augmentation,
            batch_size=run_cfg.batch_size,
            num_workers=run_cfg.num_workers,
            class_weighting=run_cfg.class_weighting,
            seed=run_cfg.seed,
            root=Path("."),
        )
        model = build_model(run_cfg.model, num_classes=len(CLASS_NAMES), dropout=run_cfg.dropout)
        state = torch.load(result["ckpt_path"], map_location=resolved_device, weights_only=True)
        model.load_state_dict(state)
        model.to(resolved_device)

        y_true, y_pred, y_prob, _ = evaluate_model(model, test_loader, resolved_device)
        test_metrics = compute_classification_metrics(y_true, y_pred, y_prob, CLASS_NAMES)

        rows.append({
            "fraction": float(fraction),
            "n_train": n_train,
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_macro_f1": float(test_metrics["macro"]["f1"]),
        })

        results_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(results_csv, index=False)

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        plot_xy(
            x=out_df["fraction"].tolist(),
            series={
                "test_accuracy": out_df["test_accuracy"].tolist(),
                "test_macro_f1": out_df["test_macro_f1"].tolist(),
            },
            out_path=_DATA_VOLUME_FIG_PATH,
            xlabel="training data fraction",
            ylabel="score",
            title="Data-volume study",
        )

    return out_df


def _resolve_split_manifest(config_path: Path, override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    raw_cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    return Path(raw_cfg.get("data", {}).get("split_manifest", "results/q1_split_manifest.csv"))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q1 staged grid search and data-volume study")
    sub = parser.add_subparsers(dest="command", required=True)

    search_p = sub.add_parser("search", help="Run the staged grid search")
    search_p.add_argument("--config", type=Path, required=True)
    search_p.add_argument("--split-manifest", type=Path, default=None)
    search_p.add_argument("--out-root", type=Path, default=Path("runs/q1/grid"))
    search_p.add_argument("--results-csv", type=Path, default=Path("results/q1_gridsearch.csv"))

    dv_p = sub.add_parser("datavolume", help="Run the data-volume study")
    dv_p.add_argument("--config", type=Path, required=True)
    dv_p.add_argument("--best", type=Path, required=True)
    dv_p.add_argument("--split-manifest", type=Path, default=None)
    dv_p.add_argument("--out-root", type=Path, default=Path("runs/q1/datavolume"))
    dv_p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    dv_p.add_argument("--results-csv", type=Path, default=Path("results/q1_datavolume.csv"))

    return parser


def main(argv: list[str] | None = None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "search":
        base = load_config(args.config, {})
        split_manifest = _resolve_split_manifest(args.config, args.split_manifest)
        split_df = pd.read_csv(split_manifest)

        best_cfg, rows_df = run_search(base, split_df, args.out_root, args.results_csv)

        summary = summarize_search(rows_df, best_cfg)
        summary_path = Path("results/q1_hyperparam_table.csv")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)

        print(f"wrote {args.results_csv} ({len(rows_df)} rows)")
        print(f"wrote {_BEST_CONFIG_PATH}")
        print(summary.to_string(index=False))
        return best_cfg, rows_df

    if args.command == "datavolume":
        best_cfg = RunConfig(**json.loads(Path(args.best).read_text()))
        split_manifest = _resolve_split_manifest(args.config, args.split_manifest)
        split_df = pd.read_csv(split_manifest)

        out_df = run_data_volume_study(
            best_cfg, split_df, args.fractions, args.out_root, args.results_csv
        )
        print(f"wrote {args.results_csv} ({len(out_df)} rows)")
        print(out_df.to_string(index=False))
        return out_df

    raise ValueError(f"unknown command: {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    main()
