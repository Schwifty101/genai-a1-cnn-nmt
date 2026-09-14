"""Q1 config-driven training entry point.

Ties the Q1 data loaders (`src.q1_cnn.datasets`), models
(`src.q1_cnn.models`), and the generic trainer (`src.common.trainer`)
into one run: `run_training` executes a single training run from a
`RunConfig` and writes `result.json` + `curves.png` into an output
directory; the CLI at the bottom of this module builds that `RunConfig`
from a YAML config file plus `--set key=value` overrides.

Model selection is by validation macro-F1 (`monitor="val_macro_f1"`,
`mode="max"`), not accuracy — the Q1 classes are imbalanced 7.6:1, so
accuracy alone rewards a model that always predicts the majority class.

The test split is never touched here: `run_training` only ever evaluates
on the validation loader. Test-set evaluation is a later, separate task,
deliberately, so hyperparameter tuning never sees test data.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.common.device import pick_device
from src.common.metrics import compute_classification_metrics
from src.common.plots import plot_training_curves
from src.common.seed import SEED, set_seed
from src.common.trainer import TrainConfig, evaluate_model, train_model
from src.q1_cnn.data_prep import CLASS_NAMES
from src.q1_cnn.datasets import build_loaders
from src.q1_cnn.models import build_model, count_parameters


@dataclass
class RunConfig:
    model: str
    batch_size: int
    lr: float
    epochs: int
    patience: int
    dropout: float
    l2_lambda: float
    l1_lambda: float
    normalization: str
    augmentation: str
    class_weighting: str
    image_size: int
    num_workers: int
    seed: int
    run_name: str


def load_config(path: Path, overrides: dict) -> RunConfig:
    """Build a `RunConfig` from a `configs/q1_*.yaml` file plus overrides.

    `overrides` wins over anything in the file. Any field missing from
    both the file and `overrides` falls back to a safe default (the
    project SEED, or "run" for `run_name`) so config files that predate a
    field (e.g. lack a `run_name`) still load.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    train_raw = raw.get("train", {}) or {}
    data_raw = raw.get("data", {}) or {}

    merged = {
        "model": train_raw.get("model"),
        "batch_size": train_raw.get("batch_size"),
        "lr": train_raw.get("lr"),
        "epochs": train_raw.get("epochs"),
        "patience": train_raw.get("patience"),
        "dropout": train_raw.get("dropout"),
        "l2_lambda": train_raw.get("l2_lambda"),
        "l1_lambda": train_raw.get("l1_lambda"),
        "normalization": train_raw.get("normalization"),
        "augmentation": train_raw.get("augmentation"),
        "class_weighting": train_raw.get("class_weighting"),
        "image_size": data_raw.get("image_size"),
        "num_workers": train_raw.get("num_workers"),
        "seed": raw.get("seed", SEED),
        "run_name": raw.get("run_name", "run"),
    }
    merged.update(overrides or {})
    return RunConfig(**merged)


def _is_head(name: str) -> bool:
    """True for parameter names belonging to a model's classification head."""
    return "classifier" in name or "fc" in name


def _build_optimizer(model: nn.Module, cfg: RunConfig) -> torch.optim.Optimizer:
    """AdamW; `_finetune` models get a head group at `lr` and an unfrozen
    backbone group at `lr * 0.1`; every other model gets one group at `lr`.
    Only `requires_grad=True` parameters are handed to the optimizer.
    """
    if cfg.model.endswith("_finetune"):
        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (head_params if _is_head(name) else backbone_params).append(param)
        return torch.optim.AdamW(
            [
                {"params": head_params, "lr": cfg.lr},
                {"params": backbone_params, "lr": cfg.lr * 0.1},
            ],
            weight_decay=cfg.l2_lambda,
        )
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.l2_lambda,
    )


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays and Paths to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_training(
    cfg: RunConfig,
    split_df: pd.DataFrame,
    out_dir: Path,
    device=None,
    root: Path | None = None,
) -> dict:
    """Run one full training + validation cycle and persist its artifacts.

    Writes `out_dir / "result.json"` and `out_dir / "curves.png"`. Only the
    validation split is ever evaluated here — the test split is left
    untouched for a later task.

    `root` is the directory the manifest's `path` column is relative to. It
    defaults to `out_dir.parent`, which matches the CLI's own
    `runs/q1/<run_name>` layout only when explicitly overridden (the CLI
    passes `root=Path(".")`, the repo root the real manifest paths are
    relative to); tests that build a self-contained `tmp_path` fixture and
    write into `tmp_path / "run"` get the correct root for free.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if root is None:
        root = out_dir.parent
    root = Path(root)

    set_seed(cfg.seed)

    resolved_device = pick_device() if device is None else device

    train_loader, val_loader, _test_loader, class_weights = build_loaders(
        split_df,
        image_size=cfg.image_size,
        normalization=cfg.normalization,
        policy=cfg.augmentation,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        class_weighting=cfg.class_weighting,
        seed=cfg.seed,
        root=root,
    )

    model = build_model(cfg.model, num_classes=len(CLASS_NAMES), dropout=cfg.dropout)
    model.to(resolved_device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(resolved_device))
    optimizer = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(1, cfg.patience // 2)
    )

    ckpt_path = out_dir / "best.pt"
    train_cfg = TrainConfig(
        epochs=cfg.epochs,
        patience=cfg.patience,
        monitor="val_macro_f1",
        mode="max",
        ckpt_path=ckpt_path,
        l1_lambda=cfg.l1_lambda,
    )

    start = time.time()
    history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        resolved_device,
        train_cfg,
        CLASS_NAMES,
        scheduler=scheduler,
    )
    train_seconds = time.time() - start

    y_true, y_pred, y_prob, _val_loss = evaluate_model(
        model, val_loader, resolved_device, criterion
    )
    val_metrics = compute_classification_metrics(y_true, y_pred, y_prob, CLASS_NAMES)

    params_total, params_trainable = count_parameters(model)

    result = {
        "config": asdict(cfg),
        "history": history,
        "val_metrics": val_metrics,
        "params_total": int(params_total),
        "params_trainable": int(params_trainable),
        "train_seconds": float(train_seconds),
        "ckpt_path": str(ckpt_path),
    }
    result = _to_jsonable(result)

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    plot_training_curves(
        history, out_dir / "curves.png", title=f"{cfg.model} ({cfg.run_name})"
    )

    return result


def _parse_set_overrides(pairs: list[str]) -> dict:
    """Parse repeated `--set key=value` pairs, coercing values via YAML."""
    overrides = {}
    for pair in pairs:
        key, _, raw_value = pair.partition("=")
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Q1 CNN training entry point")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config value, e.g. --set lr=0.001",
    )
    args = parser.parse_args(argv)

    overrides = _parse_set_overrides(args.overrides)
    if args.model is not None:
        overrides["model"] = args.model
    if args.run_name is not None:
        overrides["run_name"] = args.run_name

    cfg = load_config(args.config, overrides)

    raw_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    split_manifest = Path(
        raw_cfg.get("data", {}).get("split_manifest", "results/q1_split_manifest.csv")
    )
    split_df = pd.read_csv(split_manifest)

    out_dir = Path("runs/q1") / cfg.run_name
    result = run_training(cfg, split_df, out_dir, root=Path("."))

    n_epochs = len(result["history"]["epoch_time_s"])
    avg_epoch_s = sum(result["history"]["epoch_time_s"]) / n_epochs if n_epochs else 0.0
    print(f"wrote {out_dir / 'result.json'}")
    print(f"val_macro_f1={result['val_metrics']['macro']['f1']:.4f}")
    print(f"avg_epoch_time_s={avg_epoch_s:.3f} over {n_epochs} epochs")

    return result


if __name__ == "__main__":
    main()
