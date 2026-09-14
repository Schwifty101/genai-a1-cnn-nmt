import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.q1_cnn.data_prep import CLASS_NAMES
from src.q1_cnn.train import RunConfig, load_config, run_training


def _fixture(tmp_path, n_per_class=9):
    rows = []
    rng = np.random.default_rng(0)
    for idx, cls in enumerate(CLASS_NAMES):
        for i in range(n_per_class):
            p = tmp_path / cls / f"{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            base = 40 + idx * 70
            arr = np.clip(
                base + rng.integers(-15, 15, (32, 32)), 0, 255
            ).astype(np.uint8)
            Image.fromarray(arr, "L").save(p)
            split = "train" if i < 5 else ("val" if i < 7 else "test")
            rows.append(
                {
                    "path": str(p.relative_to(tmp_path)),
                    "label": cls,
                    "label_idx": idx,
                    "md5": f"{cls}{i}",
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


def _cfg(**kw):
    base = dict(
        model="pneumonet",
        batch_size=4,
        lr=1e-3,
        epochs=2,
        patience=3,
        dropout=0.3,
        l2_lambda=0.0,
        l1_lambda=0.0,
        normalization="imagenet",
        augmentation="light",
        class_weighting="weighted_loss",
        image_size=32,
        num_workers=0,
        seed=42,
        run_name="test",
    )
    base.update(kw)
    return RunConfig(**base)


def test_run_training_writes_result_json(tmp_path):
    df = _fixture(tmp_path)
    out = tmp_path / "run"
    result = run_training(_cfg(), df, out, device="cpu")
    assert (out / "result.json").exists()
    saved = json.loads((out / "result.json").read_text())
    assert saved["config"]["model"] == "pneumonet"


def test_result_has_all_required_keys(tmp_path):
    df = _fixture(tmp_path)
    result = run_training(_cfg(), df, tmp_path / "run", device="cpu")
    assert set(result) >= {
        "config",
        "history",
        "val_metrics",
        "params_total",
        "params_trainable",
        "train_seconds",
        "ckpt_path",
    }


def test_history_length_matches_epochs(tmp_path):
    df = _fixture(tmp_path)
    result = run_training(_cfg(epochs=3), df, tmp_path / "run", device="cpu")
    assert len(result["history"]["train_loss"]) == 3


def test_val_metrics_include_macro_f1_and_confusion_matrix(tmp_path):
    df = _fixture(tmp_path)
    result = run_training(_cfg(), df, tmp_path / "run", device="cpu")
    assert "macro" in result["val_metrics"]
    assert "f1" in result["val_metrics"]["macro"]
    assert len(result["val_metrics"]["confusion_matrix"]) == 3


def test_curves_figure_is_written(tmp_path):
    df = _fixture(tmp_path)
    out = tmp_path / "run"
    run_training(_cfg(), df, out, device="cpu")
    assert (out / "curves.png").exists()


def test_checkpoint_is_written(tmp_path):
    df = _fixture(tmp_path)
    out = tmp_path / "run"
    result = run_training(_cfg(), df, out, device="cpu")
    from pathlib import Path

    assert Path(result["ckpt_path"]).exists()


def test_same_seed_gives_same_first_epoch_loss(tmp_path):
    df = _fixture(tmp_path)
    a = run_training(_cfg(), df, tmp_path / "a", device="cpu")
    b = run_training(_cfg(), df, tmp_path / "b", device="cpu")
    assert a["history"]["train_loss"][0] == pytest.approx(
        b["history"]["train_loss"][0], rel=1e-6
    )


def test_result_json_is_serializable_without_numpy_types(tmp_path):
    df = _fixture(tmp_path)
    out = tmp_path / "run"
    run_training(_cfg(), df, out, device="cpu")
    json.loads((out / "result.json").read_text())


def test_load_config_applies_overrides(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "train:\n  model: pneumonet\n  batch_size: 32\n  lr: 0.0003\n"
        "  epochs: 30\n  patience: 5\n  dropout: 0.3\n  l2_lambda: 0.00001\n"
        "  l1_lambda: 0.0\n  normalization: imagenet\n  augmentation: light\n"
        "  class_weighting: weighted_loss\n  num_workers: 0\n"
        "seed: 42\ndata:\n  image_size: 224\n"
    )
    cfg = load_config(cfg_path, {"batch_size": 8, "lr": 0.01})
    assert cfg.batch_size == 8
    assert cfg.lr == 0.01
    assert cfg.epochs == 30
