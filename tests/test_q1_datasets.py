import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.q1_cnn.augment import build_transforms
from src.q1_cnn.data_prep import CLASS_NAMES
from src.q1_cnn.datasets import ChestXrayDataset, build_loaders


def _fixture(tmp_path, n_per_class=12):
    rows = []
    rng = np.random.default_rng(0)
    for idx, cls in enumerate(CLASS_NAMES):
        for i in range(n_per_class):
            p = tmp_path / cls / f"{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                rng.integers(0, 255, (40, 48), dtype=np.uint8), "L"
            ).save(p)
            split = "train" if i < 8 else ("val" if i < 10 else "test")
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


def test_dataset_returns_three_channel_tensor_and_int_label(tmp_path):
    df = _fixture(tmp_path)
    ds = ChestXrayDataset(
        df, build_transforms(64, "imagenet", "none", train=False), root=tmp_path
    )
    x, y = ds[0]
    assert x.shape == (3, 64, 64)
    assert isinstance(y, int)
    assert 0 <= y < len(CLASS_NAMES)


def test_dataset_length_matches_dataframe(tmp_path):
    df = _fixture(tmp_path)
    ds = ChestXrayDataset(
        df, build_transforms(64, "imagenet", "none", train=False), root=tmp_path
    )
    assert len(ds) == len(df)


def test_grayscale_source_is_replicated_across_channels(tmp_path):
    df = _fixture(tmp_path)
    ds = ChestXrayDataset(
        df, build_transforms(64, "zero_one", "none", train=False), root=tmp_path
    )
    x, _ = ds[0]
    assert torch.allclose(x[0], x[1]) and torch.allclose(x[1], x[2])


def test_build_loaders_partitions_by_split_column(tmp_path):
    df = _fixture(tmp_path)
    train, val, test, weights = build_loaders(
        df,
        image_size=64,
        normalization="imagenet",
        policy="light",
        batch_size=4,
        num_workers=0,
        class_weighting="weighted_loss",
        seed=42,
    )
    assert len(train.dataset) == 24
    assert len(val.dataset) == 6
    assert len(test.dataset) == 6
    assert weights.shape == (3,)


def test_eval_loaders_are_not_shuffled(tmp_path):
    df = _fixture(tmp_path)
    _, val, test, _ = build_loaders(
        df, 64, "imagenet", "heavy", 4, 0, "weighted_loss", seed=42, root=tmp_path
    )
    first = [y.tolist() for _, y in val]
    second = [y.tolist() for _, y in val]
    assert first == second


def test_class_weights_are_inverse_frequency(tmp_path):
    df = _fixture(tmp_path)
    df = df.drop(df[(df["label"] == "COVID19") & (df["split"] == "train")].index[:4])
    _, _, _, weights = build_loaders(
        df, 64, "imagenet", "none", 4, 0, "weighted_loss", seed=42
    )
    assert weights[0] > weights[1]


def test_sampler_mode_returns_uniform_weights(tmp_path):
    df = _fixture(tmp_path)
    train, _, _, weights = build_loaders(
        df, 64, "imagenet", "none", 4, 0, "sampler", seed=42
    )
    assert torch.allclose(weights, torch.ones(3))
    assert train.sampler is not None
