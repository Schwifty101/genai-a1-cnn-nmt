"""PyTorch Dataset and DataLoader construction for the Q1 chest X-ray splits.

Reads a manifest that already carries a `split` column (produced once by
`src.q1_cnn.splits.make_splits` and persisted to disk) — this module never
re-splits.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.common.seed import make_generator, seed_worker
from src.q1_cnn.augment import build_transforms
from src.q1_cnn.data_prep import CLASS_NAMES


class ChestXrayDataset(Dataset):
    """Wraps a split-manifest dataframe; always returns a 3-channel tensor."""

    def __init__(self, df: pd.DataFrame, transform, root: Path):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(self.root / row["path"])
        tensor = self.transform(image)
        return tensor, int(row["label_idx"])


def _inverse_frequency_weights(train_df: pd.DataFrame) -> torch.Tensor:
    n_classes = len(CLASS_NAMES)
    n_train = len(train_df)
    counts = train_df["label"].value_counts()
    weights = [
        n_train / (n_classes * counts.get(cls, 0)) if counts.get(cls, 0) > 0 else 0.0
        for cls in CLASS_NAMES
    ]
    return torch.tensor(weights, dtype=torch.float32)


def build_loaders(
    split_df: pd.DataFrame,
    image_size: int,
    normalization: str,
    policy: str,
    batch_size: int,
    num_workers: int,
    class_weighting: str,
    seed: int = 42,
    root: Path = Path("."),
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Partition `split_df` by its `split` column and build train/val/test loaders.

    `class_weighting` is "weighted_loss" (returns inverse-frequency class
    weights, shuffled loader) or "sampler" (returns uniform weights,
    train loader driven by a `WeightedRandomSampler`).
    """
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]

    train_transform = build_transforms(image_size, normalization, policy, train=True)
    eval_transform = build_transforms(image_size, normalization, policy, train=False)

    train_ds = ChestXrayDataset(train_df, train_transform, root=root)
    val_ds = ChestXrayDataset(val_df, eval_transform, root=root)
    test_ds = ChestXrayDataset(test_df, eval_transform, root=root)

    inv_freq_weights = _inverse_frequency_weights(train_df)

    if class_weighting == "sampler":
        class_weights = torch.ones(len(CLASS_NAMES), dtype=torch.float32)
        label_idx_to_weight = {
            i: inv_freq_weights[i].item() for i in range(len(CLASS_NAMES))
        }
        sample_weights = [
            label_idx_to_weight[int(label_idx)] for label_idx in train_df["label_idx"]
        ]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=make_generator(seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=make_generator(seed),
            persistent_workers=num_workers > 0,
        )
    elif class_weighting == "weighted_loss":
        class_weights = inv_freq_weights
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=make_generator(seed),
            persistent_workers=num_workers > 0,
        )
    else:
        raise ValueError(f"unknown class_weighting: {class_weighting!r}")

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader, class_weights
