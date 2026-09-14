"""Augmentation policies and normalization schemes for the Q1 CNN pipeline.

Augmentation applies to the training split only: when `train=False`, the
`policy` argument is ignored unconditionally and only resize + to-tensor +
normalize are applied. This protects the validity of every reported metric
— val/test images are never randomly perturbed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.common.plots import save_image_grid
from src.common.seed import SEED

NORMALIZATIONS: dict[str, tuple[list[float], list[float]]] = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "zero_one": ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    # Measured over the real Q1 training split (5,106 images) at 224px with
    # zero_one normalization (see task-4-report.md for the computation).
    "dataset": ([0.4919, 0.4919, 0.4919], [0.2300, 0.2300, 0.2300]),
}

_POLICIES = {"none", "light", "heavy"}


def _policy_transforms(policy: str, image_size: int) -> list:
    if policy == "none":
        return []
    if policy == "light":
        return [
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2),
        ]
    if policy == "heavy":
        return [
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=image_size, scale=(0.85, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2),
        ]
    raise ValueError(f"unknown policy: {policy!r}")


def build_transforms(
    image_size: int, normalization: str, policy: str, train: bool
) -> transforms.Compose:
    """Compose Grayscale(->3ch) -> [policy transforms if train] -> Resize -> ToTensor -> Normalize.

    `policy` is one of "none", "light", "heavy". When `train=False` the
    policy block is skipped unconditionally, regardless of what `policy` is
    set to. Unknown normalization or policy names raise `ValueError`.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"unknown normalization: {normalization!r}")
    if policy not in _POLICIES:
        raise ValueError(f"unknown policy: {policy!r}")

    ops: list = [transforms.Grayscale(num_output_channels=3)]
    if train:
        ops.extend(_policy_transforms(policy, image_size))
    ops.append(transforms.Resize((image_size, image_size)))
    ops.append(transforms.ToTensor())
    mean, std = NORMALIZATIONS[normalization]
    ops.append(transforms.Normalize(mean, std))

    return transforms.Compose(ops)


def make_augmentation_figure(
    manifest_csv: Path, out_path: Path, n: int = 4, seed: int = SEED
) -> None:
    """Render `n` training images each beside three `heavy`-policy augmented draws.

    Un-normalizes before display so the figure is human-readable, and writes
    the grid through `src.common.plots.save_image_grid`.
    """
    manifest_csv = Path(manifest_csv)
    out_path = Path(out_path)

    df = pd.read_csv(manifest_csv)
    train_df = df[df["split"] == "train"] if "split" in df.columns else df

    rng = np.random.default_rng(seed)
    chosen = train_df.sample(n=n, random_state=seed) if n < len(train_df) else train_df

    image_size = 224
    normalization = "zero_one"
    mean, std = NORMALIZATIONS[normalization]
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    to_display = transforms.Compose(
        [transforms.Grayscale(num_output_channels=3), transforms.Resize((image_size, image_size))]
    )
    aug_transform = build_transforms(image_size, normalization, "heavy", train=True)

    def unnormalize_to_hwc(tensor: torch.Tensor) -> np.ndarray:
        arr = (tensor * std_t + mean_t).clamp(0, 1)
        return arr.permute(1, 2, 0).numpy()

    images = []
    titles = []
    for _, row in chosen.iterrows():
        pil = Image.open(row["path"])
        orig_display = np.asarray(to_display(pil)) / 255.0
        images.append(orig_display)
        titles.append(f"{row['label']} (orig)")
        for k in range(3):
            aug_tensor = aug_transform(pil)
            images.append(unnormalize_to_hwc(aug_tensor))
            titles.append(f"{row['label']} (aug {k + 1})")

    save_image_grid(
        images,
        titles,
        out_path,
        ncols=4,
        suptitle="Q1 heavy-policy augmentation examples",
    )
