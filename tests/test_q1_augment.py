import numpy as np
import pytest
import torch
from PIL import Image

from src.q1_cnn.augment import NORMALIZATIONS, build_transforms
from src.common.seed import set_seed


def _pil(size=(256, 300)):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0]), dtype=np.uint8), "L")


def test_normalization_keys_are_the_three_swept_schemes():
    assert set(NORMALIZATIONS) == {"imagenet", "zero_one", "dataset"}


def test_eval_transform_output_shape_is_three_channel_and_square():
    t = build_transforms(224, "imagenet", "heavy", train=False)
    out = t(_pil())
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_train_transform_output_shape_matches_eval():
    t = build_transforms(128, "zero_one", "heavy", train=True)
    assert t(_pil()).shape == (3, 128, 128)


def test_eval_transform_is_deterministic_regardless_of_policy():
    a = build_transforms(224, "imagenet", "heavy", train=False)(_pil())
    b = build_transforms(224, "imagenet", "heavy", train=False)(_pil())
    assert torch.equal(a, b)


def test_train_transform_with_heavy_policy_is_stochastic():
    set_seed(42)
    t = build_transforms(224, "imagenet", "heavy", train=True)
    img = _pil()
    assert not torch.equal(t(img), t(img))


def test_train_transform_with_none_policy_is_deterministic():
    t = build_transforms(224, "imagenet", "none", train=True)
    img = _pil()
    assert torch.equal(t(img), t(img))


def test_zero_one_normalization_keeps_values_in_unit_range():
    out = build_transforms(64, "zero_one", "none", train=False)(_pil())
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_unknown_normalization_raises():
    with pytest.raises(ValueError):
        build_transforms(64, "not_a_scheme", "none", train=False)


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        build_transforms(64, "imagenet", "not_a_policy", train=True)
