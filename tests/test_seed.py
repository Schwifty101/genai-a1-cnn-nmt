import random

import numpy as np
import torch

from src.common.seed import SEED, make_generator, set_seed


def test_seed_constant_is_42():
    assert SEED == 42


def test_set_seed_makes_torch_reproducible():
    set_seed(42)
    a = torch.randn(16)
    set_seed(42)
    b = torch.randn(16)
    assert torch.equal(a, b)


def test_set_seed_makes_numpy_reproducible():
    set_seed(42)
    a = np.random.rand(16)
    set_seed(42)
    b = np.random.rand(16)
    assert np.array_equal(a, b)


def test_set_seed_makes_stdlib_random_reproducible():
    set_seed(42)
    a = [random.random() for _ in range(16)]
    set_seed(42)
    b = [random.random() for _ in range(16)]
    assert a == b


def test_different_seeds_differ():
    set_seed(42)
    a = torch.randn(16)
    set_seed(43)
    b = torch.randn(16)
    assert not torch.equal(a, b)


def test_make_generator_is_reproducible():
    g1 = make_generator(42)
    g2 = make_generator(42)
    a = torch.randperm(32, generator=g1)
    b = torch.randperm(32, generator=g2)
    assert torch.equal(a, b)
