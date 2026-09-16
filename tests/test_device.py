import torch

from src.common.device import pick_device


def test_pick_device_cpu_is_cpu():
    assert pick_device("cpu").type == "cpu"


def test_pick_device_auto_returns_supported_type():
    assert pick_device("auto").type in {"mps", "cpu"}


def test_pick_device_auto_prefers_mps_when_available():
    expected = "mps" if torch.backends.mps.is_available() else "cpu"
    assert pick_device("auto").type == expected


def test_tensor_can_move_to_picked_device():
    device = pick_device("auto")
    x = torch.ones(4, 4, device=device)
    assert x.sum().item() == 16.0
