"""The single place in the project where a torch device is chosen."""

from __future__ import annotations

import torch


def pick_device(prefer: str = "auto") -> torch.device:
    """Return the compute device.

    `prefer` is one of "auto", "mps", "cpu". "auto" selects MPS when it is
    available and falls back to CPU otherwise.
    """
    if prefer not in {"auto", "mps", "cpu"}:
        raise ValueError(f"prefer must be auto|mps|cpu, got {prefer!r}")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
