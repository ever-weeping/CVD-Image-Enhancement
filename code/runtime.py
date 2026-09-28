"""Runtime helpers shared by local Mac runs and CUDA server runs."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a user device string into a torch.device.

    `auto` prefers CUDA on servers, then Apple MPS on Macs, then CPU.
    Explicit values such as `cpu`, `cuda`, `cuda:0`, or `mps` are accepted.
    """
    requested = device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if resolved.type == "mps":
        has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return resolved


def should_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"
