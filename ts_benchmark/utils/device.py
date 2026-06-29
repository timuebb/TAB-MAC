# -*- coding: utf-8 -*-
import os

import torch


def is_mps_available() -> bool:
    """Return whether PyTorch can use Apple's MPS backend on this machine."""
    mps_backend = getattr(torch.backends, "mps", None)
    is_built = getattr(mps_backend, "is_built", None)
    is_available = getattr(mps_backend, "is_available", None)
    return bool(
        mps_backend is not None
        and is_built is not None
        and is_available is not None
        and is_built()
        and is_available()
    )


def get_best_device() -> torch.device:
    """
    Return the best available compute device.

    TAB_DEVICE can force a specific backend: cpu, mps, cuda, or cuda:<index>.
    Without an override the order is CUDA, MPS, then CPU.
    """
    requested_device = os.getenv("TAB_DEVICE")
    if requested_device:
        requested_device = requested_device.strip().lower()
        if requested_device == "cpu":
            return torch.device("cpu")
        if requested_device == "mps":
            if not is_mps_available():
                raise RuntimeError("TAB_DEVICE=mps was requested, but MPS is not available.")
            return torch.device("mps")
        if requested_device == "cuda" or requested_device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("TAB_DEVICE=cuda was requested, but CUDA is not available.")
            return torch.device(requested_device)
        if requested_device != "auto":
            raise ValueError(
                "TAB_DEVICE must be one of: auto, cpu, mps, cuda, cuda:<index>."
            )

    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device() -> torch.device:
    """Backward-compatible alias for the canonical device resolver."""
    return get_best_device()
