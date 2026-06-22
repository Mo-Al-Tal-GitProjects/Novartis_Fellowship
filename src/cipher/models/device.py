from __future__ import annotations

from typing import Literal

import torch

DeviceRequest = Literal["auto", "cpu", "cuda", "mps"]


def resolve_device(requested: DeviceRequest = "auto") -> str:
    """Resolve and validate an inference device without silently falling back."""
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal (MPS) was requested but is not available")
    return requested
