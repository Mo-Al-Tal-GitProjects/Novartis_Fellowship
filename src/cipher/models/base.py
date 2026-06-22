from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image

from cipher.config import ModelSpec


class ImageEncoder(Protocol):
    """Small common boundary implemented by every model adapter."""

    spec: ModelSpec
    device: str

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return one unnormalized float32 embedding per image."""
        ...
