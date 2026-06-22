from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from cipher.config import ModelSpec


class DinoEncoder:
    """Original DINO ViT backbone using its final normalized CLS token."""

    def __init__(self, spec: ModelSpec, device: str) -> None:
        if spec.embedding_output != "cls_token":
            raise ValueError("DINO adapter requires cls_token output")
        self.spec = spec
        self.device = device
        common = {
            "revision": spec.revision,
            "trust_remote_code": spec.trust_remote_code,
        }
        self.processor = ViTImageProcessor.from_pretrained(spec.model_id, **common)
        self.model = ViTModel.from_pretrained(spec.model_id, add_pooling_layer=False, **common)
        self.model.eval().to(device)

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.spec.expected_dimension), dtype=np.float32)
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            hidden_states = self.model(pixel_values=pixel_values).last_hidden_state
            embeddings = hidden_states[:, 0, :]
        result = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected = (len(images), self.spec.expected_dimension)
        if result.shape != expected:
            raise ValueError(f"DINO returned shape {result.shape}; expected {expected}")
        if not np.isfinite(result).all():
            raise ValueError("DINO returned non-finite embeddings")
        return result
