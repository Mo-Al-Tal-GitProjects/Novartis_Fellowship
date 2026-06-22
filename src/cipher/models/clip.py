from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from cipher.config import ModelSpec


class ClipEncoder:
    """Official Hugging Face CLIP vision projection with native preprocessing."""

    def __init__(self, spec: ModelSpec, device: str) -> None:
        if spec.embedding_output != "image_projection":
            raise ValueError("CLIP-compatible adapters require image_projection output")
        self.spec = spec
        self.device = device
        common = {
            "revision": spec.revision,
            "trust_remote_code": spec.trust_remote_code,
        }
        self.processor = CLIPImageProcessor.from_pretrained(spec.model_id, **common)
        self.model = CLIPVisionModelWithProjection.from_pretrained(spec.model_id, **common)
        self.model.eval().to(device)

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.spec.expected_dimension), dtype=np.float32)
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            embeddings = self.model(pixel_values=pixel_values).image_embeds
        result = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected = (len(images), self.spec.expected_dimension)
        if result.shape != expected:
            raise ValueError(f"CLIP returned shape {result.shape}; expected {expected}")
        if not np.isfinite(result).all():
            raise ValueError("CLIP returned non-finite embeddings")
        return result
