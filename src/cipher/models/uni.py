from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

from cipher.config import ModelSpec
from cipher.models.access import download_model_file


def _validate_checkpoint_config(path: Path, expected_dimension: int) -> None:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {
        "architecture": "vit_large_patch16_224",
        "dynamic_img_size": True,
        "global_pool": "token",
        "img_size": 224,
        "num_classes": 0,
        "num_features": expected_dimension,
        "patch_size": 16,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"UNI checkpoint configuration {key!r} is {config.get(key)!r}; "
                f"expected {value!r}"
            )
    preprocessing = config.get("pretrained_cfg", {})
    expected_preprocessing = {
        "crop_pct": 1,
        "input_size": [3, 224, 224],
        "interpolation": "bilinear",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    for key, value in expected_preprocessing.items():
        if preprocessing.get(key) != value:
            raise ValueError(
                f"UNI preprocessing configuration {key!r} is {preprocessing.get(key)!r}; "
                f"expected {value!r}"
            )


class UniEncoder:
    """MahmoodLab UNI ViT-L/16 using its final CLS token."""

    def __init__(self, spec: ModelSpec, device: str) -> None:
        if spec.embedding_output != "cls_token":
            raise ValueError("UNI adapter requires cls_token output")
        self.spec = spec
        self.device = device
        configuration = download_model_file(spec, "config.json")
        _validate_checkpoint_config(configuration, spec.expected_dimension)
        checkpoint = download_model_file(spec, "pytorch_model.bin")
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, antialias=True),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.spec.expected_dimension), dtype=np.float32)
        pixels = torch.stack([self.transform(image.convert("RGB")) for image in images])
        pixels = pixels.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            hidden_states = self.model.forward_features(pixels)
            embeddings = hidden_states[:, 0, :]
        result = embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
        expected = (len(images), self.spec.expected_dimension)
        if result.shape != expected:
            raise ValueError(f"UNI returned shape {result.shape}; expected {expected}")
        if not np.isfinite(result).all():
            raise ValueError("UNI returned non-finite embeddings")
        return result
