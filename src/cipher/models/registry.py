from __future__ import annotations

from cipher.config import ModelSpec
from cipher.models.base import ImageEncoder


def create_encoder(spec: ModelSpec, device: str) -> ImageEncoder:
    if spec.adapter == "clip":
        from cipher.models.clip import ClipEncoder

        return ClipEncoder(spec, device)
    if spec.adapter == "dino":
        from cipher.models.dino import DinoEncoder

        return DinoEncoder(spec, device)
    if spec.adapter == "uni":
        from cipher.models.uni import UniEncoder

        return UniEncoder(spec, device)
    raise ValueError(f"unsupported model adapter: {spec.adapter}")
