from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from PIL import Image

from cipher.provenance import sha256_file


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    grayscale = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(grayscale.tobytes())
    bits: list[bool] = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        bits.extend(
            pixels[offset + column] > pixels[offset + column + 1] for column in range(hash_size)
        )
    value = sum(bit << index for index, bit in enumerate(reversed(bits)))
    return f"{value:0{hash_size * hash_size // 4}x}"


def inspect_image(path: Path, *, compute_perceptual_hash: bool = True) -> dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            image_format = image.format or path.suffix.lstrip(".").upper()
            mode = image.mode
            channels = len(image.getbands())
            perceptual_hash = difference_hash(image) if compute_perceptual_hash else None

    return {
        "extension": path.suffix.lower(),
        "format": image_format,
        "width": width,
        "height": height,
        "mode": mode,
        "channels": channels,
        "sha256": sha256_file(path),
        "perceptual_hash": perceptual_hash,
        "decoder_warnings": " | ".join(str(item.message) for item in caught),
    }


def is_allowed_image(path: Path, allowed_extensions: list[str]) -> bool:
    return path.is_file() and path.suffix.lower() in set(allowed_extensions)
