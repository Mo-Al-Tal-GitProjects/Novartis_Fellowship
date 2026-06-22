from pathlib import Path

from PIL import Image

from cipher.data.images import difference_hash, inspect_image


def test_difference_hash_is_deterministic() -> None:
    image = Image.new("RGB", (20, 20), (1, 2, 3))
    assert difference_hash(image) == difference_hash(image.copy())
    assert len(difference_hash(image)) == 16


def test_inspect_image_records_content(tmp_path: Path) -> None:
    path = tmp_path / "sample.PNG"
    Image.new("RGBA", (17, 19), (1, 2, 3, 255)).save(path)
    record = inspect_image(path)
    assert record["extension"] == ".png"
    assert record["format"] == "PNG"
    assert record["width"] == 17
    assert record["height"] == 19
    assert record["channels"] == 4
    assert len(record["sha256"]) == 64
