import zipfile
from pathlib import Path

import pytest

from cipher.data.download import _byte_ranges, extract_pubmed_archive


def test_byte_ranges_cover_file_without_overlap() -> None:
    ranges = _byte_ranges(101, 4)
    covered = [value for start, end in ranges for value in range(start, end + 1)]
    assert covered == list(range(101))
    assert len(ranges) == 4


def test_extract_pubmed_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("pubmed_set/captions.json", "{}")
        bundle.writestr("pubmed_set/images/", "")
        bundle.writestr("pubmed_set/../escape.txt", "nope")

    with pytest.raises(ValueError, match="unsafe ZIP member"):
        extract_pubmed_archive(archive, tmp_path / "pubmed_set")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_pubmed_archive_uses_expected_layout(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("pubmed_set/captions.json", "{}")
        bundle.writestr("pubmed_set/images/example.jpg", b"image bytes")
        bundle.writestr("__MACOSX/pubmed_set/._captions.json", b"metadata")

    target = tmp_path / "pubmed_set"
    extracted = extract_pubmed_archive(archive, target)
    assert extracted == 2
    assert (target / "captions.json").is_file()
    assert (target / "images/example.jpg").is_file()
