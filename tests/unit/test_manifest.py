import pandas as pd
import pytest

from cipher.data.ground_truth import build_relevance_manifest, validate_ground_truth
from cipher.data.manifest import (
    _exclude_exact_arch_duplicates,
    build_image_manifest,
    parse_match_filename,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("387_01.jpg", ("387", "01")),
        ("1353_03.PNG", ("1353", "03")),
        ("1024_05.jpeg", ("1024", "05")),
    ],
)
def test_parse_match_filename_accepts_supported_formats(filename, expected) -> None:
    assert parse_match_filename(filename) == expected


def test_parse_match_filename_rejects_unstructured_name() -> None:
    with pytest.raises(ValueError):
        parse_match_filename("image.jpg")


def test_exact_arch_duplicate_preserves_query_and_excludes_gallery() -> None:
    frame = pd.DataFrame(
        [
            {
                "source": "arch_pubmed",
                "source_record_id": "10",
                "image_id": "arch:query",
                "sha256": "same",
                "role": "query",
                "is_canonical": True,
                "duplicate_of": None,
                "validation_status": "valid",
                "validation_notes": "",
            },
            {
                "source": "arch_pubmed",
                "source_record_id": "1",
                "image_id": "arch:gallery-copy",
                "sha256": "same",
                "role": "gallery",
                "is_canonical": True,
                "duplicate_of": None,
                "validation_status": "valid",
                "validation_notes": "",
            },
        ]
    )
    result = _exclude_exact_arch_duplicates(frame)
    query = result.loc[result["image_id"] == "arch:query"].iloc[0]
    duplicate = result.loc[result["image_id"] == "arch:gallery-copy"].iloc[0]
    assert query["role"] == "query"
    assert duplicate["role"] == "excluded"
    assert duplicate["duplicate_of"] == "arch:query"
    assert duplicate["validation_status"] == "excluded_exact_duplicate"


def test_manifest_and_relevance_pipeline(project_fixture) -> None:
    images, image_summary = build_image_manifest(project_fixture)
    relevance, relevance_summary = build_relevance_manifest(project_fixture, images)
    report = validate_ground_truth(project_fixture)

    assert len(images) == 7
    assert image_summary["role_counts"] == {"query": 2, "gallery": 4, "excluded": 1}
    assert len(relevance) == 3
    assert relevance_summary["positives_per_query"] == {"1": 2, "2": 1}
    assert report["status"] == "passed"
    assert report["positive_images"] == 3
    assert report["excluded_duplicate_files"] == ["1_01.jpg"]

    excluded = images.loc[images["role"] == "excluded"].iloc[0]
    assert excluded["filename"] == "1_01.jpg"
    assert excluded["validation_status"] == "excluded_duplicate"
    assert "match:1_01.jpg" not in set(relevance["gallery_image_id"])

    stored = pd.read_parquet(project_fixture.root / "data/manifests/images.parquet")
    assert stored["image_id"].tolist() == images["image_id"].tolist()
