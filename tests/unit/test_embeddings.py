from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cipher.config import ModelSpec, load_model_spec
from cipher.data.manifest import build_image_manifest
from cipher.embeddings.generate import generate_embeddings
from cipher.embeddings.set import assemble_embedding_set
from cipher.provenance import sha256_file


class FakeEncoder:
    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.spec = spec
        self.device = device

    def encode(self, images) -> np.ndarray:
        rows = []
        for image in images:
            mean = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
            rows.append([mean[0] + 1, mean[1] + 2, mean[2] + 3, mean.sum() + 4])
        return np.asarray(rows, dtype=np.float32)


class DimensionAwareFakeEncoder:
    def __init__(self, spec: ModelSpec, device: str) -> None:
        self.spec = spec
        self.device = device

    def encode(self, images) -> np.ndarray:
        rows = []
        for image in images:
            offset = float(np.asarray(image, dtype=np.float32).mean()) + 1.0
            rows.append(np.arange(self.spec.expected_dimension, dtype=np.float32) + offset)
        return np.asarray(rows, dtype=np.float32)


def _spec() -> ModelSpec:
    return ModelSpec(
        version=1,
        name="clip",
        adapter="clip",
        model_id="openai/clip-vit-base-patch32",
        revision="a" * 40,
        expected_dimension=4,
        batch_size=1,
        dtype="float32",
        preprocessing="model_native",
        embedding_output="image_projection",
        trust_remote_code=False,
    )


def test_embedding_artifact_is_ordered_normalized_and_cached(project_fixture) -> None:
    build_image_manifest(project_fixture)
    first = generate_embeddings(
        project_fixture,
        _spec(),
        role="query",
        device="cpu",
        encoder_factory=FakeEncoder,
    )
    assert first["status"] == "generated"
    directory = project_fixture.root / first["artifact_directory"]
    raw = np.load(directory / "embeddings.raw.npy", allow_pickle=False)
    normalized = np.load(directory / "embeddings.l2.npy", allow_pickle=False)
    rows = pd.read_parquet(directory / "rows.parquet")
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))

    assert raw.shape == (2, 4)
    assert raw.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-6)
    assert rows["image_id"].tolist() == sorted(rows["image_id"])
    assert rows["embedding_index"].tolist() == [0, 1]
    assert run["identity"]["selection"]["role"] == "query"
    assert run["file_sha256"]["embeddings.raw.npy"] == sha256_file(
        directory / "embeddings.raw.npy"
    )

    def fail_if_created(spec, device):
        raise AssertionError("verified cache should be used before constructing the encoder")

    second = generate_embeddings(
        project_fixture,
        _spec(),
        role="query",
        device="cpu",
        encoder_factory=fail_if_created,
    )
    assert second["status"] == "cache_hit"
    assert second["artifact_id"] == first["artifact_id"]


def test_limit_changes_artifact_identity(project_fixture) -> None:
    build_image_manifest(project_fixture)
    full = generate_embeddings(
        project_fixture,
        _spec(),
        role="gallery",
        device="cpu",
        encoder_factory=FakeEncoder,
    )
    limited = generate_embeddings(
        project_fixture,
        _spec(),
        role="gallery",
        device="cpu",
        limit=2,
        encoder_factory=FakeEncoder,
    )
    assert full["artifact_id"] != limited["artifact_id"]
    assert full["rows"] == 4
    assert limited["rows"] == 2


def test_preprocessing_variant_changes_artifact_identity(project_fixture) -> None:
    build_image_manifest(project_fixture)
    native = generate_embeddings(
        project_fixture,
        _spec(),
        role="query",
        device="cpu",
        encoder_factory=FakeEncoder,
    )
    grayscale = generate_embeddings(
        project_fixture,
        _spec(),
        role="query",
        device="cpu",
        preprocessing_variant="grayscale",
        encoder_factory=FakeEncoder,
    )
    assert native["artifact_id"] != grayscale["artifact_id"]
    assert grayscale["preprocessing_variant"] == "grayscale"


def test_embedding_set_verifies_cross_model_row_alignment(project_fixture) -> None:
    build_image_manifest(project_fixture)
    names = ["clip", "dino", "plip", "uni"]
    for name in names:
        _, spec = load_model_spec(name, root=project_fixture.root)
        generate_embeddings(
            project_fixture,
            spec,
            role="all",
            device="cpu",
            encoder_factory=DimensionAwareFakeEncoder,
        )

    first = assemble_embedding_set(
        project_fixture,
        model_names=names,
        role="all",
        device="cpu",
    )
    assert first["status"] == "assembled"
    assert first["aligned"] is True
    assert first["rows"] == 6
    assert first["role_counts"] == {"gallery": 4, "query": 2}
    manifest = json.loads((project_fixture.root / first["manifest"]).read_text(encoding="utf-8"))
    assert set(manifest["models"]) == set(names)
    assert manifest["models"]["dino"]["shape"] == [6, 6]
    assert manifest["models"]["uni"]["shape"] == [6, 8]

    second = assemble_embedding_set(
        project_fixture,
        model_names=names,
        role="all",
        device="cpu",
    )
    assert second["status"] == "verified_existing"
    assert second["set_id"] == first["set_id"]
