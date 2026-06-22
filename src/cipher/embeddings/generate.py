from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image

from cipher import __version__
from cipher.config import ConfigContext, ModelSpec
from cipher.embeddings.store import (
    atomic_write_npy,
    atomic_write_parquet,
    validate_cached_artifact,
)
from cipher.models.base import ImageEncoder
from cipher.provenance import (
    atomic_write_json,
    runtime_provenance,
    sha256_file,
    sha256_python_tree,
)

EmbeddingRole = Literal["query", "gallery", "all"]
PreprocessingVariant = Literal["model_native", "grayscale", "center_square"]
EncoderFactory = Callable[[ModelSpec, str], ImageEncoder]


@dataclass(frozen=True)
class PreparedEmbeddingRequest:
    manifest_path: Path
    rows: pd.DataFrame
    batch_size: int
    identity: dict[str, Any]
    identity_sha256: str
    artifact_id: str
    directory: Path


def _canonical_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _select_rows(frame: pd.DataFrame, role: EmbeddingRole, limit: int | None) -> pd.DataFrame:
    roles = ["query", "gallery"] if role == "all" else [role]
    selected = frame.loc[frame["role"].isin(roles)].sort_values("image_id", kind="stable")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected.head(limit)
    if selected.empty:
        raise ValueError(f"no active manifest rows selected for role {role!r}")
    return selected.reset_index(drop=True)


def _load_batch(
    rows: pd.DataFrame,
    root: Path,
    preprocessing_variant: PreprocessingVariant,
) -> list[Image.Image]:
    images: list[Image.Image] = []
    for relative_path in rows["relative_path"]:
        path = root / str(relative_path)
        with Image.open(path) as image:
            prepared = image.convert("RGB")
            if preprocessing_variant == "grayscale":
                prepared = prepared.convert("L").convert("RGB")
            elif preprocessing_variant == "center_square":
                width, height = prepared.size
                side = min(width, height)
                left = (width - side) // 2
                top = (height - side) // 2
                prepared = prepared.crop((left, top, left + side, top + side))
            images.append(prepared.copy())
    return images


def prepare_embedding_request(
    context: ConfigContext,
    spec: ModelSpec,
    *,
    role: EmbeddingRole,
    device: str,
    batch_size: int | None = None,
    limit: int | None = None,
    preprocessing_variant: PreprocessingVariant = "model_native",
) -> PreparedEmbeddingRequest:
    manifest_path = context.resolve(context.project.paths.manifests_root) / "images.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError("image manifest not found; run `cipher data prepare` first")
    rows = _select_rows(pd.read_parquet(manifest_path), role, limit)
    actual_batch_size = batch_size or spec.batch_size
    if actual_batch_size < 1:
        raise ValueError("batch size must be positive")
    identity = {
        "schema_version": 1,
        "cipher_version": __version__,
        "cipher_source_sha256": sha256_python_tree(context.root / "src" / "cipher"),
        "image_manifest_sha256": sha256_file(manifest_path),
        "model": spec.model_dump(mode="json"),
        "execution": {
            "device": device,
            "batch_size": actual_batch_size,
            "preprocessing_variant": preprocessing_variant,
        },
        "selection": {
            "role": role,
            "limit": limit,
            "image_ids": rows["image_id"].tolist(),
        },
    }
    identity_sha256 = _canonical_digest(identity)
    artifact_id = identity_sha256[:16]
    artifact_root = context.resolve(context.project.paths.artifacts_root)
    return PreparedEmbeddingRequest(
        manifest_path=manifest_path,
        rows=rows,
        batch_size=actual_batch_size,
        identity=identity,
        identity_sha256=identity_sha256,
        artifact_id=artifact_id,
        directory=artifact_root / "embeddings" / spec.name / artifact_id,
    )


def resolve_embedding_artifact(
    context: ConfigContext,
    spec: ModelSpec,
    *,
    role: EmbeddingRole,
    device: str,
    batch_size: int | None = None,
    limit: int | None = None,
    preprocessing_variant: PreprocessingVariant = "model_native",
) -> tuple[PreparedEmbeddingRequest, dict[str, Any]]:
    request = prepare_embedding_request(
        context,
        spec,
        role=role,
        device=device,
        batch_size=batch_size,
        limit=limit,
        preprocessing_variant=preprocessing_variant,
    )
    cached = validate_cached_artifact(request.directory, request.identity_sha256)
    if cached is None:
        raise FileNotFoundError(
            f"verified {spec.name} artifact not found for role {role!r}; run "
            f"`cipher embeddings generate --model {spec.name} --role {role} --device {device}`"
        )
    return request, cached


def generate_embeddings(
    context: ConfigContext,
    spec: ModelSpec,
    *,
    role: EmbeddingRole,
    device: str,
    batch_size: int | None = None,
    limit: int | None = None,
    force: bool = False,
    preprocessing_variant: PreprocessingVariant = "model_native",
    encoder_factory: EncoderFactory | None = None,
) -> dict[str, Any]:
    request = prepare_embedding_request(
        context,
        spec,
        role=role,
        device=device,
        batch_size=batch_size,
        limit=limit,
        preprocessing_variant=preprocessing_variant,
    )
    rows = request.rows
    directory = request.directory
    if not force and (
        cached := validate_cached_artifact(directory, request.identity_sha256)
    ):
        return {
            "status": "cache_hit",
            "artifact_id": request.artifact_id,
            "artifact_directory": str(directory.relative_to(context.root)),
            "rows": cached["shape"][0],
            "dimension": cached["shape"][1],
            "model": spec.name,
            "device": cached["device"],
        }

    if encoder_factory is None:
        from cipher.models.registry import create_encoder

        encoder_factory = create_encoder
    encoder = encoder_factory(spec, device)
    batches: list[np.ndarray] = []
    for start in range(0, len(rows), request.batch_size):
        images = _load_batch(
            rows.iloc[start : start + request.batch_size],
            context.root,
            preprocessing_variant,
        )
        batch = np.asarray(encoder.encode(images), dtype=np.float32)
        expected = (len(images), spec.expected_dimension)
        if batch.shape != expected:
            raise ValueError(f"encoder returned shape {batch.shape}; expected {expected}")
        if not np.isfinite(batch).all():
            raise ValueError("encoder returned non-finite embeddings")
        batches.append(batch)
    raw = np.ascontiguousarray(np.concatenate(batches), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encoder returned a zero-length embedding")
    normalized = np.ascontiguousarray(raw / norms, dtype=np.float32)

    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "embeddings.raw.npy"
    normalized_path = directory / "embeddings.l2.npy"
    rows_path = directory / "rows.parquet"
    row_columns = [
        "image_id",
        "relative_path",
        "role",
        "source",
        "sha256",
        "query_id",
        "match_rank",
    ]
    artifact_rows = rows.loc[:, row_columns].copy()
    artifact_rows.insert(0, "embedding_index", range(len(artifact_rows)))
    atomic_write_npy(raw_path, raw)
    atomic_write_npy(normalized_path, normalized)
    atomic_write_parquet(rows_path, artifact_rows)
    file_hashes = {
        raw_path.name: sha256_file(raw_path),
        normalized_path.name: sha256_file(normalized_path),
        rows_path.name: sha256_file(rows_path),
    }
    run = {
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_id": request.artifact_id,
        "identity": request.identity,
        "identity_sha256": request.identity_sha256,
        "model": spec.name,
        "device": encoder.device,
        "batch_size": request.batch_size,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "normalization": "raw_and_l2",
        "preprocessing_variant": preprocessing_variant,
        "file_sha256": file_hashes,
        "runtime": runtime_provenance(),
    }
    atomic_write_json(directory / "run.json", run)
    return {
        "status": "generated",
        "artifact_id": request.artifact_id,
        "artifact_directory": str(directory.relative_to(context.root)),
        "rows": len(rows),
        "dimension": spec.expected_dimension,
        "model": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "device": encoder.device,
        "batch_size": request.batch_size,
        "preprocessing_variant": preprocessing_variant,
        "file_sha256": file_hashes,
    }
