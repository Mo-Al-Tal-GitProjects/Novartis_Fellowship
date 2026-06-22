from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cipher.config import ConfigContext, load_model_spec
from cipher.embeddings.generate import (
    EmbeddingRole,
    PreprocessingVariant,
    _canonical_digest,
    resolve_embedding_artifact,
)
from cipher.provenance import atomic_write_json, runtime_provenance, sha256_file

DEFAULT_MODEL_NAMES = ("clip", "dino", "plip", "uni")
ALIGNMENT_COLUMNS = ["embedding_index", "image_id", "relative_path", "role", "source", "sha256"]


def _ordered_ids_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_arrays(directory, expected_shape: tuple[int, int]) -> tuple[float, float]:
    raw = np.load(directory / "embeddings.raw.npy", mmap_mode="r", allow_pickle=False)
    normalized = np.load(directory / "embeddings.l2.npy", mmap_mode="r", allow_pickle=False)
    if raw.shape != expected_shape or normalized.shape != expected_shape:
        raise ValueError(
            f"embedding arrays in {directory} have shapes {raw.shape} and {normalized.shape}; "
            f"expected {expected_shape}"
        )
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise ValueError(f"non-finite values found in {directory}")
    norms = np.linalg.norm(normalized, axis=1)
    minimum = float(norms.min())
    maximum = float(norms.max())
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"L2-normalized vectors in {directory} are not unit length")
    return minimum, maximum


def assemble_embedding_set(
    context: ConfigContext,
    *,
    model_names: list[str],
    role: EmbeddingRole,
    device: str,
    limit: int | None = None,
    preprocessing_variant: PreprocessingVariant = "model_native",
) -> dict[str, Any]:
    names = sorted(set(model_names))
    if not names:
        raise ValueError("at least one model is required")
    if len(names) != len(model_names):
        raise ValueError("model names must be unique")

    canonical_rows: pd.DataFrame | None = None
    canonical_source_hash: str | None = None
    canonical_manifest_hash: str | None = None
    models: dict[str, Any] = {}
    for name in names:
        _, spec = load_model_spec(name, root=context.root)
        request, run = resolve_embedding_artifact(
            context,
            spec,
            role=role,
            device=device,
            limit=limit,
            preprocessing_variant=preprocessing_variant,
        )
        rows = pd.read_parquet(request.directory / "rows.parquet")
        current = rows.loc[:, ALIGNMENT_COLUMNS].reset_index(drop=True)
        if current["embedding_index"].tolist() != list(range(len(current))):
            raise ValueError(f"non-contiguous embedding indices in {request.directory}")
        if current["image_id"].tolist() != request.rows["image_id"].tolist():
            raise ValueError(f"stored rows do not match the current selection for {name}")
        if canonical_rows is None:
            canonical_rows = current
            canonical_source_hash = request.identity["cipher_source_sha256"]
            canonical_manifest_hash = request.identity["image_manifest_sha256"]
        else:
            try:
                pd.testing.assert_frame_equal(canonical_rows, current, check_dtype=False)
            except AssertionError as error:
                raise ValueError(f"row alignment differs for model {name}: {error}") from None
            if request.identity["cipher_source_sha256"] != canonical_source_hash:
                raise ValueError(f"CIPHER source hash differs for model {name}")
            if request.identity["image_manifest_sha256"] != canonical_manifest_hash:
                raise ValueError(f"image manifest hash differs for model {name}")
        norm_min, norm_max = _validate_arrays(
            request.directory,
            (len(rows), spec.expected_dimension),
        )
        models[name] = {
            "artifact_id": request.artifact_id,
            "artifact_directory": str(request.directory.relative_to(context.root)),
            "model_id": spec.model_id,
            "revision": spec.revision,
            "dimension": spec.expected_dimension,
            "embedding_output": spec.embedding_output,
            "batch_size": request.batch_size,
            "device": run["device"],
            "shape": run["shape"],
            "dtype": run["dtype"],
            "preprocessing_variant": run.get("preprocessing_variant", "model_native"),
            "file_sha256": run["file_sha256"],
            "l2_norm_range": [norm_min, norm_max],
        }

    assert canonical_rows is not None
    ordered_ids = canonical_rows["image_id"].astype(str).tolist()
    role_counts = canonical_rows["role"].value_counts().sort_index().to_dict()
    identity = {
        "schema_version": 1,
        "cipher_source_sha256": canonical_source_hash,
        "image_manifest_sha256": canonical_manifest_hash,
        "selection": {
            "role": role,
            "limit": limit,
            "preprocessing_variant": preprocessing_variant,
            "rows": len(canonical_rows),
            "role_counts": role_counts,
            "ordered_image_ids_sha256": _ordered_ids_sha256(ordered_ids),
        },
        "models": {
            name: {
                "artifact_id": payload["artifact_id"],
                "revision": payload["revision"],
                "dimension": payload["dimension"],
            }
            for name, payload in models.items()
        },
    }
    identity_sha256 = _canonical_digest(identity)
    set_id = identity_sha256[:16]
    directory = (
        context.resolve(context.project.paths.artifacts_root) / "embedding_sets" / set_id
    )
    manifest_path = directory / "manifest.json"
    existed = False
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            existed = existing.get("identity_sha256") == identity_sha256
        except (OSError, json.JSONDecodeError):
            existed = False
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "set_id": set_id,
        "identity": identity,
        "identity_sha256": identity_sha256,
        "aligned": True,
        "models": models,
        "runtime": runtime_provenance(),
    }
    if not existed:
        atomic_write_json(manifest_path, manifest)
    return {
        "status": "verified_existing" if existed else "assembled",
        "set_id": set_id,
        "manifest": str(manifest_path.relative_to(context.root)),
        "models": names,
        "rows": len(canonical_rows),
        "role_counts": role_counts,
        "ordered_image_ids_sha256": identity["selection"]["ordered_image_ids_sha256"],
        "aligned": True,
    }


def load_embedding_set(
    context: ConfigContext,
    *,
    set_id: str | None = None,
    required_models: list[str] | tuple[str, ...] = DEFAULT_MODEL_NAMES,
) -> tuple[Path, dict[str, Any]]:
    sets_root = context.resolve(context.project.paths.artifacts_root) / "embedding_sets"
    if set_id:
        candidates = [sets_root / set_id / "manifest.json"]
    else:
        candidates = sorted(sets_root.glob("*/manifest.json"))
    required = set(required_models)
    valid: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not manifest.get("aligned"):
            continue
        if not required.issubset(manifest.get("models", {})):
            continue
        if _canonical_digest(manifest["identity"]) != manifest.get("identity_sha256"):
            continue
        valid.append((path, manifest))
    if not valid:
        requested = f" {set_id!r}" if set_id else ""
        raise FileNotFoundError(f"no verified embedding set{requested} found")
    if len(valid) > 1:
        ids = [manifest["set_id"] for _, manifest in valid]
        raise ValueError(f"multiple embedding sets are available; select one of {ids}")
    path, manifest = valid[0]
    for name in required:
        payload = manifest["models"][name]
        directory = context.root / payload["artifact_directory"]
        for filename, expected in payload["file_sha256"].items():
            artifact = directory / filename
            if not artifact.is_file() or sha256_file(artifact) != expected:
                raise ValueError(f"embedding set file integrity failed: {artifact}")
    return path, manifest
