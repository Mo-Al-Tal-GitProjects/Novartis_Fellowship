from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cipher import __version__
from cipher.config import ConfigContext, load_model_spec
from cipher.embeddings.generate import _canonical_digest
from cipher.embeddings.set import load_embedding_set
from cipher.embeddings.store import atomic_write_parquet
from cipher.provenance import (
    atomic_write_json,
    runtime_provenance,
    sha256_file,
    sha256_python_tree,
)
from cipher.retrieval.contracts import get_metric_contract
from cipher.retrieval.exact import exact_rank_one


def _validate_cached_run(directory: Path, identity_sha256: str) -> dict[str, Any] | None:
    run_path = directory / "run.json"
    rankings_path = directory / "rankings.parquet"
    if not run_path.is_file() or not rankings_path.is_file():
        return None
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run["identity_sha256"] != identity_sha256:
            return None
        if sha256_file(rankings_path) != run["rankings_sha256"]:
            return None
        rankings = pd.read_parquet(rankings_path, columns=["query_id", "rank"])
        if len(rankings) != run["ranking_rows"]:
            return None
        if rankings.groupby("query_id")["rank"].min().ne(1).any():
            return None
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    return run


def run_retrieval(
    context: ConfigContext,
    *,
    model_name: str,
    metric_name: str,
    embedding_set_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    contract = get_metric_contract(metric_name)
    set_path, embedding_set = load_embedding_set(
        context,
        set_id=embedding_set_id,
        required_models=[model_name],
    )
    _, spec = load_model_spec(model_name, root=context.root)
    model_payload = embedding_set["models"][model_name]
    if model_payload["revision"] != spec.revision:
        raise ValueError(f"embedding set uses an unexpected {model_name} revision")
    artifact_directory = context.root / model_payload["artifact_directory"]
    rows = pd.read_parquet(artifact_directory / "rows.parquet")
    array_name = "embeddings.l2.npy" if contract.use_normalized_embeddings else "embeddings.raw.npy"
    embeddings = np.load(artifact_directory / array_name, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != tuple(model_payload["shape"]):
        raise ValueError(f"embedding shape mismatch for {model_name}")

    query_rows = rows.loc[rows["role"] == "query"].copy()
    gallery_rows = rows.loc[rows["role"] == "gallery"].copy()
    if query_rows.empty or gallery_rows.empty:
        raise ValueError("embedding set must contain query and gallery roles")
    if set(query_rows["image_id"]) & set(gallery_rows["image_id"]):
        raise ValueError("query/gallery image ID overlap detected")

    manifests_root = context.resolve(context.project.paths.manifests_root)
    relevance_path = manifests_root / "relevance.parquet"
    if not relevance_path.is_file():
        raise FileNotFoundError("relevance manifest not found; run `cipher data prepare`")
    relevance = pd.read_parquet(relevance_path)
    positives = {
        str(query_id): set(group["gallery_image_id"].astype(str))
        for query_id, group in relevance.groupby("query_id")
    }
    expected_query_images = dict(
        zip(relevance["query_image_id"], relevance["query_id"].astype(str), strict=False)
    )
    if set(query_rows["image_id"]) != set(expected_query_images):
        raise ValueError("embedding queries do not match relevance-manifest queries")

    identity = {
        "schema_version": 1,
        "cipher_version": __version__,
        "cipher_source_sha256": sha256_python_tree(context.root / "src" / "cipher"),
        "embedding_set_id": embedding_set["set_id"],
        "embedding_set_identity_sha256": embedding_set["identity_sha256"],
        "embedding_set_manifest_sha256": sha256_file(set_path),
        "relevance_manifest_sha256": sha256_file(relevance_path),
        "model": model_name,
        "model_revision": spec.revision,
        "metric": contract.name,
        "metric_contract": {
            "use_normalized_embeddings": contract.use_normalized_embeddings,
            "higher_is_better": contract.higher_is_better,
            "value_kind": contract.value_kind,
        },
        "query_role": "query",
        "gallery_role": "gallery",
        "self_match_policy": "exclude_by_role_and_image_id",
        "tie_breaker": "gallery_image_id_ascending",
    }
    identity_sha256 = _canonical_digest(identity)
    run_id = identity_sha256[:16]
    directory = (
        context.resolve(context.project.paths.artifacts_root)
        / "retrieval"
        / embedding_set["set_id"]
        / model_name
        / contract.name
        / run_id
    )
    if not force and (cached := _validate_cached_run(directory, identity_sha256)):
        return {
            "status": "cache_hit",
            "run_id": run_id,
            "model": model_name,
            "metric": contract.name,
            "queries": cached["queries"],
            "gallery": cached["gallery"],
            "ranking_rows": cached["ranking_rows"],
            "artifact_directory": str(directory.relative_to(context.root)),
        }

    gallery_indices = gallery_rows["embedding_index"].to_numpy(dtype=np.int64)
    gallery_ids = gallery_rows["image_id"].astype(str).to_numpy()
    gallery_vectors = embeddings[gallery_indices]
    ranking_records: list[dict[str, Any]] = []
    sorted_queries = query_rows.sort_values(
        "query_id", key=lambda values: values.astype(int)
    )
    for query in sorted_queries.itertuples():
        query_id = str(query.query_id)
        query_vector = embeddings[int(query.embedding_index)]
        order, values = exact_rank_one(
            query_vector,
            gallery_vectors,
            gallery_ids,
            query_image_id=str(query.image_id),
            contract=contract,
        )
        relevant_ids = positives[query_id]
        ranked_gallery = gallery_rows.iloc[order].reset_index(drop=True)
        for rank, (gallery, value) in enumerate(
            zip(ranked_gallery.itertuples(), values, strict=True), start=1
        ):
            ranking_records.append(
                {
                    "model": model_name,
                    "metric": contract.name,
                    "query_id": query_id,
                    "query_image_id": str(query.image_id),
                    "gallery_image_id": str(gallery.image_id),
                    "gallery_source": str(gallery.source),
                    "gallery_relative_path": str(gallery.relative_path),
                    "rank": rank,
                    "metric_value": float(value),
                    "value_kind": contract.value_kind,
                    "higher_is_better": contract.higher_is_better,
                    "is_relevant": str(gallery.image_id) in relevant_ids,
                    "curated_for_query_id": (
                        None if pd.isna(gallery.query_id) else str(gallery.query_id)
                    ),
                    "curated_rank": (
                        None if pd.isna(gallery.match_rank) else str(gallery.match_rank)
                    ),
                }
            )
    rankings = pd.DataFrame(ranking_records)
    expected_rows = len(query_rows) * len(gallery_rows)
    if len(rankings) != expected_rows:
        raise ValueError(f"expected {expected_rows} ranking rows, found {len(rankings)}")
    found_positives = int(rankings["is_relevant"].sum())
    if found_positives != len(relevance):
        raise ValueError(
            f"full rankings contain {found_positives} positives; expected {len(relevance)}"
        )

    directory.mkdir(parents=True, exist_ok=True)
    rankings_path = directory / "rankings.parquet"
    atomic_write_parquet(rankings_path, rankings)
    rankings_sha256 = sha256_file(rankings_path)
    run = {
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "identity": identity,
        "identity_sha256": identity_sha256,
        "queries": len(query_rows),
        "gallery": len(gallery_rows),
        "ranking_rows": len(rankings),
        "relevant_rows": found_positives,
        "rankings_sha256": rankings_sha256,
        "runtime": runtime_provenance(),
    }
    atomic_write_json(directory / "run.json", run)
    return {
        "status": "generated",
        "run_id": run_id,
        "model": model_name,
        "metric": contract.name,
        "queries": len(query_rows),
        "gallery": len(gallery_rows),
        "ranking_rows": len(rankings),
        "relevant_rows": found_positives,
        "artifact_directory": str(directory.relative_to(context.root)),
        "rankings_sha256": rankings_sha256,
    }
