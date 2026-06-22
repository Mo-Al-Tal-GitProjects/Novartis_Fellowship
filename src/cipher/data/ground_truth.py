from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cipher.config import ConfigContext
from cipher.provenance import atomic_write_json, runtime_provenance, sha256_file


def _policy_notes(context: ConfigContext) -> dict[str, tuple[str, str]]:
    notes: dict[str, tuple[str, str]] = {}
    for decision in context.policy.deduplications:
        notes[decision.canonical.lower()] = ("deduplicated_canonical", decision.reason)
    for anomaly in context.policy.retained_anomalies:
        is_extra_rank = any("_06." in name.lower() for name in anomaly.files)
        status = "retained_extra_rank" if is_extra_rank else "retained_repeated_rank"
        for filename in anomaly.files:
            notes[filename.lower()] = (status, anomaly.reason)
    return notes


def build_relevance_manifest(
    context: ConfigContext, images: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifests_root = context.resolve(context.project.paths.manifests_root)
    images_path = manifests_root / "images.parquet"
    if images is None:
        if not images_path.is_file():
            raise FileNotFoundError("image manifest not found; run `cipher data prepare`")
        images = pd.read_parquet(images_path)

    query_rows = images.loc[images["role"] == "query"].copy()
    expected_queries = set(context.policy.queries)
    observed_queries = set(query_rows["query_id"].dropna().astype(str))
    if observed_queries != expected_queries:
        raise ValueError(
            "query mapping mismatch: "
            f"expected {sorted(expected_queries)}, found {sorted(observed_queries)}"
        )

    query_image_ids = dict(
        zip(query_rows["query_id"].astype(str), query_rows["image_id"], strict=True)
    )
    match_rows = images.loc[
        (images["source"] == "curated_match") & (images["role"] == "gallery")
    ].copy()
    notes = _policy_notes(context)
    rows: list[dict[str, Any]] = []
    for record in match_rows.sort_values(["query_id", "match_rank", "filename"]).to_dict("records"):
        query_id = str(record["query_id"])
        filename = str(record["filename"])
        status, note = notes.get(filename.lower(), ("observed_curated_positive", ""))
        rows.append(
            {
                "judgment_id": f"{query_id}:{record['match_rank']}:{filename.lower()}",
                "query_id": query_id,
                "query_image_id": query_image_ids[query_id],
                "gallery_image_id": record["image_id"],
                "gallery_filename": filename,
                "original_rank": str(record["match_rank"]),
                "relevance": 1,
                "judgment_source": "pathologist_curated_match_folder",
                "adjudication_status": status,
                "notes": note,
            }
        )

    relevance = pd.DataFrame(rows)
    if len(relevance) != context.policy.expected_positive_images:
        raise ValueError(
            f"expected {context.policy.expected_positive_images} distinct positives, "
            f"found {len(relevance)}"
        )
    if relevance[["query_image_id", "gallery_image_id"]].duplicated().any():
        raise ValueError("duplicate query/gallery relevance pair")

    relevance_path = manifests_root / "relevance.parquet"
    relevance.to_parquet(relevance_path, index=False)
    per_query = {
        str(query_id): int(count)
        for query_id, count in relevance.groupby("query_id").size().items()
    }
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "manifest": relevance_path.relative_to(context.root).as_posix(),
        "manifest_sha256": sha256_file(relevance_path),
        "queries": len(per_query),
        "positive_images": len(relevance),
        "positives_per_query": per_query,
        "adjudication_counts": dict(Counter(relevance["adjudication_status"])),
        "runtime": runtime_provenance(),
    }
    atomic_write_json(manifests_root / "relevance.summary.json", summary)
    return relevance, summary


def validate_ground_truth(context: ConfigContext) -> dict[str, Any]:
    manifests_root = context.resolve(context.project.paths.manifests_root)
    images_path = manifests_root / "images.parquet"
    relevance_path = manifests_root / "relevance.parquet"
    if not images_path.is_file() or not relevance_path.is_file():
        raise FileNotFoundError(
            "image and relevance manifests are required; run `cipher data prepare`"
        )

    images = pd.read_parquet(images_path)
    relevance = pd.read_parquet(relevance_path)
    errors: list[str] = []
    warnings: list[str] = []

    expected_queries = set(context.policy.queries)
    query_rows = images.loc[images["role"] == "query"]
    observed_queries = set(query_rows["query_id"].dropna().astype(str))
    if observed_queries != expected_queries:
        errors.append("image manifest query IDs do not match the policy")
    if len(query_rows) != len(expected_queries):
        errors.append(f"expected {len(expected_queries)} query images, found {len(query_rows)}")
    if len(relevance) != context.policy.expected_positive_images:
        errors.append(
            f"expected {context.policy.expected_positive_images} positives, found {len(relevance)}"
        )
    if relevance[["query_image_id", "gallery_image_id"]].duplicated().any():
        errors.append("duplicate query/gallery relevance pairs")
    if (relevance["query_image_id"] == relevance["gallery_image_id"]).any():
        errors.append("self-relevance judgment detected")

    image_roles = images.set_index("image_id")["role"].to_dict()
    missing_images = set(relevance["gallery_image_id"]) - set(image_roles)
    if missing_images:
        errors.append(f"relevance rows reference missing gallery IDs: {sorted(missing_images)}")
    invalid_gallery = [
        image_id
        for image_id in relevance["gallery_image_id"]
        if image_roles.get(image_id) != "gallery"
    ]
    if invalid_gallery:
        invalid_gallery_ids = sorted(set(invalid_gallery))
        errors.append(f"relevance rows reference non-gallery images: {invalid_gallery_ids}")

    accepted_statuses = ["valid", "excluded_duplicate", "excluded_exact_duplicate"]
    invalid_rows = images.loc[~images["validation_status"].isin(accepted_statuses)]
    if not invalid_rows.empty:
        errors.append(f"{len(invalid_rows)} image rows failed validation")

    exact_hash_groups = (
        images.loc[images["role"] != "excluded"].groupby("sha256")["image_id"].apply(list)
    )
    exact_duplicates = [group for group in exact_hash_groups if len(group) > 1]
    if exact_duplicates:
        errors.append(f"unresolved exact duplicate groups: {exact_duplicates}")

    per_query = {
        str(query_id): int(count)
        for query_id, count in relevance.groupby("query_id").size().items()
    }
    for query_id in sorted(expected_queries, key=int):
        if per_query.get(query_id, 0) == 0:
            errors.append(f"query {query_id} has no positives")
        observed_ranks = sorted(
            set(relevance.loc[relevance["query_id"] == query_id, "original_rank"].astype(str)),
            key=int,
        )
        expected_ranks = {f"{rank:02d}" for rank in range(1, 6)}
        missing_ranks = sorted(expected_ranks - set(observed_ranks), key=int)
        extra_ranks = sorted(set(observed_ranks) - expected_ranks, key=int)
        if missing_ranks:
            warnings.append(
                f"query {query_id}: missing curated ranks {missing_ranks}; not fabricated"
            )
        if extra_ranks:
            warnings.append(f"query {query_id}: retained extra curated ranks {extra_ranks}")

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "image_manifest_sha256": sha256_file(images_path),
        "relevance_manifest_sha256": sha256_file(relevance_path),
        "query_images": len(query_rows),
        "positive_images": len(relevance),
        "positives_per_query": per_query,
        "excluded_duplicate_files": images.loc[
            images["validation_status"].str.startswith("excluded_"), "filename"
        ].tolist(),
        "unjudged_gallery_policy": context.policy.unjudged_gallery_policy,
        "runtime": runtime_provenance(),
    }
    atomic_write_json(manifests_root / "ground_truth.validation.json", report)
    if errors:
        raise ValueError("ground-truth validation failed: " + "; ".join(errors))
    return report


def positive_counts(relevance: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for query_id in relevance["query_id"].astype(str):
        counts[query_id] += 1
    return dict(counts)
