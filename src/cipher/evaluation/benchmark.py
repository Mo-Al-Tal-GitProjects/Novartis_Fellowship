from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cipher import __version__
from cipher.config import ConfigContext, RetrievalConfig
from cipher.embeddings.generate import _canonical_digest
from cipher.embeddings.set import load_embedding_set
from cipher.embeddings.store import atomic_write_parquet
from cipher.evaluation.metrics import score_rankings
from cipher.provenance import (
    atomic_write_json,
    runtime_provenance,
    sha256_file,
    sha256_python_tree,
)
from cipher.retrieval.run import run_retrieval


def _cached_benchmark(directory: Path, identity_sha256: str) -> dict[str, Any] | None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity_sha256"] != identity_sha256:
            return None
        for filename, expected in manifest["file_sha256"].items():
            path = directory / filename
            if not path.is_file() or sha256_file(path) != expected:
                return None
    except (KeyError, OSError, json.JSONDecodeError):
        return None
    return manifest


def run_benchmark(
    context: ConfigContext,
    retrieval_config: RetrievalConfig,
    *,
    embedding_set_id: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    set_path, embedding_set = load_embedding_set(
        context,
        set_id=embedding_set_id,
        required_models=retrieval_config.models,
    )
    relevance_path = context.resolve(context.project.paths.manifests_root) / "relevance.parquet"
    relevance = pd.read_parquet(relevance_path)
    identity = {
        "schema_version": 1,
        "cipher_version": __version__,
        "cipher_source_sha256": sha256_python_tree(context.root / "src" / "cipher"),
        "embedding_set_id": embedding_set["set_id"],
        "embedding_set_identity_sha256": embedding_set["identity_sha256"],
        "embedding_set_manifest_sha256": sha256_file(set_path),
        "relevance_manifest_sha256": sha256_file(relevance_path),
        "retrieval_config": retrieval_config.model_dump(mode="json"),
    }
    identity_sha256 = _canonical_digest(identity)
    benchmark_id = identity_sha256[:16]
    directory = context.resolve(context.project.paths.artifacts_root) / "benchmarks" / benchmark_id
    if not force and (cached := _cached_benchmark(directory, identity_sha256)):
        return {
            "status": "cache_hit",
            "benchmark_id": benchmark_id,
            "embedding_set_id": embedding_set["set_id"],
            "models": retrieval_config.models,
            "metrics": retrieval_config.metrics,
            "summary": str((directory / "summary.json").relative_to(context.root)),
            "manifest": str((directory / "manifest.json").relative_to(context.root)),
            "retrieval_runs": cached["retrieval_runs"],
        }

    per_query_frames: list[pd.DataFrame] = []
    aggregate_rows: list[dict[str, Any]] = []
    retrieval_runs: dict[str, dict[str, str]] = {}
    for model_name in retrieval_config.models:
        retrieval_runs[model_name] = {}
        for metric_name in retrieval_config.metrics:
            report = run_retrieval(
                context,
                model_name=model_name,
                metric_name=metric_name,
                embedding_set_id=embedding_set["set_id"],
                force=force,
            )
            retrieval_runs[model_name][metric_name] = report["run_id"]
            rankings_path = context.root / report["artifact_directory"] / "rankings.parquet"
            rankings = pd.read_parquet(rankings_path)
            per_query, aggregate = score_rankings(
                rankings,
                relevance,
                retrieval_config.top_k,
            )
            per_query_frames.append(per_query)
            aggregate_rows.append(aggregate)

    per_query_results = pd.concat(per_query_frames, ignore_index=True)
    summary = pd.DataFrame(aggregate_rows)
    summary = summary.sort_values(
        ["metric", "mean_average_precision", "model"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary["map_rank_within_metric"] = (
        summary.groupby("metric").cumcount() + 1
    ).astype(int)
    directory.mkdir(parents=True, exist_ok=True)
    per_query_path = directory / "per_query.parquet"
    summary_path = directory / "summary.parquet"
    summary_json_path = directory / "summary.json"
    atomic_write_parquet(per_query_path, per_query_results)
    atomic_write_parquet(summary_path, summary)
    summary_records = json.loads(summary.to_json(orient="records"))
    atomic_write_json(
        summary_json_path,
        {
            "benchmark_id": benchmark_id,
            "embedding_set_id": embedding_set["set_id"],
            "rows": summary_records,
        },
    )
    file_sha256 = {
        per_query_path.name: sha256_file(per_query_path),
        summary_path.name: sha256_file(summary_path),
        summary_json_path.name: sha256_file(summary_json_path),
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark_id": benchmark_id,
        "identity": identity,
        "identity_sha256": identity_sha256,
        "retrieval_runs": retrieval_runs,
        "file_sha256": file_sha256,
        "summary_rows": len(summary),
        "per_query_rows": len(per_query_results),
        "runtime": runtime_provenance(),
    }
    atomic_write_json(directory / "manifest.json", manifest)
    report_k = 5 if 5 in retrieval_config.top_k else max(retrieval_config.top_k)
    primary_columns = [
        "model",
        "metric",
        "mean_average_precision",
        "mean_reciprocal_rank",
        f"precision_at_{report_k}",
        f"recall_at_{report_k}",
        f"top_{report_k}_accuracy",
        f"ndcg_at_{report_k}",
        "map_rank_within_metric",
    ]
    return {
        "status": "generated",
        "benchmark_id": benchmark_id,
        "embedding_set_id": embedding_set["set_id"],
        "models": retrieval_config.models,
        "metrics": retrieval_config.metrics,
        "summary": str(summary_json_path.relative_to(context.root)),
        "manifest": str((directory / "manifest.json").relative_to(context.root)),
        "retrieval_runs": retrieval_runs,
        "primary_results": summary.loc[:, primary_columns].to_dict("records"),
    }
