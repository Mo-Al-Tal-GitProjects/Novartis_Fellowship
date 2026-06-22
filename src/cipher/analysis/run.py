from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cipher import __version__
from cipher.config import ConfigContext
from cipher.embeddings.generate import _canonical_digest
from cipher.embeddings.store import atomic_write_parquet
from cipher.provenance import (
    atomic_write_json,
    runtime_provenance,
    sha256_file,
    sha256_python_tree,
)

ANALYSIS_FILES = (
    "uncertainty.parquet",
    "pairwise_differences.parquet",
    "model_agreement.parquet",
    "norm_summary.parquet",
    "norm_rank_correlations.parquet",
    "metric_sensitivity.parquet",
    "query_summary.parquet",
    "summary.json",
)


def load_benchmark(
    context: ConfigContext, benchmark_id: str | None
) -> tuple[Path, dict[str, Any]]:
    root = context.resolve(context.project.paths.artifacts_root) / "benchmarks"
    candidates = [root / benchmark_id / "manifest.json"] if benchmark_id else sorted(
        root.glob("*/manifest.json")
    )
    valid: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            directory = path.parent
            if any(
                not (directory / name).is_file()
                or sha256_file(directory / name) != expected
                for name, expected in manifest["file_sha256"].items()
            ):
                continue
        except (KeyError, OSError, json.JSONDecodeError):
            continue
        valid.append((path, manifest))
    if not valid:
        raise FileNotFoundError(f"verified benchmark {benchmark_id or ''!r} not found")
    if len(valid) != 1:
        ids = [manifest["benchmark_id"] for _, manifest in valid]
        raise ValueError(f"multiple benchmarks are available; select one of {ids}")
    return valid[0]


def _ranking_path(
    context: ConfigContext,
    benchmark: dict[str, Any],
    model: str,
    metric: str,
) -> Path:
    set_id = benchmark["identity"]["embedding_set_id"]
    run_id = benchmark["retrieval_runs"][model][metric]
    return (
        context.resolve(context.project.paths.artifacts_root)
        / "retrieval"
        / set_id
        / model
        / metric
        / run_id
        / "rankings.parquet"
    )


def _bootstrap_tables(
    per_query: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_ids = sorted(per_query["query_id"].astype(str).unique(), key=int)
    indices = np.random.default_rng(seed).integers(
        0, len(query_ids), size=(iterations, len(query_ids))
    )
    values: dict[tuple[str, str], np.ndarray] = {}
    uncertainty: list[dict[str, Any]] = []
    for (model, metric), group in per_query.groupby(["model", "metric"], sort=True):
        aligned = group.set_index("query_id").loc[query_ids, "average_precision"]
        observed = aligned.to_numpy(dtype=np.float64)
        values[(model, metric)] = observed
        samples = observed[indices].mean(axis=1)
        uncertainty.append(
            {
                "model": model,
                "metric": metric,
                "queries": len(query_ids),
                "map": float(observed.mean()),
                "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_iterations": iterations,
                "seed": seed,
            }
        )

    pairwise: list[dict[str, Any]] = []
    models = sorted(per_query["model"].unique())
    metrics = sorted(per_query["metric"].unique())
    for metric in metrics:
        for model_a, model_b in itertools.combinations(models, 2):
            differences = values[(model_a, metric)] - values[(model_b, metric)]
            samples = differences[indices].mean(axis=1)
            pairwise.append(
                {
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_ap_difference_a_minus_b": float(differences.mean()),
                    "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                    "bootstrap_probability_a_greater_b": float((samples > 0).mean()),
                    "query_wins_a": int((differences > 0).sum()),
                    "query_ties": int((differences == 0).sum()),
                    "query_wins_b": int((differences < 0).sum()),
                    "queries": len(query_ids),
                }
            )
    return pd.DataFrame(uncertainty), pd.DataFrame(pairwise)


def _agreement_table(
    context: ConfigContext,
    benchmark: dict[str, Any],
    models: list[str],
    top_k: tuple[int, ...],
) -> pd.DataFrame:
    rankings = {
        model: pd.read_parquet(
            _ranking_path(context, benchmark, model, "cosine"),
            columns=["query_id", "gallery_image_id", "rank"],
        )
        for model in models
    }
    records: list[dict[str, Any]] = []
    query_ids = sorted(rankings[models[0]]["query_id"].astype(str).unique(), key=int)
    for model_a, model_b in itertools.combinations(models, 2):
        for cutoff in top_k:
            scores = []
            for query_id in query_ids:
                a = set(
                    rankings[model_a].loc[
                        (rankings[model_a]["query_id"] == query_id)
                        & (rankings[model_a]["rank"] <= cutoff),
                        "gallery_image_id",
                    ]
                )
                b = set(
                    rankings[model_b].loc[
                        (rankings[model_b]["query_id"] == query_id)
                        & (rankings[model_b]["rank"] <= cutoff),
                        "gallery_image_id",
                    ]
                )
                scores.append(len(a & b) / len(a | b))
            records.append(
                {
                    "metric": "cosine",
                    "model_a": model_a,
                    "model_b": model_b,
                    "top_k": cutoff,
                    "mean_jaccard": float(np.mean(scores)),
                    "median_jaccard": float(np.median(scores)),
                    "minimum_jaccard": float(np.min(scores)),
                    "maximum_jaccard": float(np.max(scores)),
                }
            )
    return pd.DataFrame(records)


def _norm_tables(
    context: ConfigContext,
    benchmark: dict[str, Any],
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_id = benchmark["identity"]["embedding_set_id"]
    set_manifest = json.loads(
        (
            context.resolve(context.project.paths.artifacts_root)
            / "embedding_sets"
            / set_id
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    summaries: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    for model in models:
        directory = context.root / set_manifest["models"][model]["artifact_directory"]
        rows = pd.read_parquet(directory / "rows.parquet")
        embeddings = np.load(directory / "embeddings.raw.npy", mmap_mode="r")
        norms = np.linalg.norm(embeddings, axis=1).astype(np.float64)
        norm_frame = pd.DataFrame({"image_id": rows["image_id"].astype(str), "norm": norms})
        for role, indices in rows.groupby("role").groups.items():
            role_norms = norms[np.asarray(list(indices), dtype=np.int64)]
            mean = float(role_norms.mean())
            summaries.append(
                {
                    "model": model,
                    "role": role,
                    "count": len(role_norms),
                    "mean_norm": mean,
                    "std_norm": float(role_norms.std(ddof=1)),
                    "coefficient_of_variation": float(role_norms.std(ddof=1) / mean),
                    "minimum_norm": float(role_norms.min()),
                    "median_norm": float(np.median(role_norms)),
                    "maximum_norm": float(role_norms.max()),
                }
            )
        ranking = pd.read_parquet(
            _ranking_path(context, benchmark, model, "inner_product"),
            columns=["query_id", "gallery_image_id", "rank"],
        ).merge(norm_frame, left_on="gallery_image_id", right_on="image_id", how="left")
        for query_id, group in ranking.groupby("query_id", sort=False):
            norm_ranks = group["norm"].rank(method="average").to_numpy(dtype=np.float64)
            rank_values = group["rank"].to_numpy(dtype=np.float64)
            correlation = float(np.corrcoef(rank_values, norm_ranks)[0, 1])
            correlations.append(
                {
                    "model": model,
                    "metric": "inner_product",
                    "query_id": str(query_id),
                    "spearman_retrieval_rank_vs_gallery_norm": correlation,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(correlations)


def _metric_sensitivity(per_query: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    records = []
    for model in models:
        frame = per_query.loc[per_query["model"] == model].pivot(
            index="query_id", columns="metric", values="average_precision"
        )
        for alternative in ("inner_product", "euclidean"):
            delta = frame[alternative] - frame["cosine"]
            records.append(
                {
                    "model": model,
                    "baseline_metric": "cosine",
                    "alternative_metric": alternative,
                    "mean_ap_difference": float(delta.mean()),
                    "median_ap_difference": float(delta.median()),
                    "queries_improved": int((delta > 0).sum()),
                    "queries_unchanged": int((delta == 0).sum()),
                    "queries_worsened": int((delta < 0).sum()),
                }
            )
    return pd.DataFrame(records)


def _query_summary(
    context: ConfigContext,
    per_query: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    cosine = per_query.loc[per_query["metric"] == "cosine"].copy()
    hit_cutoffs = sorted(
        int(column.removeprefix("hit_at_"))
        for column in cosine.columns
        if column.startswith("hit_at_")
    )
    diagnostic_cutoff = 5 if 5 in hit_cutoffs else max(hit_cutoffs)
    hit_column = f"hit_at_{diagnostic_cutoff}"
    records = []
    images = pd.read_parquet(
        context.resolve(context.project.paths.manifests_root) / "images.parquet"
    )
    captions = images.loc[images["role"] == "query"].set_index("query_id")["caption"]
    for query_id, group in cosine.groupby("query_id", sort=False):
        ordered = group.sort_values("average_precision", ascending=False)
        records.append(
            {
                "query_id": str(query_id),
                "query_image_id": str(group["query_image_id"].iloc[0]),
                "caption": str(captions.loc[str(query_id)]),
                "positive_count": int(group["positive_count"].iloc[0]),
                "mean_cosine_ap_across_models": float(group["average_precision"].mean()),
                "minimum_cosine_ap": float(group["average_precision"].min()),
                "maximum_cosine_ap": float(group["average_precision"].max()),
                "best_model": str(ordered["model"].iloc[0]),
                "best_model_ap": float(ordered["average_precision"].iloc[0]),
                "diagnostic_cutoff": diagnostic_cutoff,
                "models_with_hit_at_cutoff": int(group[hit_column].sum()),
                "all_models_hit_at_cutoff": bool(group[hit_column].all()),
                **{
                    f"{model}_cosine_ap": float(
                        group.loc[group["model"] == model, "average_precision"].iloc[0]
                    )
                    for model in models
                },
            }
        )
    return pd.DataFrame(records).sort_values("mean_cosine_ap_across_models")


def run_analysis(
    context: ConfigContext,
    *,
    benchmark_id: str | None,
    bootstrap_iterations: int = 10_000,
    seed: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    benchmark_path, benchmark = load_benchmark(context, benchmark_id)
    actual_seed = context.project.seed if seed is None else seed
    identity = {
        "schema_version": 1,
        "cipher_version": __version__,
        "cipher_source_sha256": sha256_python_tree(context.root / "src" / "cipher"),
        "benchmark_id": benchmark["benchmark_id"],
        "benchmark_manifest_sha256": sha256_file(benchmark_path),
        "bootstrap_iterations": bootstrap_iterations,
        "seed": actual_seed,
        "agreement_top_k": [5, 10, 20, 50],
    }
    identity_sha256 = _canonical_digest(identity)
    analysis_id = identity_sha256[:16]
    directory = context.resolve(context.project.paths.artifacts_root) / "analyses" / analysis_id
    manifest_path = directory / "manifest.json"
    if not force and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["identity_sha256"] == identity_sha256 and all(
                sha256_file(directory / name) == expected
                for name, expected in manifest["file_sha256"].items()
            ):
                return {
                    "status": "cache_hit",
                    "analysis_id": analysis_id,
                    "benchmark_id": benchmark["benchmark_id"],
                    "artifact_directory": str(directory.relative_to(context.root)),
                }
        except (KeyError, OSError, json.JSONDecodeError):
            pass

    benchmark_directory = benchmark_path.parent
    per_query = pd.read_parquet(benchmark_directory / "per_query.parquet")
    summary = pd.read_parquet(benchmark_directory / "summary.parquet")
    models = sorted(summary["model"].unique())
    uncertainty, pairwise = _bootstrap_tables(
        per_query, iterations=bootstrap_iterations, seed=actual_seed
    )
    agreement = _agreement_table(context, benchmark, models, (5, 10, 20, 50))
    norm_summary, norm_correlations = _norm_tables(context, benchmark, models)
    sensitivity = _metric_sensitivity(per_query, models)
    queries = _query_summary(context, per_query, models)

    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "uncertainty.parquet": uncertainty,
        "pairwise_differences.parquet": pairwise,
        "model_agreement.parquet": agreement,
        "norm_summary.parquet": norm_summary,
        "norm_rank_correlations.parquet": norm_correlations,
        "metric_sensitivity.parquet": sensitivity,
        "query_summary.parquet": queries,
    }
    for name, frame in tables.items():
        atomic_write_parquet(directory / name, frame)
    cosine_uncertainty = uncertainty.loc[uncertainty["metric"] == "cosine"].sort_values(
        "map", ascending=False
    )
    summary_payload = {
        "analysis_id": analysis_id,
        "benchmark_id": benchmark["benchmark_id"],
        "cosine_map_with_bootstrap_ci": json.loads(
            cosine_uncertainty.to_json(orient="records")
        ),
        "hardest_query_by_mean_cosine_ap": str(queries.iloc[0]["query_id"]),
        "easiest_query_by_mean_cosine_ap": str(queries.iloc[-1]["query_id"]),
        "interpretation_boundary": (
            "Paired query bootstrap intervals are descriptive with 11 queries; "
            "unjudged images remain operational negatives."
        ),
    }
    atomic_write_json(directory / "summary.json", summary_payload)
    file_sha256 = {name: sha256_file(directory / name) for name in ANALYSIS_FILES}
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "analysis_id": analysis_id,
        "identity": identity,
        "identity_sha256": identity_sha256,
        "file_sha256": file_sha256,
        "runtime": runtime_provenance(),
    }
    atomic_write_json(manifest_path, manifest)
    return {
        "status": "generated",
        "analysis_id": analysis_id,
        "benchmark_id": benchmark["benchmark_id"],
        "artifact_directory": str(directory.relative_to(context.root)),
        "hardest_query": summary_payload["hardest_query_by_mean_cosine_ap"],
        "easiest_query": summary_payload["easiest_query_by_mean_cosine_ap"],
        "cosine_map_with_bootstrap_ci": summary_payload["cosine_map_with_bootstrap_ci"],
    }
