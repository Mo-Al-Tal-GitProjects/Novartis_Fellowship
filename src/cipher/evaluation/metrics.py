from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _average_precision(
    relevant_ranks: np.ndarray,
    positive_count: int,
    cutoff: int | None,
) -> float:
    ranks = relevant_ranks if cutoff is None else relevant_ranks[relevant_ranks <= cutoff]
    if len(ranks) == 0:
        return 0.0
    precisions = np.arange(1, len(ranks) + 1, dtype=np.float64) / ranks
    return float(precisions.sum() / positive_count)


def _ndcg_at_k(relevant_ranks: np.ndarray, positive_count: int, cutoff: int) -> float:
    observed = relevant_ranks[relevant_ranks <= cutoff]
    dcg = sum(1.0 / math.log2(int(rank) + 1) for rank in observed)
    ideal_count = min(positive_count, cutoff)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return float(dcg / idcg) if idcg else 0.0


def score_rankings(
    rankings: pd.DataFrame,
    relevance: pd.DataFrame,
    top_k: list[int],
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    models = rankings["model"].unique().tolist()
    metrics = rankings["metric"].unique().tolist()
    if len(models) != 1 or len(metrics) != 1:
        raise ValueError("score_rankings expects exactly one model and metric")
    positive_counts = relevance.groupby("query_id").size().astype(int).to_dict()
    query_rows: list[dict] = []
    for query_id, group in rankings.groupby("query_id", sort=False):
        query_id = str(query_id)
        ordered = group.sort_values("rank")
        ranks = ordered["rank"].to_numpy(dtype=np.int64)
        if not np.array_equal(ranks, np.arange(1, len(ordered) + 1)):
            raise ValueError(f"ranks are not contiguous for query {query_id}")
        positive_count = int(positive_counts[query_id])
        relevant_ranks = ranks[ordered["is_relevant"].to_numpy(dtype=bool)]
        if len(relevant_ranks) != positive_count:
            raise ValueError(
                f"query {query_id} contains {len(relevant_ranks)} positives; "
                f"expected {positive_count}"
            )
        first_rank = int(relevant_ranks[0])
        row = {
            "model": models[0],
            "metric": metrics[0],
            "query_id": query_id,
            "query_image_id": str(ordered.iloc[0]["query_image_id"]),
            "positive_count": positive_count,
            "gallery_count": len(ordered),
            "first_relevant_rank": first_rank,
            "reciprocal_rank": 1.0 / first_rank,
            "average_precision": _average_precision(relevant_ranks, positive_count, None),
        }
        for cutoff in top_k:
            if cutoff > len(ordered):
                raise ValueError(f"top-k cutoff {cutoff} exceeds gallery size {len(ordered)}")
            hits = int((relevant_ranks <= cutoff).sum())
            row[f"hits_at_{cutoff}"] = hits
            row[f"hit_at_{cutoff}"] = int(hits > 0)
            row[f"precision_at_{cutoff}"] = hits / cutoff
            row[f"recall_at_{cutoff}"] = hits / positive_count
            row[f"ap_at_{cutoff}"] = _average_precision(
                relevant_ranks, positive_count, cutoff
            )
            row[f"ndcg_at_{cutoff}"] = _ndcg_at_k(
                relevant_ranks, positive_count, cutoff
            )
        query_rows.append(row)

    per_query = pd.DataFrame(query_rows).sort_values(
        "query_id", key=lambda values: values.astype(int)
    )
    aggregate: dict[str, float | int | str] = {
        "model": models[0],
        "metric": metrics[0],
        "queries": len(per_query),
        "gallery": int(per_query["gallery_count"].iloc[0]),
        "positive_judgments": int(per_query["positive_count"].sum()),
        "mean_average_precision": float(per_query["average_precision"].mean()),
        "mean_reciprocal_rank": float(per_query["reciprocal_rank"].mean()),
        "mean_first_relevant_rank": float(per_query["first_relevant_rank"].mean()),
        "median_first_relevant_rank": float(per_query["first_relevant_rank"].median()),
    }
    for cutoff in top_k:
        aggregate[f"total_hits_at_{cutoff}"] = int(per_query[f"hits_at_{cutoff}"].sum())
        aggregate[f"top_{cutoff}_accuracy"] = float(per_query[f"hit_at_{cutoff}"].mean())
        aggregate[f"precision_at_{cutoff}"] = float(
            per_query[f"precision_at_{cutoff}"].mean()
        )
        aggregate[f"recall_at_{cutoff}"] = float(per_query[f"recall_at_{cutoff}"].mean())
        aggregate[f"map_at_{cutoff}"] = float(per_query[f"ap_at_{cutoff}"].mean())
        aggregate[f"ndcg_at_{cutoff}"] = float(per_query[f"ndcg_at_{cutoff}"].mean())
    return per_query.reset_index(drop=True), aggregate
