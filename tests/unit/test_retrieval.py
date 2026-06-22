from __future__ import annotations

import numpy as np
import pandas as pd

from cipher.evaluation.metrics import score_rankings
from cipher.retrieval.contracts import get_metric_contract
from cipher.retrieval.exact import exact_rank_one


def test_exact_ranking_excludes_self_and_breaks_ties_by_image_id() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    gallery = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    ids = np.array(["gallery:c", "query:self", "gallery:a", "gallery:b"])
    order, values = exact_rank_one(
        query,
        gallery,
        ids,
        query_image_id="query:self",
        contract=get_metric_contract("cosine"),
    )
    assert ids[order].tolist() == ["gallery:a", "gallery:c", "gallery:b"]
    np.testing.assert_array_equal(values, [1.0, 1.0, 0.0])


def test_raw_euclidean_orders_by_ascending_distance() -> None:
    query = np.array([0.0, 0.0], dtype=np.float32)
    gallery = np.array([[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    ids = np.array(["two", "one", "three"])
    order, values = exact_rank_one(
        query,
        gallery,
        ids,
        query_image_id="query",
        contract=get_metric_contract("euclidean"),
    )
    assert ids[order].tolist() == ["one", "two", "three"]
    np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


def test_normalized_euclidean_has_exact_cosine_ranking() -> None:
    query = np.asarray([0.6, 0.8], dtype=np.float32)
    gallery = np.asarray(
        [[0.6, 0.8], [0.0, 1.0], [1.0, 0.0], [-0.6, -0.8]],
        dtype=np.float32,
    )
    ids = np.asarray(["gallery:d", "gallery:c", "gallery:b", "gallery:a"])
    cosine_order, _ = exact_rank_one(
        query,
        gallery,
        ids,
        query_image_id="query:x",
        contract=get_metric_contract("cosine"),
    )
    euclidean_order, distances = exact_rank_one(
        query,
        gallery,
        ids,
        query_image_id="query:x",
        contract=get_metric_contract("euclidean_l2"),
    )

    assert euclidean_order.tolist() == cosine_order.tolist()
    assert np.all(distances[:-1] <= distances[1:])


def test_ranking_metrics_use_binary_curated_relevance() -> None:
    records = []
    relevant = {"1": {1, 3}, "2": {2, 5}}
    for query_id in ["1", "2"]:
        for rank in range(1, 6):
            records.append(
                {
                    "model": "fixture",
                    "metric": "cosine",
                    "query_id": query_id,
                    "query_image_id": f"query:{query_id}",
                    "rank": rank,
                    "is_relevant": rank in relevant[query_id],
                }
            )
    rankings = pd.DataFrame(records)
    relevance = pd.DataFrame(
        [
            {"query_id": query_id, "gallery_image_id": f"gallery:{rank}"}
            for query_id, ranks in relevant.items()
            for rank in ranks
        ]
    )
    per_query, aggregate = score_rankings(rankings, relevance, [1, 3, 5])
    np.testing.assert_allclose(
        per_query["average_precision"],
        [(1.0 + 2.0 / 3.0) / 2.0, (1.0 / 2.0 + 2.0 / 5.0) / 2.0],
    )
    np.testing.assert_allclose(aggregate["mean_average_precision"], 0.6416666666666666)
    np.testing.assert_allclose(aggregate["mean_reciprocal_rank"], 0.75)
    np.testing.assert_allclose(aggregate["top_1_accuracy"], 0.5)
    assert aggregate["total_hits_at_5"] == 4
