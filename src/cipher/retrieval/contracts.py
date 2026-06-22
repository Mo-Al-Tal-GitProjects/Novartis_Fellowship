from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricName = Literal["cosine", "inner_product", "euclidean", "euclidean_l2"]


@dataclass(frozen=True)
class MetricContract:
    name: MetricName
    use_normalized_embeddings: bool
    higher_is_better: bool
    value_kind: str


METRIC_CONTRACTS: dict[str, MetricContract] = {
    "cosine": MetricContract("cosine", True, True, "cosine_similarity"),
    "inner_product": MetricContract("inner_product", False, True, "raw_inner_product"),
    "euclidean": MetricContract("euclidean", False, False, "raw_euclidean_distance"),
    "euclidean_l2": MetricContract(
        "euclidean_l2", True, False, "l2_normalized_euclidean_distance"
    ),
}


def get_metric_contract(name: str) -> MetricContract:
    try:
        return METRIC_CONTRACTS[name]
    except KeyError:
        raise ValueError(
            f"unsupported metric {name!r}; choose from {sorted(METRIC_CONTRACTS)}"
        ) from None
