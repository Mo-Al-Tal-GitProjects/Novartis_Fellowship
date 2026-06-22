from __future__ import annotations

import numpy as np

from cipher.retrieval.contracts import MetricContract


def exact_rank_one(
    query: np.ndarray,
    gallery: np.ndarray,
    gallery_image_ids: np.ndarray,
    *,
    query_image_id: str,
    contract: MetricContract,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank one query exactly, breaking equal values by ascending gallery image ID."""
    if query.ndim != 1 or gallery.ndim != 2 or gallery.shape[1] != query.shape[0]:
        raise ValueError("query and gallery dimensions are incompatible")
    if len(gallery_image_ids) != len(gallery):
        raise ValueError("gallery IDs and vectors have different lengths")
    eligible = gallery_image_ids != query_image_id
    eligible_indices = np.flatnonzero(eligible)
    eligible_gallery = np.asarray(gallery[eligible], dtype=np.float64)
    query_vector = np.asarray(query, dtype=np.float64)
    ids = gallery_image_ids[eligible].astype(str)
    if contract.name in {"cosine", "inner_product"}:
        values = eligible_gallery @ query_vector
        order = np.lexsort((ids, -values))
    elif contract.name == "euclidean_l2":
        cosine_values = eligible_gallery @ query_vector
        values = np.sqrt(np.maximum(0.0, 2.0 - (2.0 * cosine_values)))
        order = np.lexsort((ids, -cosine_values))
    else:
        values = np.linalg.norm(eligible_gallery - query_vector, axis=1)
        order = np.lexsort((ids, values))
    return eligible_indices[order], values[order]
