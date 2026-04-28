"""Heuristics for choosing search hyperparameters from index density and
query characteristics.

Numbers are empirically tuned across BEIR datasets — each is paired with a
comment that explains its role. These are *not* user-tunable today; if a
caller wants to override, they pass explicit values to
:meth:`xtr_warp.search.XTRWarp.search`.
"""
from __future__ import annotations

import math


def optimize(
    metadata: dict | None,
    top_k: int,
    num_query_tokens: int,
) -> tuple[int, int, float, int, int] | None:
    """Suggest ``(bound, nprobe, centroid_score_threshold, max_candidates, t_prime)``.

    Returns ``None`` when *metadata* is missing — the caller is expected to
    raise.
    """
    if metadata is None:
        return None

    num_embeddings = metadata["num_embeddings"]
    num_partitions = metadata["num_partitions"]
    avg_doclen = metadata["avg_doclen"]

    density = num_embeddings / max(1, num_partitions)

    def _clamp(v: float, low: int, high: int) -> int:
        return max(low, min(int(v), high))

    if top_k <= 20:
        base_probe = 2
    elif top_k <= 100:
        base_probe = 4
    else:
        base_probe = 6

    density_boost = int(
        math.log10(max(1.0, density))
    )  # 0 for sparse, +1 per order of magnitude
    nprobe = _clamp(base_probe + density_boost, 2, min(32, num_partitions))

    # very large partition counts (e.g. 65k) tend to need more probing to keep
    # NDCG stable on long-query datasets
    if num_partitions >= 65536 and num_query_tokens >= 48:
        nprobe = max(nprobe, 12)

    # bound controls how many centroids we score before pruning
    bound = max(nprobe * 8, int(0.05 * num_partitions))

    centroid_score_threshold = 0.5
    if density > 1000 or top_k >= 50:
        centroid_score_threshold -= 0.05
    if density > 2500 or top_k >= 200:
        centroid_score_threshold -= 0.05

    # allow more candidates on dense corpora and multi-token queries
    est_candidates = density * max(1, nprobe) * max(1, num_query_tokens)
    max_candidates = int(est_candidates)
    max_candidates = max(max_candidates, top_k * 50)
    max_candidates = min(max_candidates, num_embeddings) // 2

    # t_prime controls how aggressively we estimate and correct quantization error
    # we need to bump this up for dense/long-queries
    t_prime = int(density * max(1, nprobe) * max(1, num_query_tokens // 2))

    # long-doc, low-density corpora often benefit from a smaller t':
    # otherwise the implicit "missing token" baseline becomes too harsh.
    if avg_doclen > 0 and density < 256:
        doclen_scale = 120.0 / avg_doclen
        doclen_scale = max(0.35, min(doclen_scale, 1.0))
        t_prime = int(t_prime * doclen_scale)

    t_prime = _clamp(t_prime, 5_000, 200_000)
    t_prime = min(t_prime, num_embeddings)

    return (
        bound,
        nprobe,
        centroid_score_threshold,
        max_candidates,
        t_prime,
    )
