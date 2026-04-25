"""Post-mutation index maintenance: centroid expansion and threshold
recalibration.

Both functions are called by :meth:`xtr_warp.search.XTRWarp.add` after new
embeddings are inserted. They operate purely against the on-disk index and
are stateless (no class to instantiate).
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch

from . import xtr_warp_rs
from .kmeans import compute_kmeans

logger = logging.getLogger(__name__)

EmbeddingsSource = Union[list[torch.Tensor], torch.Tensor, str, Path]


def maybe_expand_centroids(
    *,
    index: str,
    residual_norms: list[float],
    embeddings_source: EmbeddingsSource,
    device: str,
    metadata: dict | None,
    min_outliers: int = 50,
    max_growth_rate: float = 0.1,
    max_points_per_centroid: int = 256,
) -> None:
    """Expand the centroid codebook if many newly-added embeddings are outliers.

    An outlier is an embedding whose residual norm (distance to its nearest
    centroid) exceeds the cluster threshold stored at index creation time.
    No-op when the threshold file is missing, when fewer than
    *min_outliers* embeddings exceed it, or when *embeddings_source* is a
    disk path (centroid expansion against streaming sources is not yet
    supported).
    """
    threshold_path = os.path.join(index, "cluster_threshold.npy")
    if not os.path.exists(threshold_path) or not residual_norms:
        return

    threshold = float(np.load(threshold_path).item())
    norms = np.array(residual_norms, dtype=np.float32)
    outlier_mask = norms > threshold
    outlier_count = int(outlier_mask.sum())

    if outlier_count < min_outliers:
        return

    if isinstance(embeddings_source, (str, Path)):
        logger.warning(
            "Centroid expansion with disk-based embeddings not yet supported, skipping."
        )
        return

    if isinstance(embeddings_source, torch.Tensor):
        all_embs = [embeddings_source[i] for i in range(embeddings_source.shape[0])]
    else:
        all_embs = embeddings_source

    flat_embs = torch.cat(
        [e.squeeze(0) if e.dim() == 3 else e for e in all_embs], dim=0
    )
    outlier_embs = flat_embs[torch.from_numpy(outlier_mask)]

    current_centroids = metadata.get("num_centroids", 1) if metadata else 1
    target_k = math.ceil(outlier_count / max_points_per_centroid)
    max_new = max(1, int(current_centroids * max_growth_rate))
    k_new = max(1, min(target_k, max_new))

    if k_new < 1 or outlier_embs.shape[0] < k_new:
        return

    logger.info(
        "Centroid expansion: %d outliers detected, adding %d centroids",
        outlier_count,
        k_new,
    )

    new_centroids, _ = compute_kmeans(
        embeddings_source=outlier_embs,
        device=device,
        kmeans_niters=4,
        max_points_per_centroid=max_points_per_centroid,
        seed=42,
        num_partitions_override=k_new,
    )

    xtr_warp_rs.append_centroids_py(
        index=index,
        new_centroids=new_centroids,
    )


def recalibrate_threshold(
    *,
    index: str,
    residual_norms: list[float],
    metadata: dict | None,
) -> None:
    """Update ``cluster_threshold.npy`` with a count-weighted blend.

    The old threshold (weighted by pre-existing embedding count) is blended
    with the 75th-percentile of the *new* norms (weighted by the new
    embedding count).
    """
    threshold_path = os.path.join(index, "cluster_threshold.npy")
    if not os.path.exists(threshold_path) or not residual_norms:
        return

    old_threshold = float(np.load(threshold_path).item())
    new_norms = np.array(residual_norms, dtype=np.float32)
    new_count = len(new_norms)
    new_threshold = float(np.percentile(new_norms, 75))

    # metadata on disk already includes the just-added embeddings
    total = metadata.get("num_embeddings", new_count) if metadata else new_count
    old_count = max(0, total - new_count)

    if old_count + new_count > 0:
        updated = (old_threshold * old_count + new_threshold * new_count) / (
            old_count + new_count
        )
    else:
        updated = new_threshold

    np.save(threshold_path, np.float32(updated))
