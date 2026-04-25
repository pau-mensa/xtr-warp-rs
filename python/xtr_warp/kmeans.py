"""K-means centroid computation for index creation and centroid expansion."""
from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from fastkmeans import FastKMeans

from .embedding_source import create_source


def compute_kmeans(  # noqa: PLR0913
    embeddings_source: list[torch.Tensor] | torch.Tensor | Path,
    device: str,
    kmeans_niters: int,
    max_points_per_centroid: int,
    seed: int,
    n_samples_kmeans: int | None = None,
    use_triton_kmeans: bool | None = None,
    num_partitions_override: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute K-means centroids for document embeddings.

    When ``num_partitions_override`` is set, the K for K-means is forced to
    that value (used by centroid expansion). When a raw ``torch.Tensor`` is
    passed as *embeddings_source*, it is treated as a flat ``[N, dim]``
    tensor of pre-sampled embeddings (no sampling step).
    """
    # Fast path: raw tensor (used by centroid expansion).
    if isinstance(embeddings_source, torch.Tensor):
        assert embeddings_source.dim() == 2, "Centroid expansion requires 2-dim tensors"
        tensors = embeddings_source
        total_tokens = tensors.shape[0]
        dim = tensors.shape[1]
        num_partitions = num_partitions_override or max(
            1, total_tokens // max_points_per_centroid
        )
    else:
        source = create_source(embeddings_source)
        num_passages = source.get_num_passages()

        if n_samples_kmeans is None:
            n_samples_kmeans = min(
                1 + int(16 * math.sqrt(120 * num_passages)),
                num_passages,
            )

        rng = random.Random(seed)
        sampled_pids = rng.sample(range(num_passages), k=n_samples_kmeans)

        tensors, total_tokens, dim = source.sample_embeddings(sampled_pids)

        if num_partitions_override is not None:
            num_partitions = num_partitions_override
        else:
            num_partitions = (total_tokens / n_samples_kmeans) * num_passages
            num_partitions = int(
                2 ** math.floor(math.log2(16 * math.sqrt(num_partitions)))
            )

    # I don't want any surprises here.
    if tensors.is_cuda:
        tensors = tensors.to(device="cpu")
    if tensors.dtype != torch.float32:
        tensors = tensors.to(dtype=torch.float32)
    if not tensors.is_contiguous():
        tensors = tensors.contiguous()

    use_gpu = device != "cpu"

    # The triton k-means kernel in fastkmeans produces incorrect cluster
    # assignments for non-power-of-2 k, so round down.
    # TODO(pau-mensa): apparently this has been fixed for +10 months in their repo
    # they just have not released it yet, so I'll keep this here just in case
    k = min(num_partitions, total_tokens)
    if k > 0 and (k & (k - 1)) != 0:
        k = 2 ** int(math.log2(k))

    kmeans = FastKMeans(
        d=dim,
        k=k,
        niter=kmeans_niters,
        gpu=use_gpu,
        verbose=False,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        use_triton=use_triton_kmeans if use_gpu else False,
    )

    kmeans.train(data=tensors.numpy())

    centroids = torch.from_numpy(kmeans.centroids).to(
        device=device, dtype=torch.float32
    )

    return torch.nn.functional.normalize(input=centroids, dim=-1), dim
