"""Memory-bounded k-means for index creation and centroid expansion."""

from __future__ import annotations

import logging
import math
import random
from typing import TYPE_CHECKING

import torch

from .embedding_source import create_source

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DATA_CHUNK_SIZE = 32_768
_DEFAULT_CENTROID_CHUNK_SIZE = 8_192


@torch.inference_mode()
def _compute_centroids_chunked(  # noqa: PLR0913, PLR0915
    data: torch.Tensor,
    *,
    k: int,
    device: str,
    niter: int,
    seed: int,
    max_points_per_centroid: int,
    data_chunk_size: int,
    centroid_chunk_size: int,
) -> torch.Tensor:
    """Run Lloyd k-means without allocating an ``N x K`` distance matrix.

    This follows the double-chunking strategy used by fast-plaid. Training
    samples stay on CPU and only one data/centroid tile is transferred to the
    accelerator at a time.
    """
    if data.dim() != 2:
        error = f"k-means data must be 2D, got shape {tuple(data.shape)}"
        raise ValueError(error)
    if k <= 0:
        error = f"num_partitions must be positive, got {k}"
        raise ValueError(error)

    n_samples, dim = data.shape
    if n_samples < k:
        error = (
            f"K-means needs at least as many training points as centroids: "
            f"{n_samples:,} points < {k:,} centroids"
        )
        raise ValueError(error)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    max_training_points = k * max_points_per_centroid
    if n_samples > max_training_points:
        indices = torch.randperm(n_samples, generator=generator)[:max_training_points]
        data = data.index_select(0, indices)
        n_samples = data.shape[0]

    compute_device = torch.device(device)
    compute_dtype = torch.float32 if compute_device.type == "cpu" else torch.float16
    # Tiles are moved to the accelerator in float16. Unit-normalizing first
    # keeps every dot product and squared norm within float16 range; the
    # returned centroids are normalized by the caller anyway, so this matches
    # the cosine geometry the index scores with.
    data = torch.nn.functional.normalize(
        data.to(device="cpu", dtype=torch.float32), dim=-1
    ).contiguous()
    data_norms = (data * data).sum(dim=1)

    initial_indices = torch.randperm(n_samples, generator=generator)[:k]
    centroids = data.index_select(0, initial_indices).to(
        device=compute_device, dtype=compute_dtype
    )

    for _ in range(niter):
        centroid_norms = (centroids * centroids).sum(dim=1)
        cluster_sums = torch.zeros((k, dim), device=compute_device, dtype=torch.float32)
        cluster_counts = torch.zeros(k, device=compute_device, dtype=torch.float32)

        for data_start in range(0, n_samples, data_chunk_size):
            data_end = min(data_start + data_chunk_size, n_samples)
            data_chunk = data[data_start:data_end].to(
                device=compute_device, dtype=compute_dtype
            )
            norm_chunk = data_norms[data_start:data_end].to(
                device=compute_device, dtype=compute_dtype
            )
            best_distances = torch.full(
                (data_end - data_start,),
                float("inf"),
                device=compute_device,
                dtype=compute_dtype,
            )
            best_ids = torch.zeros(
                data_end - data_start, device=compute_device, dtype=torch.int64
            )

            for centroid_start in range(0, k, centroid_chunk_size):
                centroid_end = min(centroid_start + centroid_chunk_size, k)
                centroid_chunk = centroids[centroid_start:centroid_end]
                distances = norm_chunk.unsqueeze(1) + centroid_norms[
                    centroid_start:centroid_end
                ].unsqueeze(0)
                distances = distances.addmm_(
                    data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0
                )
                local_distances, local_ids = distances.min(dim=1)
                improved = local_distances < best_distances
                best_distances[improved] = local_distances[improved]
                best_ids[improved] = centroid_start + local_ids[improved]

            cluster_sums.index_add_(0, best_ids, data_chunk.float())
            cluster_counts.index_add_(
                0,
                best_ids,
                torch.ones_like(best_ids, dtype=torch.float32),
            )

        non_empty = cluster_counts > 0
        new_centroids = torch.empty_like(centroids)
        new_centroids[non_empty] = (
            cluster_sums[non_empty] / cluster_counts[non_empty].unsqueeze(1)
        ).to(compute_dtype)

        empty_ids = (~non_empty).nonzero(as_tuple=True)[0]
        if empty_ids.numel():
            replacement_indices = torch.randint(
                n_samples,
                (empty_ids.numel(),),
                generator=generator,
                device="cpu",
            )
            new_centroids[empty_ids] = data.index_select(0, replacement_indices).to(
                device=compute_device, dtype=compute_dtype
            )

        shift = torch.linalg.vector_norm(
            new_centroids.float() - centroids.float(), dim=1
        ).sum()
        centroids = new_centroids
        if shift.item() < 1e-8:
            break

    return centroids.to(device="cpu", dtype=torch.float32)


def compute_kmeans(  # noqa: C901, PLR0912, PLR0913, PLR0915
    embeddings_source: list[torch.Tensor] | torch.Tensor | Path,
    device: str,
    kmeans_niters: int,
    max_points_per_centroid: int,
    seed: int,
    n_samples_kmeans: int | None = None,
    use_triton_kmeans: bool | None = None,
    num_partitions_override: int | None = None,
    sample_workers: int | None = None,
    kmeans_sample_max_bytes: int | None = 2 * 1024**3,
    kmeans_data_chunk_size: int = _DEFAULT_DATA_CHUNK_SIZE,
    kmeans_centroid_chunk_size: int = _DEFAULT_CENTROID_CHUNK_SIZE,
) -> tuple[torch.Tensor, int]:
    """Compute centroids with bounded CPU and accelerator working sets."""
    if use_triton_kmeans:
        logger.warning(
            "use_triton_kmeans=True was requested, but the memory-bounded "
            "PyTorch implementation is used because FastKMeans's Triton path "
            "is not reliable across shapes"
        )
    if kmeans_niters <= 0:
        error = "kmeans_niters must be positive"
        raise ValueError(error)
    if max_points_per_centroid <= 0:
        error = "max_points_per_centroid must be positive"
        raise ValueError(error)
    if kmeans_data_chunk_size <= 0 or kmeans_centroid_chunk_size <= 0:
        error = "k-means chunk sizes must be positive"
        raise ValueError(error)
    if num_partitions_override is not None and num_partitions_override <= 0:
        error = "num_partitions must be positive"
        raise ValueError(error)

    if isinstance(embeddings_source, torch.Tensor):
        if embeddings_source.dim() != 2:
            error = "Centroid expansion requires a 2D tensor"
            raise ValueError(error)
        tensors = embeddings_source
        total_tokens = tensors.shape[0]
        dim = tensors.shape[1]
        num_partitions = num_partitions_override or max(
            1, total_tokens // max_points_per_centroid
        )
        sampled_docs = None
    else:
        source = create_source(embeddings_source, sample_workers=sample_workers)
        num_passages = source.get_num_passages()
        if num_passages == 0:
            error = "Cannot cluster an empty embedding source"
            raise ValueError(error)

        requested_samples = n_samples_kmeans
        if requested_samples is None:
            requested_samples = min(
                1 + int(16 * math.sqrt(120 * num_passages)), num_passages
            )
        if requested_samples <= 0:
            error = "n_samples_kmeans must be positive"
            raise ValueError(error)
        requested_samples = min(requested_samples, num_passages)

        rng = random.Random(seed)
        sampled_pids = rng.sample(range(num_passages), k=requested_samples)
        if kmeans_sample_max_bytes is not None:
            if kmeans_sample_max_bytes <= 0:
                error = "kmeans_sample_max_bytes must be positive or None"
                raise ValueError(error)
            # K-means materializes the sample as float32 [tokens, dimension].
            # Account for the complete vector rather than only one scalar.
            embedding_dim = source.get_embedding_dim()
            bytes_per_token = embedding_dim * torch.float32.itemsize
            max_tokens = kmeans_sample_max_bytes // bytes_per_token
            if max_tokens == 0:
                error = (
                    "kmeans_sample_max_bytes is smaller than one float32 "
                    f"embedding ({bytes_per_token:,} bytes)"
                )
                raise ValueError(error)
            doclens = source.get_doclens()
            bounded_pids: list[int] = []
            bounded_tokens = 0
            for pid in sampled_pids:
                doc_tokens = doclens[pid]
                if doc_tokens > max_tokens:
                    error = f"Document {pid} alone exceeds kmeans_sample_max_bytes"
                    raise ValueError(error)
                if bounded_tokens + doc_tokens > max_tokens:
                    # Stop at the first overflow rather than skipping ahead, so
                    # the sample stays an unbiased prefix of the random draw.
                    break
                bounded_pids.append(pid)
                bounded_tokens += doc_tokens
            if len(bounded_pids) < len(sampled_pids):
                logger.warning(
                    "kmeans_sample_max_bytes kept %s of %s sampled documents "
                    "(%s token embeddings)",
                    f"{len(bounded_pids):,}",
                    f"{len(sampled_pids):,}",
                    f"{bounded_tokens:,}",
                )
            sampled_pids = bounded_pids

        tensors, total_tokens, dim = source.sample_embeddings(sampled_pids)
        sampled_docs = len(sampled_pids)
        if num_partitions_override is not None:
            num_partitions = num_partitions_override
        else:
            total_corpus_tokens = sum(source.get_doclens())
            num_partitions = int(
                2 ** math.floor(math.log2(16 * math.sqrt(total_corpus_tokens)))
            )

    k = min(num_partitions, total_tokens)
    if k < num_partitions:
        logger.warning(
            "Requested %s centroids but the k-means sample contains only %s "
            "token embeddings; using %s centroids",
            f"{num_partitions:,}",
            f"{total_tokens:,}",
            f"{k:,}",
        )

    sample_gib = total_tokens * dim * 4 / 1024**3
    sample_desc = f"{sampled_docs:,} documents / " if sampled_docs is not None else ""
    logger.warning(
        "K-means plan: %s%s token embeddings (%.2f GiB float32), "
        "%s centroids, tiles <= %s x %s",
        sample_desc,
        f"{total_tokens:,}",
        sample_gib,
        f"{k:,}",
        f"{kmeans_data_chunk_size:,}",
        f"{kmeans_centroid_chunk_size:,}",
    )

    centroids = _compute_centroids_chunked(
        tensors,
        k=k,
        device=device,
        niter=kmeans_niters,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        data_chunk_size=kmeans_data_chunk_size,
        centroid_chunk_size=kmeans_centroid_chunk_size,
    )
    centroids = centroids.to(device=device, dtype=torch.float32)
    return torch.nn.functional.normalize(input=centroids, dim=-1), dim
