from __future__ import annotations

import glob
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from fastkmeans import FastKMeans

from . import xtr_warp_rs
from .filtering import MetadataStore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TorchWithCudaNotFoundError(Exception):
    """Exception raised when PyTorch with CUDA support is not found."""


def _load_torch_path(device: str) -> str:
    """Find the path to the shared library for PyTorch with CUDA."""
    search_paths = [
        os.path.join(os.path.dirname(torch.__file__), "lib", f"libtorch_{device}.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", f"libtorch_{device}.so"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", "libtorch_cuda.dylib"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", "libtorch.so"),
        os.path.join(os.path.dirname(torch.__file__), "**", "libtorch.dylib"),
        os.path.join(os.path.dirname(torch.__file__), "lib", f"torch_{device}.dll"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "torch.dll"),
        os.path.join(os.path.dirname(torch.__file__), "lib", f"c10_{device}.dll"),
        os.path.join(os.path.dirname(torch.__file__), "lib", "c10.dll"),
        os.path.join(os.path.dirname(torch.__file__), "**", f"torch_{device}.dll"),
        os.path.join(os.path.dirname(torch.__file__), "**", "torch.dll"),
    ]

    for path_pattern in search_paths:
        found_libs = glob.glob(path_pattern, recursive=True)
        if found_libs:
            return found_libs[0]

    error = """
    Could not find torch binary.
    Please ensure PyTorch is installed.
    """
    raise TorchWithCudaNotFoundError(error) from IndexError


class EmbeddingSource(Protocol):
    """Protocol for embedding sources."""

    def get_num_passages(self) -> int: ...
    def sample_embeddings(self, pids: list[int]) -> tuple[torch.Tensor, int, int]: ...


@dataclass
class InMemorySource:
    """Source for embeddings already in memory."""

    embeddings: list[torch.Tensor]

    def get_num_passages(self) -> int:
        """Get the number of passages."""
        return len(self.embeddings)

    def sample_embeddings(self, pids: list[int]) -> tuple[torch.Tensor, int, int]:
        """Sample the embeddings based on the pids."""
        samples = [self.embeddings[pid] for pid in pids]
        dim = samples[0].size(-1)
        total_tokens = sum(sample.shape[0] for sample in samples)
        tensors = torch.cat(tensors=samples)
        return tensors, total_tokens, dim


@dataclass
class DiskSource:
    """Source for embeddings stored in disk."""

    path: Path
    _files_and_doclens: list[tuple[Path, list[int]]] | None = None
    _doclens: list[int] | None = None
    _num_passages: int = 0

    def _load_metadata(self) -> None:
        if self._files_and_doclens is not None:
            return

        files = _get_all_embedding_files(self.path)
        self._doclens = []
        self._files_and_doclens = []

        for file in files:
            doclens_file = _doclens_path_for(file)
            sidecar = np.load(doclens_file)
            self._num_passages += len(sidecar)
            sidecar_list = sidecar.tolist()
            self._doclens.extend(sidecar_list)
            self._files_and_doclens.append((file, sidecar_list))

    def get_num_passages(self) -> int:
        """Get the number of passages."""
        self._load_metadata()
        return self._num_passages

    def sample_embeddings(self, pids: list[int]) -> tuple[torch.Tensor, int, int]:
        """Sample the embeddings based on the pids."""
        self._load_metadata()

        sampled_pid_set = set(pids)
        total_tokens = sum(self._doclens[pid] for pid in pids)

        tensors = None
        dim = None
        write_offset = 0
        doc_offset = 0
        remaining = len(sampled_pid_set)

        for file, sidecar in self._files_and_doclens:
            data = torch.from_numpy(np.load(file))

            if tensors is None:
                dim = data.size(-1)
                tensors = torch.empty((total_tokens, dim), dtype=data.dtype)

            offset = 0
            for doc_len in sidecar:
                if doc_offset in sampled_pid_set:
                    doc = data[offset : offset + doc_len]
                    tensors[write_offset : write_offset + doc_len].copy_(doc)
                    write_offset += doc_len
                    sampled_pid_set.remove(doc_offset)
                    remaining -= 1
                    if remaining == 0:
                        break
                offset += doc_len
                doc_offset += 1

            del data
            if remaining == 0:
                break

        if tensors is None or dim is None:
            raise ValueError("Could not sample embeddings from source")

        return tensors, total_tokens, dim


def _create_source(embeddings_source: list[torch.Tensor] | Path) -> EmbeddingSource:
    """Create appropriate source."""
    if isinstance(embeddings_source, list):
        return InMemorySource(embeddings_source)
    return DiskSource(embeddings_source)


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
    that value (used by centroid expansion).  When a raw ``torch.Tensor`` is
    passed as *embeddings_source*, it is treated as a flat [N, dim] tensor of
    pre-sampled embeddings (no sampling step).
    """
    # Fast path: raw tensor (used by centroid expansion)
    if isinstance(embeddings_source, torch.Tensor):
        assert embeddings_source.dim() == 2, "Centroid expansion requires 2-dim tensors"
        tensors = embeddings_source
        total_tokens = tensors.shape[0]
        dim = tensors.shape[1]
        num_partitions = num_partitions_override or max(
            1, total_tokens // max_points_per_centroid
        )
    else:
        source = _create_source(embeddings_source)
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

    # I don't want any surprises here
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

    return torch.nn.functional.normalize(
        input=centroids,
        dim=-1,
    ), dim


def _doclens_path_for(emb_path: Path) -> Path:
    npy_path = emb_path.with_suffix(".doclens.npy")
    if npy_path.exists():
        return npy_path
    raise ValueError(
        f"The {emb_path} embeddings file is missing its sidecar: {npy_path}"
    )


def _get_all_embedding_files(embeddings_path: Path) -> list[Path]:
    if embeddings_path.is_file():
        files = [embeddings_path]
    else:
        npy_files = list(embeddings_path.glob("*.npy"))
        files = sorted(
            [path for path in npy_files if not path.name.endswith(".doclens.npy")],
            key=_embedding_chunk_sort_key,
        )
    if not files:
        raise FileNotFoundError(f"No embedding .npy files found in {embeddings_path}")

    return files


def _embedding_chunk_sort_key(path: Path) -> tuple[int, int | str]:
    name = path.stem

    # (double extension for doclens)
    if path.name.endswith(".doclens.npy"):
        name = path.name[: -len(".doclens.npy")]

    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (0, int(parts[1]))
    return (1, name)


def search_on_device(
    search_config,
    queries_embeddings: torch.Tensor,
    loaded_index,
    torch_path: str,
    subsets: list[list[int]] | None = None,
    show_progress: bool = True,
) -> list[list[tuple[int, float]]]:
    """Perform a search on a loaded index."""
    scores = loaded_index.search(
        torch_path=torch_path,
        queries_embeddings=queries_embeddings,
        search_config=search_config,
        subsets=subsets,
        show_progress=show_progress,
    )

    return [
        [
            (passage_id, score)
            for score, passage_id in zip(score.scores, score.passage_ids)
        ]
        for score in scores
    ]


class XTRWarp:
    """A class for creating and searching a XTRWarp index.

    Args:
    ----
    index:
        Path to the directory where the index is stored or will be stored.

    """

    def __init__(
        self,
        index: str,
        device: str | None = None,
    ) -> None:
        self._searcher = None
        self.index: str = index
        self.devices: list | None = None
        self.dtype: torch.dtype | None = None
        self._torch_initialized = {}
        self._metadata: dict | None = None
        self.device: str | None = device
        self._mmap: bool = True
        self._metadata_store: MetadataStore | None = None
        self._device_arg: str | list[str] | dict[str, float] | None = None

    def _ensure_torch_initialized(self, device: str) -> str:
        """Initialize torch once per device type."""
        device_type = device.split(":")[0]  # 'cuda:0' -> 'cuda'
        if device_type not in self._torch_initialized:
            torch_path = _load_torch_path(device=device_type)
            xtr_warp_rs.initialize_torch(torch_path)
            self._torch_initialized[device_type] = torch_path
        return self._torch_initialized[device_type]

    def free(self) -> None:
        """Free the loaded index from memory."""
        if self._searcher is not None:
            self._searcher.free()
            self._searcher = None
        if self._metadata_store is not None:
            self._metadata_store.close()
            self._metadata_store = None

    def _reload_if_loaded(self) -> None:
        """Free and re-load the index using the same parameters as the last ``load()`` call."""
        if self._searcher is None:
            return
        if self._device_arg is None:
            return
        self.free()
        self._metadata = None
        self.load(device=self._device_arg, dtype=self.dtype, mmap=self._mmap)

    def __del__(self):
        """Destructor."""
        self.free()

    def create(  # noqa: PLR0913
        self,
        embeddings_source: list[torch.Tensor] | torch.Tensor | str | Path,
        device: str,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        nbits: int = 4,
        n_samples_kmeans: int | None = None,
        seed: int = 42,
        use_triton_kmeans: bool | None = None,
        metadata: list[dict] | None = None,
        show_progress: bool = True,
    ) -> "XTRWarp":
        """Create and saves the XTRWarp index.

        Args:
        ----
        embeddings_source:
            A list of document embeddings or the path to a folder where the embeddings
            are stored. The stored embeddings must be in `.npy` format,
            in a 2D tensor `[total_len, dim]` with a matching `.doclens.npy` sidecar.
        stream:
            Whether to stream embeddings from disk during index creation. If True,
            `embeddings_source` must be a str and embeddings will be read from disk
            during encoding.
        device:
            The device to use for the indexing (eg. cpu, cuda, mps, etc.)
        kmeans_niters:
            Number of iterations for the K-means algorithm.
        max_points_per_centroid:
            The maximum number of points per centroid for K-means.
        nbits:
            Number of bits to use for quantization (default is 4).
        n_samples_kmeans:
            Number of samples to use for K-means. If None, it will be calculated based
            on the number of documents.
        seed:
            Optional seed for the random number generator used in index creation.
        use_triton_kmeans:
            Whether to use the Triton-based K-means implementation. If None, it will be
            set to True if the device is not "cpu".

        """
        self.device = device
        torch_path = self._ensure_torch_initialized(device)

        embeddings_path = None
        documents_embeddings = None

        if isinstance(embeddings_source, (list, torch.Tensor)):
            if isinstance(embeddings_source, torch.Tensor):
                documents_embeddings = [
                    embeddings_source[i] for i in range(embeddings_source.shape[0])
                ]
            elif isinstance(embeddings_source, list):
                documents_embeddings = embeddings_source

            documents_embeddings = [
                embedding.squeeze(0) if embedding.dim() == 3 else embedding
                for embedding in documents_embeddings
            ]
        else:
            embeddings_path = Path(embeddings_source)

        self._prepare_index_directory(index_path=self.index)

        centroids, dim = compute_kmeans(
            embeddings_source=embeddings_path or documents_embeddings,
            kmeans_niters=kmeans_niters,
            device=device,
            max_points_per_centroid=max_points_per_centroid,
            n_samples_kmeans=n_samples_kmeans,
            seed=seed,
            use_triton_kmeans=use_triton_kmeans,
        )

        xtr_warp_rs.create(
            index=self.index,
            torch_path=torch_path,
            device=device,
            nbits=nbits,
            centroids=centroids,
            embeddings=documents_embeddings or str(embeddings_path),
            embedding_dim=dim,
            seed=seed,
            show_progress=show_progress,
        )

        if metadata is not None:
            store = MetadataStore(self.index)
            store.create(metadata, start_pid=0)
            store.close()

        return self

    @staticmethod
    def _prepare_index_directory(index_path: str) -> None:
        """Prepare the index directory by cleaning or creating it."""
        if os.path.exists(index_path) and os.path.isdir(index_path):
            for json_file in glob.glob(os.path.join(index_path, "*.json")):
                try:
                    os.remove(json_file)
                except OSError:
                    pass

            for npy_file in glob.glob(os.path.join(index_path, "*.npy")):
                try:
                    os.remove(npy_file)
                except OSError:
                    pass

            for pt_file in glob.glob(os.path.join(index_path, "*.pt")):
                try:
                    os.remove(pt_file)
                except OSError:
                    pass

            for duckdb_file in glob.glob(os.path.join(index_path, "*.duckdb*")):
                try:
                    os.remove(duckdb_file)
                except OSError:
                    pass
        elif not os.path.exists(index_path):
            try:
                os.makedirs(index_path)
            except OSError as e:
                raise e

    def delete(
        self,
        passage_ids: list[int],
        compact_threshold: float | None = 0.2,
    ) -> "XTRWarp":
        """Delete passages by ID. O(1) tombstone operation.

        Search automatically filters deleted passages. To physically
        remove deleted data, call ``compact()`` afterward, or set
        ``compact_threshold`` to trigger compaction when the tombstone
        ratio exceeds the ratio.

        Args:
        ----
        passage_ids:
            List of passage IDs to mark as deleted.
        compact_threshold:
            Fraction of deleted passages that triggers auto-compaction
            (default 0.2 = 20%). If set to None the compaction does not run

        """
        xtr_warp_rs.delete(self.index, passage_ids)
        if self._searcher is not None:
            self._searcher.update_tombstones(passage_ids)

        if self._metadata_store is not None:
            self._metadata_store.delete(passage_ids)

        if compact_threshold is not None:
            meta = self._load_metadata()
            if meta and meta.get("num_passages", 0) > 0:
                deleted_path = os.path.join(self.index, "deleted_pids.npy")
                num_deleted = (
                    len(np.load(deleted_path)) if os.path.exists(deleted_path) else 0
                )
                total = meta["num_passages"] + num_deleted
                ratio = num_deleted / total if total > 0 else 0.0
                if ratio >= compact_threshold:
                    logger.warning(
                        "Tombstone ratio %.1f%% >= %.0f%% threshold, running auto-compaction",
                        ratio * 100,
                        compact_threshold * 100,
                    )
                    self.compact()

        return self

    def add(
        self,
        embeddings_source: list[torch.Tensor] | torch.Tensor | str | Path,
        reload: bool = True,
        min_outliers: int = 50,
        max_growth_rate: float = 0.1,
        max_points_per_centroid: int = 256,
        metadata: list[dict] | None = None,
        show_progress: bool = True,
    ) -> list[int]:
        """Add new passages. Encodes new documents and recompacts the index.

        Args:
        ----
        embeddings_source:
            New document embeddings (same formats as ``create``).
        reload:
            If True (default) and the index was loaded, automatically
            free and re-load so searches reflect the new data.
            Set to False when batching several mutations before a
            manual ``load()`` call.
        min_outliers:
            Minimum number of outliers to cause centroid expansion
        max_growth_rate:
            Ratio of the maximum number of centroids relative to the index
            size that will be added
        max_points_per_centroid:
            The number of points per centroid to use

        Returns:
        -------
        List of newly assigned passage IDs.

        """
        device = self._resolve_device(None)
        torch_path = self._ensure_torch_initialized(device)
        embeddings = self._prepare_embeddings(embeddings_source)
        was_loaded = self._searcher is not None
        if was_loaded:
            self.free()
        result = xtr_warp_rs.add(
            index=self.index,
            torch_path=torch_path,
            device=device,
            embeddings=embeddings,
            show_progress=show_progress,
        )
        new_ids = result["new_passage_ids"]

        # Centroid expansion: detect outliers and grow the codebook
        self._maybe_expand_centroids(
            residual_norms=result["residual_norms"],
            embeddings_source=embeddings_source,
            device=device,
            min_outliers=min_outliers,
            max_growth_rate=max_growth_rate,
            max_points_per_centroid=max_points_per_centroid,
        )

        # Recalibrate the outlier threshold so it reflects the proper data distribution
        self._recalibrate_threshold(result["residual_norms"])

        if metadata is not None:
            store = MetadataStore(self.index)
            store.add(metadata, start_pid=new_ids[0])
            store.close()

        if reload and was_loaded:
            self._metadata = None
            self.load(
                device=self._device_arg or self.devices,
                dtype=self.dtype,
                mmap=self._mmap,
            )
        else:
            self._metadata = None
        return new_ids

    def update(
        self,
        passage_ids: list[int],
        embeddings_source: list[torch.Tensor] | torch.Tensor | str | Path,
        reload: bool = True,
        show_progress: bool = True,
    ) -> "XTRWarp":
        """Update passages in-place: new embeddings, same IDs.

        Args:
        ----
        passage_ids:
            IDs of passages to update.
        embeddings_source:
            Replacement embeddings (one per passage ID).
        reload:
            If True (default), automatically re-load after mutation.

        """
        device = self._resolve_device(None)
        torch_path = self._ensure_torch_initialized(device)
        embeddings = self._prepare_embeddings(embeddings_source)
        was_loaded = self._searcher is not None
        if was_loaded:
            self.free()
        xtr_warp_rs.update(
            index=self.index,
            torch_path=torch_path,
            device=device,
            passage_ids=passage_ids,
            embeddings=embeddings,
            show_progress=show_progress,
        )
        if reload and was_loaded:
            self._metadata = None
            self.load(
                device=self._device_arg or self.devices,
                dtype=self.dtype,
                mmap=self._mmap,
            )
        else:
            self._metadata = None
        return self

    def compact(self, reload: bool = True, show_progress: bool = True) -> "XTRWarp":
        """Rebuild index excluding deleted passages.

        Use after ``delete()`` to physically reclaim space.

        Args:
        ----
        device:
            Compute device. Defaults to the device set at init or load time.
        reload:
            If True (default), automatically re-load after mutation.

        """
        device = self._resolve_device(None)
        torch_path = self._ensure_torch_initialized(device)
        was_loaded = self._searcher is not None
        if was_loaded:
            self.free()

        # Read tombstones before compact (compact clears the tombstone file)
        metadata_db = os.path.join(self.index, "metadata.duckdb")
        deleted_path = os.path.join(self.index, "deleted_pids.npy")
        tombstones: list[int] = []
        if os.path.exists(metadata_db) and os.path.exists(deleted_path):
            tombstones = np.load(deleted_path).tolist()

        xtr_warp_rs.compact(
            index=self.index,
            torch_path=torch_path,
            device=device,
            show_progress=show_progress,
        )

        # Delete metadata after successful compact
        if tombstones:
            store = MetadataStore(self.index)
            store.delete(tombstones)
            store.close()
        if reload and was_loaded:
            self._metadata = None
            self.load(
                device=self._device_arg or self.devices,
                dtype=self.dtype,
                mmap=self._mmap,
            )
        else:
            self._metadata = None
        return self

    def _maybe_expand_centroids(
        self,
        residual_norms: list[float],
        embeddings_source: list[torch.Tensor] | torch.Tensor | str | Path,
        device: str,
        min_outliers: int = 50,
        max_growth_rate: float = 0.1,
        max_points_per_centroid: int = 256,
    ) -> None:
        """Expand the centroid codebook if many new embeddings are outliers.

        An outlier is an embedding whose residual norm (distance to its
        nearest centroid) exceeds the cluster threshold stored at index
        creation time.
        """
        threshold_path = os.path.join(self.index, "cluster_threshold.npy")
        if not os.path.exists(threshold_path) or not residual_norms:
            return

        threshold = float(np.load(threshold_path).item())
        norms = np.array(residual_norms, dtype=np.float32)
        outlier_mask = norms > threshold
        outlier_count = int(outlier_mask.sum())

        if outlier_count < min_outliers:
            return

        # Collect outlier embeddings from the source
        if isinstance(embeddings_source, (str, Path)):
            logger.warning(
                "Centroid expansion with disk-based embeddings not yet supported, skipping."
            )
            return

        if isinstance(embeddings_source, torch.Tensor):
            all_embs = [embeddings_source[i] for i in range(embeddings_source.shape[0])]
        else:
            all_embs = embeddings_source

        # Flatten all embeddings and select outliers
        flat_embs = torch.cat(
            [e.squeeze(0) if e.dim() == 3 else e for e in all_embs], dim=0
        )
        outlier_embs = flat_embs[torch.from_numpy(outlier_mask)]

        # Determine K for new centroids
        meta = self._load_metadata()
        current_centroids = meta.get("num_centroids", 1) if meta else 1
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

        # Run K-means on outlier embeddings
        new_centroids, _ = compute_kmeans(
            embeddings_source=outlier_embs,
            device=device,
            kmeans_niters=4,
            max_points_per_centroid=max_points_per_centroid,
            seed=42,
            num_partitions_override=k_new,
        )

        # Append to codebook via Rust
        xtr_warp_rs.append_centroids_py(
            index=self.index,
            new_centroids=new_centroids,
        )
        self._metadata = None

    def _recalibrate_threshold(self, residual_norms: list[float]) -> None:
        """Update cluster_threshold.npy using a weighted average.

        We blend the old threshold (weighted by pre-existing embedding count)
        with the 75th-percentile of the new norms (weighted by the new
        embedding count).
        """
        threshold_path = os.path.join(self.index, "cluster_threshold.npy")
        if not os.path.exists(threshold_path) or not residual_norms:
            return

        old_threshold = float(np.load(threshold_path).item())
        new_norms = np.array(residual_norms, dtype=np.float32)
        new_count = len(new_norms)
        new_threshold = float(np.percentile(new_norms, 75))

        # metadata on disk already includes the just-added embeddings
        meta = self._load_metadata()
        total = meta.get("num_embeddings", new_count) if meta else new_count
        old_count = max(0, total - new_count)

        if old_count + new_count > 0:
            updated = (old_threshold * old_count + new_threshold * new_count) / (
                old_count + new_count
            )
        else:
            updated = new_threshold

        np.save(threshold_path, np.float32(updated))

    def _resolve_device(self, device: str | None) -> str:
        """Return *device* if given, otherwise fall back to ``self.device``."""
        if device is not None:
            return device
        if self.device is not None:
            return self.device
        raise ValueError(
            "No device specified. Pass device= or set it via __init__/load()."
        )

    def _prepare_embeddings(
        self,
        embeddings_source: list[torch.Tensor] | torch.Tensor | str | Path,
    ) -> list[torch.Tensor] | str:
        """Normalise embeddings_source into the format expected by Rust."""
        if isinstance(embeddings_source, torch.Tensor):
            return [embeddings_source[i] for i in range(embeddings_source.shape[0])]
        if isinstance(embeddings_source, list):
            return [e.squeeze(0) if e.dim() == 3 else e for e in embeddings_source]
        return str(embeddings_source)

    def _load_metadata(self) -> dict | None:
        """Load index metadata from disk if available."""
        if self._metadata is not None:
            return self._metadata

        metadata_path = os.path.join(self.index, "metadata.json")
        try:
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)
        except FileNotFoundError:
            logger.warning(
                "metadata.json not found in %s; using heuristic defaults", self.index
            )
            self._metadata = None
        except OSError as exc:
            logger.warning("Failed to load metadata from %s: %s", metadata_path, exc)
            self._metadata = None

        return self._metadata

    def _prepare_subsets(
        self,
        queries_embeddings: torch.Tensor,
        subset: list[list[int]] | list[int],
    ) -> list[list[int]]:
        """Validate and normalize subsets for search."""
        n_queries = queries_embeddings.shape[0]

        if len(subset) == 0:
            return [[]]

        # Shared subset: list[int] → wrap for broadcast
        if isinstance(subset[0], int):
            return [list(subset)]  # type: ignore[arg-type]

        # Per-query subsets: list[list[int]]
        if len(subset) != n_queries:
            error = (
                f"When passing per-query subsets, the number of subset lists "
                f"({len(subset)}) must match the number of queries ({n_queries})"
            )
            raise ValueError(error)
        return [list(s) for s in subset]  # type: ignore[arg-type]

    def load(
        self,
        device: str | list[str] | dict[str, float] = "auto",
        dtype: torch.dtype = torch.float32,
        mmap: bool = True,
    ) -> "XTRWarp":
        """Load an index to a specific device with the specified precision.

        Args:
        ----
        device:
            'auto', 'cpu', 'cuda', 'mps', or a list of devices, or a dict
            mapping device strings to ratios (e.g. ``{"cuda:0": 0.6, "cpu": 0.4}``).

            - ``str``: single device, same behavior as before.
            - ``list[str]``: auto-compute ratios (fill accelerator VRAM first,
              remainder on CPU). Enables index sharding.
            - ``dict[str, float]``: explicit ratios for each device. Enables
              index sharding across devices.
        dtype:
            valid torch dtype
        mmap:
            If True, memory-map the large index tensors (codes and residuals)
            instead of loading them into memory. Applied to CPU shards only
            when sharding is enabled.

        """
        if self._searcher is not None:
            logger.warning(
                "Index is already loaded, use free() before calling load() again."
            )
            return self

        if (
            (isinstance(device, str) and device == "mps")
            or (isinstance(device, list) and "mps" in device)
            or (isinstance(device, dict) and "mps" in device)
        ):
            raise ValueError("MPS is not supported")

        self._device_arg = device
        self.dtype = dtype
        self._mmap = mmap

        _ = self._load_metadata()

        # explicit ratios
        if isinstance(device, dict):
            return self._load_sharded(device, mmap=True)

        # auto-compute ratios
        if isinstance(device, list):
            ratios = self._compute_device_ratios(device)
            return self._load_sharded(ratios, mmap=True)

        # single device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        if mmap and device != "cpu":
            logger.warning(
                "mmap=True is only supported when device='cpu', disabling it"
            )
            mmap = False
            self._mmap = mmap

        return self._load_sharded({device: 1.0}, mmap=mmap)

    def _load_sharded(
        self,
        ratios: dict[str, float],
        mmap: bool = True,
    ) -> "XTRWarp":
        """Load the index, optionally sharded across multiple devices."""
        total = sum(ratios.values())
        if total <= 0:
            raise ValueError("Device ratios must sum to a positive number")
        ratios = {d: r / total for d, r in ratios.items()}

        # Soft VRAM check
        try:
            mem_est = xtr_warp_rs.estimate_index_memory(self.index)
            shardable = mem_est.get("pids", 0) + mem_est.get("residuals", 0)
            for dev, ratio in ratios.items():
                if dev.startswith("cuda") and torch.cuda.is_available():
                    dev_idx = int(dev.split(":")[-1]) if ":" in dev else 0
                    free, _ = torch.cuda.mem_get_info(dev_idx)
                    needed = ratio * shardable
                    overhead = mem_est.get("centroids", 0) + mem_est.get(
                        "bucket_weights", 0
                    )
                    if needed + overhead > free:
                        logger.warning(
                            "Device %s: estimated shard size %.0f MB + overhead %.0f MB "
                            "exceeds free VRAM %.0f MB — may OOM",
                            dev,
                            needed / 1e6,
                            overhead / 1e6,
                            free / 1e6,
                        )
        except Exception:
            pass  # best-effort check

        for dev in ratios:
            _ = self._ensure_torch_initialized(dev)

        device_ratios_list = list(ratios.items())
        searcher = xtr_warp_rs.ShardedSearcher(self.index, device_ratios_list, mmap)
        searcher.load()

        self._searcher = searcher
        self.devices = list(ratios.keys())
        self.device = self.devices[0]

        metadata_db = os.path.join(self.index, "metadata.duckdb")
        if os.path.exists(metadata_db):
            self._metadata_store = MetadataStore(self.index)

        return self

    def _compute_device_ratios(self, devices: list[str]) -> dict[str, float]:
        """Compute shard ratios: fill accelerator VRAM first, remainder to CPU."""
        try:
            mem_est = xtr_warp_rs.estimate_index_memory(self.index)
        except Exception as e:
            logger.warning(
                "Could not estimate index memory: %s — using equal ratios", e
            )
            n = len(devices)
            return {d: 1.0 / n for d in devices}

        shardable = mem_est.get("pids", 0) + mem_est.get("residuals", 0)
        if shardable == 0:
            n = len(devices)
            return {d: 1.0 / n for d in devices}

        accel_overhead = (
            mem_est.get("centroids", 0)
            + mem_est.get("bucket_weights", 0)
            + 50 * 1024 * 1024  # 50 MB reserve for matmul + allocator
        )

        accelerators = [d for d in devices if d != "cpu"]
        has_cpu = "cpu" in devices

        ratios: dict[str, float] = {}
        remaining = 1.0

        for i, dev in enumerate(accelerators):
            if remaining <= 0:
                break
            try:
                dev_idx = int(dev.split(":")[-1]) if ":" in dev else 0
                free, _ = torch.cuda.mem_get_info(dev_idx)
            except Exception:
                continue

            usable = free - (accel_overhead if i == 0 else 0)
            usable = max(0, usable)
            ratio = min(usable / shardable, remaining)
            ratios[dev] = ratio
            remaining -= ratio

        if remaining > 0 and has_cpu:
            ratios["cpu"] = remaining
        elif remaining > 0 and not has_cpu:
            # Distribute remainder proportionally across existing devices
            if ratios:
                assigned = sum(ratios.values())
                if assigned > 0:
                    for dev in ratios:
                        ratios[dev] /= assigned
            else:
                # Fallback: equal split
                for dev in devices:
                    ratios[dev] = 1.0 / len(devices)

        return ratios

    def estimate_index_memory(self) -> dict[str, int]:
        """Estimate memory in bytes for each index component.

        Returns a dict with keys: 'centroids', 'bucket_weights', 'pids',
        'residuals', 'sizes_and_offsets', 'total'.
        """
        return xtr_warp_rs.estimate_index_memory(self.index)

    def recommend_device_map(
        self,
        devices: list[str],
    ) -> dict[str, float]:
        """Suggest a device map based on available memory.

        Args:
        ----
        devices:
            List of devices to consider (e.g. ``["cuda:0", "cpu"]``).

        Returns:
        -------
        Dict mapping device → ratio.

        """
        return self._compute_device_ratios(devices)

    def optimize_hyperparams(
        self, top_k: int, queries_embeddings: torch.Tensor
    ) -> tuple[int, int, float, int, int] | None:
        """Optimize the search hyperparams based on search config and index density."""
        if self._metadata is None:
            return None

        num_embeddings = self._metadata["num_embeddings"]
        num_partitions = self._metadata["num_partitions"]
        avg_doclen = self._metadata["avg_doclen"]
        num_tokens = queries_embeddings.size(1)

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
        if num_partitions >= 65536 and num_tokens >= 48:
            nprobe = max(nprobe, 12)

        # bound controls how many centroids we score before pruning
        bound = max(nprobe * 8, int(0.05 * num_partitions))

        centroid_score_threshold = 0.5
        if density > 1000 or top_k >= 50:
            centroid_score_threshold -= 0.05
        if density > 2500 or top_k >= 200:
            centroid_score_threshold -= 0.05

        # allow more candidates on dense corpora and multi-token queries
        est_candidates = density * max(1, nprobe) * max(1, num_tokens)
        max_candidates = int(est_candidates)
        max_candidates = max(max_candidates, top_k * 50)
        max_candidates = min(max_candidates, num_embeddings) // 2

        # t_prime controls how aggressively we estimate and correct quantization error
        # we need to bump this up for dense/long-queries
        t_prime = int(density * max(1, nprobe) * max(1, num_tokens // 2))

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

    def filter(
        self,
        condition: str,
        parameters: list | tuple | None = None,
    ) -> list[int]:
        """Return passage IDs matching a metadata filter condition.

        Requires metadata to have been provided during ``create()`` or ``add()``.

        Args:
        ----
        condition:
            SQL WHERE clause fragment, e.g. ``"category = ? AND age > ?"``.
            DuckDB native functions are supported (``list_contains``, struct
            dot-notation, etc.).
        parameters:
            Values for ``?`` placeholders in *condition*.

        Returns:
        -------
        List of matching passage IDs.

        """
        if self._metadata_store is None:
            raise RuntimeError(
                "No metadata available. Ensure metadata was provided during "
                "create() or add(), and that the index is loaded."
            )
        return self._metadata_store.filter(condition, parameters)

    def search(
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int,
        num_threads: int | None = 1,
        bound: int | None = None,
        t_prime: int | None = None,
        nprobe: int | None = None,
        max_candidates: int | None = None,
        centroid_score_threshold: float | None = None,
        batch_size: int | None = 8192,
        subset: list[list[int]] | list[int] | None = None,
        show_progress: bool = True,
    ) -> list[list[tuple[int, float]]]:
        """Search the index for the given query embeddings.

        Args:
        ----
        queries_embeddings:
            Embeddings of the queries to search for.
        top_k:
            Number of top results to return.
        num_threads:
            Upper bound of threads to use for the search.
            Used only if index is loaded in cpu. Defaults to 1.
        bound:
            The number of centroids to consider per query. Defaults to None.
        nprobe:
            Number of inverted file probes to use. Defaults to None.
        t_prime:
            Value to use for the t_prime policy. Defaults to None.
        max_candidates:
            Maximum number of candidates to consider before the final sort.
        centroid_score_threshold:
            Threshold for centroid scores, from 0 to 1. Defaults to None.
        batch_size:
            Batch size for the query matmul against the centroids. Defaults to 8192.
        subset:
            Passage IDs to restrict the search to. Can be:
            - ``None``: no filtering (default).
            - ``list[int]``: a single subset applied to every query.
            - ``list[list[int]]``: per-query subsets whose length must equal
              the number of queries.

        """
        if self._searcher is None or self.devices is None:
            error = "Index not loaded, call load() first"
            raise RuntimeError(error)

        # Single-device CUDA uses rayon threads to fan per-query merger
        # launches across a CUDA stream pool, so num_threads > 1 lifts
        # throughput substantially. For single-query workloads it adds a
        # small per-call overhead (thread-pool dispatch + BLAS guard)
        if (
            not getattr(self, "_warned_cuda_threads_latency", False)
            and len(self.devices) == 1
            and num_threads is not None
            and num_threads > 1
            and self.devices[0].startswith("cuda")
        ):
            logger.warning(
                f"num_threads={num_threads} on single-device cuda maximizes batch "
                "throughput but adds ~0.3-0.4 ms overhead per search vs "
                "num_threads=1; use num_threads=1 if you are optimizing for "
                "latency serving"
            )
            self._warned_cuda_threads_latency = True

        if isinstance(queries_embeddings, list):
            queries_embeddings = torch.nn.utils.rnn.pad_sequence(
                sequences=[
                    embedding[0] if embedding.dim() == 3 else embedding
                    for embedding in queries_embeddings
                ],
                batch_first=True,
                padding_value=0.0,
            )

        if queries_embeddings.dim() == 2:
            queries_embeddings = queries_embeddings.unsqueeze(0)
        elif queries_embeddings.dim() != 3:
            error = f"Expected 2D or 3D tensor, got {queries_embeddings.dim()}D tensor"
            raise ValueError(error)

        device = self.devices[0].split(":")[0]

        if device != queries_embeddings.device.type:
            queries_embeddings = queries_embeddings.to(device)

        if self.dtype != queries_embeddings.dtype:
            queries_embeddings = queries_embeddings.to(self.dtype)

        if subset is not None:
            subset = self._prepare_subsets(queries_embeddings, subset)

        optimized = self.optimize_hyperparams(top_k, queries_embeddings)

        if optimized is None:
            err = "Index metadata could not be accessed"
            raise RuntimeError(err)

        if bound is None:
            bound = optimized[0]
        if nprobe is None:
            nprobe = optimized[1]
        if centroid_score_threshold is None:
            centroid_score_threshold = optimized[2]
        if max_candidates is None:
            max_candidates = optimized[3]
        if t_prime is None:
            t_prime = optimized[4]

        logger.debug(
            "Search hyperparams - bound=%s nprobe=%s centroid_score_threshold=%s max_candidates=%s t_prime=%s",
            bound,
            nprobe,
            centroid_score_threshold,
            max_candidates,
            t_prime,
        )

        search_config = xtr_warp_rs.SearchConfig(
            k=top_k,
            device=device,
            nprobe=nprobe,
            t_prime=t_prime,
            bound=bound,
            batch_size=batch_size,
            num_threads=num_threads,
            centroid_score_threshold=centroid_score_threshold,
            max_codes_per_centroid=None,
            max_candidates=max_candidates,
        )
        torch_path = self._ensure_torch_initialized(device)

        scores = search_on_device(
            torch_path=torch_path,
            queries_embeddings=queries_embeddings,
            search_config=search_config,
            loaded_index=self._searcher,
            subsets=subset,
            show_progress=show_progress,
        )

        return scores
