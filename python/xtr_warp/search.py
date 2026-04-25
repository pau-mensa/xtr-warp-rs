"""Public ``XTRWarp`` class: orchestrates index creation, mutation, loading,
and search.

Internals are split across:

- :mod:`xtr_warp.embedding_source` — passage embedding sources (memory / disk)
- :mod:`xtr_warp.kmeans` — centroid k-means
- :mod:`xtr_warp.hyperparams` — automatic search hyperparameter tuning
- :mod:`xtr_warp.device_planner` — VRAM-aware shard ratios
- :mod:`xtr_warp.index_maintenance` — centroid expansion and threshold recalibration
"""
from __future__ import annotations

import glob
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from . import xtr_warp_rs
from .device_planner import compute_device_ratios, warn_on_vram_overflow
from .filtering import MetadataStore
from .hyperparams import optimize as _optimize_hyperparams
from .index_maintenance import maybe_expand_centroids, recalibrate_threshold
from .kmeans import compute_kmeans

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
        self._torch_initialized: dict[str, str] = {}
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

    @contextmanager
    def _with_reload(self, *, reload: bool = True) -> Iterator[None]:
        """Free the index for the body of the with-block, optionally re-loading after.

        Used by ``add`` / ``update`` / ``compact`` to mutate the on-disk
        index while no searcher is holding it open. ``self._metadata`` is
        invalidated unconditionally — the on-disk metadata may have changed
        regardless of whether the index gets reloaded.
        """
        was_loaded = self._searcher is not None
        if was_loaded:
            self.free()
        try:
            yield
        finally:
            self._metadata = None
            if reload and was_loaded:
                self.load(
                    device=self._device_arg or self.devices,
                    dtype=self.dtype,
                    mmap=self._mmap,
                )

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
        sample_workers: int | None = None,
    ) -> "XTRWarp":
        """Create and saves the XTRWarp index.

        Args:
        ----
        embeddings_source:
            A list of document embeddings or the path to a folder where the embeddings
            are stored. The stored embeddings must be in `.npy` format,
            in a 2D tensor `[total_len, dim]` with a matching `.doclens.npy` sidecar.
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
            sample_workers=sample_workers,
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
            for pattern in ("*.json", "*.npy", "*.pt", "*.duckdb*"):
                for stale in glob.glob(os.path.join(index_path, pattern)):
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
        elif not os.path.exists(index_path):
            os.makedirs(index_path)

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

        with self._with_reload(reload=reload):
            result = xtr_warp_rs.add(
                index=self.index,
                torch_path=torch_path,
                device=device,
                embeddings=embeddings,
                show_progress=show_progress,
            )
            new_ids = result["new_passage_ids"]

            # Centroid expansion: detect outliers and grow the codebook.
            maybe_expand_centroids(
                index=self.index,
                residual_norms=result["residual_norms"],
                embeddings_source=embeddings_source,
                device=device,
                metadata=self._load_metadata(),
                min_outliers=min_outliers,
                max_growth_rate=max_growth_rate,
                max_points_per_centroid=max_points_per_centroid,
            )

            # Recalibrate the outlier threshold so it reflects the proper
            # data distribution.
            recalibrate_threshold(
                index=self.index,
                residual_norms=result["residual_norms"],
                metadata=self._load_metadata(),
            )

            if metadata is not None:
                store = MetadataStore(self.index)
                store.add(metadata, start_pid=new_ids[0])
                store.close()

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
        with self._with_reload(reload=reload):
            xtr_warp_rs.update(
                index=self.index,
                torch_path=torch_path,
                device=device,
                passage_ids=passage_ids,
                embeddings=embeddings,
                show_progress=show_progress,
            )
        return self

    def compact(self, reload: bool = True, show_progress: bool = True) -> "XTRWarp":
        """Rebuild index excluding deleted passages.

        Use after ``delete()`` to physically reclaim space.

        Args:
        ----
        reload:
            If True (default), automatically re-load after mutation.

        """
        device = self._resolve_device(None)
        torch_path = self._ensure_torch_initialized(device)

        with self._with_reload(reload=reload):
            # Read tombstones before compact (it clears the tombstone file).
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

            # Delete metadata after successful compact.
            if tombstones:
                store = MetadataStore(self.index)
                store.delete(tombstones)
                store.close()

        return self

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

        # Shared subset: list[int] → wrap for broadcast.
        if isinstance(subset[0], int):
            return [list(subset)]  # type: ignore[arg-type]

        # Per-query subsets: list[list[int]].
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

        # Explicit ratios.
        if isinstance(device, dict):
            return self._load_sharded(device, mmap=True)

        # Auto-compute ratios.
        if isinstance(device, list):
            ratios = compute_device_ratios(device, self.index)
            return self._load_sharded(ratios, mmap=True)

        # Single device.
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

        warn_on_vram_overflow(ratios, self.index)

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
        return compute_device_ratios(devices, self.index)

    def optimize_hyperparams(
        self, top_k: int, queries_embeddings: torch.Tensor
    ) -> tuple[int, int, float, int, int] | None:
        """Optimize the search hyperparams based on search config and index density."""
        return _optimize_hyperparams(
            metadata=self._metadata,
            top_k=top_k,
            num_query_tokens=queries_embeddings.size(1),
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
        # small per-call overhead (thread-pool dispatch + BLAS guard).
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
