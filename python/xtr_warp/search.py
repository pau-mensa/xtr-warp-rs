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
import torch.multiprocessing as mp
from fastkmeans import FastKMeans

from . import xtr_warp_rs

# from ..filtering import create, delete, update
#
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
    embeddings_source: list[torch.Tensor] | Path,
    device: str,
    kmeans_niters: int,
    max_points_per_centroid: int,
    seed: int,
    n_samples_kmeans: int | None = None,
    use_triton_kmeans: bool | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute K-means centroids for document embeddings."""
    source = _create_source(embeddings_source)
    num_passages = source.get_num_passages()

    if n_samples_kmeans is None:
        n_samples_kmeans = min(
            1 + int(16 * math.sqrt(120 * num_passages)),
            num_passages,
        )

    # Sample passages
    rng = random.Random(seed)
    sampled_pids = rng.sample(range(num_passages), k=n_samples_kmeans)

    tensors, total_tokens, dim = source.sample_embeddings(sampled_pids)

    num_partitions = (total_tokens / n_samples_kmeans) * num_passages
    num_partitions = int(2 ** math.floor(math.log2(16 * math.sqrt(num_partitions))))

    # I don't want any surprises here
    if tensors.is_cuda:
        tensors = tensors.to(device="cpu")
    if tensors.dtype != torch.float32:
        tensors = tensors.to(dtype=torch.float32)
    if not tensors.is_contiguous():
        tensors = tensors.contiguous()

    kmeans = FastKMeans(
        d=dim,
        k=min(num_partitions, total_tokens),
        niter=kmeans_niters,
        gpu=device != "cpu",
        verbose=False,
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        use_triton=use_triton_kmeans,
    )

    kmeans.train(data=tensors.numpy())

    centroids = torch.from_numpy(
        kmeans.centroids,
    ).to(
        device=device,
        dtype=torch.float32,
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
) -> list[list[tuple[int, float]]]:
    """Perform a search on a loaded index."""
    scores = loaded_index.search(
        torch_path=torch_path,
        queries_embeddings=queries_embeddings,
        search_config=search_config,
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
    ) -> None:
        self._loaded_searchers: list | None = None
        self.index: str = index
        self.devices: list | None = None
        self.dtype: torch.dtype | None = None
        self._torch_initialized = {}
        self._metadata: dict | None = None
        self.device: str | None = None
        self._mmap: bool = True

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
        if self._loaded_searchers is not None:
            for searcher in self._loaded_searchers:
                searcher.free()
            self._loaded_searchers = None

    def _reload_if_loaded(self) -> None:
        """Free and re-load the index using the same parameters as the last ``load()`` call."""
        if self._loaded_searchers is None or self.devices is None:
            return
        self.free()
        self._metadata = None
        self.load(device=self.devices, dtype=self.dtype, mmap=self._mmap)

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
        )

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
        elif not os.path.exists(index_path):
            try:
                os.makedirs(index_path)
            except OSError as e:
                raise e

    def delete(self, passage_ids: list[int]) -> "XTRWarp":
        """Delete passages by ID. O(1) tombstone operation.

        Search automatically filters deleted passages. To physically
        remove deleted data, call ``compact()`` afterward.

        Args:
        ----
        passage_ids:
            List of passage IDs to mark as deleted.

        """
        xtr_warp_rs.delete(self.index, passage_ids)
        if self._loaded_searchers:
            for s in self._loaded_searchers:
                s.update_tombstones(passage_ids)
        return self

    def add(
        self,
        embeddings_source: list[torch.Tensor] | torch.Tensor | str | Path,
        device: str,
        reload: bool = True,
    ) -> list[int]:
        """Add new passages. Encodes new documents and recompacts the index.

        Args:
        ----
        embeddings_source:
            New document embeddings (same formats as ``create``).
        device:
            Compute device (e.g. "cpu", "cuda:0").
        reload:
            If True (default) and the index was loaded, automatically
            free and re-load so searches reflect the new data.
            Set to False when batching several mutations before a
            manual ``load()`` call.

        Returns:
        -------
        List of newly assigned passage IDs.

        """
        torch_path = self._ensure_torch_initialized(device)
        embeddings = self._prepare_embeddings(embeddings_source)
        new_ids = xtr_warp_rs.add(
            index=self.index,
            torch_path=torch_path,
            device=device,
            embeddings=embeddings,
        )
        if reload:
            self._reload_if_loaded()
        else:
            if self._loaded_searchers:
                self.free()
            self._metadata = None
        return new_ids

    def update(
        self,
        passage_ids: list[int],
        embeddings_source: list[torch.Tensor] | torch.Tensor | str | Path,
        device: str,
        reload: bool = True,
    ) -> "XTRWarp":
        """Update passages in-place: new embeddings, same IDs.

        Args:
        ----
        passage_ids:
            IDs of passages to update.
        embeddings_source:
            Replacement embeddings (one per passage ID).
        device:
            Compute device.
        reload:
            If True (default), automatically re-load after mutation.

        """
        torch_path = self._ensure_torch_initialized(device)
        embeddings = self._prepare_embeddings(embeddings_source)
        xtr_warp_rs.update(
            index=self.index,
            torch_path=torch_path,
            device=device,
            passage_ids=passage_ids,
            embeddings=embeddings,
        )
        if reload:
            self._reload_if_loaded()
        else:
            if self._loaded_searchers:
                self.free()
            self._metadata = None
        return self

    def compact(self, device: str, reload: bool = True) -> "XTRWarp":
        """Rebuild index excluding deleted passages.

        Use after ``delete()`` to physically reclaim space.

        Args:
        ----
        device:
            Compute device.
        reload:
            If True (default), automatically re-load after mutation.

        """
        torch_path = self._ensure_torch_initialized(device)
        xtr_warp_rs.compact(
            index=self.index,
            torch_path=torch_path,
            device=device,
        )
        if reload:
            self._reload_if_loaded()
        else:
            if self._loaded_searchers:
                self.free()
            self._metadata = None
        return self

    def _prepare_embeddings(
        self,
        embeddings_source: list[torch.Tensor] | torch.Tensor | str | Path,
    ) -> list[torch.Tensor] | str:
        """Normalise embeddings_source into the format expected by Rust."""
        if isinstance(embeddings_source, torch.Tensor):
            return [
                embeddings_source[i] for i in range(embeddings_source.shape[0])
            ]
        if isinstance(embeddings_source, list):
            return [
                e.squeeze(0) if e.dim() == 3 else e for e in embeddings_source
            ]
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

    def load(
        self,
        device: str | list[str] = "auto",
        dtype: torch.dtype = torch.float32,
        mmap: bool = True,
    ) -> "XTRWarp":
        """Load an index to a specific device with the specified precision.

        Args:
        ----
        device:
            'auto', 'cpu', 'cuda', 'mps', or a list of cuda devices
                (eg. ['cuda:0', 'cuda:1'])
            auto, cuda, mps and a list of cuda devices keep the index on CPU
                but run centroid scoring on the accelerator.
        dtype:
            valid torch dtype
        mmap:
            If True, memory-map the large index tensors (codes and residuals)
            instead of loading them into memory. Only supported on CPU.

        """
        if self._loaded_searchers is not None:
            logger.warning(
                "Index is already loaded, use free() before calling load() again."
            )
            return self

        devices = [device] if isinstance(device, str) else device
        dtype_str = str(dtype).split(".")[1]
        self.dtype = dtype

        _ = self._load_metadata()
        self.devices = devices

        if device == "auto":
            inferred_device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            self.devices = [inferred_device]
        else:
            self.devices = devices

        if mmap and any(d != "cpu" for d in self.devices):
            logger.warning(
                "mmap=True is only supported when device='cpu', disabling it"
            )
            mmap = False

        self._mmap = mmap
        self._loaded_searchers = []
        for d in self.devices:
            _ = self._ensure_torch_initialized(d)
            searcher = xtr_warp_rs.LoadedSearcher(self.index, d, dtype_str, mmap)
            searcher.load()
            self._loaded_searchers.append(searcher)

        return self

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

        """
        if self._loaded_searchers is None or self.devices is None:
            error = "Index not loaded, call load() first"
            raise RuntimeError(error)

        if (
            num_threads is not None
            and num_threads > 1
            and self.devices[0].startswith("cuda")
        ):
            warning = (
                "num_threads > 1 is not supported for cuda devices, defaulting to 1"
            )
            logger.warning(warning)
            num_threads = 1

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
            dtype=str(self.dtype).split(".")[1],
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
        if len(self.devices) == 1:
            scores = search_on_device(
                torch_path=torch_path,
                queries_embeddings=queries_embeddings,
                search_config=search_config,
                loaded_index=self._loaded_searchers[0],
            )
        else:
            num_queries = queries_embeddings.shape[0]
            split_size = (num_queries // len(self.devices)) + 1
            queries_embeddings_splits = torch.split(
                tensor=queries_embeddings, split_size_or_sections=split_size
            )

            args_for_starmap = [
                (search_config, dev_queries, loaded_index, torch_path)
                for loaded_index, dev_queries in zip(
                    self._loaded_searchers, queries_embeddings_splits
                )
            ]

            scores_devices = []

            context = mp.get_context()
            with context.Pool(processes=len(args_for_starmap)) as pool:
                scores_devices = pool.starmap(
                    func=search_on_device, iterable=args_for_starmap
                )
            scores = []
            for scores_device in scores_devices:
                scores.extend(scores_device)

        return scores
