from __future__ import annotations

import glob
import json
import logging
import math
import os
import random
from typing import Literal

import torch
import torch.multiprocessing as mp
from fastkmeans import FastKMeans
from joblib import Parallel, delayed

from . import xtr_warp_rust

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


def compute_kmeans(  # noqa: PLR0913
    documents_embeddings: list[torch.Tensor],
    dim: int,
    device: str,
    kmeans_niters: int,
    max_points_per_centroid: int,
    seed: int,
    n_samples_kmeans: int | None = None,
    use_triton_kmeans: bool | None = None,
) -> torch.Tensor:
    """Compute K-means centroids for document embeddings."""
    num_passages = len(documents_embeddings)

    if n_samples_kmeans is None:
        n_samples_kmeans = min(
            1 + int(16 * math.sqrt(120 * num_passages)),
            num_passages,
        )

    n_samples_kmeans = min(num_passages, n_samples_kmeans)

    sampled_pids = random.sample(
        population=range(n_samples_kmeans),
        k=n_samples_kmeans,
    )

    samples: list[torch.Tensor] = [
        documents_embeddings[pid] for pid in set(sampled_pids)
    ]

    total_tokens = sum([sample.shape[0] for sample in samples])
    num_partitions = (total_tokens / len(samples)) * len(documents_embeddings)
    num_partitions = int(2 ** math.floor(math.log2(16 * math.sqrt(num_partitions))))

    tensors = torch.cat(tensors=samples)
    if tensors.is_cuda:
        tensors = tensors.to(device="cpu", dtype=torch.float32)

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
    )


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
        self.dtype: str | None = None
        self._torch_initialized = {}
        self._metadata: dict | None = None
        self.selector_device: str | None = None
        self.decompress_device: str | None = None

    def _ensure_torch_initialized(self, device: str) -> str:
        """Initialize torch once per device type."""
        device_type = device.split(":")[0]  # 'cuda:0' -> 'cuda'
        if device_type not in self._torch_initialized:
            torch_path = _load_torch_path(device=device_type)
            xtr_warp_rust.initialize_torch(torch_path)
            self._torch_initialized[device_type] = torch_path
        return self._torch_initialized[device_type]

    def free(self) -> None:
        """Free the loaded index from memory."""
        if self._loaded_searchers is not None:
            for searcher in self._loaded_searchers:
                searcher.free()
            self._loaded_searchers = None

    def __del__(self):
        """Destructor."""
        self.free()

    def create(  # noqa: PLR0913
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
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
        documents_embeddings:
            A list of document embedding tensors to be indexed.
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
        if isinstance(documents_embeddings, torch.Tensor):
            documents_embeddings = [
                documents_embeddings[i] for i in range(documents_embeddings.shape[0])
            ]

        documents_embeddings = [
            embedding.squeeze(0) if embedding.dim() == 3 else embedding
            for embedding in documents_embeddings
        ]

        self._prepare_index_directory(index_path=self.index)

        dim = documents_embeddings[0].shape[-1]

        centroids = compute_kmeans(
            documents_embeddings=documents_embeddings,
            dim=dim,
            kmeans_niters=kmeans_niters,
            device=device,
            max_points_per_centroid=max_points_per_centroid,
            n_samples_kmeans=n_samples_kmeans,
            seed=seed,
            use_triton_kmeans=use_triton_kmeans,
        )

        xtr_warp_rust.create(
            index=self.index,
            torch_path=torch_path,
            device=device,
            nbits=nbits,
            embeddings=documents_embeddings,
            centroids=centroids,
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
        decompress_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "XTRWarp":
        """Load an index to a specific device with the specified precision.

        Args:
        ----
        device:
            'auto', 'cpu', 'cuda', 'mps', or a list of cuda devices
                (eg. ['cuda:0', 'cuda:1']
            auto, cuda, mps and a list of cuda devices keep the index on CPU
                but run centroid scoring on the accelerator.
        dtype:
            valid torch dtype

        """
        if self._loaded_searchers is not None:
            logger.warning(
                "Index is already loaded, use free() before calling load() again."
            )
            return self

        devices = [device] if isinstance(device, str) else device
        dtype_str = str(dtype).split(".")[1]
        self.dtype = dtype_str

        _ = self._load_metadata()
        self.devices = devices

        if device == "auto":
            self.selector_device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            self.decompress_device = decompress_device
            self.devices = [self.decompress_device]
            _ = self._ensure_torch_initialized(self.selector_device)
        elif isinstance(device, list):
            self.selector_device = devices[0]
            self.decompress_device = devices[0]
            self.devices = devices
        else:
            self.selector_device = device
            self.decompress_device = decompress_device
            # Keep the loaded searcher on the decompression device (e.g., cuda for speed).
            self.devices = [self.decompress_device]

        self._loaded_searchers = []
        # Ensure torch is initialized on both selector and decompression devices if they differ.
        _ = self._ensure_torch_initialized(self.selector_device)
        if self.decompress_device != self.selector_device:
            _ = self._ensure_torch_initialized(self.decompress_device)

        for d in self.devices:
            searcher = xtr_warp_rust.LoadedSearcher(self.index, d, dtype_str)
            searcher.load()
            self._loaded_searchers.append(searcher)

        return self

    def optimize_hyperparams(
        self, top_k: int, queries_embeddings: torch.Tensor
    ) -> tuple[int, int, float, int] | None:
        """Optimize the search hyperparams based on search config and index density."""
        if self._metadata is None:
            return None

        density = self._metadata.get("num_embeddings", 0) / max(
            1, self._metadata.get("num_partitions", 1)
        )
        num_tokens = queries_embeddings.size(1)
        num_partitions = self._metadata.get("num_partitions", 128)

        def _clamp(v: float, low: int, high: int) -> int:
            return max(low, min(int(v), high))

        # Probe more aggressively on denser indices even for small top_k
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

        # Bound controls how many centroids we score before pruning
        bound = max(nprobe * 8, int(0.05 * num_partitions))

        centroid_score_threshold = 0.5
        if density > 1000 or top_k >= 50:
            centroid_score_threshold -= 0.05
        if density > 2500 or top_k >= 200:
            centroid_score_threshold -= 0.05

        # Allow more candidates on dense corpora and multi-token queries
        est_candidates = density * max(1, nprobe) * max(1, num_tokens) * 1.5
        max_candidates = int(est_candidates)
        max_candidates = max(max_candidates, top_k * 50)
        max_candidates = min(
            max_candidates,
            self._metadata.get("num_embeddings", max_candidates),
        )

        return bound, nprobe, centroid_score_threshold, max_candidates

    def search(
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int,
        num_threads: int = 1,
        bound: int | None = None,
        t_prime: int | None = None,
        nprobe: int | None = None,
        max_candidates: int | None = None,
        centroid_score_threshold: float | None = None,
    ) -> list[list[tuple[int, float]]]:
        """Search the index for the given query embeddings.

        Args:
        ----
        queries_embeddings:
            Embeddings of the queries to search for.
        top_k:
            Number of top results to return.
        num_threads:
            Number of threads to use for the search.
            Used only if index is loaded in cpu. Defaults to 1.
        bound:
            The number of centroids to consider per query. Defaults to None.
        nprobe:
            Number of inverted file probes to use.
        t_prime:
            Value to use for the t_prime policy. Defaults to None.
        max_candidates:
            Maximum number of candidates to consider before the final sort.
        centroid_score_threshold:
            Threshold for centroid scores.

        """
        if (
            self._loaded_searchers is None
            or self.devices is None
            or self.selector_device is None
            or self.decompress_device is None
        ):
            error = "Index not loaded, call load() first"
            raise RuntimeError(error)

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

        if self.decompress_device != "cpu":
            if queries_embeddings.device.type != self.decompress_device.split(":")[0]:
                queries_embeddings = queries_embeddings.to(self.decompress_device)
        else:
            # Ensure queries are on CPU when decompression runs on CPU to avoid device mismatch.
            if queries_embeddings.is_cuda:
                queries_embeddings = queries_embeddings.to("cpu")

        optimized = self.optimize_hyperparams(top_k, queries_embeddings)

        if optimized is None:
            err = "Index metadata could not be accessed"
            raise RuntimeError(err)

        print(
            f"Bound: {optimized[0]}, nprobe: {optimized[1]}, centroid score threshold: {optimized[2]}, max candidates: {optimized[3]}"
        )

        if bound is None:
            bound = optimized[0]
        if nprobe is None:
            nprobe = optimized[1]
        if centroid_score_threshold is None:
            centroid_score_threshold = optimized[2]
        if max_candidates is None:
            max_candidates = optimized[3]

        search_config = xtr_warp_rust.SearchConfig(
            k=top_k,
            selector_device=self.selector_device,
            decompress_device=self.decompress_device,
            dtype=self.dtype,
            nprobe=nprobe,
            t_prime=t_prime,
            bound=bound,
            num_threads=num_threads,
            centroid_score_threshold=centroid_score_threshold,
            max_codes_per_centroid=None,
            max_candidates=max_candidates,
        )
        torch_path = self._ensure_torch_initialized(self.selector_device)
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
