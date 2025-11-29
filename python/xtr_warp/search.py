from __future__ import annotations

import glob
import logging
import math
import os
import random
from typing import Literal

import torch
import torch.multiprocessing as mp
import xtr_warp_rust
from fastkmeans import FastKMeans
from joblib import Parallel, delayed

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


def search_on_device(  # noqa: PLR0913
    device: str,
    dtype: Literal["float64", "float32", "float16", "bfloat16"],
    queries_embeddings: torch.Tensor,
    top_k: int,
    nprobe: int,
    index: str,
    torch_path: str,
    t_prime: int | None,
    bound: int,
    max_candidates: int,
    centroid_score_threshold: float | None = None,
) -> list[list[tuple[int, float]]]:
    """Perform a search on a single specified device."""
    # Ensure queries_embeddings is 3D [batch, num_tokens, dim]
    if queries_embeddings.dim() == 2:
        queries_embeddings = queries_embeddings.unsqueeze(0)  # [1, tokens, dim]
    elif queries_embeddings.dim() != 3:
        raise ValueError(
            f"Expected 2D or 3D tensor, got {queries_embeddings.dim()}D tensor"
        )

    search_config = xtr_warp_rust.SearchConfig(
        k=top_k,
        device=device,
        dtype=dtype,
        nprobe=nprobe,
        t_prime=t_prime,
        bound=bound,
        parallel=False,
        num_threads=1,
        centroid_score_threshold=centroid_score_threshold,
        max_codes_per_centroid=None,
        max_candidates=max_candidates,
    )

    scores = xtr_warp_rust.load_and_search(
        index=index,
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
    device:
        The device(s) to use for computation (e.g., "cuda", ["cuda:0", "cuda:1"]).
        If None, defaults to ["cuda"].

    """

    def __init__(
        self,
        index: str,
        device: str | list[str] | None = None,
    ) -> None:
        self.multiple_gpus = False
        if (
            isinstance(device, list)
            and len(device) > 1
            and torch.cuda.device_count() > 1
        ):
            self.multiple_gpus = True
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method(method="spawn", force=True)

        if device is None and torch.cuda.is_available():
            self.devices = ["cuda"]
        elif not torch.cuda.is_available():
            cpu_count = os.cpu_count()
            if cpu_count is None:
                error = """
                No CPU cores available. Please check your system configuration.
                >>> import os; print(os.cpu_count())
                Returns None.
                """
                raise RuntimeError(error)
            self.devices = ["cpu"] * cpu_count
        elif isinstance(device, str):
            self.devices = [device]
        elif isinstance(device, list):
            self.devices = device
        else:
            error = "Device must be a string, a list of strings, or None."
            raise ValueError(error)

        self.torch_path = _load_torch_path(device=self.devices[0])
        self.index = index

        if self.multiple_gpus:
            return

        xtr_warp_rust.initialize_torch(
            torch_path=self.torch_path,
        )

    def create(  # noqa: PLR0913
        self,
        documents_embeddings: list[torch.Tensor] | torch.Tensor,
        kmeans_niters: int = 4,
        max_points_per_centroid: int = 256,
        nbits: int = 4,
        nprobe: int = 8,
        t_prime: int | None = None,
        n_samples_kmeans: int | None = None,
        seed: int = 42,
        use_triton_kmeans: bool | None = None,
    ) -> "XTRWarp":
        """Create and saves the XTRWarp index.

        Args:
        ----
        documents_embeddings:
            A list of document embedding tensors to be indexed.
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
        metadata:
            Optional list of dictionaries containing metadata for each document.

        """
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
            device=self.devices[0],
            max_points_per_centroid=max_points_per_centroid,
            n_samples_kmeans=n_samples_kmeans,
            seed=seed,
            use_triton_kmeans=use_triton_kmeans,
        )

        xtr_warp_rust.create(
            index=self.index,
            torch_path=self.torch_path,
            device=self.devices[0],
            nbits=nbits,
            nprobe=nprobe,
            embeddings=documents_embeddings,
            centroids=centroids,
            t_prime=t_prime,
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

    def search(  # noqa: PLR0913, C901
        self,
        queries_embeddings: torch.Tensor | list[torch.Tensor],
        top_k: int,
        bound: int = 128,  # Could be set dynamically using the index size
        t_prime: int | None = None,
        nprobe: int | None = None,
        max_candidates: int | None = None,
        centroid_score_threshold: float | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[list[tuple[int, float]]]:
        """Search the index for the given query embeddings.

        Args:
        ----
        queries_embeddings:
            Embeddings of the queries to search for.
        top_k:
            Number of top results to return.
        bound:
            The number of centroids to consider per query. Defaults to 128.
        nprobe:
            Number of inverted file probes to use.
        t_prime:
            Value to use for the t_prime policy. Defaults to None.
        max_candidates:
            Maximum number of candidates to consider before the final sort.
        centroid_score_threshold:
            Threshold for centroid scores.
        dtype:
            Data type to use for the search.

        """
        if top_k <= 10:
            if nprobe is None:
                nprobe = 1
            if max_candidates is None:
                max_candidates = 256
            if centroid_score_threshold is None:
                centroid_score_threshold = 0.5
        elif top_k <= 100:
            if nprobe is None:
                nprobe = 2
            if max_candidates is None:
                max_candidates = 1024
            if centroid_score_threshold is None:
                centroid_score_threshold = 0.45
        else:
            if nprobe is None:
                nprobe = 4
            if max_candidates is None:
                max_candidates = max(top_k * 4, 4096)
            if centroid_score_threshold is None:
                centroid_score_threshold = 0.4

        if isinstance(queries_embeddings, list):
            queries_embeddings = torch.nn.utils.rnn.pad_sequence(
                sequences=[
                    embedding[0] if embedding.dim() == 3 else embedding
                    for embedding in queries_embeddings
                ],
                batch_first=True,
                padding_value=0.0,
            )

        num_queries = queries_embeddings.shape[0]
        dtype = dtype or torch.float32

        if dtype != queries_embeddings.dtype:
            logger.warning(
                f"Query embeddings and dtype selection mismatch ({dtype} != {queries_embeddings.dtype}). Casting embeddings to avoid errors."
            )
            queries_embeddings = queries_embeddings.to(dtype=dtype)

        dtype_str: str = str(dtype).split(".")[1]

        if not self.multiple_gpus and len(self.devices) > 1:
            split_size = (num_queries // len(self.devices)) + 1
            queries_embeddings_splits = torch.split(
                tensor=queries_embeddings,
                split_size_or_sections=split_size,
            )

            tasks = [
                delayed(function=search_on_device)(
                    device=device,
                    dtype=dtype_str,
                    queries_embeddings=dev_queries,
                    top_k=top_k,
                    bound=bound,
                    nprobe=nprobe,
                    index=self.index,
                    t_prime=t_prime,
                    torch_path=self.torch_path,
                    centroid_score_threshold=centroid_score_threshold,
                    max_candidates=max_candidates,
                )
                for device, dev_queries in zip(self.devices, queries_embeddings_splits)
            ]

            scores_per_device = Parallel(n_jobs=len(self.devices))(tasks)

            scores = []
            for device_scores in scores_per_device:
                scores.extend(device_scores)

            return scores

        if not self.multiple_gpus:
            return search_on_device(
                device=self.devices[0],
                dtype=dtype_str,
                queries_embeddings=queries_embeddings,
                top_k=top_k,
                bound=bound,
                nprobe=nprobe,
                index=self.index,
                t_prime=t_prime,
                torch_path=self.torch_path,
                centroid_score_threshold=centroid_score_threshold,
                max_candidates=max_candidates,
            )

        split_size = (num_queries // len(self.devices)) + 1
        queries_embeddings_splits = torch.split(
            tensor=queries_embeddings,
            split_size_or_sections=split_size,
        )

        args_for_starmap = [
            (
                device,
                dtype_str,
                dev_queries,
                top_k,
                bound,
                nprobe,
                self.index,
                t_prime,
                self.torch_path,
                centroid_score_threshold,
                max_candidates,
            )
            for device, dev_queries in zip(self.devices, queries_embeddings_splits)
        ]

        scores_devices = []

        context = mp.get_context()
        with context.Pool(processes=len(args_for_starmap)) as pool:
            scores_devices = pool.starmap(
                func=search_on_device,
                iterable=args_for_starmap,
            )

        scores = []
        for scores_device in scores_devices:
            scores.extend(scores_device)

        return scores
