"""Passage embedding sources used by index creation and k-means sampling.

Two backing stores:

- :class:`InMemorySource` — a list of in-memory tensors, one per passage.
- :class:`DiskSource` — a directory of ``.npy`` files with ``.doclens.npy``
  sidecars. Reads are parallelised across the files that own sampled passage
  ids, using mmap so threads release the GIL during page faults.

:func:`create_source` is the factory used by callers that don't care which
backing store they get.
"""
from __future__ import annotations

import os
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import torch


class EmbeddingSource(Protocol):
    """Protocol for embedding sources used during index creation."""

    def get_num_passages(self) -> int: ...
    def sample_embeddings(self, pids: list[int]) -> tuple[torch.Tensor, int, int]: ...


@dataclass
class InMemorySource:
    """Source for embeddings already in memory."""

    embeddings: list[torch.Tensor]

    def get_num_passages(self) -> int:
        return len(self.embeddings)

    def sample_embeddings(self, pids: list[int]) -> tuple[torch.Tensor, int, int]:
        samples = [self.embeddings[pid] for pid in pids]
        dim = samples[0].size(-1)
        total_tokens = sum(sample.shape[0] for sample in samples)
        tensors = torch.cat(tensors=samples)
        return tensors, total_tokens, dim


@dataclass
class DiskSource:
    """Source for embeddings stored as ``.npy`` files with ``.doclens.npy`` sidecars."""

    path: Path
    # Max threads for parallel mmap reads during k-means sampling. ``None``
    # picks ``min(num_files, cpu_count, 8)``.
    sample_workers: Optional[int] = None
    _files_and_doclens: list[tuple[Path, np.ndarray]] | None = None
    _doclens: list[int] | None = None
    _num_passages: int = 0
    # Cumulative pid count up to (but not including) each file, so
    # `bisect_right(_file_start_pids, pid) - 1` resolves a pid to the
    # owning file in O(log num_files). Lets the parallel sampler skip
    # files that own no sampled pids.
    _file_start_pids: list[int] | None = None
    # Per-file `cumsum(sidecar)` with a leading 0, so a doc's row range
    # in its file is `token_offsets[i:i+2]`. Avoids re-walking sidecars
    # in the inner loop.
    _file_token_offsets: list[np.ndarray] | None = None

    def _load_metadata(self) -> None:
        if self._files_and_doclens is not None:
            return

        files = _get_all_embedding_files(self.path)
        self._doclens = []
        self._files_and_doclens = []
        self._file_start_pids = []
        self._file_token_offsets = []
        self._num_passages = 0

        pid_offset = 0
        for file in files:
            doclens_file = _doclens_path_for(file)
            sidecar = np.load(doclens_file).astype(np.int64, copy=False)
            self._num_passages += int(len(sidecar))
            self._doclens.extend(sidecar.tolist())
            self._files_and_doclens.append((file, sidecar))
            self._file_start_pids.append(pid_offset)

            token_offsets = np.empty(len(sidecar) + 1, dtype=np.int64)
            token_offsets[0] = 0
            if len(sidecar) > 0:
                np.cumsum(sidecar, out=token_offsets[1:])
            self._file_token_offsets.append(token_offsets)
            pid_offset += int(len(sidecar))

    def get_num_passages(self) -> int:
        self._load_metadata()
        return self._num_passages

    def _resolve_sample_workers(self, num_files: int) -> int:
        if self.sample_workers is not None:
            return max(1, int(self.sample_workers))
        return min(num_files, os.cpu_count() or 1, 8)

    def sample_embeddings(
        self, pids: list[int]
    ) -> tuple[torch.Tensor, int, int]:
        """Read only files that own a sampled pid; copy per-doc slices via
        mmap; parallelise across files on a thread pool. NumPy disk reads /
        mmap page faults release the GIL, so threads parallelise this
        I/O-bound work effectively.
        """
        self._load_metadata()
        if not pids:
            raise ValueError("No passage IDs provided for sampling")
        assert self._files_and_doclens is not None
        assert self._file_start_pids is not None
        assert self._file_token_offsets is not None
        assert self._doclens is not None

        # Ascending-pid iteration order so the output tensor layout is
        # stable for callers that depend on it (e.g. seeded k-means).
        ordered_pids = sorted(set(pids))
        total_tokens = sum(self._doclens[pid] for pid in ordered_pids)

        requests_by_file: dict[int, list[tuple[int, int, int]]] = {}
        write_offset = 0
        for pid in ordered_pids:
            file_idx = bisect_right(self._file_start_pids, pid) - 1
            local_doc_idx = pid - self._file_start_pids[file_idx]
            doc_len = int(self._doclens[pid])
            requests_by_file.setdefault(file_idx, []).append(
                (local_doc_idx, write_offset, doc_len)
            )
            write_offset += doc_len

        active_file_indices = sorted(requests_by_file.keys())
        if not active_file_indices:
            raise ValueError("Could not sample embeddings from source")

        # Probe the first hit file for dim/dtype (mmap a single row, copy
        # to a fresh buffer to avoid the mmap-readonly torch warning).
        first_file_path = self._files_and_doclens[active_file_indices[0]][0]
        first_data = np.load(first_file_path, mmap_mode="r")
        dim = int(first_data.shape[-1])
        dtype = torch.from_numpy(np.array(first_data[:1], copy=True)).dtype
        del first_data

        tensors = torch.empty((total_tokens, dim), dtype=dtype)

        max_workers = self._resolve_sample_workers(len(active_file_indices))

        def _copy_file_samples(file_idx: int) -> None:
            file_path, _ = self._files_and_doclens[file_idx]
            token_offsets = self._file_token_offsets[file_idx]
            data = np.load(file_path, mmap_mode="r")
            for local_doc_idx, out_off, doc_len in requests_by_file[file_idx]:
                tok_off = int(token_offsets[local_doc_idx])
                doc = torch.from_numpy(
                    np.array(data[tok_off : tok_off + doc_len], copy=True)
                )
                tensors[out_off : out_off + doc_len].copy_(doc)

        if max_workers <= 1 or len(active_file_indices) == 1:
            for file_idx in active_file_indices:
                _copy_file_samples(file_idx)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_copy_file_samples, file_idx)
                    for file_idx in active_file_indices
                ]
                for fut in as_completed(futures):
                    fut.result()

        return tensors, total_tokens, dim


def create_source(
    embeddings_source: list[torch.Tensor] | Path,
    *,
    sample_workers: Optional[int] = None,
) -> EmbeddingSource:
    """Wrap *embeddings_source* in the appropriate :class:`EmbeddingSource`.

    ``sample_workers`` only applies to the disk-backed branch.
    """
    if isinstance(embeddings_source, list):
        return InMemorySource(embeddings_source)
    return DiskSource(embeddings_source, sample_workers=sample_workers)


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
