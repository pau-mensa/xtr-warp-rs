import os
import shutil

import numpy as np
import torch
from xtr_warp.search import XTRWarp


def test():
    """Ensure that the WARP search index can be created and queried correctly."""
    index_name = ".indices/test_index"

    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    index = XTRWarp(index=index_name)

    documents_embeddings = [torch.randn(300, 128, device="cpu") for _ in range(100)]

    queries_embeddings = torch.randn(10, 30, 128, device="cpu")
    index.create(
        embeddings_source=documents_embeddings,
        kmeans_niters=4,
        max_points_per_centroid=256,
        nbits=4,
        seed=42,
        device="cpu",
    )

    index.load("cpu")

    results = index.search(
        queries_embeddings=queries_embeddings,
        top_k=10,
        num_threads=1,
    )

    assert len(results) == 10, (
        f"Expected 10 sets of query results, but got {len(results)}"
    )

    assert all(len(query_res) == 10 for query_res in results), (
        "Expected each query to have 10 results"
    )

    shutil.rmtree(index_name, ignore_errors=True)


def test_embeddings_from_path():
    """Ensure that embeddings can be loaded from a path."""
    index_name = ".indices/test_index_path"
    emb_dir = ".test_embeddings"

    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    if os.path.exists(emb_dir):
        shutil.rmtree(emb_dir)
    os.makedirs(emb_dir, exist_ok=True)

    # Create 100 documents and split into two files (50 each)
    documents_embeddings = [torch.randn(300, 128, device="cpu") for _ in range(100)]

    # Split 1: first 50 documents
    split1_docs = documents_embeddings[:50]
    split1_data = torch.cat(split1_docs, dim=0).numpy()
    split1_doclens = np.array([doc.shape[0] for doc in split1_docs])
    np.save(os.path.join(emb_dir, "emb_0.npy"), split1_data)
    np.save(os.path.join(emb_dir, "emb_0.doclens.npy"), split1_doclens)

    # Split 2: last 50 documents
    split2_docs = documents_embeddings[50:]
    split2_data = torch.cat(split2_docs, dim=0).numpy()
    split2_doclens = np.array([doc.shape[0] for doc in split2_docs])
    np.save(os.path.join(emb_dir, "emb_1.npy"), split2_data)
    np.save(os.path.join(emb_dir, "emb_1.doclens.npy"), split2_doclens)

    index = XTRWarp(index=index_name)

    queries_embeddings = torch.randn(10, 30, 128, device="cpu")
    index.create(
        embeddings_source=emb_dir,
        kmeans_niters=4,
        max_points_per_centroid=256,
        nbits=4,
        seed=42,
        device="cpu",
    )

    index.load("cpu")

    results = index.search(
        queries_embeddings=queries_embeddings,
        top_k=10,
        num_threads=1,
    )

    assert len(results) == 10, (
        f"Expected 10 sets of query results, but got {len(results)}"
    )

    assert all(len(query_res) == 10 for query_res in results), (
        "Expected each query to have 10 results"
    )

    shutil.rmtree(index_name, ignore_errors=True)
    shutil.rmtree(emb_dir, ignore_errors=True)


def test_nprobe_and_bound_larger_than_num_centroids():
    """Ensure search does not panic when nprobe or bound exceed the number of centroids."""
    index_name = ".indices/test_index_clamp"

    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    index = XTRWarp(index=index_name)

    # Small corpus → few centroids (well under 1000)
    documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(20)]
    queries_embeddings = torch.randn(2, 10, 128, device="cpu")

    index.create(
        embeddings_source=documents_embeddings,
        kmeans_niters=4,
        max_points_per_centroid=256,
        nbits=4,
        seed=42,
        device="cpu",
    )

    index.load("cpu")

    # nprobe and bound far exceed the actual number of centroids
    results = index.search(
        queries_embeddings=queries_embeddings,
        top_k=5,
        num_threads=1,
        nprobe=9999,
        bound=9999,
    )

    assert len(results) == 2, f"Expected 2 query results, got {len(results)}"
    assert all(len(r) <= 5 for r in results), (
        "Each query should return at most top_k results"
    )

    shutil.rmtree(index_name, ignore_errors=True)


def test_subset_with_nonexistent_pids():
    """Search with a subset containing only PIDs not in the index should return empty results, not panic."""
    index_name = ".indices/test_index_bogus_subset"

    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    index = XTRWarp(index=index_name)

    documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(20)]
    queries_embeddings = torch.randn(2, 10, 128, device="cpu")

    index.create(
        embeddings_source=documents_embeddings,
        kmeans_niters=4,
        max_points_per_centroid=256,
        nbits=4,
        seed=42,
        device="cpu",
    )

    index.load("cpu")

    # Subset with PIDs that don't exist in the index — non-empty so it won't short-circuit
    results = index.search(
        queries_embeddings=queries_embeddings,
        top_k=5,
        num_threads=1,
        subset=[999999, 999998],
    )

    assert len(results) == 2, f"Expected 2 query results, got {len(results)}"
    for r in results:
        assert len(r) == 0, f"Expected empty results for bogus subset, got {len(r)}"

    shutil.rmtree(index_name, ignore_errors=True)


def test_max_candidates_does_not_discard_top_results():
    """max_candidates should still return the globally best results.

    Regression test: partial_sort_results used to partition only the first
    `max_candidates` indices instead of the full candidate array, so the true
    top-k could be silently dropped when they happened to sit beyond that prefix
    in the merged stride.
    """
    index_name = ".indices/test_index_max_candidates"

    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    index = XTRWarp(index=index_name)

    n_docs = 500
    documents_embeddings = [torch.randn(50, 128, device="cpu") for _ in range(n_docs)]
    queries_embeddings = torch.randn(5, 20, 128, device="cpu")

    index.create(
        embeddings_source=documents_embeddings,
        kmeans_niters=4,
        max_points_per_centroid=256,
        nbits=4,
        seed=42,
        device="cpu",
    )
    index.load("cpu")

    top_k = 10

    # Baseline: no cap on candidates (effectively unlimited)
    baseline = index.search(
        queries_embeddings=queries_embeddings,
        top_k=top_k,
        num_threads=1,
        max_candidates=999999,
    )

    # Constrained: very tight cap forces the old bug to surface
    constrained = index.search(
        queries_embeddings=queries_embeddings,
        top_k=top_k,
        num_threads=1,
        max_candidates=top_k,
    )

    for q_idx in range(len(baseline)):
        baseline_top1_score = baseline[q_idx][0][1]
        constrained_top1_score = constrained[q_idx][0][1]
        assert constrained_top1_score >= baseline_top1_score - 1e-3, (
            f"Query {q_idx}: constrained top-1 score {constrained_top1_score:.4f} "
            f"is worse than baseline {baseline_top1_score:.4f}. "
            "partial_sort_results may be discarding high-scoring candidates."
        )

    shutil.rmtree(index_name, ignore_errors=True)
