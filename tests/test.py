import os
import shutil

import torch
from xtr_warp.search import XTRWarp


def test():
    """Ensure that the Fast-PLAiD search index can be created and queried correctly."""
    index_name = ".indices/test_index"

    if os.path.exists(index_name):
        shutil.rmtree(index_name)
    index = XTRWarp(index=index_name)

    documents_embeddings = [torch.randn(300, 128, device="cpu") for _ in range(100)]

    queries_embeddings = torch.randn(10, 30, 128, device="cpu")
    index.create(
        documents_embeddings=documents_embeddings,
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

    print("✅ Test passed: Results have the correct shape (10, 10).")

    shutil.rmtree(index_name, ignore_errors=True)
