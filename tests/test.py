import os
import shutil

import numpy as np
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

    print("✅ Test passed: Results have the correct shape (10, 10).")

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

    print("✅ Test passed: Path-based embeddings work correctly (10, 10).")

    shutil.rmtree(index_name, ignore_errors=True)
    shutil.rmtree(emb_dir, ignore_errors=True)
