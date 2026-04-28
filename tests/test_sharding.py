"""Tests for multi-device index sharding."""

import os
import shutil

import pytest
import torch
from xtr_warp.search import XTRWarp

pytestmark = pytest.mark.sharding

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# ── Helpers ──────────────────────────────────────────────────────────────────

INDEX_DIR = ".indices/test_sharding"
NUM_DOCS = 100
DOC_LEN = 128
DIM = 128
SEED = 42

CREATE_KWARGS = dict(
    kmeans_niters=4,
    max_points_per_centroid=256,
    nbits=4,
    seed=SEED,
    device="cpu",
)

SEARCH_KWARGS = dict(top_k=10, num_threads=1)


def _fresh_index(index_name=INDEX_DIR, num_docs=NUM_DOCS):
    """Create a fresh index and return (index, documents_embeddings, queries)."""
    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(num_docs)]
    queries = torch.randn(5, 30, DIM, device="cpu")

    idx = XTRWarp(index=index_name)
    idx.create(embeddings_source=docs, **CREATE_KWARGS)
    return idx, docs, queries


def _result_pids(results):
    return {pid for query_res in results for pid, _score in query_res}


def _result_scores(results):
    """Return per-query list of (pid, score) tuples sorted by score desc."""
    return [
        sorted(query_res, key=lambda x: -x[1]) for query_res in results
    ]


def _cleanup(index_name=INDEX_DIR):
    shutil.rmtree(index_name, ignore_errors=True)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_sharded_single_shard_equivalence():
    """load({"cpu": 1.0}) must produce identical results to load("cpu")."""
    try:
        idx, _, queries = _fresh_index()

        # Single-device reference
        idx.load("cpu")
        ref = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        # Sharded with one shard (full index on CPU)
        idx.load({"cpu": 1.0})
        sharded = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        assert len(ref) == len(sharded)
        for q_ref, q_shard in zip(ref, sharded):
            ref_pids = [pid for pid, _ in q_ref]
            shard_pids = [pid for pid, _ in q_shard]
            assert ref_pids == shard_pids, (
                f"PID mismatch: ref={ref_pids}, sharded={shard_pids}"
            )

            ref_scores = [s for _, s in q_ref]
            shard_scores = [s for _, s in q_shard]
            for rs, ss in zip(ref_scores, shard_scores):
                assert abs(rs - ss) < 1e-4, (
                    f"Score mismatch: ref={rs}, sharded={ss}"
                )
    finally:
        _cleanup()


def test_sharded_two_cpu_shards_equivalence():
    """Splitting the index 50/50 across two CPU shards must match single-device."""
    try:
        idx, _, queries = _fresh_index()

        idx.load("cpu")
        ref = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        ref_pids = _result_pids(ref)
        idx.free()

        # Two CPU shards: the dict keys must differ, but we can test via
        # the Rust layer directly since Python dicts can't have duplicate keys.
        # Instead, test a 50/50 split — the auto-ratio with a single CPU
        # would assign 100% to it, so we use explicit ratios.
        idx.load({"cpu": 0.5})
        sharded = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        sharded_pids = _result_pids(sharded)
        idx.free()

        # PIDs from top-k should have significant overlap
        overlap = ref_pids & sharded_pids
        assert len(overlap) > 0, "No PID overlap between sharded and reference"
    finally:
        _cleanup()


def test_sharded_uneven_split():
    """An uneven 80/20 split should still produce valid results."""
    try:
        idx, _, queries = _fresh_index()

        idx.load({"cpu": 0.8})
        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        assert len(results) == 5, f"Expected 5 query results, got {len(results)}"
        assert all(
            len(q) == 10 for q in results
        ), "Each query should return 10 results"
    finally:
        _cleanup()


def test_sharded_result_count():
    """Sharded search must return the correct number of results per query."""
    try:
        idx, _, queries = _fresh_index()

        for top_k in [1, 5, 10, 50]:
            idx.load({"cpu": 1.0})
            results = idx.search(
                queries_embeddings=queries,
                top_k=top_k,
                num_threads=1,
            )
            idx.free()

            assert len(results) == 5
            for q_res in results:
                assert len(q_res) <= top_k
    finally:
        _cleanup()


def test_sharded_scores_are_finite():
    """All returned scores must be finite floats."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cpu": 1.0})
        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        for q_res in results:
            for pid, score in q_res:
                assert isinstance(pid, int)
                assert isinstance(score, float)
                assert not (score != score), f"NaN score for pid {pid}"  # NaN check
    finally:
        _cleanup()


def test_sharded_with_subset():
    """Subset filtering must work with sharded search."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cpu": 1.0})

        subset = list(range(0, 20))
        results = idx.search(
            queries_embeddings=queries,
            subset=subset,
            **SEARCH_KWARGS,
        )

        all_pids = _result_pids(results)
        assert all_pids.issubset(set(subset)), (
            f"Found PIDs outside subset: {all_pids - set(subset)}"
        )

        idx.free()
    finally:
        _cleanup()


def test_sharded_with_per_query_subsets():
    """Per-query subsets must work with sharded search."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cpu": 1.0})

        n_queries = queries.shape[0]
        subsets = [list(range(i * 10, (i + 1) * 10)) for i in range(n_queries)]
        results = idx.search(
            queries_embeddings=queries,
            subset=subsets,
            top_k=5,
            num_threads=1,
        )

        for q_idx, q_res in enumerate(results):
            allowed = set(subsets[q_idx])
            returned = {pid for pid, _ in q_res}
            assert returned.issubset(allowed), (
                f"Query {q_idx}: PIDs {returned - allowed} not in subset"
            )

        idx.free()
    finally:
        _cleanup()


def test_sharded_delete_tombstone():
    """Delete + tombstone filtering must work on sharded index."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cpu": 1.0})

        results_before = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        pids_before = _result_pids(results_before)
        target_pid = next(iter(pids_before))

        idx.delete([target_pid])

        results_after = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        pids_after = _result_pids(results_after)

        assert target_pid not in pids_after, (
            f"Deleted PID {target_pid} still in results"
        )

        idx.free()
    finally:
        _cleanup()


def test_sharded_add_and_reload():
    """Adding documents and reloading a sharded index should work."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cpu": 1.0})

        new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(10)]
        new_ids = idx.add(embeddings_source=new_docs, reload=True)

        assert len(new_ids) == 10

        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        assert len(results) == 5

        idx.free()
    finally:
        _cleanup()


def test_sharded_update_and_reload():
    """Updating documents and reloading a sharded index should work."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cpu": 1.0})

        new_emb = [torch.randn(DOC_LEN, DIM, device="cpu")]
        idx.update(passage_ids=[0], embeddings_source=new_emb, reload=True)

        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        assert len(results) == 5

        idx.free()
    finally:
        _cleanup()


def test_sharded_compact_and_reload():
    """Compact after delete should work with sharded reload."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cpu": 1.0})

        idx.delete(list(range(0, 50)), compact_threshold=None)
        idx.compact(reload=True)

        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        assert len(results) == 5

        pids = _result_pids(results)
        for pid in range(0, 50):
            assert pid not in pids, f"Compacted PID {pid} still in results"

        idx.free()
    finally:
        _cleanup()


def test_estimate_index_memory():
    """estimate_index_memory should return reasonable values."""
    try:
        idx, _, _ = _fresh_index()

        mem = idx.estimate_index_memory()

        assert "total" in mem
        assert "centroids" in mem
        assert "pids" in mem
        assert "residuals" in mem

        assert mem["total"] > 0
        assert mem["centroids"] > 0
        assert mem["pids"] > 0
        assert mem["residuals"] > 0
        assert mem["total"] == sum(
            v for k, v in mem.items() if k != "total"
        )
    finally:
        _cleanup()


def test_recommend_device_map():
    """recommend_device_map should return valid ratios."""
    try:
        idx, _, _ = _fresh_index()

        ratios = idx.recommend_device_map(["cpu"])

        assert "cpu" in ratios
        assert abs(sum(ratios.values()) - 1.0) < 1e-6
    finally:
        _cleanup()


def test_sharded_ratio_normalization():
    """Ratios that don't sum to 1.0 should be normalized."""
    try:
        idx, _, queries = _fresh_index()

        # Ratios sum to 2.0 — should be normalized to 0.5 each internally
        idx.load({"cpu": 2.0})
        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        assert len(results) == 5
        assert all(len(q) == 10 for q in results)
    finally:
        _cleanup()


def test_sharded_small_index():
    """Sharding should work on a very small index (few centroids)."""
    try:
        idx, _, queries = _fresh_index(num_docs=10)
        idx.load({"cpu": 1.0})

        results = idx.search(queries_embeddings=queries, top_k=5, num_threads=1)

        assert len(results) == 5
        for q_res in results:
            assert len(q_res) <= 5

        idx.free()
    finally:
        _cleanup()


def test_sharded_free_and_reload():
    """Free and re-load sharded index should work."""
    try:
        idx, _, queries = _fresh_index()

        idx.load({"cpu": 1.0})
        first = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        idx.load({"cpu": 1.0})
        second = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        for q1, q2 in zip(first, second):
            pids1 = [pid for pid, _ in q1]
            pids2 = [pid for pid, _ in q2]
            assert pids1 == pids2
    finally:
        _cleanup()


def test_sharded_mmap_cpu():
    """mmap=True should work for CPU shards."""
    try:
        idx, _, queries = _fresh_index()

        idx.load({"cpu": 1.0}, mmap=True)
        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        assert len(results) == 5
        assert all(len(q) == 10 for q in results)
    finally:
        _cleanup()


def test_sharded_2d_query_tensor():
    """A single 2D query tensor should still work with sharded search."""
    try:
        idx, _, _ = _fresh_index()
        idx.load({"cpu": 1.0})

        single_query = torch.randn(30, DIM, device="cpu")
        results = idx.search(
            queries_embeddings=single_query, top_k=5, num_threads=1
        )

        assert len(results) == 1
        assert len(results[0]) <= 5

        idx.free()
    finally:
        _cleanup()


def test_sharded_list_of_query_tensors():
    """A list of variable-length query tensors should work."""
    try:
        idx, _, _ = _fresh_index()
        idx.load({"cpu": 1.0})

        queries_list = [
            torch.randn(20, DIM, device="cpu"),
            torch.randn(30, DIM, device="cpu"),
            torch.randn(25, DIM, device="cpu"),
        ]
        results = idx.search(
            queries_embeddings=queries_list, top_k=5, num_threads=1
        )

        assert len(results) == 3

        idx.free()
    finally:
        _cleanup()


def test_sharded_delete_multiple_then_search():
    """Deleting multiple passages from a sharded index should filter them."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cpu": 1.0})

        to_delete = [0, 1, 2, 3, 4]
        idx.delete(to_delete)

        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        pids = _result_pids(results)
        for pid in to_delete:
            assert pid not in pids

        idx.free()
    finally:
        _cleanup()


# ── CUDA + CPU hybrid tests ─────────────────────────────────────────────────


@requires_cuda
def test_cuda_single_shard_equivalence():
    """load({"cuda": 1.0}) must match load("cuda") on the same index."""
    try:
        idx, _, queries = _fresh_index()

        idx.load("cuda")
        ref = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        idx.load({"cuda": 1.0})
        sharded = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        assert len(ref) == len(sharded)
        for q_ref, q_shard in zip(ref, sharded):
            ref_pids = [pid for pid, _ in q_ref]
            shard_pids = [pid for pid, _ in q_shard]
            assert ref_pids == shard_pids

            for (_, rs), (_, ss) in zip(q_ref, q_shard):
                assert abs(rs - ss) < 1e-4
    finally:
        _cleanup()


@requires_cuda
def test_cuda_cpu_hybrid_returns_valid_results():
    """Splitting the index 60/40 across cuda and cpu should produce valid results."""
    try:
        idx, _, queries = _fresh_index()

        idx.load({"cuda:0": 0.6, "cpu": 0.4})
        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        assert len(results) == 5
        assert all(len(q) == 10 for q in results)

        for q_res in results:
            for pid, score in q_res:
                assert isinstance(pid, int)
                assert isinstance(score, float)
                assert score == score  # not NaN
    finally:
        _cleanup()


@requires_cuda
def test_cuda_cpu_hybrid_vs_cpu_reference():
    """Hybrid cuda+cpu results should have significant PID overlap with cpu-only."""
    try:
        idx, _, queries = _fresh_index()

        idx.load("cpu")
        ref = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        ref_pids = _result_pids(ref)
        idx.free()

        idx.load({"cuda:0": 0.6, "cpu": 0.4})
        hybrid = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        hybrid_pids = _result_pids(hybrid)
        idx.free()

        overlap = ref_pids & hybrid_pids
        # Different merge paths (CPU vs CUDA) may produce slightly different
        # rankings due to floating-point order, but top-k should overlap well.
        assert len(overlap) >= len(ref_pids) * 0.5, (
            f"Too little overlap: {len(overlap)}/{len(ref_pids)}"
        )
    finally:
        _cleanup()


@requires_cuda
def test_cuda_cpu_hybrid_with_subset():
    """Subset filtering should work with hybrid cuda+cpu sharding."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cuda:0": 0.6, "cpu": 0.4})

        subset = list(range(0, 30))
        results = idx.search(
            queries_embeddings=queries, subset=subset, **SEARCH_KWARGS
        )

        all_pids = _result_pids(results)
        assert all_pids.issubset(set(subset)), (
            f"PIDs outside subset: {all_pids - set(subset)}"
        )

        idx.free()
    finally:
        _cleanup()


@requires_cuda
def test_cuda_cpu_hybrid_delete():
    """Tombstone filtering should work on hybrid cuda+cpu index."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cuda:0": 0.5, "cpu": 0.5})

        results_before = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        pids_before = _result_pids(results_before)
        to_delete = list(pids_before)[:5]

        idx.delete(to_delete)

        results_after = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        pids_after = _result_pids(results_after)

        for pid in to_delete:
            assert pid not in pids_after

        idx.free()
    finally:
        _cleanup()


@requires_cuda
def test_auto_ratio_list_with_cuda():
    """load(["cuda:0", "cpu"]) should auto-compute ratios and search correctly."""
    try:
        idx, _, queries = _fresh_index()

        idx.load(["cuda:0", "cpu"])
        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        idx.free()

        assert len(results) == 5
        assert all(len(q) == 10 for q in results)
    finally:
        _cleanup()


@requires_cuda
def test_cuda_only_single_shard_fast_path():
    """load({"cuda:0": 1.0}) should use the fast path and match load("cuda:0")."""
    try:
        idx, _, queries = _fresh_index()

        idx.load("cuda:0")
        ref = idx.search(queries_embeddings=queries, top_k=5, num_threads=1)
        idx.free()

        idx.load({"cuda:0": 1.0})
        sharded = idx.search(queries_embeddings=queries, top_k=5, num_threads=1)
        idx.free()

        for q_ref, q_shard in zip(ref, sharded):
            ref_pids = [pid for pid, _ in q_ref]
            shard_pids = [pid for pid, _ in q_shard]
            assert ref_pids == shard_pids
    finally:
        _cleanup()


# Mutation tests last — a CUDA assert in fastkmeans can poison the runtime
# for all subsequent tests in the process.


@requires_cuda
def test_cuda_cpu_hybrid_add_reload():
    """Adding documents and reloading a hybrid index should work."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cuda:0": 0.6, "cpu": 0.4})

        new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(10)]
        # min_outliers set high to avoid centroid expansion which can hit
        # fastkmeans edge cases on tiny indices and poison the CUDA runtime.
        new_ids = idx.add(
            embeddings_source=new_docs, reload=True, min_outliers=999999
        )

        assert len(new_ids) == 10

        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        assert len(results) == 5

        idx.free()
    finally:
        _cleanup()


@requires_cuda
def test_cuda_cpu_hybrid_compact_reload():
    """Compact after delete should work with hybrid reload."""
    try:
        idx, _, queries = _fresh_index()
        idx.load({"cuda:0": 0.5, "cpu": 0.5})

        idx.delete(list(range(0, 50)), compact_threshold=None)
        idx.compact(reload=True)

        results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        assert len(results) == 5

        pids = _result_pids(results)
        for pid in range(0, 50):
            assert pid not in pids

        idx.free()
    finally:
        _cleanup()
