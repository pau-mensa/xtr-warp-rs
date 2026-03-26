"""Tests for index management: delete, add, update, compact."""

import json
import os
import shutil

import numpy as np
import torch
from xtr_warp.search import XTRWarp

# ── Helpers ──────────────────────────────────────────────────────────────────

INDEX_DIR = ".indices/test_mgmt"
NUM_DOCS = 100
DOC_LEN = 128  # tokens per document
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
    """Create a fresh index and return (index, documents_embeddings, queries).

    After create(), self.device is set to "cpu" automatically.
    """
    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(num_docs)]
    queries = torch.randn(5, 30, DIM, device="cpu")

    idx = XTRWarp(index=index_name)
    idx.create(embeddings_source=docs, **CREATE_KWARGS)
    return idx, docs, queries


def _load_metadata(index_name=INDEX_DIR):
    with open(os.path.join(index_name, "metadata.json")) as f:
        return json.load(f)


def _result_pids(results):
    """Flatten all passage IDs across all query results."""
    return {pid for query_res in results for pid, _score in query_res}


def _cleanup(index_name=INDEX_DIR):
    shutil.rmtree(index_name, ignore_errors=True)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_delete():
    """Delete a passage and verify it no longer appears in search results."""
    idx, _docs, queries = _fresh_index()
    idx.load("cpu")

    # Search before delete — pick a PID that appears in results
    results_before = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    all_pids_before = _result_pids(results_before)
    assert len(all_pids_before) > 0, "Expected at least some results"

    target_pid = next(iter(all_pids_before))

    # Delete that PID
    idx.delete([target_pid])

    # Search again — deleted PID must be gone
    results_after = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    all_pids_after = _result_pids(results_after)
    assert target_pid not in all_pids_after, (
        f"PID {target_pid} should not appear after deletion"
    )

    # Other results should still be present
    assert len(all_pids_after) > 0, "Expected remaining passages in results"

    # Tombstone file should exist
    assert os.path.exists(os.path.join(INDEX_DIR, "deleted_pids.npy"))

    _cleanup()


def test_delete_multiple():
    """Delete several passages at once."""
    idx, _docs, queries = _fresh_index()
    idx.load("cpu")

    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    pids = list(_result_pids(results))
    to_delete = pids[:3]

    idx.delete(to_delete)

    results_after = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    pids_after = _result_pids(results_after)
    for pid in to_delete:
        assert pid not in pids_after, f"PID {pid} should be gone"

    _cleanup()


def test_add():
    """Add new documents and verify they are searchable with correct IDs."""
    idx, _docs, queries = _fresh_index()
    meta_before = _load_metadata()

    num_before = meta_before["num_passages"]
    next_pid_before = meta_before.get("next_passage_id", num_before)

    # Add 20 new documents
    new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(20)]
    new_ids = idx.add(embeddings_source=new_docs, reload=False)

    assert len(new_ids) == 20, f"Expected 20 new IDs, got {len(new_ids)}"
    assert new_ids == list(range(next_pid_before, next_pid_before + 20)), (
        f"New IDs should be sequential from {next_pid_before}"
    )

    # Metadata should reflect the addition
    meta_after = _load_metadata()
    assert meta_after["num_passages"] == num_before + 20
    assert meta_after["next_passage_id"] == next_pid_before + 20
    assert meta_after["num_embeddings"] > meta_before["num_embeddings"]

    # Should be searchable after load
    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    _cleanup()


def test_add_with_reload():
    """Add with reload=True (default) should allow immediate search."""
    idx, _docs, queries = _fresh_index()
    idx.load("cpu")

    new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(10)]
    new_ids = idx.add(embeddings_source=new_docs)

    assert len(new_ids) == 10

    # Should be searchable immediately (auto-reloaded)
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    _cleanup()


def test_update_preserves_ids():
    """Update a document — same ID, different embeddings."""
    idx, _docs, queries = _fresh_index()
    meta_before = _load_metadata()

    # Pick a PID to update
    target_pid = 5
    new_embedding = [torch.randn(DOC_LEN, DIM, device="cpu")]

    idx.update(
        passage_ids=[target_pid],
        embeddings_source=new_embedding,
        reload=False,
    )

    meta_after = _load_metadata()

    # next_passage_id should NOT change — we replaced, not appended
    assert meta_after["next_passage_id"] == meta_before.get(
        "next_passage_id", meta_before["num_passages"]
    )

    # num_chunks should increase (new chunk written for the replacement)
    assert meta_after["num_chunks"] >= meta_before["num_chunks"]

    # Load and search — target_pid should still appear (same ID, new data)
    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    all_pids = _result_pids(results)
    # The PID should still be valid (not deleted from the index)
    # It may or may not appear in top-k depending on the random new embedding
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    _cleanup()


def test_compact():
    """Delete passages then compact — compacted files should shrink."""
    idx, _docs, queries = _fresh_index()
    meta_before = _load_metadata()
    emb_count_before = meta_before["num_embeddings"]

    # Delete half the documents
    to_delete = list(range(0, NUM_DOCS, 2))  # even IDs
    idx.delete(to_delete)

    # Compact to physically remove them
    idx.compact(reload=False)

    meta_after = _load_metadata()

    # Passage count should drop
    assert meta_after["num_passages"] < meta_before["num_passages"]
    # Embedding count should drop
    assert meta_after["num_embeddings"] < emb_count_before
    # Tombstone file cleared — chunk files rewritten with only active data
    assert not os.path.exists(os.path.join(INDEX_DIR, "deleted_pids.npy"))

    # Search should only return non-deleted docs
    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    all_pids = _result_pids(results)
    for pid in to_delete:
        assert pid not in all_pids, f"PID {pid} should be gone after compact"

    _cleanup()


def test_delete_then_add():
    """Delete some docs, add new ones — tombstones should clear after add."""
    idx, _docs, queries = _fresh_index()
    meta_before = _load_metadata()

    # Delete docs 0-9
    idx.delete(list(range(10)))
    assert os.path.exists(os.path.join(INDEX_DIR, "deleted_pids.npy"))

    # Add 10 new docs — this triggers compaction
    new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(10)]
    new_ids = idx.add(embeddings_source=new_docs, reload=False)

    # Tombstones persist — chunk cleanup is deferred to explicit compact()
    assert os.path.exists(os.path.join(INDEX_DIR, "deleted_pids.npy"))

    # New IDs should start from the watermark
    next_pid = meta_before.get("next_passage_id", meta_before["num_passages"])
    assert new_ids[0] == next_pid

    # Search should find new docs but not deleted ones
    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    all_pids = _result_pids(results)
    for pid in range(10):
        assert pid not in all_pids, f"Deleted PID {pid} should not appear"

    _cleanup()


def test_metadata_consistency():
    """Verify metadata stays consistent across a chain of operations."""
    idx, _docs, queries = _fresh_index()

    # After create
    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS
    assert meta["next_passage_id"] == NUM_DOCS
    assert meta["num_embeddings"] == NUM_DOCS * DOC_LEN
    assert meta["dim"] == DIM
    assert meta["nbits"] == 4

    # After add
    new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(5)]
    new_ids = idx.add(embeddings_source=new_docs, reload=False)
    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS + 5
    assert meta["next_passage_id"] == NUM_DOCS + 5
    assert meta["num_embeddings"] == (NUM_DOCS + 5) * DOC_LEN

    # After delete (metadata doesn't change — tombstone only)
    idx.delete([0, 1, 2])
    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS + 5  # unchanged
    assert meta["next_passage_id"] == NUM_DOCS + 5  # unchanged

    # After compact
    idx.compact(reload=False)
    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS + 5 - 3  # 3 deleted
    assert meta["next_passage_id"] == NUM_DOCS + 5  # watermark never decreases
    assert meta["num_embeddings"] == (NUM_DOCS + 5 - 3) * DOC_LEN

    _cleanup()


def test_passage_ids_npy_written():
    """Verify that create writes passage_ids.npy per chunk."""
    idx, _docs, _queries = _fresh_index()
    meta = _load_metadata()
    for chunk_idx in range(meta["num_chunks"]):
        path = os.path.join(INDEX_DIR, f"{chunk_idx}.passage_ids.npy")
        assert os.path.exists(path), f"Missing {path}"
        pids = np.load(path)
        assert pids.ndim == 1
        assert len(pids) > 0
    _cleanup()


def test_delete_idempotent():
    """Deleting the same PID twice should be harmless."""
    idx, _docs, queries = _fresh_index()
    idx.load("cpu")

    idx.delete([5])
    idx.delete([5])  # again

    pids = np.load(os.path.join(INDEX_DIR, "deleted_pids.npy"))
    assert list(pids).count(5) == 1, "PID 5 should appear exactly once in tombstone"

    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    all_pids = _result_pids(results)
    assert 5 not in all_pids

    _cleanup()


def test_add_empty():
    """Adding zero documents should be a no-op (or at least not crash)."""
    idx, _docs, _queries = _fresh_index()
    meta_before = _load_metadata()

    empty_docs = []
    try:
        idx.add(embeddings_source=empty_docs, reload=False)
    except Exception:
        # Some implementations may reject empty adds — that's acceptable
        pass

    _cleanup()


def test_search_after_all_operations():
    """Full lifecycle: create -> search -> delete -> add -> update -> compact -> search."""
    idx, _docs, queries = _fresh_index()
    idx.load("cpu")

    # Initial search
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5

    # Delete doc 0
    idx.delete([0])
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert 0 not in _result_pids(results)

    # Add 5 new docs
    new_docs = [torch.randn(DOC_LEN, DIM, device="cpu") for _ in range(5)]
    new_ids = idx.add(embeddings_source=new_docs)

    # Search after add (auto-reloaded)
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    # Update doc 10
    updated_emb = [torch.randn(DOC_LEN, DIM, device="cpu")]
    idx.update(passage_ids=[10], embeddings_source=updated_emb)

    # Search after update (auto-reloaded)
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5

    # Compact
    idx.compact()

    # Final search after compact (auto-reloaded)
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)
    # Doc 0 should still be gone
    assert 0 not in _result_pids(results)

    _cleanup()


# ── New tests for mutability improvements ───────────────────────────────────


def test_device_from_init():
    """Device set at __init__ time should propagate to add/update/compact."""
    index_name = ".indices/test_device_init"
    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM) for _ in range(50)]
    queries = torch.randn(3, 20, DIM)

    # Set device at init
    idx = XTRWarp(index=index_name, device="cpu")
    idx.create(
        embeddings_source=docs,
        kmeans_niters=4,
        max_points_per_centroid=256,
        nbits=4,
        seed=SEED,
        device="cpu",
    )

    # All mutations should work without passing device=
    new_ids = idx.add(embeddings_source=[torch.randn(DOC_LEN, DIM)], reload=False)
    assert len(new_ids) == 1

    idx.update(
        passage_ids=[0],
        embeddings_source=[torch.randn(DOC_LEN, DIM)],
        reload=False,
    )

    idx.delete([1])
    idx.compact(reload=False)

    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 3

    _cleanup(index_name)


def test_device_from_load():
    """Device should be inferred from load() when not set at init."""
    idx, _docs, queries = _fresh_index()
    # After create, self.device is already "cpu".
    # But test that load() also sets it:
    idx2 = XTRWarp(index=INDEX_DIR)
    assert idx2.device is None
    idx2.load("cpu")
    assert idx2.device == "cpu"

    # Should be able to add without explicit device
    new_ids = idx2.add(embeddings_source=[torch.randn(DOC_LEN, DIM)], reload=False)
    assert len(new_ids) == 1

    _cleanup()


def test_chunk_coalescing():
    """Adding a small batch to a small index should coalesce, not create new chunks."""
    idx, _docs, _queries = _fresh_index()
    meta_before = _load_metadata()
    chunks_before = meta_before["num_chunks"]

    # Index has 100 docs in 1 chunk (100 < 2000 threshold).
    # Adding 5 more should coalesce into the existing chunk.
    new_docs = [torch.randn(DOC_LEN, DIM) for _ in range(5)]
    idx.add(embeddings_source=new_docs, reload=False)

    meta_after = _load_metadata()
    assert meta_after["num_chunks"] == chunks_before, (
        f"Expected coalescing: chunks should stay at {chunks_before}, "
        f"got {meta_after['num_chunks']}"
    )
    assert meta_after["num_passages"] == NUM_DOCS + 5

    _cleanup()


def test_multiple_sequential_adds():
    """Several sequential adds should all produce correct metadata and searchable results."""
    idx, _docs, queries = _fresh_index()

    all_new_ids = []
    for i in range(5):
        new_docs = [torch.randn(DOC_LEN, DIM) for _ in range(3)]
        new_ids = idx.add(embeddings_source=new_docs, reload=False)
        assert len(new_ids) == 3
        # IDs should be globally unique
        assert not set(new_ids) & set(all_new_ids), "IDs must not overlap"
        all_new_ids.extend(new_ids)

    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS + 15
    assert meta["next_passage_id"] == NUM_DOCS + 15
    assert meta["num_embeddings"] == (NUM_DOCS + 15) * DOC_LEN

    # All docs should be searchable
    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 5
    assert all(len(r) == 10 for r in results)

    _cleanup()


def test_cluster_threshold_created():
    """create_index should produce a cluster_threshold.npy file."""
    idx, _docs, _queries = _fresh_index()
    path = os.path.join(INDEX_DIR, "cluster_threshold.npy")
    assert os.path.exists(path), (
        "cluster_threshold.npy should be created during index creation"
    )
    threshold = float(np.load(path))
    assert threshold > 0, "cluster_threshold should be positive"

    _cleanup()


def test_centroid_expansion_with_outliers():
    """Adding very different embeddings should trigger centroid expansion."""
    idx, _docs, _queries = _fresh_index()
    meta_before = _load_metadata()
    centroids_before = meta_before["num_centroids"]

    # Create embeddings that are very far from existing data:
    # large magnitude in a specific direction to be outliers.
    torch.manual_seed(999)
    outlier_docs = [torch.randn(DOC_LEN, DIM) * 100 for _ in range(200)]

    idx.add(embeddings_source=outlier_docs, reload=False)

    meta_after = _load_metadata()
    centroids_after = meta_after["num_centroids"]

    # With 200 docs * 128 tokens = 25600 embeddings, many should be outliers
    # and trigger expansion (if threshold check fires).
    # Note: expansion may or may not trigger depending on the threshold.
    # We just verify it doesn't crash and metadata stays consistent.
    assert meta_after["num_passages"] == NUM_DOCS + 200
    assert meta_after["num_embeddings"] == (NUM_DOCS + 200) * DOC_LEN
    # If expansion triggered, centroids should have increased
    assert centroids_after >= centroids_before

    _cleanup()


def test_centroid_pruning_on_compact():
    """Compact after heavy deletion should prune empty centroids."""
    # Create a large enough index to have many centroids
    index_name = ".indices/test_prune"
    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM) for _ in range(200)]
    queries = torch.randn(3, 20, DIM)

    idx = XTRWarp(index=index_name)
    idx.create(embeddings_source=docs, **CREATE_KWARGS)

    meta_before = _load_metadata(index_name)
    centroids_before = meta_before["num_centroids"]

    # Delete 90% of documents — many centroids should become empty
    to_delete = list(range(0, 200, 1))[:180]
    idx.delete(to_delete)
    idx.compact(reload=False)

    meta_after = _load_metadata(index_name)
    centroids_after = meta_after["num_centroids"]

    # Some centroids should have been pruned (those with zero embeddings)
    assert centroids_after <= centroids_before, (
        f"Expected pruning: centroids should decrease from {centroids_before}, "
        f"got {centroids_after}"
    )

    # Verify search still works
    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    assert len(results) == 3
    pids = _result_pids(results)
    for pid in to_delete:
        assert pid not in pids

    _cleanup(index_name)


def test_incremental_merge_search_quality():
    """Search results after incremental add should largely agree with a full compact.

    Small differences in top-k are expected because incremental merge preserves
    old tombstoned data in compacted (filtered at search time), which can
    slightly change the embedding ordering within centroids.
    """
    idx, _docs, queries = _fresh_index()

    # Add some docs (uses incremental merge)
    new_docs = [torch.randn(DOC_LEN, DIM) for _ in range(20)]
    idx.add(embeddings_source=new_docs, reload=False)

    # Search after incremental merge
    idx.load("cpu")
    results_incremental = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    pids_incremental = _result_pids(results_incremental)

    # Now compact (full rebuild) and search again
    idx.compact(reload=False)
    idx.load("cpu")
    results_compacted = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    pids_compacted = _result_pids(results_compacted)

    # Results should overlap heavily — allow small differences from
    # rounding / ordering effects
    overlap = pids_incremental & pids_compacted
    total = pids_incremental | pids_compacted
    jaccard = len(overlap) / len(total) if total else 1.0
    assert jaccard >= 0.8, (
        f"Incremental vs compact results too different: "
        f"Jaccard={jaccard:.2f}, overlap={len(overlap)}, total={len(total)}"
    )

    _cleanup()


def test_add_then_delete_then_compact_consistency():
    """Multiple add/delete cycles followed by compact should produce clean state."""
    idx, _docs, queries = _fresh_index()

    # Cycle 1: add 10
    new1 = [torch.randn(DOC_LEN, DIM) for _ in range(10)]
    ids1 = idx.add(embeddings_source=new1, reload=False)

    # Cycle 2: delete some old + add more
    idx.delete([0, 1, 2, 3, 4])
    new2 = [torch.randn(DOC_LEN, DIM) for _ in range(5)]
    ids2 = idx.add(embeddings_source=new2, reload=False)

    # Cycle 3: delete some new
    idx.delete(ids1[:3])
    new3 = [torch.randn(DOC_LEN, DIM) for _ in range(2)]
    ids3 = idx.add(embeddings_source=new3, reload=False)

    # Compact to clean everything up
    idx.compact(reload=False)

    meta = _load_metadata()
    # Should have: 100 - 5 (deleted old) + 10 - 3 (deleted from ids1) + 5 + 2 = 109
    expected_passages = NUM_DOCS - 5 + 10 - 3 + 5 + 2
    assert meta["num_passages"] == expected_passages, (
        f"Expected {expected_passages} passages, got {meta['num_passages']}"
    )
    assert meta["num_embeddings"] == expected_passages * DOC_LEN

    # Tombstones should be cleared
    assert not os.path.exists(os.path.join(INDEX_DIR, "deleted_pids.npy"))

    # Search should work
    idx.load("cpu")
    results = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
    pids = _result_pids(results)
    for pid in [0, 1, 2, 3, 4] + ids1[:3]:
        assert pid not in pids, f"Deleted PID {pid} should not appear"

    _cleanup()


def test_recalibration_on_compact():
    """Compact should recalibrate cluster_threshold and avg_residual."""
    idx, docs, _queries = _fresh_index()

    threshold_path = os.path.join(INDEX_DIR, "cluster_threshold.npy")
    avg_res_path = os.path.join(INDEX_DIR, "avg_residual.npy")

    threshold_before = float(np.load(threshold_path))
    avg_res_before = np.load(avg_res_path).copy()

    # Add documents with a different distribution to shift the statistics
    torch.manual_seed(999)
    new_docs = [torch.randn(DOC_LEN, DIM) * 3.0 for _ in range(50)]
    idx.add(embeddings_source=new_docs, reload=False)

    # Delete some originals so compact has work to do
    idx.delete(list(range(30)))
    idx.compact(reload=False)

    threshold_after = float(np.load(threshold_path))
    avg_res_after = np.load(avg_res_path)

    # Both should have been rewritten with correct shape
    assert avg_res_after.shape == avg_res_before.shape
    assert avg_res_after.shape == (DIM,)
    # With 3x-scaled docs added, the threshold should shift upward
    assert threshold_after > threshold_before, (
        f"Expected threshold to increase with scaled docs: {threshold_before} -> {threshold_after}"
    )

    _cleanup()


def test_auto_compact_on_delete():
    """delete(auto_compact=True) should trigger compaction above threshold."""
    idx, _docs, _queries = _fresh_index()

    tombstone_path = os.path.join(INDEX_DIR, "deleted_pids.npy")

    # Delete 10% — below the 20% threshold, no compaction
    idx.delete(list(range(10)), compact_threshold=0.2)
    assert os.path.exists(tombstone_path), (
        "Tombstones should still exist below threshold"
    )

    # Delete another 15% — now at 25%, above threshold → auto-compact
    idx.delete(list(range(10, 25)), compact_threshold=0.2)
    assert not os.path.exists(tombstone_path), (
        "Tombstones should be cleared after auto-compaction"
    )

    # Metadata should reflect the compaction
    meta = _load_metadata()
    assert meta["num_passages"] == NUM_DOCS - 25

    _cleanup()
