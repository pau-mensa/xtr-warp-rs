"""Tests for metadata filtering and subset search."""

from __future__ import annotations

import os
import shutil

import torch
from xtr_warp import XTRWarp

INDEX_DIR = ".indices/test_filtering"
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


def _fresh_index(
    index_name=INDEX_DIR,
    num_docs=NUM_DOCS,
    metadata=None,
):
    """Create a fresh index and return (index, docs, queries)."""
    torch.manual_seed(SEED)
    docs = [torch.randn(DOC_LEN, DIM) for _ in range(num_docs)]
    queries = torch.randn(5, 30, DIM)

    idx = XTRWarp(index=index_name)
    idx.create(embeddings_source=docs, metadata=metadata, **CREATE_KWARGS)
    idx.load("cpu")

    return idx, docs, queries


def _result_pids(results):
    """Extract all passage IDs from results."""
    return {pid for query_res in results for pid, _ in query_res}


def _cleanup(index_name=INDEX_DIR):
    if os.path.exists(index_name):
        shutil.rmtree(index_name)


# ---------------------------------------------------------------------------
# Subset filtering tests (no metadata store needed)
# ---------------------------------------------------------------------------


def test_search_with_subset():
    """Searching with a subset should only return PIDs from that subset."""
    try:
        idx, _docs, queries = _fresh_index()
        subset = list(range(0, 20))

        results = idx.search(queries_embeddings=queries, subset=subset, **SEARCH_KWARGS)

        assert len(results) == 5
        all_pids = _result_pids(results)
        assert all_pids.issubset(set(subset)), (
            f"Returned PIDs {all_pids - set(subset)} not in subset"
        )
    finally:
        _cleanup()


def test_search_with_empty_subset():
    """Empty subset should return empty results."""
    try:
        idx, _docs, queries = _fresh_index()

        results = idx.search(queries_embeddings=queries, subset=[], **SEARCH_KWARGS)

        assert len(results) == 5
        for query_res in results:
            assert len(query_res) == 0
    finally:
        _cleanup()


def test_search_with_full_subset():
    """Full subset should match unfiltered results."""
    try:
        idx, _docs, queries = _fresh_index()
        full_subset = list(range(NUM_DOCS))

        results_unfiltered = idx.search(queries_embeddings=queries, **SEARCH_KWARGS)
        results_filtered = idx.search(
            queries_embeddings=queries, subset=full_subset, **SEARCH_KWARGS
        )

        pids_unfiltered = _result_pids(results_unfiltered)
        pids_filtered = _result_pids(results_filtered)
        assert pids_unfiltered == pids_filtered
    finally:
        _cleanup()


def test_search_with_per_query_subsets():
    """Each query should only return PIDs from its own subset."""
    try:
        idx, _docs, queries = _fresh_index()

        subsets = [
            list(range(0, 20)),
            list(range(20, 40)),
            list(range(40, 60)),
            list(range(60, 80)),
            list(range(80, 100)),
        ]

        results = idx.search(
            queries_embeddings=queries, subset=subsets, **SEARCH_KWARGS
        )

        assert len(results) == 5
        for query_res, query_subset in zip(results, subsets):
            pids = {pid for pid, _ in query_res}
            assert pids.issubset(set(query_subset)), (
                f"Returned PIDs {pids - set(query_subset)} not in query subset"
            )
    finally:
        _cleanup()


def test_per_query_subsets_with_empty():
    """Per-query subsets where some are empty should return empty for those queries."""
    try:
        idx, _docs, queries = _fresh_index()

        subsets = [
            list(range(0, 20)),
            [],
            list(range(40, 60)),
            [],
            list(range(80, 100)),
        ]

        results = idx.search(
            queries_embeddings=queries, subset=subsets, **SEARCH_KWARGS
        )

        assert len(results) == 5
        assert len(results[1]) == 0
        assert len(results[3]) == 0

        for i in [0, 2, 4]:
            pids = {pid for pid, _ in results[i]}
            assert pids.issubset(set(subsets[i]))
    finally:
        _cleanup()


def test_per_query_subsets_length_mismatch():
    """Mismatched number of subsets and queries should raise ValueError."""
    try:
        idx, _docs, queries = _fresh_index()

        subsets = [list(range(0, 20)), list(range(20, 40))]  # 2 subsets, 5 queries

        try:
            idx.search(queries_embeddings=queries, subset=subsets, **SEARCH_KWARGS)
            assert False, "Expected ValueError"
        except ValueError:
            pass
    finally:
        _cleanup()


def test_per_query_subsets_match_shared():
    """Per-query subsets all identical should match a single shared subset."""
    try:
        idx, _docs, queries = _fresh_index()
        shared = list(range(0, 30))

        results_shared = idx.search(
            queries_embeddings=queries, subset=shared, **SEARCH_KWARGS
        )
        results_per_query = idx.search(
            queries_embeddings=queries, subset=[shared] * 5, **SEARCH_KWARGS
        )

        for shared_res, pq_res in zip(results_shared, results_per_query):
            shared_pids = {pid for pid, _ in shared_res}
            pq_pids = {pid for pid, _ in pq_res}
            assert shared_pids == pq_pids
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Metadata store + filtering tests
# ---------------------------------------------------------------------------


def _make_metadata(num_docs=NUM_DOCS):
    """Generate sample metadata for testing."""
    return [
        {
            "category": "science" if i % 3 == 0 else "history" if i % 3 == 1 else "art",
            "score": float(i),
            "is_featured": i % 5 == 0,
        }
        for i in range(num_docs)
    ]


def test_metadata_create_and_filter():
    """Create index with metadata, then filter by condition."""
    try:
        metadata = _make_metadata()
        idx, _docs, _ = _fresh_index(metadata=metadata)

        # Filter for science category (every 3rd doc starting from 0)
        subset = idx.filter("category = ?", ["science"])
        expected = [i for i in range(NUM_DOCS) if i % 3 == 0]
        assert sorted(subset) == expected

        # Filter with numeric condition
        subset = idx.filter("score >= ?", [50.0])
        expected = [i for i in range(NUM_DOCS) if i >= 50]
        assert sorted(subset) == expected
    finally:
        _cleanup()


def test_metadata_filter_combined_conditions():
    """Test filtering with combined conditions."""
    try:
        metadata = _make_metadata()
        idx, _docs, _ = _fresh_index(metadata=metadata)

        subset = idx.filter("category = ? AND is_featured = ?", ["science", True])
        expected = [i for i in range(NUM_DOCS) if i % 3 == 0 and i % 5 == 0]
        assert sorted(subset) == expected
    finally:
        _cleanup()


def test_metadata_add_and_filter():
    """Add documents with metadata, then filter across all docs."""
    try:
        metadata = _make_metadata()
        idx, _docs, _ = _fresh_index(metadata=metadata)

        new_docs = [torch.randn(DOC_LEN, DIM) for _ in range(10)]
        new_metadata = [
            {"category": "new_cat", "score": 999.0, "is_featured": True}
        ] * 10
        new_ids = idx.add(embeddings_source=new_docs, metadata=new_metadata)

        subset = idx.filter("category = ?", ["new_cat"])
        assert sorted(subset) == sorted(new_ids)
    finally:
        _cleanup()


def test_metadata_delete_removes_from_store():
    """Deleting documents should remove their metadata."""
    try:
        metadata = _make_metadata()
        idx, _docs, _ = _fresh_index(metadata=metadata)

        idx.delete([0, 1, 2])

        subset = idx.filter("score < ?", [3.0])
        # PIDs 0, 1, 2 had scores 0.0, 1.0, 2.0 but are now deleted
        assert subset == []
    finally:
        _cleanup()


def test_search_with_filtered_subset():
    """End-to-end: create with metadata, filter, search with subset."""
    try:
        metadata = _make_metadata()
        idx, _docs, queries = _fresh_index(metadata=metadata)

        subset = idx.filter("category = ?", ["science"])
        results = idx.search(queries_embeddings=queries, subset=subset, **SEARCH_KWARGS)

        assert len(results) == 5
        all_pids = _result_pids(results)
        assert all_pids.issubset(set(subset))
    finally:
        _cleanup()


def test_nested_metadata_with_native_duckdb_functions():
    """Index with deeply nested metadata, filter using DuckDB native functions."""
    try:
        torch.manual_seed(SEED)
        num_docs = 20
        docs = [torch.randn(DOC_LEN, DIM) for _ in range(num_docs)]
        queries = torch.randn(3, 30, DIM)

        # Nested metadata: lists, structs (dicts), list-of-structs
        metadata = [
            {
                "tags": ["ml", "search"],
                "author": {"name": "Alice", "org": "ACME"},
                "scores": [0.9, 0.8, 0.7],
                "sections": [
                    {"title": "intro", "word_count": 100},
                    {"title": "methods", "word_count": 500},
                ],
            },
            {
                "tags": ["ml", "vision"],
                "author": {"name": "Bob", "org": "ACME"},
                "scores": [0.5],
                "sections": [{"title": "abstract", "word_count": 50}],
            },
            {
                "tags": ["search", "nlp"],
                "author": {"name": "Carol", "org": "Beta"},
                "scores": [0.3, 0.9],
                "sections": [
                    {"title": "intro", "word_count": 200},
                    {"title": "results", "word_count": 800},
                ],
            },
            {
                "tags": ["vision"],
                "author": {"name": "Dave", "org": "Beta"},
                "scores": [0.1],
                "sections": [],
            },
            {
                "tags": ["ml", "search", "vision"],
                "author": {"name": "Eve", "org": "ACME"},
                "scores": [1.0, 0.95, 0.9, 0.85],
                "sections": [
                    {"title": "intro", "word_count": 150},
                    {"title": "methods", "word_count": 300},
                    {"title": "discussion", "word_count": 600},
                ],
            },
        ]
        # Repeat pattern for remaining docs
        full_metadata = [metadata[i % len(metadata)] for i in range(num_docs)]

        idx = XTRWarp(index=INDEX_DIR)
        idx.create(embeddings_source=docs, metadata=full_metadata, **CREATE_KWARGS)
        idx.load("cpu")

        # --- list_contains: docs tagged "search" ---
        subset = idx.filter("list_contains(tags, ?)", ["search"])
        expected = [i for i in range(num_docs) if "search" in full_metadata[i]["tags"]]
        assert sorted(subset) == expected

        # --- len on list: docs with 3+ tags ---
        subset = idx.filter("len(tags) >= ?", [3])
        expected = [i for i in range(num_docs) if len(full_metadata[i]["tags"]) >= 3]
        assert sorted(subset) == expected

        # --- struct dot notation: author.org = 'Beta' ---
        subset = idx.filter("author.org = ?", ["Beta"])
        expected = [
            i for i in range(num_docs) if full_metadata[i]["author"]["org"] == "Beta"
        ]
        assert sorted(subset) == expected

        # --- list_has_any: docs with 'vision' or 'nlp' tags ---
        subset = idx.filter("list_has_any(tags, ['vision', 'nlp'])")
        expected = [
            i
            for i in range(num_docs)
            if set(full_metadata[i]["tags"]) & {"vision", "nlp"}
        ]
        assert sorted(subset) == expected

        # --- combined: struct + list ---
        subset = idx.filter(
            "author.org = ? AND list_contains(tags, ?)", ["ACME", "search"]
        )
        expected = [
            i
            for i in range(num_docs)
            if full_metadata[i]["author"]["org"] == "ACME"
            and "search" in full_metadata[i]["tags"]
        ]
        assert sorted(subset) == expected

        # --- lambda on nested list-of-structs: any section with word_count > 400 ---
        subset = idx.filter("len(list_filter(sections, x -> x.word_count > 400)) > 0")
        expected = [
            i
            for i in range(num_docs)
            if any(s["word_count"] > 400 for s in full_metadata[i]["sections"])
        ]
        assert sorted(subset) == expected

        # --- list_aggregate: sum of scores > 2.0 ---
        subset = idx.filter("list_aggregate(scores, 'sum') > ?", [2.0])
        expected = [i for i in range(num_docs) if sum(full_metadata[i]["scores"]) > 2.0]
        assert sorted(subset) == expected

        # --- end-to-end: filter then search ---
        subset = idx.filter("author.org = ?", ["ACME"])
        results = idx.search(queries_embeddings=queries, subset=subset, **SEARCH_KWARGS)
        assert len(results) == 3
        all_pids = _result_pids(results)
        assert all_pids.issubset(set(subset))
    finally:
        _cleanup()


def test_inconsistent_schemas_create():
    """Documents with different attribute sets should coexist (NULLs for missing)."""
    try:
        torch.manual_seed(SEED)
        num_docs = 30
        docs = [torch.randn(DOC_LEN, DIM) for _ in range(num_docs)]

        metadata = []
        for i in range(num_docs):
            row = {"title": f"doc_{i}"}
            if i % 3 == 0:
                row["category"] = "science"
            if i % 2 == 0:
                row["rating"] = float(i)
            if i % 5 == 0:
                row["tags"] = ["featured"]
            metadata.append(row)

        idx = XTRWarp(index=INDEX_DIR)
        idx.create(embeddings_source=docs, metadata=metadata, **CREATE_KWARGS)
        idx.load("cpu")

        # Filter on a field that only some docs have — missing docs should be excluded
        subset = idx.filter("category = ?", ["science"])
        expected = [i for i in range(num_docs) if i % 3 == 0]
        assert sorted(subset) == expected

        # Filter on rating — only even-indexed docs have it
        subset = idx.filter("rating >= ?", [10.0])
        expected = [i for i in range(num_docs) if i % 2 == 0 and float(i) >= 10.0]
        assert sorted(subset) == expected

        # NULL-aware: docs without rating should not match any comparison
        subset = idx.filter("rating IS NULL")
        expected = [i for i in range(num_docs) if i % 2 != 0]
        assert sorted(subset) == expected

        # Combine fields with different coverage
        subset = idx.filter("category = ? AND rating >= ?", ["science", 0.0])
        expected = [i for i in range(num_docs) if i % 3 == 0 and i % 2 == 0]
        assert sorted(subset) == expected
    finally:
        _cleanup()


def test_inconsistent_schemas_add():
    """Adding docs with new columns not in the original schema."""
    try:
        metadata = _make_metadata(NUM_DOCS)
        idx, _docs, _ = _fresh_index(metadata=metadata)

        # Add docs with a brand-new "language" field not in original metadata
        new_docs = [torch.randn(DOC_LEN, DIM) for _ in range(10)]
        new_metadata = [
            {"category": "science", "score": 500.0, "is_featured": False, "language": "en"}
            if i % 2 == 0
            else {"category": "art", "score": 600.0, "is_featured": True, "language": "fr"}
            for i in range(10)
        ]
        new_ids = idx.add(embeddings_source=new_docs, metadata=new_metadata)

        # New column should be queryable on new docs
        subset = idx.filter("language = ?", ["en"])
        expected = [pid for i, pid in enumerate(new_ids) if i % 2 == 0]
        assert sorted(subset) == sorted(expected)

        # Original docs should have NULL for the new column
        subset = idx.filter("language IS NULL")
        expected = list(range(NUM_DOCS))
        assert sorted(subset) == expected

        # Original columns still work across all docs
        subset = idx.filter("category = ?", ["science"])
        original_science = [i for i in range(NUM_DOCS) if i % 3 == 0]
        new_science = [pid for i, pid in enumerate(new_ids) if i % 2 == 0]
        assert sorted(subset) == sorted(original_science + new_science)
    finally:
        _cleanup()


def test_metadata_survives_compact():
    """Metadata should survive compaction (tombstoned rows removed)."""
    try:
        metadata = _make_metadata()
        idx, _docs, _ = _fresh_index(metadata=metadata)

        # Delete some docs and compact
        to_delete = list(range(10))
        idx.delete(to_delete, compact_threshold=None)
        idx.compact()

        # Filter should not return deleted PIDs
        subset = idx.filter("score < ?", [10.0])
        for pid in to_delete:
            assert pid not in subset
    finally:
        _cleanup()
