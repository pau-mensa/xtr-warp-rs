# /// script
# dependencies = [
#    "pylate>=1.3.3",
#    "beir>=2.2.0",
#    "fast_plaid",
#    "ranx",
#    "psutil"
# ]
# ///

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_PKG = REPO_ROOT / "python"

for entry in (REPO_ROOT, PY_PKG):
    entry_str = str(entry)
    if entry_str not in sys.path:
        sys.path.insert(0, entry_str)

import numpy as np
import psutil
import torch
from fast_plaid import search
from pylate import models
from xtr_warp.evaluation import evaluate, load_beir

# from xtr_warp_rust import evaluation, search
from xtr_warp.search import XTRWarp

DEVICE = "cuda"
TEST_BOUND = 128
# Use a modest t_prime that the reduced test index can satisfy; the default
# heuristic (>=1000) assumes production-scale indexes and undervalues missing
# similarities in this small diagnostic setup.
TEST_T_PRIME = None

parser = argparse.ArgumentParser(
    description="Run Fast-PLAiD evaluation on a BEIR dataset with optional limits for testing."
)
parser.add_argument(
    "--dataset",
    type=str,
    default="scifact",
    help="Name of the dataset to process from the BEIR benchmark.",
)
parser.add_argument(
    "--n-docs",
    type=int,
    default=None,
    help="Limit the number of documents to embed (useful for testing before full evaluation).",
)
parser.add_argument(
    "--n-queries",
    type=int,
    default=None,
    help="Limit the number of queries to test (useful for testing before full evaluation).",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print per-query relevance diagnostics (which retrieved docs overlap qrels).",
)
args = parser.parse_args()
dataset_name = args.dataset


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB


def get_memory_baseline():
    """Get a stable baseline memory reading."""
    # Take multiple readings to get a stable baseline
    readings = []
    for _ in range(5):
        readings.append(get_memory_usage())
        time.sleep(0.01)
    return sum(readings) / len(readings)


class PeakMemoryMonitor:
    """Monitor peak memory usage during a function execution."""

    def __init__(self, pre_operation_baseline=None):
        self.peak_memory = 0
        self.initial_memory = 0
        self.pre_operation_baseline = pre_operation_baseline
        self.monitoring = False
        self.monitor_thread = None

    def _monitor_memory(self):
        """Continuously monitor memory usage in a separate thread."""
        while self.monitoring:
            current_memory = get_memory_usage()
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
            time.sleep(0.001)  # Check every 1ms for better granularity

    def start_monitoring(self):
        """Start monitoring memory usage."""
        # Wait a moment to ensure thread starts cleanly
        time.sleep(0.01)
        self.initial_memory = get_memory_baseline()
        self.peak_memory = self.initial_memory
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        # Ensure thread is actually running
        time.sleep(0.01)

    def stop_monitoring(self):
        """Stop monitoring and return peak memory increase."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        total_increase = (
            self.peak_memory - self.pre_operation_baseline
            if self.pre_operation_baseline
            else 0
        )

        return self.peak_memory, total_increase


def log_query_hits(
    engine: str, queries_dict: dict, ranked, qrels_map: dict, available_docs: set[str]
) -> None:
    """Print per-query relevance intersections for debugging."""
    print(f"\n[{engine}] per-query relevance summary:")
    for (query_id, query_text), query_results in zip(
        queries_dict.items(), ranked, strict=True
    ):
        retrieved_ids = [entry["id"] for entry in query_results]
        relevant_ids = set(qrels_map.get(query_text, {}))
        relevant_in_corpus = relevant_ids & available_docs
        hits = relevant_in_corpus & set(retrieved_ids)
        print(
            f"  query {query_id} · {len(hits)}/{len(relevant_in_corpus)} hits "
            f"(accessible rel={len(relevant_in_corpus)}/{len(relevant_ids)})"
        )
        if hits:
            print(f"    hit doc ids: {sorted(hits)}")
        else:
            print(f"    top retrieved ids: {retrieved_ids[:5]}")


query_length = {
    "quora": 32,
    "climate-fever": 64,
    "nq": 32,
    "msmarco": 32,
    "hotpotqa": 32,
    "nfcorpus": 32,
    "scifact": 48,
    "trec-covid": 48,
    "fiqa": 32,
    "arguana": 64,
    "scidocs": 48,
    "dbpedia-entity": 32,
    "webis-touche2020": 32,
    "fever": 32,
}

print(f"🚀 Starting evaluation for dataset: {dataset_name}")

model = models.ColBERT(
    model_name_or_path="answerdotai/answerai-colbert-small-v1",
    query_length=query_length.get(dataset_name, 32),
    document_length=300,
)

shutil.rmtree(dataset_name, ignore_errors=True)
os.makedirs(dataset_name, exist_ok=True)
shutil.rmtree(f"{dataset_name}_pylate", ignore_errors=True)

print(f"📚 Loading BEIR dataset: {dataset_name}")
documents, queries, qrels, documents_ids = load_beir(
    dataset_name=dataset_name,
    split="dev" if "msmarco" in dataset_name else "test",
)
print(f"📚 Loaded {len(documents)} documents and {len(queries)} queries")
available_doc_ids = {document["id"] for document in documents}

# Limit documents and queries if specified
print(f"Document sample: {documents[0]}")
if args.n_docs is not None:
    print(f"🔢 Limiting documents to first {args.n_docs} documents")
    documents = documents[: args.n_docs]
    # Rebuild documents_ids dictionary for the limited documents
    documents_ids = {index: document["id"] for index, document in enumerate(documents)}
    available_doc_ids = {document["id"] for document in documents}
print(f"Query sample: {list(queries.items())[0]}")
if args.n_queries is not None:
    print(f"🔢 Limiting queries to first {args.n_queries} queries")
    queries_items = list(queries.items())[: args.n_queries]
    queries = dict(queries_items)
    # Filter qrels to only include the limited queries
    qrels = {
        query_text: {
            doc_id: relevance
            for doc_id, relevance in qrels[query_text].items()
            if doc_id in available_doc_ids
        }
        for query_text in queries.values()
        if query_text in qrels
    }
else:
    # Ensure qrels only contains documents that exist in the (possibly truncated) corpus
    qrels = {
        query_text: {
            doc_id: relevance
            for doc_id, relevance in query_docs.items()
            if doc_id in available_doc_ids
        }
        for query_text, query_docs in qrels.items()
    }
print("-" * 150)
num_queries = len(queries)
print(f"📊 Processing {len(documents)} documents and {num_queries} queries")

if False:
    print(f"🧠 Encoding documents for {dataset_name}...")
    documents_embeddings = model.encode(
        [document["text"] for document in documents],
        batch_size=256,
        show_progress_bar=True,
        is_query=False,
    )

    print(f"🧠 Encoding queries for {dataset_name}...")
    queries_embeddings = model.encode(
        list(queries.values()),
        batch_size=256,
        show_progress_bar=True,
        is_query=True,
    )

    queries_embeddings = torch.Tensor(np.array(queries_embeddings))
    documents_embeddings = [torch.tensor(doc_emb) for doc_emb in documents_embeddings]
    queries_embeddings = torch.cat(tensors=[queries_embeddings], dim=0)

    torch.save(documents_embeddings, f"documents_embeddings_{dataset_name}.pt")
    torch.save(queries_embeddings, f"queries_embeddings_{dataset_name}.pt")
else:
    documents_embeddings = torch.load(f"documents_embeddings_{dataset_name}.pt")
    queries_embeddings = torch.load(f"queries_embeddings_{dataset_name}.pt")

# FastPlaid
if False:
    print(f"\n=== 🚀 FastPlaid Evaluation ===")

    # Get baseline memory before creating index
    pre_index_memory = get_memory_baseline()
    print(f"🧠 Memory before FastPlaid index: {pre_index_memory:.2f} MB")

    index = search.FastPlaid(
        index=os.path.join("benchmark", dataset_name), device=DEVICE
    )
    print(f"🏗️  Building index for {dataset_name}...")
    start_index = time.time()
    index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
    end_index = time.time()
    indexing_time = end_index - start_index
    post_index_memory = get_memory_baseline()
    print(
        f"🧠 Memory after indexing: {post_index_memory:.2f} MB (↗️ +{post_index_memory - pre_index_memory:.2f} MB)"
    )
    print(f"\t✅ {dataset_name} indexing: {indexing_time:.2f} seconds")

    print(f"🔍 Searching on {dataset_name}...")
    # Monitor peak memory during search
    memory_monitor = PeakMemoryMonitor(pre_operation_baseline=pre_index_memory)
    memory_monitor.start_monitoring()
    start_search = time.time()
    scores = index.search(
        queries_embeddings=queries_embeddings,
        top_k=500,
        n_ivf_probe=32,
        n_full_scores=4096,
    )
    end_search = time.time()
    search_time = end_search - start_search
    peak_memory, total_increase = memory_monitor.stop_monitoring()
    print(
        f"🧠 FastPlaid peak memory during search: {peak_memory:.2f} MB (↗️ +{total_increase:.2f} MB from baseline)"
    )

    large_queries_embeddings = torch.cat(
        ([queries_embeddings] * ((1000 // queries_embeddings.shape[0]) + 1))[:1000]
    )

    print(f"🔍 50_000 queries on {dataset_name}...")
    start_search = time.time()
    _ = index.search(
        queries_embeddings=large_queries_embeddings,
        top_k=500,
        n_ivf_probe=32,
        n_full_scores=4096,
    )
    end_search = time.time()
    heavy_search_time = end_search - start_search
    queries_per_second = large_queries_embeddings.shape[0] / heavy_search_time
    print(
        f"\t✅ {dataset_name} search: {heavy_search_time:.2f} seconds ({queries_per_second:.2f} QPS)"
    )

    results = []
    for (query_id, _), query_scores in zip(queries.items(), scores, strict=True):
        results.append(
            [
                {"id": documents_ids[document_id], "score": score}
                for document_id, score in query_scores
                if documents_ids[document_id] != query_id
            ]
        )

    print(f"📊 Calculating metrics for {dataset_name}...")
    evaluation_scores = evaluate(
        scores=results,
        qrels=qrels,
        queries=list(queries.values()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(f"\n--- 📈 Final Scores for {dataset_name} ---")
    print(evaluation_scores)

    output_dir = "./benchmark"
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        "dataset": dataset_name,
        "indexing": round(indexing_time, 3),
        "search": round(search_time, 3),
        "qps": round(queries_per_second, 2),
        "size": len(documents),
        "queries": num_queries,
        "scores": evaluation_scores,
        "memory": {
            "peak_search_mb": round(peak_memory, 2),
            "total_increase_mb": round(total_increase, 2),
        },
    }

    output_filepath = os.path.join(output_dir, f"{dataset_name}_fastplaid.json")
    print(f"💾 Exporting results to {output_filepath}")
    with open(output_filepath, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"🎉 Finished evaluation for dataset: {dataset_name}\n")

# XTR-Warp

print(f"\n=== 🚀 XTR-Warp Evaluation ===")

# Get baseline memory before creating index
pre_index_memory = get_memory_baseline()
print(f"🧠 Memory before XTR-Warp index: {pre_index_memory:.2f} MB")

index = XTRWarp(index=os.path.join(".indexes", dataset_name))
print(f"🏗️  Building index for {dataset_name}...")
print(f"Document shape: {documents_embeddings[0].shape}")
start_index = time.time()
index.create(
    documents_embeddings=documents_embeddings,
    kmeans_niters=4,
    max_points_per_centroid=256,
    nbits=4,
    seed=42,
    device="cuda",
)
end_index = time.time()
indexing_time = end_index - start_index
post_index_memory = get_memory_baseline()
print(
    f"🧠 Memory after indexing: {post_index_memory:.2f} MB (↗️ +{post_index_memory - pre_index_memory:.2f} MB)"
)
print(f"\t✅ {dataset_name} indexing: {indexing_time:.2f} seconds")

print(f"🔍 Searching on {dataset_name}...")
# Monitor peak memory during search
memory_monitor = PeakMemoryMonitor(pre_operation_baseline=pre_index_memory)
memory_monitor.start_monitoring()
index.load("cpu")
# queries_embeddings = queries_embeddings.to("cuda")
start_search = time.time()
scores = index.search(
    queries_embeddings=queries_embeddings,
    top_k=500,
    nprobe=32,
    centroid_score_threshold=0.5,
    max_candidates=4096,
    num_threads=64,
    # t_prime=TEST_T_PRIME,
    # bound=128,
    # dtype=torch.float16,
)
end_search = time.time()
search_time = end_search - start_search
peak_memory, total_increase = memory_monitor.stop_monitoring()
print(
    f"🧠 XTR-Warp peak memory during search: {peak_memory:.2f} MB (↗️ +{total_increase:.2f} MB from baseline)"
)

large_queries_embeddings = torch.cat(
    ([queries_embeddings] * ((1000 // queries_embeddings.shape[0]) + 1))[:1000]
)  # .to(torch.float16)

print(f"🔍 50_000 queries on {dataset_name} - {large_queries_embeddings.shape}...")
start_search = time.time()
_ = index.search(
    queries_embeddings=large_queries_embeddings,
    top_k=500,
    nprobe=32,
    centroid_score_threshold=0.5,
    max_candidates=4096,
    num_threads=64,
    # t_prime=TEST_T_PRIME,
    # bound=TEST_BOUND,
    # dtype=torch.float16,
)
end_search = time.time()
heavy_search_time = end_search - start_search
queries_per_second = large_queries_embeddings.shape[0] / heavy_search_time
print(
    f"\t✅ {dataset_name} search: {heavy_search_time:.2f} seconds ({queries_per_second:.2f} QPS)"
)

results = []
for (query_id, _), query_scores in zip(queries.items(), scores, strict=True):
    results.append(
        [
            {"id": documents_ids[document_id], "score": score}
            for document_id, score in query_scores
            if documents_ids[document_id] != query_id
        ]
    )

if args.debug:
    log_query_hits("XTR-WARP", queries, results, qrels, available_doc_ids)

print(f"📊 Calculating metrics for {dataset_name}...")
print(qrels)
evaluation_scores = evaluate(
    scores=results,
    qrels=qrels,
    queries=list(queries.values()),
    metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
)

print(f"\n--- 📈 Final Scores for {dataset_name} ---")
print(evaluation_scores)

output_dir = "./benchmark"
os.makedirs(output_dir, exist_ok=True)

output_data = {
    "dataset": dataset_name,
    "indexing": round(indexing_time, 3),
    "search": round(search_time, 3),
    "qps": round(queries_per_second, 2),
    "size": len(documents),
    "queries": num_queries,
    "scores": evaluation_scores,
    "memory": {
        "peak_search_mb": round(peak_memory, 2),
        "total_increase_mb": round(total_increase, 2),
    },
}

output_filepath = os.path.join(output_dir, f"{dataset_name}.json")
print(f"💾 Exporting results to {output_filepath}")
with open(output_filepath, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"🎉 Finished evaluation for dataset: {dataset_name}\n")

# Pylate
if False:
    print(f"\n=== 🚀 Pylate Evaluation ===")

    # Get baseline memory before creating index
    pre_index_memory = get_memory_baseline()
    print(f"🧠 Memory before Pylate index: {pre_index_memory:.2f} MB")

    from pylate import evaluation, indexes, retrieve

    index = indexes.PLAID(
        override=True,
        index_name=f"{dataset_name}_pylate",
        embedding_size=96,
        nbits=4,
        device=DEVICE,
    )

    retriever = retrieve.ColBERT(index=index)

    start = time.time()
    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )
    end = time.time()
    indexing_time = end - start
    post_index_memory = get_memory_baseline()
    print(
        f"🧠 Memory after indexing: {post_index_memory:.2f} MB (↗️ +{post_index_memory - pre_index_memory:.2f} MB)"
    )
    print(f"🏗️  Pylate index on {dataset_name}: {end - start:.2f} seconds")

    # Monitor peak memory during search
    memory_monitor = PeakMemoryMonitor(pre_operation_baseline=pre_index_memory)
    memory_monitor.start_monitoring()
    start = time.time()
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=20)
    end = time.time()
    search_time = end - start
    peak_memory, total_increase = memory_monitor.stop_monitoring()
    print(f"🔍 Pylate search on {dataset_name}: {search_time:.2f} seconds")
    print(
        f"🧠 Pylate peak memory during search: {peak_memory:.2f} MB (↗️ +{total_increase:.2f} MB from baseline)"
    )

    start = time.time()
    _ = retriever.retrieve(queries_embeddings=large_queries_embeddings, k=20)
    end = time.time()
    heavy_search_time = end - start
    queries_per_second = large_queries_embeddings.shape[0] / heavy_search_time

    for (query_id, query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                # Remove the query_id from the score
                query_scores.remove(score)

    if args.debug:
        log_query_hits("PLAID", queries, scores, qrels, available_doc_ids)

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.values()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(f"\n--- 📈 Final Scores for {dataset_name} (Pylate) ---")
    print(evaluation_scores)

    output_data = {
        "dataset": dataset_name,
        "indexing": round(indexing_time, 3),
        "search": round(search_time, 3),
        "qps": round(queries_per_second, 2),
        "size": len(documents),
        "queries": num_queries,
        "scores": evaluation_scores,
        "memory": {
            "peak_search_mb": round(peak_memory, 2),
            "total_increase_mb": round(total_increase, 2),
        },
    }

    output_filepath = os.path.join(output_dir, f"{dataset_name}_pylate.json")
    with open(output_filepath, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"💾 Exporting Pylate results to {output_filepath}")
