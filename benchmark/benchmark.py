# /// script
# dependencies = [
#    "pylate>=1.3.3",
#    "beir>=2.2.0",
#    "fast_plaid",
#    "ranx",
#    "psutil",
#    "pyyaml"
# ]
# ///

import argparse
import gc
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import yaml

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
from xtr_warp.search import XTRWarp


def get_cpu_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_gpu_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def reset_gpu_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_stable_baseline():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    readings = []
    for _ in range(10):
        readings.append(get_cpu_memory_mb())
        time.sleep(0.02)
    return sum(readings) / len(readings)


class MemoryMonitor:
    def __init__(self, pre_operation_baseline=None):
        self.cpu_baseline = pre_operation_baseline if pre_operation_baseline else 0
        self.cpu_peak = 0
        self.gpu_peak = 0
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None

    def _monitor(self):
        while self.monitoring:
            current_cpu = get_cpu_memory_mb()
            current_gpu = get_gpu_memory_mb()

            if current_cpu > self.cpu_peak:
                self.cpu_peak = current_cpu
            if current_gpu > self.gpu_peak:
                self.gpu_peak = current_gpu

            time.sleep(0.005)

    def start(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            reset_gpu_peak()

        time.sleep(0.05)
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self.monitor_thread.start()
        time.sleep(0.05)
        self.start_time = time.time()

    def stop(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_cpu_peak = self.cpu_peak
        final_gpu_peak = self.gpu_peak

        torch_gpu_peak = get_gpu_peak_mb()
        final_gpu_peak = max(final_gpu_peak, torch_gpu_peak)

        cpu_increase = final_cpu_peak - self.cpu_baseline if self.cpu_baseline else 0

        duration = time.time() - self.start_time if self.start_time else 0

        return {
            "cpu_peak_mb": round(final_cpu_peak, 2),
            "cpu_increase_mb": round(cpu_increase, 2),
            "gpu_peak_mb": round(final_gpu_peak, 2),
            "gpu_increase_mb": round(final_gpu_peak, 2),
            "duration_seconds": round(duration, 3),
        }


QUERY_LENGTH = {
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


def run_xtr_warp(
    config,
    documents,
    queries,
    qrels,
    documents_ids,
    documents_embeddings,
    queries_embeddings,
    dataset_name,
):
    print(f"\n=== 🚀 XTR-Warp Evaluation ===")

    pre_index_memory = get_stable_baseline()
    print(f"🧠 Memory before XTR-Warp index: {pre_index_memory:.2f} MB")

    index_dir = os.path.join(".indices", dataset_name)
    index = XTRWarp(index=index_dir)

    print(f"🏗️  Building index for {dataset_name}...")
    print(f"Document shape: {documents_embeddings[0].shape}")

    index_monitor = MemoryMonitor(pre_operation_baseline=pre_index_memory)
    index_monitor.start()

    start_index = time.time()
    if config["device"] == "cuda":
        index.create(
            documents_embeddings=documents_embeddings,
            kmeans_niters=4,
            max_points_per_centroid=256,
            nbits=4,
            seed=42,
            device=config["device"],
        )
    end_index = time.time()

    index_memory = index_monitor.stop()
    indexing_time = end_index - start_index

    print(f"\t✅ {dataset_name} indexing: {indexing_time:.2f} seconds")
    print(
        f"🧠 Indexing memory - CPU: +{index_memory['cpu_increase_mb']:.2f} MB, GPU: +{index_memory['gpu_increase_mb']:.2f} MB"
    )

    print(f"🔍 Searching on {dataset_name}...")

    precision_dtype = getattr(torch, config["precision"])
    index.load(config["device"], dtype=precision_dtype)

    search_monitor = MemoryMonitor(pre_operation_baseline=pre_index_memory)
    search_monitor.start()

    start_search = time.time()
    scores = index.search(
        queries_embeddings=queries_embeddings,
        top_k=config["top_k"],
        num_threads=config.get("num_threads", 1),
        batch_size=4096,
        nprobe=config.get("nprobe"),
        bound=config.get("bound"),
        t_prime=config.get("t_prime"),
        max_candidates=config.get("max_candidates"),
        centroid_score_threshold=config.get("centroid_score_threshold"),
    )
    end_search = time.time()
    search_time = end_search - start_search

    search_memory = search_monitor.stop()
    print(
        f"🧠 Search memory - CPU: +{search_memory['cpu_increase_mb']:.2f} MB, GPU: +{search_memory['gpu_increase_mb']:.2f} MB"
    )

    large_queries_embeddings = torch.cat(
        ([queries_embeddings] * ((1000 // queries_embeddings.shape[0]) + 1))[:1000]
    )[:100]

    print(
        f"🔍 {large_queries_embeddings.shape[0]} queries on {dataset_name} - {large_queries_embeddings.shape}..."
    )
    start_search = time.time()
    _ = index.search(
        queries_embeddings=large_queries_embeddings,
        top_k=config["top_k"],
        num_threads=config.get("num_threads", 1),
        nprobe=config.get("nprobe"),
        bound=config.get("bound"),
        t_prime=config.get("t_prime"),
        max_candidates=config.get("max_candidates"),
        centroid_score_threshold=config.get("centroid_score_threshold"),
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

    output_data = {
        "dataset": dataset_name,
        "framework": "xtr-warp",
        "config": config,
        "indexing": round(indexing_time, 3),
        "search": round(search_time, 3),
        "qps": round(queries_per_second, 2),
        "size": len(documents),
        "queries": len(queries),
        "scores": evaluation_scores,
        "memory": {
            "indexing": {
                "cpu_increase_mb": index_memory["cpu_increase_mb"],
                "gpu_increase_mb": index_memory["gpu_increase_mb"],
            },
            "search": {
                "cpu_increase_mb": search_memory["cpu_increase_mb"],
                "gpu_increase_mb": search_memory["gpu_increase_mb"],
            },
        },
    }

    return output_data


def run_pylate(
    config,
    documents,
    queries,
    qrels,
    documents_embeddings,
    queries_embeddings,
    dataset_name,
    output_dir,
):
    print(f"\n=== 🚀 Pylate Evaluation ===")

    from pylate import evaluation as pylate_evaluation
    from pylate import indexes, retrieve

    pre_index_memory = get_stable_baseline()
    print(f"🧠 Memory before Pylate index: {pre_index_memory:.2f} MB")

    index_dir = os.path.join(output_dir, f"{dataset_name}_pylate")
    index = indexes.PLAID(
        override=True,
        index_name=index_dir,
        embedding_size=config["embedding_size"],
        nbits=4,
        device=config["device"],
    )

    retriever = retrieve.ColBERT(index=index)

    index_monitor = MemoryMonitor(pre_operation_baseline=pre_index_memory)
    index_monitor.start()

    start = time.time()
    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )
    end = time.time()

    index_memory = index_monitor.stop()
    indexing_time = end - start

    print(f"\t✅ {dataset_name} indexing: {indexing_time:.2f} seconds")
    print(
        f"🧠 Indexing memory - CPU: +{index_memory['cpu_increase_mb']:.2f} MB, GPU: +{index_memory['gpu_increase_mb']:.2f} MB"
    )

    search_monitor = MemoryMonitor(pre_operation_baseline=pre_index_memory)
    search_monitor.start()

    start = time.time()
    scores = retriever.retrieve(
        queries_embeddings=queries_embeddings, k=config["top_k"]
    )
    end = time.time()
    search_time = end - start

    search_memory = search_monitor.stop()
    print(f"🔍 Pylate search on {dataset_name}: {search_time:.2f} seconds")
    print(
        f"🧠 Search memory - CPU: +{search_memory['cpu_increase_mb']:.2f} MB, GPU: +{search_memory['gpu_increase_mb']:.2f} MB"
    )

    large_queries_embeddings = torch.cat(
        ([queries_embeddings] * ((1000 // queries_embeddings.shape[0]) + 1))[:1000]
    )

    start = time.time()
    _ = retriever.retrieve(
        queries_embeddings=large_queries_embeddings, k=config["top_k"]
    )
    end = time.time()
    heavy_search_time = end - start
    queries_per_second = large_queries_embeddings.shape[0] / heavy_search_time

    for (query_id, query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                query_scores.remove(score)

    evaluation_scores = pylate_evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.values()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(f"\n--- 📈 Final Scores for {dataset_name} (Pylate) ---")
    print(evaluation_scores)

    output_data = {
        "dataset": dataset_name,
        "framework": "pylate",
        "config": config,
        "indexing": round(indexing_time, 3),
        "search": round(search_time, 3),
        "qps": round(queries_per_second, 2),
        "size": len(documents),
        "queries": len(queries),
        "scores": evaluation_scores,
        "memory": {
            "indexing": {
                "cpu_increase_mb": index_memory["cpu_increase_mb"],
                "gpu_increase_mb": index_memory["gpu_increase_mb"],
            },
            "search": {
                "cpu_increase_mb": search_memory["cpu_increase_mb"],
                "gpu_increase_mb": search_memory["gpu_increase_mb"],
            },
        },
    }

    return output_data


def run_fast_plaid(
    config,
    documents,
    queries,
    qrels,
    documents_ids,
    documents_embeddings,
    queries_embeddings,
    dataset_name,
    output_dir,
):
    print(f"\n=== 🚀 FastPlaid Evaluation ===")

    pre_index_memory = get_stable_baseline()
    print(f"🧠 Memory before FastPlaid index: {pre_index_memory:.2f} MB")

    index_dir = os.path.join(output_dir, dataset_name)
    index = search.FastPlaid(index=index_dir, device=config["device"])

    print(f"🏗️  Building index for {dataset_name}...")

    index_monitor = MemoryMonitor(pre_operation_baseline=pre_index_memory)
    index_monitor.start()

    start_index = time.time()
    if config["device"] == "cuda":
        index.create(documents_embeddings=documents_embeddings, kmeans_niters=4)
    end_index = time.time()

    index_memory = index_monitor.stop()
    indexing_time = end_index - start_index

    print(f"\t✅ {dataset_name} indexing: {indexing_time:.2f} seconds")
    print(
        f"🧠 Indexing memory - CPU: +{index_memory['cpu_increase_mb']:.2f} MB, GPU: +{index_memory['gpu_increase_mb']:.2f} MB"
    )

    print(f"🔍 Searching on {dataset_name}...")

    search_monitor = MemoryMonitor(pre_operation_baseline=pre_index_memory)
    search_monitor.start()

    start_search = time.time()
    scores = index.search(
        queries_embeddings=queries_embeddings,
        top_k=config["top_k"],
        n_ivf_probe=config.get("n_ivf_probe", 8),
        n_full_scores=config.get("n_full_scores", 4096),
    )
    end_search = time.time()
    search_time = end_search - start_search

    search_memory = search_monitor.stop()
    print(
        f"🧠 Search memory - CPU: +{search_memory['cpu_increase_mb']:.2f} MB, GPU: +{search_memory['gpu_increase_mb']:.2f} MB"
    )

    large_queries_embeddings = torch.cat(
        ([queries_embeddings] * ((1000 // queries_embeddings.shape[0]) + 1))[:1000]
    )

    print(f"🔍 {queries_embeddings.shape[0]} queries on {dataset_name}...")
    start_search = time.time()
    _ = index.search(
        queries_embeddings=large_queries_embeddings,
        top_k=config["top_k"],
        n_ivf_probe=config["n_ivf_probe"],
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

    output_data = {
        "dataset": dataset_name,
        "framework": "fast-plaid",
        "config": config,
        "indexing": round(indexing_time, 3),
        "search": round(search_time, 3),
        "qps": round(queries_per_second, 2),
        "size": len(documents),
        "queries": len(queries),
        "scores": evaluation_scores,
        "memory": {
            "indexing": {
                "cpu_increase_mb": index_memory["cpu_increase_mb"],
                "gpu_increase_mb": index_memory["gpu_increase_mb"],
            },
            "search": {
                "cpu_increase_mb": search_memory["cpu_increase_mb"],
                "gpu_increase_mb": search_memory["gpu_increase_mb"],
            },
        },
    }

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments from YAML configuration files."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file (e.g., benchmark/experiments/xtr-warp_0.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
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
    args = parser.parse_args()

    config_path = Path(args.config)
    config_name = config_path.stem
    framework, experiment_id = config_name.rsplit("_", 1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = args.dataset
    experiments_dir = Path(config_path).parent
    results_dir = experiments_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting evaluation for dataset: {dataset_name}")
    print(f"Framework: {framework}, Experiment ID: {experiment_id}")
    print(f"Config: {config}")

    model = models.ColBERT(
        model_name_or_path="answerdotai/answerai-colbert-small-v1",
        query_length=QUERY_LENGTH.get(dataset_name, 32),
        document_length=300,
    )

    shutil.rmtree(dataset_name, ignore_errors=True)
    os.makedirs(dataset_name, exist_ok=True)

    print(f"📚 Loading BEIR dataset: {dataset_name}")
    documents, queries, qrels, documents_ids = load_beir(
        dataset_name=dataset_name,
        split="dev" if "msmarco" in dataset_name else "test",
    )
    print(f"📚 Loaded {len(documents)} documents and {len(queries)} queries")
    available_doc_ids = {document["id"] for document in documents}

    if args.n_docs is not None:
        print(f"🔢 Limiting documents to first {args.n_docs} documents")
        documents = documents[: args.n_docs]
        documents_ids = {
            index: document["id"] for index, document in enumerate(documents)
        }
        available_doc_ids = {document["id"] for document in documents}

    if args.n_queries is not None:
        print(f"🔢 Limiting queries to first {args.n_queries} queries")
        queries_items = list(queries.items())[: args.n_queries]
        queries = dict(queries_items)
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
        qrels = {
            query_text: {
                doc_id: relevance
                for doc_id, relevance in query_docs.items()
                if doc_id in available_doc_ids
            }
            for query_text, query_docs in qrels.items()
        }

    print("-" * 150)
    print(f"📊 Processing {len(documents)} documents and {len(queries)} queries")

    embeddings_dir = Path("embeddings")
    embeddings_dir.mkdir(exist_ok=True)
    docs_emb_path = embeddings_dir / f"documents_embeddings_{dataset_name}.pt"
    queries_emb_path = embeddings_dir / f"queries_embeddings_{dataset_name}.pt"

    if docs_emb_path.exists() and queries_emb_path.exists():
        print(f"📂 Loading cached embeddings from {embeddings_dir}")
        # documents_embeddings = np.array([[]]) # placeholder for vram issues
        documents_embeddings = torch.load(docs_emb_path)
        queries_embeddings = torch.load(queries_emb_path)
        if args.n_docs is not None:
            # documents_embeddings = np.array([[]]) # placeholder for vram issues
            documents_embeddings = documents_embeddings[: args.n_docs]
        if args.n_queries is not None:
            queries_embeddings = queries_embeddings[: args.n_queries]
    else:
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
        documents_embeddings = [
            torch.tensor(doc_emb) for doc_emb in documents_embeddings
        ]
        queries_embeddings = torch.cat(tensors=[queries_embeddings], dim=0)

        torch.save(documents_embeddings, docs_emb_path)
        torch.save(queries_embeddings, queries_emb_path)
        print(f"💾 Saved embeddings to {embeddings_dir}")

    if framework == "xtr-warp":
        output_data = run_xtr_warp(
            config,
            documents,
            queries,
            qrels,
            documents_ids,
            documents_embeddings,
            queries_embeddings,
            dataset_name,
        )
    elif framework == "pylate":
        output_data = run_pylate(
            config,
            documents,
            queries,
            qrels,
            documents_embeddings,
            queries_embeddings,
            dataset_name,
            str(experiments_dir),
        )
    elif framework == "fast-plaid":
        output_data = run_fast_plaid(
            config,
            documents,
            queries,
            qrels,
            documents_ids,
            documents_embeddings,
            queries_embeddings,
            dataset_name,
            str(experiments_dir),
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")

    output_filepath = results_dir / f"{dataset_name}_{framework}_{experiment_id}.json"
    print(f"💾 Exporting results to {output_filepath}")
    with open(output_filepath, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"🎉 Finished evaluation for dataset: {dataset_name}\n")


if __name__ == "__main__":
    main()
