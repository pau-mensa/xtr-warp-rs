<div align="center">
  <h1>Warp</h1>
  <p align="center">
    <img src="assets/logo.png" alt="Warp Logo" width="400">
  </p>
</div>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" alt="Python Versions">
  <img src="https://github.com/pau-mensa/xtr-warp-rs/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  <img src="https://img.shields.io/badge/Platform-Ubuntu%7C%20macOS%20%7C%20Windows-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  <a href="https://github.com/rust-lang/rust"><img src="https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="rust"></a>
  <a href="https://github.com/pyo3"><img src="https://img.shields.io/badge/PyO₃-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="PyO₃"></a>
  <a href="https://github.com/LaurentMazare/tch-rs"><img src="https://img.shields.io/badge/tch--rs-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white" alt="tch-rs"></a>
</p>
<div align="center">
    The Multi-Vector Search Engine To Rule Them All
</div>

&nbsp;

## ⭐️ Overview

xtr-warp-rs is a high-performance implementation of the **WARP** engine for multi-vector retrieval, as described in the [WARP paper (SIGIR 2025)](https://arxiv.org/abs/2501.17788). Originally built with [XTR models (NeurIPS 2023)](https://arxiv.org/abs/2304.01982) in mind, as it turns out, it significantly outperforms all other multi-vector search engines while keeping retrieval metrics competitive.

Compared to the current SOTA (FastPlaid), xtr-warp-rs focuses on doing less work per query while staying close in quality: it prunes the centroid/posting-list space per token, uses an error-aware merge that keeps ranking stable with fewer examined candidates, and keeps the hot path (selection → decompression → merge) highly optimized and parallel friendly.

**Speed**: Achieves **10-40x** speedup on CUDA and **4-130x** on CPU (depending on dataset and thread count) vs FastPlaid.

**Memory**: During search WARP reduces peak CPU memory by **60%** on average vs FastPlaid, reaching **82%** on individual datasets. The sharded execution mode treats RAM+VRAM as a unified pool and cuts peak GPU memory by another **38%** on average over pure CUDA. During index creation the VRAM usage is around **10%** less, with an optional streaming mode that reduces it further by **66%** at a **20-25%** speed cost.

Check the [benchmark section](#benchmarks) for detailed comparisons.

&nbsp;

## Installation

```bash
uv pip install xtr-warp-rs
```

## PyTorch Compatibility

xtr-warp-rs supports three torch versions:

| xtr-warp-rs Version | PyTorch Version | Installation Command                |
| ------------------- | --------------- | ----------------------------------- |
| 2.0.2.2110        | 2.11.0          | `uv pip install xtr-warp-rs==2.0.2.2110` |
| 2.0.2.2100        | 2.10.0          | `uv pip install xtr-warp-rs==2.0.2.2100` |
| 2.0.2.290         | 2.9.0           | `uv pip install xtr-warp-rs==2.0.2.290` |
| 2.0.2.280         | 2.8.0           | `uv pip install xtr-warp-rs==2.0.2.280` |
| 2.0.2.270         | 2.7.0           | `uv pip install xtr-warp-rs==2.0.2.270` |

### Build from Source

**Install Rust:**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Install `uv`:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

**Clone and build the repo:**

```bash
git clone git@github.com:pau-mensa/xtr-warp-rs.git
cd xtr-warp-rs
make install # or make install-gpu if you have a GPU available
make build
```

## ⚡️ Quick Start

```python
import torch
from xtr_warp import XTRWarp

idx = XTRWarp(index="my_index")
embedding_dim = 128

# 1. Create the index with document metadata
metadata = [
    {"category": "science", "year": 2024, "tags": ["ml", "search"]},
    {"category": "history", "year": 2023, "tags": ["medieval"]},
    # ... one dict per document
]
idx.create(
    embeddings_source=[torch.randn(300, embedding_dim) for _ in range(100)],
    device="cpu",
    metadata=metadata,
)
idx.load(device="cpu", dtype=torch.float32)

# 2. Search (returns list of lists of (passage_id, score) tuples)
results = idx.search(queries_embeddings=torch.randn(2, 50, embedding_dim), top_k=10)

# 3. Filter by metadata, then search only matching documents
subset = idx.filter("category = ? AND year >= ?", ["science", 2024])
results = idx.search(queries_embeddings=torch.randn(2, 50, embedding_dim), top_k=10, subset=subset)

# 4. Incremental mutations (metadata can be attached to new documents too)
new_ids = idx.add(new_docs, metadata=[{"category": "art", "year": 2025}])
idx.update(passage_ids=[5], embeddings_source=[torch.randn(200, embedding_dim)])
idx.delete(passage_ids=[2, 8])

# 5. Searches immediately reflect all changes
results = idx.search(queries_embeddings=torch.randn(2, 50, embedding_dim), top_k=10)

# 6. Compact when convenient
idx.compact()
```

&nbsp;

## Benchmarks

- `qps` stands for **Queries Per Second** (higher is better)
- `indexing` stands for the time it took the engine to build the index (lower is better)

### CUDA

| Dataset (Size) | Metric | fast-plaid | xtr-warp-rs |
|----------------|--------|------------|-------------|
| arguana (8,674) | qps | 78.27 | 2706.59 (+3358.4%) |
|  | indexing | 3.01s | 0.94s |
|  | ndcg@10 | 0.47 | 0.50 |
|  | recall@10 | 0.73 | 0.76 |
| fiqa (57,638) | qps | 60.81 | 2423.17 (+3884.8%) |
|  | indexing | 6.02s | 3.92s |
|  | ndcg@10 | 0.41 | 0.36 |
|  | recall@10 | 0.48 | 0.42 |
| nfcorpus (3,633) | qps | 81.20 | 3208.68 (+3851.0%) |
|  | indexing | 2.28s | 0.61s |
|  | ndcg@10 | 0.37 | 0.37 |
|  | recall@10 | 0.18 | 0.17 |
| quora (522,931) | qps | 131.38 | 2404.28 (+1730.0%) |
|  | indexing | 6.57s | 9.78s |
|  | ndcg@10 | 0.88 | 0.86 |
|  | recall@10 | 0.95 | 0.94 |
| scidocs (25,657) | qps | 76.79 | 1830.73 (+2284.4%) |
|  | indexing | 5.29s | 3.01s |
|  | ndcg@10 | 0.19 | 0.18 |
|  | recall@10 | 0.19 | 0.19 |
| scifact (5,183) | qps | 72.38 | 3063.71 (+4133.0%) |
|  | indexing | 3.03s | 0.84s |
|  | ndcg@10 | 0.74 | 0.73 |
|  | recall@10 | 0.86 | 0.85 |
| trec-covid (171,332) | qps | 32.68 | 340.14 (+940.7%) |
|  | indexing | 17.28s | 21.79s |
|  | ndcg@10 | 0.84 | 0.80 |
|  | recall@10 | 0.02 | 0.02 |
| webis-touche2020 (382,545) | qps | 36.41 | 741.70 (+1937.1%) |
|  | indexing | 29.25s | 38.71s |
|  | ndcg@10 | 0.25 | 0.24 |
|  | recall@10 | 0.16 | 0.16 |

### CPU

#### Search Speed

| Dataset (Size) | QPS fast-plaid | QPS xtr-warp-rs (Single) | QPS xtr-warp-rs (Multi) |
|----------------|----------------|--------------------------|-------------------------|
| arguana (8,674) | 7.84 | 159.13 (+1929.7%) | 726.00 (+9159.6%) |
| fiqa (57,638) | 6.67 | 100.61 (+1408.4%) | 611.98 (+9075.9%) |
| nfcorpus (3,633) | 15.34 | 134.13 (+774.4%) | 1977.71 (+12792.5%) |
| quora (522,931) | 22.27 | 94.26 (+323.3%) | 613.87 (+2657.0%) |
| scidocs (25,657) | 8.58 | 80.03 (+832.7%) | 469.10 (+5367.4%) |
| scifact (5,183) | 11.27 | 233.75 (+1974.1%) | 973.79 (+8540.7%) |
| trec-covid (171,332) | 2.73 | 21.84 (+700.0%) | 186.08 (+6716.5%) |
| webis-touche2020 (382,545) | 3.58 | 40.59 (+1033.8%) | 303.57 (+8378.8%) |

#### Search Memory

| Dataset (Size) | Peak fast-plaid (GB) | Peak xtr-warp-rs (GB) |
|----------------|----------------------|-----------------------|
| arguana (8,674) | 8.43 | 2.79 (-66.91%) |
| fiqa (57,638) | 9.08 | 3.54 (-61.07%) |
| nfcorpus (3,633) | 10.05 | 1.83 (-81.78%) |
| quora (522,931) | 7.76 | 6.52 (-15.99%) |
| scidocs (25,657) | 9.70 | 3.70 (-61.85%) |
| scifact (5,183) | 9.91 | 2.48 (-74.99%) |
| trec-covid (171,332) | 14.57 | 6.95 (-52.27%) |
| webis-touche2020 (382,545) | 25.34 | 8.78 (-65.36%) |

### Sharded Index (RAM + VRAM as unified memory)

When the index is too large for the available VRAM — or when you want to free up GPU memory for other workloads — xtr-warp-rs can shard it between RAM and GPU memory, treating both as a single unified pool. The split is configured at load time via the `device` argument to `load()`, which accepts three forms:

- `str` — a single device (e.g. `"cuda"`, `"cpu"`, or `"auto"`). No sharding.
- `list[str]` — a list of devices (e.g. `["cuda:0", "cpu"]`). Ratios are auto-computed to fill accelerator VRAM first and place the remainder on CPU.
- `dict[str, float]` — explicit ratios per device (e.g. `{"cuda": 0.1, "cpu": 0.9}` keeps 10% of the work on GPU and 90% on CPU).

You can also call `recommend_device_map(devices)` to get a suggested ratio dict based on available memory, or `estimate_index_memory()` to inspect per-component byte sizes before deciding on a split.

The table below compares pure CUDA against a 10/90 GPU/CPU shard. Sharded mode trades roughly **40% of CUDA's QPS** for a substantial cut in peak GPU memory — and is still **8-23× faster than fast-plaid CUDA** on every dataset.

| Dataset (Size) | Peak GPU CUDA (GB) | Peak GPU Sharded (GB) | QPS CUDA | QPS Sharded |
|----------------|--------------------|-----------------------|----------|-------------|
| arguana (8,674) | 6.28 | 5.68 (-10%) | 2706.59 | 1341.49 |
| fiqa (57,638) | 3.72 | 2.73 (-27%) | 2423.17 | 1334.50 |
| nfcorpus (3,633) | 0.85 | 0.46 (-46%) | 3208.68 | 1732.16 |
| quora (522,931) | 18.54 | 16.38 (-12%) | 2404.28 | 1155.82 |
| scidocs (25,657) | 7.13 | 6.06 (-15%) | 1830.73 | 1117.98 |
| scifact (5,183) | 1.42 | 1.04 (-27%) | 3063.71 | 1666.75 |
| trec-covid (171,332) | 6.74 | 1.38 (-80%) | 340.14 | 254.14 |
| webis-touche2020 (382,545) | 5.78 | 0.86 (-85%) | 741.70 | 615.12 |

### Streamed Indexing

To showcase the benefits and tradeoffs of the stream mode during index creation I ran a benchmark using the `webis-touche2020` dataset (~380K documents). The objective was to split the dataset embeddings into multiple files, achieving a fixed number of documents per file, with a hard cap on 25k documents per file:
- 128k documents per file, 2 splits
- 64k documents per file, 4 splits
- 32k documents per file, 8 splits
- 25k documents per file, 16 splits

<p align="center">
  <img src="assets/memory_benchmark.png" alt="Memory Benchmark" width="1200">
</p>

The experiment results demonstrate a **20-25%** speed decrease that stays constant across all split sizes, but a memory usage that decreases the more splits we have, ranging from **38%** savings on the least aggresive split (only 2) to **66%** on the most aggressive one (16 splits).

&nbsp;

> [!NOTE]  
> These benchmarks were run on an NVIDIA 5090 with an AMD Ryzen 9950 CPU and using `float32` memory mapped tensors

### Reproducibility

Check the [docs](benchmark/README.md) on how to run the benchmark scripts in order to reproduce the results.

## Usage

### Automatic Hyperparameter Optimization

When search parameters are set to `None`, xtr-warp-rs automatically optimizes them based on index metadata and query characteristics. The optimization considers:

- **Index density** (`num_embeddings / num_partitions`): Determines how many embeddings are distributed across clusters
- **Corpus statistics**: Including total embeddings, number of partitions, and average document length
- **Query characteristics**: Number of tokens and desired `top_k` results
- **Dataset properties**: Dense vs sparse distributions, long vs short queries

The optimizer balances recall/accuracy against latency by adjusting parameters like `nprobe` (more probes for dense corpora or long queries), `bound` (larger for high partition counts), `t_prime` (adaptive based on corpus density and query length), and `max_candidates` (scaled with expected candidates).

### Search

```python
Parameter                   Default     Description
nprobe                      None        Number of centroids probed per query token (e.g 8)
bound                       None        Centroids considered before selecting top nprobe (e.g 128)
t_prime                     None        Missing-token penalty (larger = harsher, smaller = more forgiving) (e.g 5000)
centroid_score_threshold    None        Per-token filter to skip weak tokens, from 0 to 1 (e.g 0.5)
max_candidates              None        Cap on document candidates before final selection (e.g 64000)
batch_size                  8192        Batch size for centroid scoring (watch out for VRAM spike in large indices)
num_threads                 1           Upper bound for CPU parallelism during search (not used on CUDA)
subset                      None        List of passage IDs to restrict search to (from filter())
cuda_streams                None        Per-device CUDA stream pool size for merger + decompress fan-out (default 8; 0 or 1 disables fan-out; ignored on CPU)
```

### Indexing

The `create()` method accepts embeddings from multiple sources:

- **In-memory**: `list[torch.Tensor]` or `torch.Tensor` - embeddings already loaded in memory
- **Path-based**: `str` or `Path` - path to a directory or file containing `.npy` embeddings (enables streaming)

```python
Parameter                  Default     Description
embeddings_source          required    Source of document embeddings (see above)
device                     required    Device to use for index creation (e.g., "cpu", "cuda")
kmeans_niters              4           K-means iterations for clustering
max_points_per_centroid    256         Maximum points per centroid during K-means
nbits                      4           Product quantization bits for compression
n_samples_kmeans           None        Samples for K-means clustering
seed                       42          Random seed for reproducibility
use_triton_kmeans          None        Whether to use Triton-based K-means
metadata                   None        List of dicts (one per document) for metadata filtering
```

> [!IMPORTANT]
> Highly recommended to build the index using `cuda` devices. For a large corpus using `cpu` can take forever.

> [!NOTE]
> When using path-based inputs, embeddings files must be 2D tensors (not 3D padded tensors) with accompanying `.doclens.npy` sidecars. See the streaming subsection below for format details.

#### Streaming from Disk

For large datasets, you can stream embeddings directly from disk instead of loading everything into memory when creating the index. The memory savings can be controlled by the number of documents per file, with the max possible saving being 25k documents, because that's the chunk size used during index creation. What this effectively means is that splitting files by less than 25k documents will **not** result in more memory savings.

```python
from xtr_warp import XTRWarp

xtr_warp = XTRWarp(index="index")

# Stream embeddings from a directory containing .npy files
xtr_warp.create(
    embeddings_source="/path/to/embeddings",
    device="cuda",
)

# Load the index
xtr_warp.load(device="cpu", dtype=torch.float32)

# Search for 2 queries, each with 50 tokens, each token is a 128-dim vector
scores = xtr_warp.search(
    queries_embeddings=torch.randn(2, 50, embedding_dim),
    top_k=10,
)

print(scores)
```

**Required format for path-based inputs:**
- Embeddings must be stored as `.npy` files (2D tensors with shape `[total_tokens, embedding_dim]`)
- Each embeddings file must have a corresponding `.doclens.npy` sidecar file containing a 1D array of document lengths. This pattern is adopted to avoid forcing the padding of documents
- The order of the streaming is controlled by the filenames, it is recommended that they end in `..._idx.npy` or `..._idx.doclens.npy`

Example structure:
```
/path/to/embeddings/
├── embeddings_0.npy          # Shape: [total_tokens, 128]
├── embeddings_0.doclens.npy  # Shape: [num_docs], sum(doclens) = total_tokens
├── embeddings_1.npy
├── embeddings_1.doclens.npy
...
```

### Loading

To help with memory management the API also exposes the `load` and `free` methods, which, as the name implies, load and free the index from memory respectively.

```python
Parameter                 Default        Description
device                    "auto"         Where to load the index. Accepts a single device str ("cpu", "cuda", "cuda:0", "auto"), a list[str] (auto-computed shard ratios filling accelerator VRAM first), or a dict[str, float] mapping each device to an explicit ratio
dtype                     torch.float32  Dtype to use for the centroids and bucket weights. Lowers the memory footprint but can cause alterations in retrieval metrics
mmap                      True           Whether or not to load the large tensors ("codes" and "residuals") using memory mapping. Applied to CPU shards only when sharding is enabled
```

### Index Mutability

WARP supports incremental index mutations without requiring a full rebuild. Documents can be added, updated, or deleted after the initial `create()` call, and all changes are immediately reflected in search results.

#### How it works

- **Deletions** are O(1) tombstone operations: passage IDs are written to a `deleted_pids.npy` file and filtered out at search time. No data is physically removed until compaction runs.
- **Additions** encode new embeddings into chunks and incrementally merge them into the existing compacted structures. Only the new chunks are processed — old data is left untouched. Small additions (< 2,000 documents) are coalesced into the last chunk to avoid file fragmentation.
- **Updates** combine both: the old passage IDs are tombstoned and the replacement embeddings are encoded with the same IDs, so the document identity is preserved.
- **Compaction** physically removes tombstoned data by rewriting chunks, pruning empty centroids, rebuilding the compacted layout, and recalibrating internal thresholds.

After each `add()`, the engine also checks for *outlier* embeddings (those far from any existing centroid). If enough outliers are found, it runs K-means on them and expands the centroid codebook, keeping cluster quality stable as the distribution shifts. The outlier detection threshold is recalibrated via a weighted average so it adapts to the evolving data.

> [!WARNING]
> If your index keeps growing with updates and you created the index with less than **3k docs** consider rebuilding when you have at least **3k docs** or when the initial size you used for indexing is less than **50%** of the total size of the index. Retrieval metrics can degrade otherwise due to a bad initialization of the centroids.

#### Delete

```python
xtr_warp.delete(passage_ids=[0, 3, 7])
```

```python
Parameter                   Default     Description
passage_ids                 required    List of passage IDs to mark as deleted
compact_threshold           0.2         Fraction of deleted passages that triggers auto-compaction. Set to None to disable
```

When the ratio of tombstoned passages exceeds `compact_threshold`, compaction runs automatically.

#### Add

```python
new_ids = xtr_warp.add(
    embeddings_source=[torch.randn(200, 128) for _ in range(10)],
)
```

```python
Parameter                   Default     Description
embeddings_source           required    New document embeddings (same formats as create)
reload                      True        Auto-reload index after mutation so searches reflect the new data
min_outliers                50          Minimum outlier count to trigger centroid expansion
max_growth_rate             0.1         Maximum ratio of new centroids relative to current codebook size
max_points_per_centroid     256         Points per centroid for expansion K-means
metadata                    None        List of dicts (one per new document) for metadata filtering
```

Returns the list of newly assigned passage IDs.

#### Update

```python
xtr_warp.update(
    passage_ids=[0, 1],
    embeddings_source=[torch.randn(200, 128), torch.randn(150, 128)],
)
```

```python
Parameter                   Default     Description
passage_ids                 required    IDs of passages to replace
embeddings_source           required    Replacement embeddings (one per passage ID)
reload                      True        Auto-reload index after mutation
```

#### Compact

```python
xtr_warp.compact()
```

```python
Parameter                   Default     Description
reload                      True        Auto-reload index after compaction
```

Compaction rewrites all chunks to exclude deleted passages, prunes centroids that no longer have any assigned embeddings, rebuilds the centroid-sorted layout, and recalibrates the cluster threshold and average residual. Use it after a batch of deletions or updates to reclaim disk space and keep the index tight.

### Metadata Filtering

WARP supports document-level metadata filtering backed by [DuckDB](https://duckdb.org/). Metadata is stored as a DuckDB database alongside the index. Column types are inferred automatically from Python objects: scalars, lists, dicts (structs), and nested combinations are all supported.

`filter()` accepts a SQL WHERE clause fragment with `?` parameter placeholders. The full [DuckDB function set](https://duckdb.org/docs/sql/functions/overview) is available:

```python
idx.filter("category = ?", ["science"])                                      # equality
idx.filter("year >= ?", [2024])                                              # range
idx.filter("author.org = ?", ["ACME"])                                       # struct access
idx.filter("list_contains(tags, ?)", ["ml"])                                 # list operations
idx.filter("category = ? AND year >= ?", ["science", 2024])                  # combined
idx.filter("len(list_filter(sections, x -> x.word_count > 500)) > 0")       # lambdas
```

```python
Parameter                   Default     Description
condition                   required    SQL WHERE clause fragment (e.g. "category = ? AND year > ?")
parameters                  None        Values for ? placeholders in condition
```

Metadata is automatically kept in sync with index mutations: `delete()` and `compact()` remove metadata for deleted passages, and documents with different attribute sets coexist (missing fields are stored as `NULL`).

&nbsp;

## Citation

You can cite **xtr-warp-rs** in your work as follows:

```bibtex
@software{xtrwarprs,
  author = {Montserrat, Pau},
  title = {WARP: The Multi-Vector Search Engine To Rule Them All},
  year = {2025},
  url = {https://github.com/pau-mensa/xtr-warp-rs}
}
```

And for WARP (arXiv entry):

```bibtex
@misc{warp2025,
  title = {WARP: An Efficient Engine for Multi-Vector Retrieval},
  author = {Scheerer, Jan-Luca and Zaharia, Matei and Potts, Christopher and Alonso, Gustavo and Khattab, Omar},
  year = {2025},
  eprint = {2501.17788},
  archivePrefix = {arXiv},
  primaryClass = {cs.IR},
  url = {https://arxiv.org/abs/2501.17788}
}
```

## Contributing

This is an active in development project. Contributions are welcome, particularly in:
- Multi-gpu indexing

## Acknowledgments

I would like to personally acknowledge the creators and maintainers of the [FastPlaid](https://github.com/lightonai/fast-plaid/tree/main) library, from which I took most of the boilerplate code used here. Also give thanks to [Rohan Jha](https://github.com/robro612) for the support with XTR models and some algorithmic improvements.
