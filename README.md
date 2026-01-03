<div align="center">
  <h1>Warp</h1>
</div>
<div align="center">
    The Multi-Vector Search Engine To Rule Them All
</div>

&nbsp;

## ⭐️ Overview

xtr-warp-rs is a high-performance implementation of the **WARP** engine for multi-vector retrieval, as described in the [WARP paper (SIGIR 2025)](https://arxiv.org/abs/2501.17788). Originally built with [XTR models (NeurIPS 2023)](https://arxiv.org/abs/2304.01982) in mind, as it turns out, it significantly outperforms all other multi-vector search engines while keeping retrieval metrics competitive.

Compared to the current SOTA (FastPlaid), xtr-warp-rs focuses on doing less work per query while staying close in quality: it prunes the centroid/posting-list space per token, uses an error-aware merge that keeps ranking stable with fewer examined candidates, and keeps the hot path (selection → decompression → merge) highly optimized and parallel friendly.

The engine achieves a speedup of **3-10x** on CUDA and of **4-70x** on CPU (depending on the dataset and number of threads used) vs FastPlaid. Check the [benchmark section](#benchmarks) for a detailed comparison.

> [!IMPORTANT]  
> While the package is usable today, it is worth noting that the index creation algorithm and memory management in general use very naive approaches, resulting in 10X slower indexing times than FastPlaid and around 20% more memory usage. This is under active development and contributions are welcome!

&nbsp;

## Installation

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

Get started with creating an index and performing a search in just a few lines of Python.

```python
import torch

from xtr_warp import XTRWarp

xtr_warp = XTRWarp(index="index")

embedding_dim = 128

# Index 100 documents, each with 300 tokens, each token is a 128-dim vector.
xtr_warp.create(
    documents_embeddings=[torch.randn(300, embedding_dim) for _ in range(100)],
    device="cpu",
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

The output will be a list of lists, where each inner list contains tuples of (document_index, similarity_score) for the top_k results for each query.

&nbsp;

## Benchmarks

- `qps` stands for **Queries Per Second** (higher is better)
- `indexing` stands for the time it took the engine to build the index (lower is better)

### CUDA Performance

| Dataset (Size) | Metric | fast-plaid | xtr-warp-rs |
|----------------|--------|------------|-------------|
| arguana (8,674) | qps | 110.26 | 1008.69 (+814.8%) |
|  | indexing | 1.67s | 15.27s |
|  | ndcg@10 | 0.47 | 0.49 |
|  | recall@10 | 0.73 | 0.75 |
| fiqa (57,638) | qps | 87.16 | 943.08 (+982.0%) |
|  | indexing | 4.90s | 70.28s |
|  | ndcg@10 | 0.41 | 0.36 |
|  | recall@10 | 0.48 | 0.42 |
| nfcorpus (3,633) | qps | 123.87 | 1155.00 (+832.4%) |
|  | indexing | 0.90s | 8.71s |
|  | ndcg@10 | 0.37 | 0.36 |
|  | recall@10 | 0.18 | 0.17 |
| quora (522,931) | qps | 217.47 | 927.92 (+326.7%) |
|  | indexing | 10.51s | 79.30s |
|  | ndcg@10 | 0.88 | 0.86 |
|  | recall@10 | 0.95 | 0.94 |
| scidocs (25,657) | qps | 97.49 | 861.50 (+783.7%) |
|  | indexing | 3.93s | 42.95s |
|  | ndcg@10 | 0.19 | 0.18 |
|  | recall@10 | 0.19 | 0.19 |
| scifact (5,183) | qps | 112.30 | 1133.98 (+909.8%) |
|  | indexing | 1.42s | 12.61s |
|  | ndcg@10 | 0.74 | 0.73 |
|  | recall@10 | 0.86 | 0.85 |
| trec-covid (171,332) | qps | 43.16 | 282.75 (+555.1%) |
|  | indexing | 17.44s | 252.40s |
|  | ndcg@10 | 0.84 | 0.80 |
|  | recall@10 | 0.02 | 0.02 |
| webis-touche2020 (300,000) | qps | 67.95 | 637.24 (+837.8%) |
|  | indexing | 23.44s | 403.23s |
|  | ndcg@10 | 0.25 | 0.24 |
|  | recall@10 | 0.18 | 0.18 |

### CPU Performance

| Dataset (Size) | QPS fast-plaid | QPS xtr-warp (Single) | QPS xtr-warp-rs (Multi) |
|----------------|----------------|-----------------------|-------------------------|
| arguana (8,674) | 4.79 | 82.93 (+1631.3%) | 247.93 (+5076.0%) |
| fiqa (57,638) | 4.78 | 77.86 (+1528.9%) | 202.74 (+4141.4%) |
| nfcorpus (3,633) | 6.69 | 109.74 (+1540.4%) | 482.13 (+7106.7%) |
| quora (522,931) | 8.60 | 55.26 (+542.6%) | 190.26 (+2112.3%) |
| scidocs (25,657) | 4.52 | 47.00 (+939.8%) | 156.41 (+3360.4%) |
| scifact (5,183) | 6.14 | 169.09 (+2653.9%) | 300.06 (+4787.0%) |
| trec-covid (171,332) | 1.82 | 9.45 (+419.2%) | 24.44 (+1242.9%) |
| webis-touche2020 (300,000) | 3.80 | 34.10 (+797.4%) | 103.60 (+2626.3%) |

&nbsp;

> [!NOTE]  
> These benchmarks were run on an NVIDIA 5090 with an AMD Ryzen 9950 CPU. Due to VRAM constraints, the `webis-touche2020` dataset had to be limited to 300k documents instead of the 380k original ones.

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
batch_size                  8192        Batch size for centroid scoring on CUDA
num_threads                 1           Upper bound for CPU parallelism during search (not used on CUDA)
```

### Indexing

```python
Parameter                  Default     Description
device                     required    Device to use for index creation (e.g., "cpu", "cuda", "mps")
kmeans_niters              4           K-means iterations for clustering
max_points_per_centroid    256         Maximum points per centroid during K-means
nbits                      4           Product quantization bits for compression
n_samples_kmeans           None        Samples for K-means clustering
seed                       42          Random seed for reproducibility
use_triton_kmeans          None        Whether to use Triton-based K-means
```

> [!IMPORTANT]  
> Highly recommended to build the index using `cuda` devices. For a large corpus using `cpu` or even `mps` can take forever.

### Loading

To help with memory management the API also exposes the `load` and `free` methods, which, as the name implies, load and free the index from memory respectively.

```python
Parameter                 Default        Description
device                    required       Device where to load the index (e.g., "cpu", "cuda", "mps")
dtype                     torch.float32  Dtype to use for the centroids and bucket weights. Lowers the memory footprint but can cause alterations in retrieval metrics
```

> [!TIP]
> While loading on `cuda` is the fastest way to serve inference, `cpu` is still extremely competitive, especially when multiple threads are used.

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
- Improving index building performance and memory management
- Adding a reranking step at the end of the search pipeline can stabilize the retrieval metrics, especially for datasets like `fiqa`
- `mps` and multiple `cuda` devices support is untested

## Acknowledgments

I would like to personally acknowledge the creators and maintainers of the [FastPlaid](https://github.com/lightonai/fast-plaid/tree/main) library, from which I took most of the boilerplate code used here.
