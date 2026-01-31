# Benchmark Suite

## Quick Start

Run all benchmarks across all datasets and configurations:
```bash
uv run benchmark/run_benchmark.py
```

Run a specific experiment for a single dataset:
```bash
uv run benchmark/benchmark.py benchmark/experiments/xtr-warp_0.yaml --dataset scifact
```

## Directory Structure

```
benchmark/
├── benchmark.py              # Main benchmark script for individual experiments
├── run_benchmark.py          # Orchestrator script for running all benchmarks
└── experiments/              # Configuration files and results
    ├── *.yaml                # Experiment configurations
    └── results/              # JSON output files with benchmark results
```

## Configuration Format (YAML)

Experiment configurations are stored as YAML files in the `experiments/` directory. The filename format should be `{framework}_{experiment_id}.yaml` (`xtr-warp_0.yaml` for example).

### Common Parameters

All frameworks support these basic parameters:
- `top_k`: Number of top results to retrieve (e.g., 10, 20, 100)
- `device`: Computing device ("cpu" or "cuda")

### Framework-Specific Parameters

**xtr-warp:**
```yaml
top_k: 100
device: "cuda"
precision: "float32"
num_threads: 1
bound: 128
t_prime: 10000
nprobe: 8
max_candidates: 64000
centroid_score_threshold: 0.5
```

**pylate:**
```yaml
top_k: 100
device: "cuda"
```

**fast-plaid:**
```yaml
top_k: 100
device: "cuda"
n_ivf_probe: 8
n_full_scores: 4096
```

> [!TIP]
> Any parameter not specified will use the default value.

## Running Benchmarks

### Full Benchmark Suite

The complete benchmarking suite evaluates all YAML configurations in the `experiments/` directory across the following BEIR datasets:
- scifact
- nfcorpus
- arguana
- quora
- scidocs
- fiqa
- trec-covid
- webis-touche2020

```bash
uv run benchmark/run_benchmark.py
```

This will:
1. Iterate through all datasets
2. Run each YAML configuration for each dataset
3. Display progress with ETA calculations
4. Save results to `experiments/results/`

### Individual Experiments

To run a specific configuration on a single dataset:

```bash
uv run benchmark/benchmark.py [config_path] --dataset [dataset_name] [options]
```

**Arguments:**
- `config_path`: Path to YAML configuration file
- `--dataset`: BEIR dataset name (required)
- `--n-docs`: Limit number of documents (optional, for testing)
- `--n-queries`: Limit number of queries (optional, for testing)

**Examples:**
```bash
# Run xtr-warp configuration on scifact dataset
uv run benchmark/benchmark.py benchmark/experiments/xtr-warp_8.yaml --dataset scifact

# Test with limited data
uv run benchmark/benchmark.py benchmark/experiments/xtr-warp_8.yaml --dataset nfcorpus --n-docs 1000 --n-queries 50
```

## Results Format

Results are saved as JSON files in `experiments/results/` with the naming convention:
```
{dataset_name}_{framework}_{experiment_id}.json
```

### Result JSON Structure

```json
{
    "dataset": "scifact",
    "framework": "xtr-warp",
    "config": {
        "top_k": 20,
        "device": "cuda",
        "precision": "float32",
        "num_threads": 64
    },
    "indexing": 12.61,           // Indexing time in seconds
    "search": 0.308,             // Total search time in seconds
    "qps": 1057.52,             // Queries per second
    "size": 5183,               // Number of documents
    "queries": 300,             // Number of queries processed
    "scores": {
        "map": 0.6856,          // Mean Average Precision
        "ndcg@10": 0.7253,      // NDCG at 10
        "ndcg@100": 0.7356,     // NDCG at 100
        "recall@10": 0.8445,    // Recall at 10
        "recall@100": 0.8845    // Recall at 100
    },
    "memory": {
        "indexing": {
            "cpu_increase_mb": 1110.25,   // CPU memory increase during indexing
            "gpu_increase_mb": 3370.27    // GPU memory increase during indexing
        },
        "search": {
            "cpu_increase_mb": 1236.38,   // CPU memory increase during search
            "gpu_increase_mb": 1520.11    // GPU memory increase during search
        }
    }
}
```

## Embeddings Cache

The benchmark system caches document and query embeddings to speed up repeated experiments:
- Embeddings are stored in the `embeddings/` directory
- Files are named: `{documents|queries}_embeddings_{dataset_name}.pt`
- Delete these files to force re-computation of embeddings

## Memory Monitoring

The benchmark automatically tracks memory usage:
- CPU memory via `psutil`
- The implementation of the cpu memory monitor (RAM) is not very accurate, do not rely too much on it.
- GPU memory via `torch.cuda` (when using CUDA)
- Memory is measured before and after indexing/search phases
- Peak memory usage is recorded for GPU operations

## Adding New Configurations

1. Create a new YAML file in `experiments/` following the naming convention
2. Include all required parameters for the framework
3. Run the benchmark for testing:
   ```bash
   uv run benchmark/benchmark.py experiments/your_config.yaml --dataset scifact
   ```

## Generating Comparison Reports

After running benchmarks, generate comparison tables:
```bash
uv run benchmark/generate_comparison.py
```

This creates formatted comparison reports from the results in `experiments/results/`.

## Dependencies

The benchmark suite requires the following Python packages (automatically managed by `uv`):
- pylate>=1.3.3
- beir>=2.2.0
- fast_plaid
- ranx
- psutil
- pyyaml
- torch
- numpy

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` in the benchmark script
- Use `--n-docs` and `--n-queries` to test with smaller datasets
- Switch to CPU if GPU memory is insufficient
