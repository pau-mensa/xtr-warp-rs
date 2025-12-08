// Key types needed:
// - Configuration structures for index and search
// - ID types for passages, centroids, embeddings
// - Query and result representations
// - Index components structure
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tch::{Device, Kind, Tensor};

/// Represents a passage/document ID in the index
pub type PassageId = i64;

/// Represents a centroid ID in the clustering
pub type CentroidId = i64;

/// Represents an embedding ID in the index
pub type EmbeddingId = i64;

/// Query ID type
pub type QueryId = i64;

/// Score type for ranking
pub type Score = f32;

/// Configuration for the WARP index
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Path to the index directory
    pub index_path: PathBuf,

    /// Device to use (e.g. "cpu", "cuda:0")
    pub device: Device,

    /// Whether to load index with memory mapping
    // pub load_with_mmap: bool,

    /// Number of bits for residual compression (2 or 4)
    pub nbits: i64,

    /// Embedding dimension
    pub embedding_dim: u32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_path: PathBuf::from("./index"),
            device: Device::Cpu,
            // load_with_mmap: false,
            nbits: 4,
            embedding_dim: 128,
        }
    }
}

/// Search configuration parameters
#[pyclass]
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of top results to return
    #[pyo3(get, set)]
    pub k: usize,

    /// Device to perform the search on
    #[pyo3(get, set)]
    pub device: String,

    /// Overall device mode: cpu | cuda | mps | hybrid
    #[pyo3(get, set)]
    pub device_mode: String,

    /// Device to run centroid selection/matmul on
    #[pyo3(get, set)]
    pub selector_device: String,

    /// Device to run decompression/merge on
    #[pyo3(get, set)]
    pub decompress_device: String,

    /// Dtype to use for the search
    #[pyo3(get, set)]
    pub dtype: String,

    /// Number of centroids to probe during search
    #[pyo3(get, set)]
    pub nprobe: u32,

    /// Optional adaptive threshold (t')
    #[pyo3(get, set)]
    pub t_prime: Option<usize>,

    /// Maximum number of centroids to inspect per token
    #[pyo3(get, set)]
    pub bound: usize,

    /// Optional number of threads for parallel search
    #[pyo3(get, set)]
    pub num_threads: Option<usize>,

    /// Threshold for centroid scores
    #[pyo3(get, set)]
    pub centroid_score_threshold: Option<f32>,

    /// Maximum codes per centroid
    #[pyo3(get, set)]
    pub max_codes_per_centroid: Option<u32>,

    /// The number of candidates to consider before the sorting
    #[pyo3(get, set)]
    pub max_candidates: Option<usize>,

    /// Enable inner parallelism in decompression/merge (currently off by default)
    #[pyo3(get, set)]
    pub enable_inner_parallelism: Option<bool>,

    /// Enable per-query batch parallelism on CPU/hybrid
    #[pyo3(get, set)]
    pub enable_batch_parallelism: Option<bool>,
}

#[pymethods]
impl SearchConfig {
    /// Creates a new SearchConfig instance from Python
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        k,
        device,
        dtype=None,
        nprobe=None,
        t_prime=None,
        bound=None,
        num_threads=None,
        centroid_score_threshold=None,
        max_codes_per_centroid=None,
        max_candidates=None,
        device_mode=None,
        selector_device=None,
        decompress_device=None,
        enable_inner_parallelism=None,
        enable_batch_parallelism=None,
    ))]
    fn new(
        k: usize,
        device: String,
        dtype: Option<String>,
        nprobe: Option<u32>,
        t_prime: Option<usize>,
        bound: Option<usize>,
        num_threads: Option<usize>,
        centroid_score_threshold: Option<f32>,
        max_codes_per_centroid: Option<u32>,
        max_candidates: Option<usize>,
        device_mode: Option<String>,
        selector_device: Option<String>,
        decompress_device: Option<String>,
        enable_inner_parallelism: Option<bool>,
        enable_batch_parallelism: Option<bool>,
    ) -> Self {
        let mode = device_mode.unwrap_or_else(|| "cpu".to_string());
        let selector = selector_device.unwrap_or_else(|| device.clone());
        // If hybrid, default decompression to CPU; otherwise match the selector
        let decompress = decompress_device.unwrap_or_else(|| {
            if mode.to_lowercase() == "hybrid" {
                "cpu".to_string()
            } else {
                device.clone()
            }
        });
        Self {
            k,
            device,
            device_mode: mode,
            selector_device: selector,
            decompress_device: decompress,
            dtype: dtype.unwrap_or("float32".to_string()),
            nprobe: nprobe.unwrap_or(32),
            t_prime,
            bound: bound.unwrap_or(128),
            num_threads,
            centroid_score_threshold,
            max_codes_per_centroid,
            max_candidates,
            enable_inner_parallelism,
            enable_batch_parallelism,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            k: 100,
            device: "cpu".to_string(),
            device_mode: "cpu".to_string(),
            selector_device: "cpu".to_string(),
            decompress_device: "cpu".to_string(),
            dtype: "float32".to_string(),
            nprobe: 32,
            t_prime: None,
            bound: 128,
            num_threads: Some(1usize),
            centroid_score_threshold: None,
            max_codes_per_centroid: None,
            max_candidates: None,
            enable_inner_parallelism: Some(false),
            enable_batch_parallelism: Some(true),
        }
    }
}

/// Represents a ranked search result
#[pyclass]
#[derive(Serialize, Debug, Clone)]
pub struct SearchResult {
    #[pyo3(get)]
    pub passage_ids: Vec<PassageId>,
    #[pyo3(get)]
    pub scores: Vec<Score>,
    #[pyo3(get)]
    pub query_id: usize,
}

/// Index metadata stored alongside the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub num_passages: usize,
    pub num_embeddings: usize,
    pub num_centroids: usize,
    pub dim: usize,
    pub nbits: u8,
    pub created_at: String,
    pub index_version: String,
}

/// Components of a loaded index
pub struct LoadedIndex {
    /// Centroids tensor [num_centroids, dim]
    pub centroids: Tensor,

    /// Bucket weights for scoring
    pub bucket_weights: Tensor,

    /// Compacted sizes per centroid
    pub sizes_compacted: Tensor,

    /// Compacted codes for embeddings
    pub codes_compacted: Tensor,

    /// Compacted residuals (compressed)
    pub residuals_compacted: Tensor,

    /// Offsets for each centroid in the compacted arrays
    pub offsets_compacted: Tensor,

    /// Index of the dummy centroid (smallest)
    pub kdummy_centroid: CentroidId,

    /// Metadata about the index
    pub metadata: IndexMetadata,
}

impl Clone for LoadedIndex {
    fn clone(&self) -> Self {
        Self {
            centroids: self.centroids.shallow_clone(),
            bucket_weights: self.bucket_weights.shallow_clone(),
            sizes_compacted: self.sizes_compacted.shallow_clone(),
            codes_compacted: self.codes_compacted.shallow_clone(),
            residuals_compacted: self.residuals_compacted.shallow_clone(),
            offsets_compacted: self.offsets_compacted.shallow_clone(),
            kdummy_centroid: self.kdummy_centroid,
            metadata: self.metadata.clone(),
        }
    }
}

/// Read-only wrapper that marks the loaded index as safe to share across threads.
/// The tensors are never mutated after load, so we can treat them as Sync.
pub struct ReadOnlyIndex(pub LoadedIndex);

impl std::ops::Deref for ReadOnlyIndex {
    type Target = LoadedIndex;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Sync for ReadOnlyIndex {}

impl Clone for ReadOnlyIndex {
    fn clone(&self) -> Self {
        ReadOnlyIndex(self.0.clone())
    }
}

/// Query representation for search
pub struct Query {
    pub embeddings: Tensor, // Always [batch, num_tokens, dim]
}

/// Batch of queries for efficient processing
pub struct QueryBatch {
    pub queries: Vec<Query>,
    pub max_tokens: usize,
}

/// Read-only tensor wrapper to opt into Sync when we guarantee no mutation.
pub struct ReadOnlyTensor(pub Tensor);

impl std::ops::Deref for ReadOnlyTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Sync for ReadOnlyTensor {}

impl Clone for ReadOnlyTensor {
    fn clone(&self) -> Self {
        ReadOnlyTensor(self.0.shallow_clone())
    }
}

/// Selected centroids for a query token
#[derive(Debug)]
pub struct SelectedCentroids {
    pub centroid_ids: Tensor,
    pub scores: Tensor,
    pub mse_estimate: Tensor,
}

/// Decompressed centroid data
pub struct DecompressedCentroid {
    pub centroid_id: CentroidId,
    pub passage_ids: Vec<PassageId>,
    pub scores: Vec<Score>,
}

pub struct DecompressedCentroidsOutput {
    pub capacities: Tensor, // Total capacity of each centroid (ends - begins)
    pub sizes: Tensor,      // Actual sizes after deduplication
    pub passage_ids: Tensor,
    pub scores: Tensor,
    pub offsets: Tensor,
}

/// Candidate for final ranking
#[derive(Debug, Clone)]
pub struct RankingCandidate {
    pub passage_id: PassageId,
    pub score: Score,
    pub centroid_id: CentroidId,
}

/// T-prime policy for adaptive early termination
#[derive(Debug, Clone)]
pub enum TPrimePolicy {
    Fixed(usize),
    Max,
}

impl TPrimePolicy {
    pub fn value(&self, k: usize) -> usize {
        match self {
            TPrimePolicy::Fixed(value) => *value,
            TPrimePolicy::Max => {
                if k > 100 {
                    100_000
                } else {
                    50_000
                }
            },
        }
    }
}

/// Parses a string identifier into a `tch::Device`.
///
/// Supports simple device strings like "cpu", "cuda", and indexed CUDA devices
/// such as "cuda:0".
pub fn parse_device(device: &str) -> anyhow::Result<Device> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)), // Default to the first CUDA device.
        s if s.starts_with("cuda:") => {
            let parts: Vec<&str> = s.split(':').collect();
            if parts.len() == 2 {
                parts[1]
                    .parse::<usize>()
                    .map(Device::Cuda)
                    .map_err(|_| anyhow::anyhow!("Invalid CUDA device index: '{}'", parts[1]))
            } else {
                Err(anyhow::anyhow!(
                    "Invalid CUDA device format. Expected 'cuda:N'."
                ))
            }
        },
        _ => Err(anyhow::anyhow!("Unsupported device string: '{}'", device)),
    }
}

/// Parses a string identifier into a `tch::Kind`.
///
/// Supports simple strings like "float32", "float16"
pub fn parse_dtype(dtype: &str) -> anyhow::Result<Kind> {
    match dtype.to_lowercase().as_str() {
        "float32" => Ok(Kind::Float),
        "float16" => Ok(Kind::Half),
        "float64" => Ok(Kind::Double),
        "bfloat16" => Ok(Kind::BFloat16),
        _ => Err(anyhow::anyhow!("Unsupported dtype string: '{}', should be 'float32', 'float16', 'float64', or 'bfloat16'", dtype)),
    }
}
