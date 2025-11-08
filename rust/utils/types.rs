// XTR-WARP Implementation Reference:
// - Config types: xtr-warp/warp/parameters.py (Run parameters and configuration)
// - ColBERT config: xtr-warp/warp/infra/config/config.py (ColBERTConfig class)
// - Index metadata: xtr-warp/warp/indexing/collection_indexer.py (metadata handling)
// - Search config: xtr-warp/warp/engine/config.py (WARPRunConfig class)
//
// Fast-PLAID Rust Reference:
// - Config structs: fast-plaid/rust/search/search.rs (SearchParameters struct)
// - Metadata: fast-plaid/rust/search/load.rs (Metadata struct)
// - Query results: fast-plaid/rust/search/search.rs (QueryResult struct)
//
// Key types needed:
// - Configuration structures for index and search
// - ID types for passages, centroids, embeddings
// - Query and result representations
// - Index components structure
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tch::{Device, Tensor};

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
///
/// Based on:
/// - XTR-WARP: ColBERTConfig in xtr-warp/warp/infra/config/config.py
/// - XTR-WARP: Various parameters from xtr-warp/warp/parameters.py
/// - Fast-PLAID: Configuration loaded in metadata.json
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Path to the index directory
    pub index_path: PathBuf,

    /// Device to use (e.g. "cpu", "cuda:0")
    pub device: Device,

    /// Whether to load index with memory mapping
    pub load_with_mmap: bool,

    /// Number of bits for residual compression (2 or 4)
    pub nbits: i64,

    /// Number of probes for search
    pub nprobe: u32,

    /// Embedding dimension
    pub embedding_dim: u32,

    /// Optional t_prime parameter for search
    pub t_prime: u32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_path: PathBuf::from("./index"),
            device: Device::Cpu,
            load_with_mmap: false,
            nbits: 2,
            nprobe: 32,
            t_prime: 10000,
            embedding_dim: 128,
        }
    }
}
/*#[pyclass]
#[derive(Serialize, Debug, Clone)]
pub struct SearchResult {
    #[pyo3(get)]
    pub passage_ids: Vec<PassageId>,
    #[pyo3(get)]
    pub scores: Vec<Score>,
    #[pyo3(get)]
    pub query_id: usize,
}*/
/// Search configuration parameters
///
/// Based on:
/// - XTR-WARP: WARPRunConfig in xtr-warp/warp/engine/config.py
/// - Fast-PLAID: SearchParameters struct in fast-plaid/rust/search/search.rs
#[pyclass]
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of top results to return
    #[pyo3(get, set)]
    pub k: usize,

    /// Number of centroids to probe during search
    #[pyo3(get, set)]
    pub nprobe: u32,

    /// Optional adaptive threshold (t')
    #[pyo3(get, set)]
    pub t_prime: Option<usize>,

    /// Maximum number of centroids to inspect per token
    #[pyo3(get, set)]
    pub bound: usize,

    /// Whether to enable parallel search
    #[pyo3(get, set)]
    pub parallel: bool,

    /// Optional number of threads for parallel search
    #[pyo3(get, set)]
    pub num_threads: Option<usize>,

    /// Threshold for centroid scores
    #[pyo3(get, set)]
    pub centroid_score_threshold: Option<f32>,

    /// Maximum codes per centroid
    #[pyo3(get, set)]
    pub max_codes_per_centroid: Option<u32>,
}

#[pymethods]
impl SearchConfig {
    /// Creates a new SearchConfig instance from Python
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (k, nprobe=None, t_prime=None, bound=None, parallel=None, num_threads=None, centroid_score_threshold=None, max_codes_per_centroid=None))]
    fn new(
        k: usize,
        nprobe: Option<u32>,
        t_prime: Option<usize>,
        bound: Option<usize>,
        parallel: Option<bool>,
        num_threads: Option<usize>,
        centroid_score_threshold: Option<f32>,
        max_codes_per_centroid: Option<u32>,
    ) -> Self {
        Self {
            k,
            nprobe: nprobe.unwrap_or(32),
            t_prime,
            bound: bound.unwrap_or(128),
            parallel: parallel.unwrap_or(true),
            num_threads,
            centroid_score_threshold,
            max_codes_per_centroid,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            k: 100,
            nprobe: 32,
            t_prime: None,
            bound: 128,
            parallel: true,
            num_threads: None,
            centroid_score_threshold: None,
            max_codes_per_centroid: None,
        }
    }
}

/// Represents a ranked search result
///
/// Based on:
/// - XTR-WARP: WARPRankingItem in xtr-warp/warp/data/ranking.py
/// - Fast-PLAID: Return type of search function (Vec<(i64, f32)>)
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
///
/// Based on:
/// - XTR-WARP: metadata.json structure in indices
/// - Fast-PLAID: Metadata struct in fast-plaid/rust/search/load.rs
/// - Contains essential index properties for validation
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
///
/// Based on:
/// - XTR-WARP: Components loaded in IndexLoaderWARP/IndexScorerWARP
/// - Fast-PLAID: LoadedIndex struct in fast-plaid/rust/search/load.rs
/// - Contains all necessary tensors for search operations
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

/// Query representation for search
///
/// Based on:
/// - XTR-WARP: Query encoding in Searcher.encode
/// - XTR-WARP: WARPQueries in xtr-warp/warp/data/queries.py
/// - Fast-PLAID: query_embeddings parameter in search function
pub struct Query {
    pub embeddings: Tensor, // Always [batch, num_tokens, dim]
}

/// Batch of queries for efficient processing
///
/// Based on:
/// - XTR-WARP: Batched search in WARPSearcher._search_all_batched
/// - Optimizes multi-query processing
pub struct QueryBatch {
    pub queries: Vec<Query>,
    pub max_tokens: usize,
}

/// Selected centroids for a query token
///
/// Based on:
/// - XTR-WARP: Return type of _warp_select_centroids (cells, scores, mse)
/// - Contains centroids selected by WARP algorithm with scores
#[derive(Debug)]
pub struct SelectedCentroids {
    pub centroid_ids: Tensor,
    pub scores: Tensor,
    pub mse_estimate: Tensor,
}

/// Decompressed centroid data
///
/// Based on:
/// - XTR-WARP: Output of _decompress_centroids
/// - Fast-PLAID: decompressed_embs in search function
/// - Contains decompressed embeddings and passage information
pub struct DecompressedCentroid {
    pub centroid_id: CentroidId,
    pub passage_ids: Vec<PassageId>,
    pub scores: Vec<Score>,
}

pub struct DecompressedCentroidsOutput {
    //pub centroid_ids: Tensor,
    pub capacities: Tensor, // Total capacity of each centroid (ends - begins)
    pub sizes: Tensor,      // Actual sizes after deduplication
    pub passage_ids: Tensor,
    pub scores: Tensor,
    pub offsets: Tensor,
}

/// Candidate for final ranking
///
/// Based on:
/// - XTR-WARP: Candidates generated during search
/// - Used for final scoring and ranking
#[derive(Debug, Clone)]
pub struct RankingCandidate {
    pub passage_id: PassageId,
    pub score: Score,
    pub centroid_id: CentroidId,
}

/// T-prime policy for adaptive early termination
///
/// Based on:
/// - XTR-WARP: TPrimePolicy in xtr-warp/warp/engine/constants.py
/// - XTR-WARP: t_prime computation in IndexScorerWARP.__init__ (lines 103-109)
/// - Formula: sqrt(8 * num_embeddings) / 1000 * 1000
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
