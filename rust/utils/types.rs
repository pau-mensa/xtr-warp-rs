// Key types needed:
// - Configuration structures for index and search
// - ID types for passages, centroids, embeddings
// - Query and result representations
// - Index components structure
use memmap2::Mmap;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tch::{Device, Tensor};

/// Represents a passage/document ID in the index
pub type PassageId = i64;

/// Represents a centroid ID in the clustering
pub type CentroidId = i64;


/// Score type for ranking
pub type Score = f32;

/// Configuration for the WARP index
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Path to the index directory
    pub index_path: PathBuf,

    /// Device to use (e.g. "cpu", "cuda:0")
    pub device: Device,

    /// Number of bits for residual compression (2 or 4)
    pub nbits: u8,

    /// Embedding dimension
    pub embedding_dim: u32,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_path: PathBuf::from("./index"),
            device: Device::Cpu,
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

    /// Device to use for the search
    #[pyo3(get, set)]
    pub device: String,

    /// Number of centroids to probe during search
    #[pyo3(get, set)]
    pub nprobe: u32,

    /// Optional adaptive threshold (t')
    #[pyo3(get, set)]
    pub t_prime: Option<usize>,

    /// Maximum number of centroids to inspect per token
    #[pyo3(get, set)]
    pub bound: usize,

    /// The batch size for centroid matmul
    #[pyo3(get, set)]
    pub batch_size: i64,

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

    /// Size of the per-device CUDA stream pool used by the Pass A3
    /// merger and multi-CUDA Phase-2 fan-out. `None` uses the built-in
    /// default. Set to 0 or 1 to disable (fall back to default stream).
    #[pyo3(get, set)]
    pub merger_streams: Option<usize>,

    /// When true (default), the per-shard sharded scorer fans
    /// decompresses across multiple streams when more than one query
    /// in the batch has work on a shard. Set to false to force the
    /// single-stream path.
    #[pyo3(get, set)]
    pub decompress_parallel: bool,
}

#[pymethods]
impl SearchConfig {
    /// Creates a new SearchConfig instance from Python
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        k,
        device,
        nprobe=None,
        t_prime=None,
        bound=None,
        batch_size=None,
        num_threads=None,
        centroid_score_threshold=None,
        max_codes_per_centroid=None,
        max_candidates=None,
        merger_streams=None,
        decompress_parallel=true,
    ))]
    fn new(
        k: usize,
        device: String,
        nprobe: Option<u32>,
        t_prime: Option<usize>,
        bound: Option<usize>,
        batch_size: Option<i64>,
        num_threads: Option<usize>,
        centroid_score_threshold: Option<f32>,
        max_codes_per_centroid: Option<u32>,
        max_candidates: Option<usize>,
        merger_streams: Option<usize>,
        decompress_parallel: bool,
    ) -> Self {
        Self {
            k,
            device: device,
            nprobe: nprobe.unwrap_or(4),
            t_prime,
            bound: bound.unwrap_or(128),
            batch_size: batch_size.unwrap_or(8192i64),
            num_threads,
            centroid_score_threshold,
            max_codes_per_centroid,
            max_candidates,
            merger_streams,
            decompress_parallel,
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            k: 100,
            device: "cpu".to_string(),
            nprobe: 4,
            t_prime: None,
            bound: 128,
            batch_size: 8192i64,
            num_threads: Some(1usize),
            centroid_score_threshold: None,
            max_codes_per_centroid: None,
            max_candidates: None,
            merger_streams: None,
            decompress_parallel: true,
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

/// Represents the plan outlined to create an index
pub struct IndexPlan {
    pub n_docs: usize,
    pub num_chunks: usize,
    pub avg_doc_len: f64,
    pub est_total_embs: i64,
    pub nbits: u8,
}

/// Canonical on-disk + in-memory representation of `metadata.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub num_chunks: usize,
    pub nbits: u8,
    pub num_partitions: i64,
    pub num_embeddings: i64,
    pub avg_doclen: f64,
    pub num_passages: usize,
    /// Watermark for the next passage ID to assign.
    pub next_passage_id: i64,
    pub num_centroids: usize,
    pub dim: usize,
    pub created_at: String,
}

impl IndexMetadata {
    /// Load metadata from `metadata.json` in the given index directory.
    pub fn load(index_path: &std::path::Path) -> anyhow::Result<Self> {
        let path = index_path.join("metadata.json");
        let file = std::fs::File::open(&path)
            .map_err(|e| anyhow::anyhow!("Failed to open {}: {}", path.display(), e))?;
        let meta: IndexMetadata = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", path.display(), e))?;
        Ok(meta)
    }

    /// Persist metadata to `metadata.json` atomically (write tmp, rename).
    pub fn save(&self, index_path: &std::path::Path) -> anyhow::Result<()> {
        let path = index_path.join("metadata.json");
        let tmp_path = index_path.join("metadata.json.tmp");
        let file = std::fs::File::create(&tmp_path)
            .map_err(|e| anyhow::anyhow!("Failed to create {}: {}", tmp_path.display(), e))?;
        serde_json::to_writer_pretty(std::io::BufWriter::new(file), self)?;
        std::fs::rename(&tmp_path, &path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to rename {} -> {}: {}",
                tmp_path.display(),
                path.display(),
                e
            )
        })?;
        Ok(())
    }
}

/// A single shard of the index: owns a contiguous centroid range and the
/// corresponding slices of pids_compacted and residuals_compacted.
///
/// # Safety
/// Marked `Sync` because shard tensors are never mutated after load.
pub struct IndexShard {
    /// Inclusive start of the global centroid range owned by this shard.
    pub centroid_start: usize,
    /// Exclusive end of the global centroid range owned by this shard.
    pub centroid_end: usize,
    /// Device this shard's tensors live on.
    pub device: Device,
    /// Sizes per centroid for this shard's range: [shard_num_centroids].
    pub sizes_compacted: Tensor,
    /// Passage IDs for embeddings in this shard: [shard_num_embeddings].
    pub pids_compacted: Tensor,
    /// Compressed residuals for embeddings in this shard.
    pub residuals_compacted: Tensor,
    /// Local offsets (starts at 0): [shard_num_centroids + 1].
    pub offsets_compacted: Tensor,
    /// Bucket weights replicated on this shard's device. Avoids an
    /// implicit cross-device transfer per decompress call when the shard
    /// is not on `scoring_device` (matters for multi-CUDA configs).
    pub bucket_weights: Tensor,
    /// Mmap handles that must outlive the tensors they back.
    pub _mmap_handles: Arc<Vec<Mmap>>,
}

// SAFETY: IndexShard tensors are never mutated after load. All search
// access is read-only (narrow, index_select).
unsafe impl Sync for IndexShard {}

/// Shared state across all shards: small tensors replicated on the scoring
/// accelerator device.
pub struct SharedIndexState {
    /// Centroids tensor on scoring_device: [num_centroids, dim].
    pub centroids: Tensor,
    /// Bucket weights on scoring_device.
    pub bucket_weights: Tensor,
    /// Full sizes_compacted on CPU (small, needed by CentroidSelector).
    pub sizes_compacted: Tensor,
    /// Dummy centroid index (smallest centroid, for masked tokens).
    pub kdummy_centroid: CentroidId,
    /// Index metadata.
    pub metadata: IndexMetadata,
    /// The device used for centroid scoring (first accelerator).
    pub scoring_device: Device,
}

/// A fully sharded index: shared state + per-device shards.
///
/// # Safety
/// Marked `Sync` because shard tensors are never mutated after load — same
/// invariant as `IndexShard`.
pub struct ShardedIndex {
    pub shared: SharedIndexState,
    pub shards: Vec<IndexShard>,
}

// SAFETY: see struct docs — read-only after load.
unsafe impl Sync for ShardedIndex {}

/// Query representation for search
pub struct Query {
    pub embeddings: Tensor, // Always [batch, num_tokens, dim]
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

pub struct DecompressedCentroidsOutput {
    pub capacities: Tensor, // Total capacity of each centroid (ends - begins)
    pub sizes: Tensor,      // Actual sizes after deduplication
    pub passage_ids: Tensor,
    pub scores: Tensor,
    pub offsets: Tensor,
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

/// Stats from index compaction
pub struct CompactStats {
    pub total_embeddings: i64,
    pub num_active_passages: usize,
}

/// Result from adding documents to an index
pub struct AddResult {
    pub new_passage_ids: Vec<i64>,
    /// Per-embedding residual norms (for centroid expansion outlier detection).
    pub residual_norms: Vec<f32>,
    /// Embedding dimension.
    pub embedding_dim: usize,
}

/// Parses a string identifier into a `tch::Device`.
///
/// Supports simple device strings like "cpu", "cuda", and indexed CUDA devices
/// such as "cuda:0".
pub fn parse_device(device: &str) -> anyhow::Result<Device> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "mps" => Ok(Device::Mps),
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

/// Bitset for O(1) passage ID membership checks during subset filtering.
pub struct PassageBitset {
    bits: Vec<u64>,
}

impl PassageBitset {
    pub fn new(ids: &[i64]) -> Self {
        let max_id = ids.iter().copied().max().unwrap_or(-1);
        if max_id < 0 {
            return Self { bits: Vec::new() };
        }
        let num_words = (max_id as usize / 64) + 1;
        let mut bits = vec![0u64; num_words];
        for &id in ids {
            if id >= 0 {
                bits[id as usize / 64] |= 1u64 << (id as usize % 64);
            }
        }
        Self { bits }
    }

    #[inline]
    pub fn contains(&self, pid: i64) -> bool {
        if pid < 0 {
            return false;
        }
        let word = pid as usize / 64;
        word < self.bits.len() && (self.bits[word] & (1u64 << (pid as usize % 64))) != 0
    }
}

impl IndexShard {
    /// Translate global centroid IDs into this shard's local ID space.
    /// For a shard starting at 0 this is a no-op; otherwise subtracts
    /// `centroid_start`.
    pub fn localize_centroid_ids(&self, global_ids: &Tensor) -> Tensor {
        if self.centroid_start == 0 {
            global_ids.shallow_clone()
        } else {
            global_ids - (self.centroid_start as i64)
        }
    }
}


