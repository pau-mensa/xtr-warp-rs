use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};

use crate::utils::types::{CentroidId, IndexMetadata, LoadedIndex};

/// Index loader responsible for loading WARP index components from disk
pub struct IndexLoader {
    index_path: PathBuf,
    load_with_mmap: bool,
    device: Device,
}

impl IndexLoader {
    /// Create a new index loader
    pub fn new(index_path: impl AsRef<Path>, device: Device, load_with_mmap: bool) -> Result<Self> {
        let path = index_path.as_ref();

        if !path.exists() {
            return Err(anyhow!("Index path {:?} does not exist", path));
        }

        if !path.is_dir() {
            return Err(anyhow!("Index path {:?} is not a directory", path));
        }

        Ok(Self {
            index_path: path.to_path_buf(),
            load_with_mmap,
            device,
        })
    }

    /// Load the complete index from disk
    pub fn load(&self) -> Result<LoadedIndex> {
        let index_path = self.index_path.as_path();

        // Load bucket weights (for scoring)
        println!("#> Loading bucket weights...");
        let bucket_weights = self.load_torch_tensor(index_path.join("bucket_weights.pt"))?;

        // Load centroids
        println!("#> Loading centroids...");
        let centroids = self.load_torch_tensor(index_path.join("centroids.pt"))?;

        // Load compacted sizes per centroid
        println!("#> Loading sizes...");
        let sizes_compacted = self.load_torch_tensor(index_path.join("sizes.compacted.pt"))?;

        // Load compacted codes
        println!("#> Loading codes...");
        let codes_compacted = self.load_torch_tensor(index_path.join("codes.compacted.pt"))?;

        // Load residuals - use repacked version if available for better memory access
        println!("#> Loading residuals...");
        let residuals_path = if index_path.join("residuals.repacked.compacted.pt").exists() {
            index_path.join("residuals.repacked.compacted.pt")
        } else {
            index_path.join("residuals.compacted.pt")
        };
        let residuals_compacted = self.load_torch_tensor(residuals_path)?;

        // Validate dimensions
        let num_centroids = centroids.size()[0];
        assert_eq!(
            sizes_compacted.size()[0],
            num_centroids,
            "Sizes tensor must have same length as number of centroids"
        );

        let num_embeddings = residuals_compacted.size()[0];
        let sizes_sum: i64 = sizes_compacted.sum(Kind::Int64).int64_value(&[]);
        assert_eq!(
            sizes_sum, num_embeddings,
            "Sum of sizes must equal number of embeddings"
        );
        assert_eq!(
            codes_compacted.size()[0],
            num_embeddings,
            "Codes must have same length as residuals"
        );

        // Compute offsets from sizes using cumulative sum
        let offsets_compacted = self.compute_offsets(&sizes_compacted)?;

        // Find kdummy_centroid (the centroid with smallest size)
        let kdummy_centroid = self.find_kdummy_centroid(&sizes_compacted)?;

        // Get metadata
        let dim = centroids
            .size()
            .get(1)
            .copied()
            .ok_or_else(|| anyhow!("Centroids tensor must be 2D"))? as usize;
        let metadata_fallback = IndexMetadata {
            num_passages: 0,
            num_embeddings: num_embeddings as usize,
            num_centroids: num_centroids as usize,
            dim,
            nbits: 2,
            created_at: Utc::now().to_rfc3339(),
            index_version: "xtr-warp-1.0".to_string(),
        };

        let metadata = self.load_metadata(index_path, metadata_fallback)?;

        println!("#> Index loaded successfully!");
        println!("#> - {} centroids", num_centroids);
        println!("#> - {} embeddings", num_embeddings);
        println!("#> - kdummy_centroid: {}", kdummy_centroid);

        Ok(LoadedIndex {
            centroids,
            bucket_weights,
            sizes_compacted,
            codes_compacted,
            residuals_compacted,
            offsets_compacted,
            kdummy_centroid,
            metadata,
        })
    }

    /// Memory-map a large tensor file for efficient loading
    ///
    /// XTR-WARP: ResidualEmbeddings.load_chunks supports mmap (residual_embeddings.py:25-37)
    /// Fast-PLAID: Does not explicitly use mmap for tensors, but could benefit from it
    fn mmap_tensor_file(&self, path: PathBuf) -> Result<Tensor> {
        use memmap2::Mmap;
        use std::fs::File;

        if self.load_with_mmap {
            let file =
                File::open(&path).map_err(|e| anyhow!("Failed to open file for mmap: {}", e))?;
            let _mmap =
                unsafe { Mmap::map(&file) }.map_err(|e| anyhow!("Failed to mmap file: {}", e))?;

            // TODO: Support mmap-backed tensors. For now fall back to regular loading.
        }

        self.load_torch_tensor(path)
    }

    /// Load a PyTorch tensor file
    ///
    /// Uses native PyTorch format for efficiency
    fn load_torch_tensor(&self, path: PathBuf) -> Result<Tensor> {
        let tensor =
            Tensor::load(&path).map_err(|e| anyhow!("Failed to load tensor {:?}: {}", path, e))?;

        Ok(tensor.to_device(self.device))
    }

    /// Compute offsets from sizes for efficient indexing
    ///
    /// XTR-WARP: IndexLoaderWARP._load_codec lines 66-68
    /// Creates offsets for indexing into compacted arrays
    fn compute_offsets(&self, sizes: &Tensor) -> Result<Tensor> {
        use tch::{Kind, Tensor};

        let num_centroids = sizes.size()[0];

        // Create tensor of size [num_centroids + 1]
        let offsets = Tensor::zeros(&[num_centroids + 1], (Kind::Int64, self.device));

        // Compute cumulative sum into offsets[1:]
        // First element remains 0
        let cumsum = sizes.cumsum(0, Kind::Int64);
        offsets.narrow(0, 1, num_centroids).copy_(&cumsum);

        Ok(offsets)
    }

    /// Find the centroid with minimum size (dummy centroid)
    ///
    /// XTR-WARP: IndexLoaderWARP._load_codec line 70
    /// Used to assign masked/invalid query tokens to a minimal centroid
    fn find_kdummy_centroid(&self, sizes: &Tensor) -> Result<CentroidId> {
        // Find the index of the minimum value
        let kdummy_idx = sizes.argmin(0, false).int64_value(&[]) as u32;

        Ok(kdummy_idx as CentroidId)
    }

    fn load_metadata(&self, index_path: &Path, fallback: IndexMetadata) -> Result<IndexMetadata> {
        let metadata_path = index_path.join("metadata.json");
        if !metadata_path.exists() {
            return Ok(fallback);
        }

        #[derive(Debug, Default, serde::Deserialize)]
        struct DiskMetadata {
            num_passages: Option<usize>,
            num_embeddings: Option<usize>,
            num_centroids: Option<usize>,
            dim: Option<usize>,
            nbits: Option<u8>,
            created_at: Option<String>,
            index_version: Option<String>,
        }

        let file = File::open(&metadata_path)
            .with_context(|| format!("Failed to open {:?}", metadata_path))?;
        let reader = BufReader::new(file);
        let disk: DiskMetadata = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse {:?}", metadata_path))?;

        Ok(IndexMetadata {
            num_passages: disk.num_passages.unwrap_or(fallback.num_passages),
            num_embeddings: disk.num_embeddings.unwrap_or(fallback.num_embeddings),
            num_centroids: disk.num_centroids.unwrap_or(fallback.num_centroids),
            dim: disk.dim.unwrap_or(fallback.dim),
            nbits: disk.nbits.unwrap_or(fallback.nbits),
            created_at: disk
                .created_at
                .unwrap_or_else(|| fallback.created_at.clone()),
            index_version: disk
                .index_version
                .unwrap_or_else(|| fallback.index_version.clone()),
        })
    }
}

/// Statistics about the loaded index
#[derive(Debug)]
pub struct IndexStats {
    pub num_centroids: usize,
    pub num_embeddings: usize,
    pub num_passages: usize,
    pub avg_embeddings_per_centroid: f32,
    pub median_embeddings_per_centroid: f32,
    pub memory_usage_bytes: usize,
    pub compression_ratio: f32,
}
