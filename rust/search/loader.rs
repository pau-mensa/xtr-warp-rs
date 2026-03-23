use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

use crate::utils::types::{CentroidId, IndexMetadata, LoadedIndex};

/// Parse a NPY file header, returning (data_offset, shape, tch_kind).
fn parse_npy_header(path: &Path) -> Result<(u64, Vec<i64>, Kind)> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read NPY file {:?}", path))?;

    // Magic: \x93NUMPY
    anyhow::ensure!(
        data.len() >= 10 && &data[..6] == b"\x93NUMPY",
        "Not a valid NPY file: {:?}",
        path
    );

    let major = data[6];
    let header_len: usize;
    let header_start: usize;

    if major == 1 {
        header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
        header_start = 10;
    } else if major == 2 {
        anyhow::ensure!(data.len() >= 12, "NPY v2 header too short");
        header_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        header_start = 12;
    } else {
        anyhow::bail!("Unsupported NPY version {}", major);
    }

    let data_offset = (header_start + header_len) as u64;
    let header_str = std::str::from_utf8(&data[header_start..header_start + header_len])
        .context("NPY header is not valid UTF-8")?;

    // Parse 'descr'
    let descr = extract_npy_field(header_str, "descr")
        .ok_or_else(|| anyhow!("Missing 'descr' in NPY header"))?;
    let kind = match descr.as_str() {
        "<i4" | "=i4" => Kind::Int,
        "<i8" | "=i8" => Kind::Int64,
        "|u1" | "<u1" | ">u1" => Kind::Uint8,
        "<f4" | "=f4" => Kind::Float,
        "<f2" | "=f2" => Kind::Half,
        "<f8" | "=f8" => Kind::Double,
        other => anyhow::bail!("Unsupported NPY dtype: '{}'", other),
    };

    // Parse 'fortran_order'
    let fortran = extract_npy_field(header_str, "fortran_order")
        .unwrap_or_else(|| "False".to_string());
    anyhow::ensure!(
        fortran == "False",
        "Fortran-order NPY files are not supported"
    );

    // Parse 'shape'
    let shape_str = extract_npy_field(header_str, "shape")
        .ok_or_else(|| anyhow!("Missing 'shape' in NPY header"))?;
    let shape: Vec<i64> = shape_str
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() { None } else { s.parse().ok() }
        })
        .collect();

    anyhow::ensure!(!shape.is_empty(), "Empty shape in NPY header");

    Ok((data_offset, shape, kind))
}

/// Extract a value for a given key from the NPY header dict string.
/// The header looks like: {'descr': '<f4', 'fortran_order': False, 'shape': (100, 128), }
fn extract_npy_field(header: &str, key: &str) -> Option<String> {
    let pattern = format!("'{}':", key);
    let idx = header.find(&pattern)?;
    let rest = &header[idx + pattern.len()..];
    let rest = rest.trim_start();

    if rest.starts_with('\'') {
        // String value
        let end = rest[1..].find('\'')?;
        Some(rest[1..1 + end].to_string())
    } else if rest.starts_with('(') {
        // Tuple value
        let end = rest.find(')')?;
        Some(rest[..=end].to_string())
    } else {
        // Bare value (e.g. True/False)
        let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
        Some(rest[..end].trim().to_string())
    }
}

/// Compute C-contiguous strides for a given shape.
fn compute_c_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Index loader responsible for loading WARP index components from disk
pub struct IndexLoader {
    index_path: PathBuf,
    device: Device,
    dtype: Kind,
    use_mmap: bool,
}

impl IndexLoader {
    /// Create a new index loader
    pub fn new(
        index_path: impl AsRef<Path>,
        device: Device,
        dtype: Kind,
        use_mmap: bool,
    ) -> Result<Self> {
        let path = index_path.as_ref();

        if !path.exists() {
            return Err(anyhow!("Index path {:?} does not exist", path));
        }

        if !path.is_dir() {
            return Err(anyhow!("Index path {:?} is not a directory", path));
        }

        Ok(Self {
            index_path: path.to_path_buf(),
            device,
            dtype,
            use_mmap,
        })
    }

    /// Load the complete index from disk
    pub fn load(&self) -> Result<LoadedIndex> {
        let index_path = self.index_path.as_path();

        // Load bucket weights (for scoring)
        let bucket_weights = self
            .load_torch_tensor(index_path.join("bucket_weights.npy"))
            .unwrap()
            .to_dtype(self.dtype, false, false);

        // Load centroids
        let centroids = self
            .load_torch_tensor(index_path.join("centroids.npy"))
            .unwrap()
            .to_dtype(self.dtype, false, false);

        // Load compacted sizes per centroid
        let sizes_compacted = self.load_torch_tensor(index_path.join("sizes.compacted.npy"))?;

        // Load compacted codes and residuals (optionally memory-mapped)
        let codes_path = index_path.join("codes.compacted.npy");
        let residuals_path = if index_path.join("residuals.repacked.compacted.npy").exists() {
            index_path.join("residuals.repacked.compacted.npy")
        } else {
            index_path.join("residuals.compacted.npy")
        };

        let (pids_compacted, residuals_compacted, mmap_handles) = if self.use_mmap {
            anyhow::ensure!(
                self.device == Device::Cpu,
                "mmap is only supported on CPU"
            );
            let (codes, mmap1) = self.load_tensor_mmap(&codes_path)?;
            let (residuals, mmap2) = self.load_tensor_mmap(&residuals_path)?;
            (codes, residuals, vec![mmap1, mmap2])
        } else {
            let codes = self.load_torch_tensor(codes_path)?;
            let residuals = self.load_torch_tensor(residuals_path)?;
            (codes, residuals, vec![])
        };

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
            pids_compacted.size()[0],
            num_embeddings,
            "Codes must have same length as residuals"
        );

        // Compute offsets from sizes using cumulative sum
        let offsets_compacted = self.compute_offsets(&sizes_compacted)?;

        // Find kdummy_centroid (the centroid with smallest size)
        let kdummy_centroid = self.find_kdummy_centroid(&sizes_compacted)?;

        // Get metadata
        let metadata = IndexMetadata::load(index_path)?;

        Ok(LoadedIndex {
            centroids,
            bucket_weights,
            sizes_compacted,
            pids_compacted,
            residuals_compacted,
            offsets_compacted,
            kdummy_centroid,
            metadata,
            _mmap_handles: Arc::new(mmap_handles),
        })
    }

    /// Load a PyTorch tensor file
    ///
    /// Uses native PyTorch format for efficiency
    fn load_torch_tensor(&self, path: PathBuf) -> Result<Tensor> {
        let tensor = Tensor::read_npy(&path)
            .map_err(|e| anyhow!("Failed to load tensor {:?}: {}", path, e))?;

        Ok(tensor.to_device(self.device))
    }

    /// Load a tensor via memory-mapping the NPY file (zero-copy).
    ///
    /// # Safety
    /// The returned `Mmap` handle must outlive the tensor — it backs the tensor's data.
    fn load_tensor_mmap(&self, path: &Path) -> Result<(Tensor, Mmap)> {
        let (data_offset, shape, kind) = parse_npy_header(path)?;
        let file = File::open(path)
            .with_context(|| format!("Failed to open {:?} for mmap", path))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap {:?}", path))?;
        let data_ptr = mmap[data_offset as usize..].as_ptr();
        let strides = compute_c_strides(&shape);
        let tensor = unsafe {
            Tensor::from_blob(data_ptr, &shape, &strides, kind, Device::Cpu)
        };
        Ok((tensor, mmap))
    }

    /// Compute offsets from sizes for efficient indexing
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
    /// Used to assign masked/invalid query tokens to a minimal centroid
    fn find_kdummy_centroid(&self, sizes: &Tensor) -> Result<CentroidId> {
        // Find the index of the minimum value
        let kdummy_idx = sizes.argmin(0, false).int64_value(&[]) as u32;

        Ok(kdummy_idx as CentroidId)
    }

}
