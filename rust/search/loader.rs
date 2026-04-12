use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

use crate::utils::types::{
    CentroidId, IndexMetadata, IndexShard, LoadedIndex, SharedIndexState, ShardedIndex,
};

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

    // -----------------------------------------------------------------------
    // Sharded loading
    // -----------------------------------------------------------------------

    /// Load a contiguous row slice from an NPY file into a tensor on
    /// `target_device`. Reads only the needed bytes from disk.
    fn load_tensor_npy_slice(
        &self,
        path: &Path,
        row_start: i64,
        row_count: i64,
        target_device: Device,
    ) -> Result<Tensor> {
        use std::io::{Read as _, Seek, SeekFrom};

        let (data_offset, shape, kind) = parse_npy_header(path)?;
        anyhow::ensure!(!shape.is_empty(), "Empty shape for {:?}", path);
        anyhow::ensure!(
            row_start + row_count <= shape[0],
            "Slice [{}, {}) out of bounds for shape[0]={}",
            row_start,
            row_start + row_count,
            shape[0]
        );

        let row_stride_bytes: i64 =
            shape[1..].iter().product::<i64>() * kind_element_size(kind) as i64;
        let slice_byte_offset = data_offset as i64 + row_start * row_stride_bytes;
        let slice_byte_len = (row_count * row_stride_bytes) as usize;

        let mut file = File::open(path)
            .with_context(|| format!("Failed to open {:?} for slice read", path))?;
        file.seek(SeekFrom::Start(slice_byte_offset as u64))?;
        let mut buf = vec![0u8; slice_byte_len];
        file.read_exact(&mut buf)?;

        // Build the sliced shape: [row_count, shape[1], shape[2], ...]
        let mut sliced_shape = shape.clone();
        sliced_shape[0] = row_count;

        let cpu_tensor = Tensor::from_data_size(&buf, &sliced_shape, kind);
        Ok(cpu_tensor.to_device(target_device))
    }

    /// Memory-map a contiguous row slice from an NPY file (zero-copy, CPU only).
    ///
    /// # Safety
    /// The returned `Mmap` must outlive the tensor.
    fn load_tensor_mmap_slice(
        &self,
        path: &Path,
        row_start: i64,
        row_count: i64,
    ) -> Result<(Tensor, Mmap)> {
        let (data_offset, shape, kind) = parse_npy_header(path)?;
        anyhow::ensure!(!shape.is_empty(), "Empty shape for {:?}", path);
        anyhow::ensure!(
            row_start + row_count <= shape[0],
            "Slice [{}, {}) out of bounds for shape[0]={}",
            row_start,
            row_start + row_count,
            shape[0]
        );

        let row_stride_bytes: i64 =
            shape[1..].iter().product::<i64>() * kind_element_size(kind) as i64;
        let slice_byte_offset = data_offset as i64 + row_start * row_stride_bytes;

        let file = File::open(path)
            .with_context(|| format!("Failed to open {:?} for mmap slice", path))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap {:?}", path))?;

        let data_ptr = mmap[slice_byte_offset as usize..].as_ptr();

        let mut sliced_shape = shape.clone();
        sliced_shape[0] = row_count;
        let strides = compute_c_strides(&sliced_shape);

        let tensor =
            unsafe { Tensor::from_blob(data_ptr, &sliced_shape, &strides, kind, Device::Cpu) };

        Ok((tensor, mmap))
    }

    /// Load a sharded index: shared state on `scoring_device`, per-shard
    /// data on each shard's device.
    pub fn load_sharded(
        &self,
        device_ratios: &[(Device, f64)],
        scoring_device: Device,
    ) -> Result<ShardedIndex> {
        let index_path = self.index_path.as_path();

        // Shared small tensors — load onto scoring device
        let bucket_weights = Tensor::read_npy(index_path.join("bucket_weights.npy"))
            .map_err(|e| anyhow!("bucket_weights: {}", e))?
            .to_dtype(self.dtype, false, false)
            .to_device(scoring_device);

        let centroids = Tensor::read_npy(index_path.join("centroids.npy"))
            .map_err(|e| anyhow!("centroids: {}", e))?
            .to_dtype(self.dtype, false, false)
            .to_device(scoring_device);

        // Sizes on CPU (small, needed for split computation and selector)
        let sizes_compacted =
            Tensor::read_npy(index_path.join("sizes.compacted.npy"))
                .map_err(|e| anyhow!("sizes: {}", e))?
                .to_device(Device::Cpu);

        let kdummy_centroid = self.find_kdummy_centroid(&sizes_compacted)?;
        let metadata = IndexMetadata::load(index_path)?;

        // Compute shard boundaries by cumulative embedding count
        let boundaries = compute_shard_boundaries(&sizes_compacted, device_ratios)?;

        // Resolve file paths for codes and residuals
        let codes_path = index_path.join("codes.compacted.npy");
        let residuals_path = if index_path.join("residuals.repacked.compacted.npy").exists() {
            index_path.join("residuals.repacked.compacted.npy")
        } else {
            index_path.join("residuals.compacted.npy")
        };

        // Cumulative sum to translate centroid ranges → embedding row ranges
        let cumsum = sizes_compacted.cumsum(0, Kind::Int64);

        let mut shards = Vec::with_capacity(boundaries.len());
        for (device, start, end) in &boundaries {
            let start = *start;
            let end = *end;
            let num_centroids_shard = (end - start) as i64;

            // Embedding row range for this shard
            let emb_start = if start == 0 {
                0i64
            } else {
                cumsum.int64_value(&[start as i64 - 1])
            };
            let emb_end = if num_centroids_shard == 0 {
                emb_start
            } else {
                cumsum.int64_value(&[end as i64 - 1])
            };
            let emb_count = emb_end - emb_start;

            let shard_sizes = sizes_compacted.narrow(0, start as i64, num_centroids_shard);
            let use_mmap = self.use_mmap && *device == Device::Cpu;

            let (pids, residuals, mmap_handles) = if emb_count == 0 {
                (
                    Tensor::zeros(&[0], (Kind::Int64, *device)),
                    Tensor::zeros(&[0], (Kind::Uint8, *device)),
                    vec![],
                )
            } else if use_mmap {
                let (p, m1) =
                    self.load_tensor_mmap_slice(&codes_path, emb_start, emb_count)?;
                let (r, m2) =
                    self.load_tensor_mmap_slice(&residuals_path, emb_start, emb_count)?;
                (p, r, vec![m1, m2])
            } else {
                let p =
                    self.load_tensor_npy_slice(&codes_path, emb_start, emb_count, *device)?;
                let r = self.load_tensor_npy_slice(
                    &residuals_path,
                    emb_start,
                    emb_count,
                    *device,
                )?;
                (p, r, vec![])
            };

            // Local offsets for this shard (cumsum of shard sizes, starts at 0)
            let offsets = {
                let o = Tensor::zeros(&[num_centroids_shard + 1], (Kind::Int64, *device));
                if num_centroids_shard > 0 {
                    let cs = shard_sizes.to_device(*device).cumsum(0, Kind::Int64);
                    o.narrow(0, 1, num_centroids_shard).copy_(&cs);
                }
                o
            };

            shards.push(IndexShard {
                centroid_start: start,
                centroid_end: end,
                device: *device,
                sizes_compacted: shard_sizes.to_device(*device),
                pids_compacted: pids,
                residuals_compacted: residuals,
                offsets_compacted: offsets,
                _mmap_handles: Arc::new(mmap_handles),
            });
        }

        Ok(ShardedIndex {
            shared: SharedIndexState {
                centroids,
                bucket_weights,
                sizes_compacted,
                kdummy_centroid,
                metadata,
                scoring_device,
            },
            shards,
        })
    }
}

/// Compute per-shard centroid boundaries from device ratios.
///
/// Splits by **cumulative embedding count** so memory is proportional to ratio.
/// Returns `(device, start_centroid_inclusive, end_centroid_exclusive)` per shard.
fn compute_shard_boundaries(
    sizes: &Tensor,
    device_ratios: &[(Device, f64)],
) -> Result<Vec<(Device, usize, usize)>> {
    let num_centroids = sizes.size()[0] as usize;
    let total_embeddings: i64 = sizes.sum(Kind::Int64).int64_value(&[]);

    if num_centroids == 0 || total_embeddings == 0 || device_ratios.is_empty() {
        return Ok(device_ratios
            .iter()
            .map(|(d, _)| (*d, 0usize, 0usize))
            .collect());
    }

    // Normalize ratios to sum to 1.0
    let ratio_sum: f64 = device_ratios.iter().map(|(_, r)| r).sum();
    let ratios: Vec<(Device, f64)> = device_ratios
        .iter()
        .map(|(d, r)| (*d, r / ratio_sum))
        .collect();

    let cumsum = sizes.cumsum(0, Kind::Int64);
    let cumsum_vec: Vec<i64> = cumsum.try_into()?;

    let mut boundaries = Vec::with_capacity(ratios.len());
    let mut running_ratio = 0.0;
    let mut prev_end = 0usize;

    for (i, (device, ratio)) in ratios.iter().enumerate() {
        let start = prev_end;
        let is_last = i == ratios.len() - 1;

        if is_last {
            // Last shard gets everything remaining
            boundaries.push((*device, start, num_centroids));
        } else {
            running_ratio += ratio;
            let target = (running_ratio * total_embeddings as f64).ceil() as i64;
            // Find first centroid index where cumsum >= target
            let end = cumsum_vec
                .iter()
                .position(|&cs| cs >= target)
                .map(|idx| idx + 1) // position is 0-indexed, end is exclusive
                .unwrap_or(num_centroids);
            let end = end.max(start); // ensure end >= start
            boundaries.push((*device, start, end));
            prev_end = end;
        }
    }

    Ok(boundaries)
}

/// Estimate index memory usage by reading only metadata and NPY headers.
/// Returns a map of component name → size in bytes.
pub fn estimate_index_memory(index_path: &Path) -> Result<std::collections::HashMap<String, u64>> {
    let metadata = IndexMetadata::load(index_path)?;
    let mut result = std::collections::HashMap::new();

    let codes_path = index_path.join("codes.compacted.npy");
    let residuals_path = if index_path.join("residuals.repacked.compacted.npy").exists() {
        index_path.join("residuals.repacked.compacted.npy")
    } else {
        index_path.join("residuals.compacted.npy")
    };

    // Centroids: [num_centroids, dim] at float32 (4 bytes)
    let centroids_bytes = (metadata.num_centroids * metadata.dim * 4) as u64;
    result.insert("centroids".into(), centroids_bytes);

    // Bucket weights: read NPY header for exact size
    let bw_path = index_path.join("bucket_weights.npy");
    if bw_path.exists() {
        let (_, shape, kind) = parse_npy_header(&bw_path)?;
        let elems: i64 = shape.iter().product();
        result.insert(
            "bucket_weights".into(),
            elems as u64 * kind_element_size(kind) as u64,
        );
    }

    // PIDs: [num_embeddings] at i64 (8 bytes)
    if codes_path.exists() {
        let (_, shape, kind) = parse_npy_header(&codes_path)?;
        let elems: i64 = shape.iter().product();
        result.insert("pids".into(), elems as u64 * kind_element_size(kind) as u64);
    }

    // Residuals: [num_embeddings, packed_dim]
    if residuals_path.exists() {
        let (_, shape, kind) = parse_npy_header(&residuals_path)?;
        let elems: i64 = shape.iter().product();
        result.insert(
            "residuals".into(),
            elems as u64 * kind_element_size(kind) as u64,
        );
    }

    // Sizes + offsets: small
    let sizes_bytes = (metadata.num_centroids * 8) as u64; // i64
    let offsets_bytes = ((metadata.num_centroids + 1) * 8) as u64;
    result.insert("sizes_and_offsets".into(), sizes_bytes + offsets_bytes);

    let total: u64 = result.values().sum();
    result.insert("total".into(), total);

    Ok(result)
}

/// Byte size of a single element of the given tch::Kind.
fn kind_element_size(kind: Kind) -> usize {
    match kind {
        Kind::Uint8 | Kind::Int8 | Kind::Bool => 1,
        Kind::Half | Kind::BFloat16 => 2,
        Kind::Float | Kind::Int => 4,
        Kind::Double | Kind::Int64 => 8,
        _ => 4, // safe fallback
    }
}
