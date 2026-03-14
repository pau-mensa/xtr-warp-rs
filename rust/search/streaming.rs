use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use tch::{Device, Kind, Tensor};

use crate::utils::types::{CentroidId, IndexMetadata};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpyDtype {
    I64,
    U8,
}

impl NpyDtype {
    fn item_size(self) -> usize {
        match self {
            NpyDtype::I64 => 8,
            NpyDtype::U8 => 1,
        }
    }
}

struct NpyHeader {
    dtype: NpyDtype,
    shape: Vec<usize>,
    fortran_order: bool,
}

pub(crate) struct NpyReader {
    path: PathBuf,
    file: Mutex<File>,
    data_offset: u64,
    shape: Vec<usize>,
    row_len: usize,
    row_size: usize,
    dtype: NpyDtype,
}

impl NpyReader {
    pub(crate) fn open(path: PathBuf) -> Result<Self> {
        let mut file = File::open(&path)
            .map_err(|e| anyhow!("Failed to open {:?}: {}", path, e))?;
        let (header, data_offset) = Self::read_header(&mut file)?;

        if header.fortran_order {
            return Err(anyhow!("Fortran-order .npy arrays are not supported"));
        }

        let row_len = if header.shape.len() > 1 {
            header.shape[1..].iter().product()
        } else {
            1
        };
        let row_size = row_len * header.dtype.item_size();

        Ok(Self {
            path,
            file: Mutex::new(file),
            data_offset,
            shape: header.shape,
            row_len,
            row_size,
            dtype: header.dtype,
        })
    }

    fn read_header(file: &mut File) -> Result<(NpyHeader, u64)> {
        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)?;
        if &magic != b"\x93NUMPY" {
            return Err(anyhow!("Invalid NPY magic header"));
        }

        let mut version = [0u8; 2];
        file.read_exact(&mut version)?;
        let major = version[0];
        let _minor = version[1];

        let header_len = if major == 1 {
            let mut len_bytes = [0u8; 2];
            file.read_exact(&mut len_bytes)?;
            u16::from_le_bytes(len_bytes) as usize
        } else if major == 2 || major == 3 {
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            u32::from_le_bytes(len_bytes) as usize
        } else {
            return Err(anyhow!("Unsupported NPY version {}", major));
        };

        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;
        let header_str = std::str::from_utf8(&header_bytes)
            .map_err(|e| anyhow!("Invalid NPY header utf8: {}", e))?
            .trim()
            .trim_end_matches('\n');

        let header = Self::parse_header(header_str)?;

        let data_offset = match major {
            1 => 6 + 2 + 2 + header_len,
            2 | 3 => 6 + 2 + 4 + header_len,
            _ => unreachable!(),
        } as u64;

        Ok((header, data_offset))
    }

    fn parse_header(header: &str) -> Result<NpyHeader> {
        let descr = Self::extract_quoted_value(header, "'descr'")
            .ok_or_else(|| anyhow!("Missing 'descr' in NPY header"))?;
        let fortran_raw = Self::extract_unquoted_value(header, "'fortran_order'")
            .ok_or_else(|| anyhow!("Missing 'fortran_order' in NPY header"))?;
        let shape_raw = Self::extract_paren_value(header, "'shape'")
            .ok_or_else(|| anyhow!("Missing 'shape' in NPY header"))?;

        let dtype = Self::parse_descr(&descr)?;
        let fortran_order = match fortran_raw.trim() {
            "True" => true,
            "False" => false,
            other => return Err(anyhow!("Invalid fortran_order value: {}", other)),
        };

        let mut shape = Vec::new();
        for part in shape_raw.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            let val: usize = trimmed
                .parse()
                .map_err(|_| anyhow!("Invalid shape entry '{}'", trimmed))?;
            shape.push(val);
        }

        if shape.is_empty() {
            return Err(anyhow!("Empty shape in NPY header"));
        }

        Ok(NpyHeader {
            dtype,
            shape,
            fortran_order,
        })
    }

    fn extract_quoted_value(header: &str, key: &str) -> Option<String> {
        let idx = header.find(key)?;
        let after = &header[idx + key.len()..];
        let colon = after.find(':')?;
        let after_colon = after[colon + 1..].trim();
        let start = after_colon.find('"').or_else(|| after_colon.find('\''))?;
        let quote = after_colon.chars().nth(start)?;
        let after_quote = &after_colon[start + 1..];
        let end = after_quote.find(quote)?;
        Some(after_quote[..end].to_string())
    }

    fn extract_unquoted_value(header: &str, key: &str) -> Option<String> {
        let idx = header.find(key)?;
        let after = &header[idx + key.len()..];
        let colon = after.find(':')?;
        let after_colon = after[colon + 1..].trim();
        let end = after_colon.find(',').unwrap_or(after_colon.len());
        Some(after_colon[..end].trim().to_string())
    }

    fn extract_paren_value(header: &str, key: &str) -> Option<String> {
        let idx = header.find(key)?;
        let after = &header[idx + key.len()..];
        let colon = after.find(':')?;
        let after_colon = after[colon + 1..].trim();
        let start = after_colon.find('(')?;
        let after_paren = &after_colon[start + 1..];
        let end = after_paren.find(')')?;
        Some(after_paren[..end].to_string())
    }

    fn parse_descr(descr: &str) -> Result<NpyDtype> {
        let mut chars = descr.chars();
        let endian = chars.next().ok_or_else(|| anyhow!("Empty descr"))?;
        let kind = chars
            .next()
            .ok_or_else(|| anyhow!("Invalid descr '{}'", descr))?;
        let size_str: String = chars.collect();
        let item_size: usize = size_str
            .parse()
            .map_err(|_| anyhow!("Invalid descr size '{}'", descr))?;

        if endian == '>' {
            return Err(anyhow!("Big-endian NPY arrays are not supported"));
        }

        match (kind, item_size) {
            ('i', 8) => Ok(NpyDtype::I64),
            ('u', 1) => Ok(NpyDtype::U8),
            _ => Err(anyhow!("Unsupported NPY dtype '{}'", descr)),
        }
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.shape[0]
    }

    fn read_rows_raw(&self, start: usize, len: usize) -> Result<Vec<u8>> {
        if start + len > self.num_rows() {
            return Err(anyhow!(
                "Read out of bounds for {:?}: start={}, len={}, rows={}",
                self.path,
                start,
                len,
                self.num_rows()
            ));
        }

        let offset = self.data_offset + (start as u64) * (self.row_size as u64);
        let byte_len = len * self.row_size;
        let mut buf = vec![0u8; byte_len];

        let mut file = self
            .file
            .lock()
            .map_err(|_| anyhow!("Failed to lock NPY reader for {:?}", self.path))?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut buf)?;

        Ok(buf)
    }

    fn read_i64_rows(&self, start: usize, len: usize) -> Result<Vec<i64>> {
        if self.dtype != NpyDtype::I64 {
            return Err(anyhow!("Expected i64 dtype for {:?}", self.path));
        }
        let raw = self.read_rows_raw(start, len)?;
        let mut out = Vec::with_capacity(len * self.row_len);
        for chunk in raw.chunks_exact(8) {
            let val = i64::from_le_bytes(chunk.try_into().unwrap());
            out.push(val);
        }
        Ok(out)
    }

    fn read_u8_rows(&self, start: usize, len: usize) -> Result<Vec<u8>> {
        if self.dtype != NpyDtype::U8 {
            return Err(anyhow!("Expected u8 dtype for {:?}", self.path));
        }
        self.read_rows_raw(start, len)
    }

    pub(crate) fn row_len(&self) -> usize {
        self.row_len
    }
}

struct CentroidChunk {
    pids: Vec<i64>,
    residuals: Vec<u8>,
}

impl CentroidChunk {
    fn bytes_len(&self) -> usize {
        self.pids.len() * std::mem::size_of::<i64>() + self.residuals.len()
    }
}

struct CacheEntry {
    chunk: std::sync::Arc<CentroidChunk>,
    bytes: usize,
}

struct CentroidCache {
    max_bytes: usize,
    current_bytes: usize,
    map: HashMap<CentroidId, CacheEntry>,
    order: VecDeque<CentroidId>,
}

impl CentroidCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            current_bytes: 0,
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, centroid_id: CentroidId) -> Option<std::sync::Arc<CentroidChunk>> {
        let chunk = self.map.get(&centroid_id).map(|entry| entry.chunk.clone());
        if chunk.is_some() {
            self.touch(centroid_id);
        }
        chunk
    }

    fn insert(&mut self, centroid_id: CentroidId, chunk: std::sync::Arc<CentroidChunk>) {
        if self.max_bytes == 0 {
            return;
        }

        let bytes = chunk.bytes_len();
        if bytes > self.max_bytes {
            return;
        }

        while self.current_bytes + bytes > self.max_bytes {
            if let Some(oldest) = self.order.pop_front() {
                if let Some(entry) = self.map.remove(&oldest) {
                    self.current_bytes = self.current_bytes.saturating_sub(entry.bytes);
                }
            } else {
                break;
            }
        }

        self.order.push_back(centroid_id);
        self.current_bytes += bytes;
        self.map.insert(
            centroid_id,
            CacheEntry {
                chunk,
                bytes,
            },
        );
    }

    fn touch(&mut self, centroid_id: CentroidId) {
        if let Some(pos) = self.order.iter().position(|id| *id == centroid_id) {
            self.order.remove(pos);
            self.order.push_back(centroid_id);
        }
    }
}

pub struct StreamingIndex {
    centroids: Tensor,
    bucket_weights: Tensor,
    sizes_compacted: Tensor,
    offsets_compacted: Tensor,
    kdummy_centroid: CentroidId,
    metadata: IndexMetadata,
    codes_reader: NpyReader,
    residuals_reader: NpyReader,
    cache: Option<Mutex<CentroidCache>>,
}

// Tensors are never mutated after load, so treat the index as Sync.
unsafe impl Sync for StreamingIndex {}

impl StreamingIndex {
    pub fn new(
        centroids: Tensor,
        bucket_weights: Tensor,
        sizes_compacted: Tensor,
        offsets_compacted: Tensor,
        kdummy_centroid: CentroidId,
        metadata: IndexMetadata,
        codes_reader: NpyReader,
        residuals_reader: NpyReader,
        cache_bytes: usize,
    ) -> Result<Self> {
        let residual_row_len = residuals_reader.row_len();
        let dim = metadata.dim;
        let nbits = metadata.nbits as usize;
        let expected_residual_len = (dim * nbits) / 8;
        if residual_row_len != expected_residual_len {
            return Err(anyhow!(
                "Residual row length mismatch: expected {}, got {}",
                expected_residual_len,
                residual_row_len
            ));
        }

        Ok(Self {
            centroids,
            bucket_weights,
            sizes_compacted,
            offsets_compacted,
            kdummy_centroid,
            metadata,
            codes_reader,
            residuals_reader,
            cache: if cache_bytes > 0 {
                Some(Mutex::new(CentroidCache::new(cache_bytes)))
            } else {
                None
            },
        })
    }

    pub fn centroids(&self) -> &Tensor {
        &self.centroids
    }

    pub fn bucket_weights(&self) -> &Tensor {
        &self.bucket_weights
    }

    pub fn sizes_compacted(&self) -> &Tensor {
        &self.sizes_compacted
    }

    pub fn offsets_compacted(&self) -> &Tensor {
        &self.offsets_compacted
    }

    pub fn kdummy_centroid(&self) -> CentroidId {
        self.kdummy_centroid
    }

    pub fn metadata(&self) -> &IndexMetadata {
        &self.metadata
    }

    pub fn read_codes_range(&self, start: i64, len: i64) -> Result<Vec<i64>> {
        if start < 0 || len < 0 {
            return Err(anyhow!("Invalid range start={} len={}", start, len));
        }
        self.codes_reader
            .read_i64_rows(start as usize, len as usize)
    }

    pub fn read_residuals_range(&self, start: i64, len: i64) -> Result<Vec<u8>> {
        if start < 0 || len < 0 {
            return Err(anyhow!("Invalid range start={} len={}", start, len));
        }
        self.residuals_reader
            .read_u8_rows(start as usize, len as usize)
    }

    pub fn read_centroid_chunk(
        &self,
        centroid_id: CentroidId,
        start: i64,
        len: i64,
    ) -> Result<(Vec<i64>, Vec<u8>)> {
        if let Some(cache) = &self.cache {
            if let Ok(mut guard) = cache.lock() {
                if let Some(chunk) = guard.get(centroid_id) {
                    return Ok((chunk.pids.clone(), chunk.residuals.clone()));
                }
            }
        }

        let pids = self.read_codes_range(start, len)?;
        let residuals = self.read_residuals_range(start, len)?;
        let chunk = std::sync::Arc::new(CentroidChunk {
            pids: pids.clone(),
            residuals: residuals.clone(),
        });

        if let Some(cache) = &self.cache {
            if let Ok(mut guard) = cache.lock() {
                guard.insert(centroid_id, chunk);
            }
        }

        Ok((pids, residuals))
    }
}

// Helper for loader to create readers
pub(crate) fn open_codes_reader(path: &Path) -> Result<NpyReader> {
    NpyReader::open(path.to_path_buf())
}

pub(crate) fn open_residuals_reader(path: &Path) -> Result<NpyReader> {
    NpyReader::open(path.to_path_buf())
}

// Streaming should be CPU-only for now; keep a helper to validate.
pub(crate) fn ensure_cpu_device(device: Device) -> Result<()> {
    if device != Device::Cpu {
        return Err(anyhow!(
            "Streaming search currently supports CPU only; got device {:?}",
            device
        ));
    }
    Ok(())
}

// Ensure dtype is supported for CPU streaming
pub(crate) fn ensure_cpu_dtype(dtype: Kind) -> Result<()> {
    match dtype {
        Kind::Float | Kind::Double | Kind::Half | Kind::BFloat16 => Ok(()),
        _ => Err(anyhow!("Unsupported dtype {:?} for streaming", dtype)),
    }
}
