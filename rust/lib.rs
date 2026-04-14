// XTR-WARP Rust Implementation with Python Bindings

use anyhow::{anyhow, Result};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
use std::collections::HashSet;
use std::ffi::CString;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;
use tch::{Device, Kind};

#[cfg(windows)]
use winapi::um::errhandlingapi::GetLastError;
#[cfg(windows)]
use winapi::um::libloaderapi::LoadLibraryA;

// Module declarations
pub mod index;
pub mod search;
pub mod utils;

// Re-exports for convenience
use crate::index::create::create_index;
use crate::index::source::{DiskEmbeddingSource, EmbeddingSource, InMemoryEmbeddingSource};
use search::{IndexLoader, ShardedScorer};
use utils::types::{IndexConfig, Query, ReadOnlyShardedIndex, SearchConfig, SearchResult};

/// Dynamically loads the native Torch shared library (e.g., `libtorch.so` or `torch.dll`).
///
/// This is a workaround to ensure Torch's symbols are available in memory,
/// which can prevent linking errors when `tch-rs` is used within a
/// Python extension module.
fn call_torch(torch_path: String) -> Result<(), anyhow::Error> {
    let torch_path_cstr = CString::new(torch_path.clone())
        .map_err(|e| anyhow!("Failed to create CString for libtorch path: {}", e))?;

    #[cfg(unix)]
    {
        let handle = unsafe { libc::dlopen(torch_path_cstr.as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return Err(anyhow!(
                "Failed to load Torch library '{}' via dlopen. Check the path and permissions.",
                torch_path
            ));
        }
    }

    #[cfg(windows)]
    {
        let handle = unsafe { LoadLibraryA(torch_path_cstr.as_ptr()) };
        if handle.is_null() {
            let error_code = unsafe { GetLastError() };
            return Err(anyhow!(
                "Failed to load Torch library '{}' via LoadLibraryA. Windows error code: {}",
                torch_path,
                error_code
            ));
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        return Err(anyhow!(
            "Dynamic library loading is not supported on this operating system."
        ));
    }

    Ok(())
}

fn get_device(device: &str) -> Result<Device, PyErr> {
    utils::types::parse_device(device).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn get_dtype(dtype: &str) -> Result<Kind, PyErr> {
    utils::types::parse_dtype(dtype).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Filter tombstoned PIDs and truncate to k.
/// Results arrive sorted by score descending from the merger, so we
/// stop as soon as we collect k non-deleted entries.
fn filter_tombstones(results: &mut [SearchResult], deleted_pids: &HashSet<i64>, k: usize) {
    for result in results {
        if !deleted_pids.is_empty() {
            let mut filtered_pids = Vec::with_capacity(k);
            let mut filtered_scores = Vec::with_capacity(k);
            for (pid, score) in result.passage_ids.iter().zip(result.scores.iter()) {
                if !deleted_pids.contains(pid) {
                    filtered_pids.push(*pid);
                    filtered_scores.push(*score);
                    if filtered_pids.len() == k {
                        break;
                    }
                }
            }
            result.passage_ids = filtered_pids;
            result.scores = filtered_scores;
        } else {
            result.passage_ids.truncate(k);
            result.scores.truncate(k);
        }
    }
}

/// Pre-loads the native Torch library from a specified path.
///
/// Call this function once at the start of a Python script if you encounter
/// linking issues with the Torch library, which can occur in complex deployment
/// environments.
///
/// Args:
///     torch_path (str): The file path to the Torch shared library,
///         e.g., `/path/to/libtorch_cuda.so`.
#[pyfunction]
fn initialize_torch(_py: Python<'_>, torch_path: String) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize Torch: {}", e)))
}

enum EmbeddingsInput {
    Direct(Vec<PyTensor>),
    FromPath(String),
}

impl<'source> FromPyObject<'source> for EmbeddingsInput {
    fn extract_bound(ob: &Bound<'source, PyAny>) -> PyResult<Self> {
        if let Ok(path) = ob.extract::<String>() {
            return Ok(EmbeddingsInput::FromPath(path));
        }
        if let Ok(embeddings) = ob.extract::<Vec<PyTensor>>() {
            return Ok(EmbeddingsInput::Direct(embeddings));
        }
        Err(PyValueError::new_err(
            "embeddings must be a list of torch.Tensor or a string path",
        ))
    }
}

impl EmbeddingsInput {
    fn into_source(self) -> Result<Box<dyn EmbeddingSource>> {
        match self {
            EmbeddingsInput::Direct(embeddings) => {
                let embeddings: Vec<_> = embeddings.into_iter().map(|tensor| tensor.0).collect();
                Ok(Box::new(InMemoryEmbeddingSource::new(embeddings)))
            },
            EmbeddingsInput::FromPath(path) => {
                Ok(Box::new(DiskEmbeddingSource::new(Path::new(&path))?))
            },
        }
    }
}

/// Creates and saves a new xtr-warp index.
///
/// Args:
///     index (str): The directory path where the new index will be saved.
///     torch_path (str): Path to the Torch shared library (e.g., `libtorch.so`).
///     device (str): The compute device to use for index creation (e.g., "cpu", "cuda:0").
///     nbits (int): The number of bits to use for residual quantization.
///     embeddings (list[torch.Tensor] | str): List of 2D tensors, one per document,
///         or a path to batched embedding files.
///     centroids (torch.Tensor): A 2D tensor of shape `[num_centroids, embedding_dim]`.
///     embedding_dim (int): The dimensionality of the embeddings.
///     seed (int, optional): Optional seed for the random number generator.
#[pyfunction]
#[pyo3(signature = (
    index,
    torch_path,
    device,
    nbits,
    centroids,
    embeddings,
    embedding_dim=None,
    seed=None,
    show_progress=true,
))]
fn create(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    nbits: i64,
    centroids: PyTensor,
    embeddings: EmbeddingsInput,
    embedding_dim: Option<u32>,
    seed: Option<u64>,
    show_progress: bool,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

    let device = get_device(&device)?;
    let nbits: u8 = nbits
        .try_into()
        .map_err(|_| PyValueError::new_err("nbits must be in 0..=255"))?;
    let centroids = centroids.to_device(device);

    let mut source = embeddings
        .into_source()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read embeddings: {}", e)))?;

    create_index(
        &IndexConfig {
            index_path: Path::new(&index).to_path_buf(),
            device,
            nbits,
            embedding_dim: embedding_dim.unwrap_or(128),
        },
        source.as_mut(),
        centroids,
        seed,
        show_progress,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index: {}", e)))
}

/// Delete passages by ID. O(1) tombstone operation — no index rewrite.
/// Search automatically filters deleted passages.
#[pyfunction]
fn delete(_py: Python<'_>, index: String, passage_ids: Vec<i64>) -> PyResult<()> {
    crate::index::delete::delete_from_index(&passage_ids, Path::new(&index))
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to delete: {}", e)))
}

/// Add new passages to an existing index. Encodes + incrementally merges.
/// Returns a dict with `new_passage_ids`, `residual_norms`, and `embedding_dim`.
#[pyfunction]
#[pyo3(signature = (index, torch_path, device, embeddings, show_progress=true))]
fn add(
    py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    embeddings: EmbeddingsInput,
    show_progress: bool,
) -> PyResult<PyObject> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;
    let device = get_device(&device)?;
    let mut source = embeddings
        .into_source()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read embeddings: {}", e)))?;
    let result = crate::index::update::add_to_index(
        source.as_mut(),
        Path::new(&index),
        device,
        show_progress,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to add to index: {}", e)))?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("new_passage_ids", result.new_passage_ids)?;
    dict.set_item("residual_norms", result.residual_norms)?;
    dict.set_item("embedding_dim", result.embedding_dim)?;
    Ok(dict.into_any().unbind())
}

/// Append new centroids to the codebook (called after Python-side K-means).
#[pyfunction]
fn append_centroids_py(_py: Python<'_>, index: String, new_centroids: PyTensor) -> PyResult<()> {
    crate::index::update::append_centroids(Path::new(&index), &new_centroids)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to append centroids: {}", e)))
}

/// Update passages in-place: new embeddings, same IDs.
/// Reads embedding_dim from the existing index metadata.
#[pyfunction]
#[pyo3(signature = (index, torch_path, device, passage_ids, embeddings, show_progress=true))]
fn update(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    passage_ids: Vec<i64>,
    embeddings: EmbeddingsInput,
    show_progress: bool,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;
    let device = get_device(&device)?;
    let mut source = embeddings
        .into_source()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read embeddings: {}", e)))?;
    crate::index::update::update_in_index(
        &passage_ids,
        source.as_mut(),
        Path::new(&index),
        device,
        show_progress,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to update index: {}", e)))
}

/// Rebuild index excluding deleted passages (compact without adding new data).
#[pyfunction]
#[pyo3(signature = (index, torch_path, device, show_progress=true))]
fn compact(
    _py: Python<'_>,
    index: String,
    torch_path: String,
    device: String,
    show_progress: bool,
) -> PyResult<()> {
    call_torch(torch_path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;
    let device = get_device(&device)?;
    crate::index::update::compact_standalone(Path::new(&index), device, show_progress)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to compact index: {}", e)))
}

#[pyclass(unsendable)]
struct ShardedSearcher {
    sharded_index: Option<Arc<ReadOnlyShardedIndex>>,
    index_path: String,
    device_ratios: Vec<(Device, f64)>,
    scoring_device: Device,
    dtype: Kind,
    use_mmap: bool,
    deleted_pids: HashSet<i64>,
}

#[pymethods]
impl ShardedSearcher {
    #[new]
    #[pyo3(signature = (index_path, device_ratios, dtype, use_mmap=true))]
    fn new(
        index_path: String,
        device_ratios: Vec<(String, f64)>,
        dtype: String,
        use_mmap: bool,
    ) -> PyResult<Self> {
        let dtype = get_dtype(&dtype)?;
        let parsed: Vec<(Device, f64)> = device_ratios
            .iter()
            .map(|(d, r)| get_device(d).map(|dev| (dev, *r)))
            .collect::<Result<Vec<_>, _>>()?;

        let scoring_device = parsed
            .iter()
            .find(|(d, _)| d.is_cuda())
            .map(|(d, _)| *d)
            .unwrap_or(parsed[0].0);

        Ok(Self {
            sharded_index: None,
            index_path,
            device_ratios: parsed,
            scoring_device,
            dtype,
            use_mmap,
            deleted_pids: HashSet::new(),
        })
    }

    fn load(&mut self) -> PyResult<()> {
        let loader = IndexLoader::new(&self.index_path, Device::Cpu, self.dtype, self.use_mmap)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loader: {}", e)))?;

        let sharded_index = loader
            .load_sharded(&self.device_ratios, self.scoring_device)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load index: {}", e)))?;

        self.deleted_pids = crate::index::delete::load_tombstones(Path::new(&self.index_path))
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load tombstones: {}", e)))?;

        self.sharded_index = Some(Arc::new(ReadOnlyShardedIndex(sharded_index)));
        Ok(())
    }

    #[pyo3(signature = (torch_path, queries_embeddings, search_config, subsets=None, show_progress=true))]
    fn search(
        &self,
        torch_path: String,
        queries_embeddings: PyTensor,
        search_config: SearchConfig,
        subsets: Option<Vec<Vec<i64>>>,
        show_progress: bool,
    ) -> PyResult<Vec<SearchResult>> {
        call_torch(torch_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch: {}", e)))?;

        let shape = queries_embeddings.size();
        if shape.len() != 3 {
            return Err(PyRuntimeError::new_err(format!(
                "Expected 3D tensor, got {}D tensor with shape {:?}",
                shape.len(),
                shape
            )));
        }

        let scorer = ShardedScorer::new(
            self.sharded_index
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("Index not loaded. Call load() first."))?,
            search_config.clone(),
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create scorer: {}", e)))?;

        let k = search_config.k;

        let mut results = scorer
            .rank(
                &Query {
                    embeddings: queries_embeddings.deref().shallow_clone(),
                },
                subsets.as_deref(),
                show_progress,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {}", e)))?;

        filter_tombstones(&mut results, &self.deleted_pids, k);
        Ok(results)
    }

    fn update_tombstones(&mut self, passage_ids: Vec<i64>) -> PyResult<()> {
        self.deleted_pids.extend(passage_ids);
        Ok(())
    }

    fn free(&mut self) {
        self.sharded_index = None;
        self.deleted_pids.clear();
    }
}

/// Estimate index memory usage by reading only metadata and NPY headers.
/// Returns a dict of component name → size in bytes.
#[pyfunction]
fn estimate_index_memory(
    _py: Python<'_>,
    index_path: String,
) -> PyResult<std::collections::HashMap<String, u64>> {
    search::loader::estimate_index_memory(Path::new(&index_path))
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to estimate memory: {}", e)))
}

/// A high-performance document retrieval toolkit using a ColBERT-style late
/// interaction model, implemented in Rust with Python bindings.
///
/// This module provides functions for creating, updating, and searching indexes,
/// along with the necessary data classes `SearchParameters` and `QueryResult`
/// to interact with the library from Python.
#[pymodule]
#[pyo3(name = "xtr_warp_rs")]
fn python_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchConfig>()?;
    m.add_class::<SearchResult>()?;
    m.add_class::<ShardedSearcher>()?;

    m.add_function(wrap_pyfunction!(initialize_torch, m)?)?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    m.add_function(wrap_pyfunction!(delete, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(update, m)?)?;
    m.add_function(wrap_pyfunction!(compact, m)?)?;
    m.add_function(wrap_pyfunction!(append_centroids_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_index_memory, m)?)?;
    Ok(())
}
