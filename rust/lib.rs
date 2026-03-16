// XTR-WARP Rust Implementation with Python Bindings

use anyhow::{anyhow, Result};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_tch::PyTensor;
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
use search::{IndexLoader, Searcher};
use utils::types::{IndexConfig, Query, ReadOnlyIndex, SearchConfig, SearchResult};

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

/// Parses a string identifier into a `tch::Device`.
///
/// Supports simple device strings like "cpu", "cuda", and indexed CUDA devices
/// such as "cuda:0".
fn get_device(device: &str) -> Result<Device, PyErr> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "mps" => Ok(Device::Mps),
        "cuda" => Ok(Device::Cuda(0)), // Default to the first CUDA device.
        s if s.starts_with("cuda:") => {
            let parts: Vec<&str> = s.split(':').collect();
            if parts.len() == 2 {
                parts[1].parse::<usize>().map(Device::Cuda).map_err(|_| {
                    PyValueError::new_err(format!("Invalid CUDA device index: '{}'", parts[1]))
                })
            } else {
                Err(PyValueError::new_err(
                    "Invalid CUDA device format. Expected 'cuda:N'.",
                ))
            }
        },
        _ => Err(PyValueError::new_err(format!(
            "Unsupported device string: '{}'",
            device
        ))),
    }
}

/// Parses a string identifier into a `tch::Kind`.
///
/// Supports simple strings like "float32", "float16"
fn get_dtype(dtype: &str) -> Result<Kind, PyErr> {
    match dtype.to_lowercase().as_str() {
        "float32" => Ok(Kind::Float),
        "float16" => Ok(Kind::Half),
        "float64" => Ok(Kind::Double),
        "bfloat16" => Ok(Kind::BFloat16),
        _ => Err(PyValueError::new_err(format!(
            "Unsupported dtype string: '{}', should be 'float32', 'float16', 'float64', or 'bfloat16'",
            dtype
        ))),
    }
}

/// Represents a loaded index
#[pyclass(unsendable)]
struct LoadedSearcher {
    loaded_index: Option<Arc<ReadOnlyIndex>>,
    index_path: String,
    device: Device,
    dtype: Kind,
    use_mmap: bool,
}

#[pymethods]
impl LoadedSearcher {
    #[new]
    #[pyo3(signature = (index_path, device, dtype, use_mmap=true))]
    fn new(index_path: String, device: String, dtype: String, use_mmap: bool) -> PyResult<Self> {
        let device = get_device(&device)?;
        let dtype = get_dtype(&dtype)?;

        Ok(Self {
            loaded_index: None,
            index_path,
            device,
            dtype,
            use_mmap,
        })
    }

    /// Load the index in memory
    fn load(&mut self) -> PyResult<()> {
        let index_loader =
            IndexLoader::new(&self.index_path, self.device, self.dtype, self.use_mmap)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create loader: {}", e)))?;
        let loaded_index = Arc::new(
            index_loader
                .load()
                .map(ReadOnlyIndex)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to load index: {}", e)))?,
        );
        self.loaded_index = Some(loaded_index);
        Ok(())
    }

    /// Main search entrypoint
    fn search(
        &self,
        torch_path: String,
        queries_embeddings: PyTensor,
        search_config: SearchConfig,
    ) -> PyResult<Vec<SearchResult>> {
        call_torch(torch_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load Torch library: {}", e)))?;

        // Always expect 3D tensor
        let shape = queries_embeddings.size();
        if shape.len() != 3 {
            return Err(PyRuntimeError::new_err(format!(
                "Expected 3D tensor, got {}D tensor with shape {:?}",
                shape.len(),
                shape
            )));
        }

        let searcher = Searcher::new(self.loaded_index.as_ref().unwrap(), &search_config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create searcher: {}", e)))?;

        // process batch
        let results = searcher
            .search(Query {
                embeddings: queries_embeddings.deref().shallow_clone(),
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Search failed: {}", e)))?;

        Ok(results)
    }

    /// Free the loaded index
    fn free(&mut self) {
        self.loaded_index = None;
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
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create index: {}", e)))
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
    m.add_class::<LoadedSearcher>()?;

    m.add_function(wrap_pyfunction!(initialize_torch, m)?)?;
    m.add_function(wrap_pyfunction!(create, m)?)?;
    Ok(())
}
