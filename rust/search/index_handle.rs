use std::sync::Arc;

use anyhow::Result;
use tch::Tensor;

use crate::search::streaming::StreamingIndex;
use crate::utils::types::{CentroidId, IndexMetadata, ReadOnlyIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexAccessMode {
    InMemory,
    Streaming,
}

/// Unified handle to an index, either fully loaded or streaming.
#[derive(Clone)]
pub enum IndexHandle {
    InMemory(Arc<ReadOnlyIndex>),
    Streaming(Arc<StreamingIndex>),
}

impl IndexHandle {
    pub fn access_mode(&self) -> IndexAccessMode {
        match self {
            IndexHandle::InMemory(_) => IndexAccessMode::InMemory,
            IndexHandle::Streaming(_) => IndexAccessMode::Streaming,
        }
    }

    pub fn is_streaming(&self) -> bool {
        matches!(self, IndexHandle::Streaming(_))
    }

    pub fn centroids(&self) -> &Tensor {
        match self {
            IndexHandle::InMemory(index) => &index.centroids,
            IndexHandle::Streaming(index) => index.centroids(),
        }
    }

    pub fn bucket_weights(&self) -> &Tensor {
        match self {
            IndexHandle::InMemory(index) => &index.bucket_weights,
            IndexHandle::Streaming(index) => index.bucket_weights(),
        }
    }

    pub fn sizes_compacted(&self) -> &Tensor {
        match self {
            IndexHandle::InMemory(index) => &index.sizes_compacted,
            IndexHandle::Streaming(index) => index.sizes_compacted(),
        }
    }

    pub fn offsets_compacted(&self) -> &Tensor {
        match self {
            IndexHandle::InMemory(index) => &index.offsets_compacted,
            IndexHandle::Streaming(index) => index.offsets_compacted(),
        }
    }

    pub fn kdummy_centroid(&self) -> CentroidId {
        match self {
            IndexHandle::InMemory(index) => index.kdummy_centroid,
            IndexHandle::Streaming(index) => index.kdummy_centroid(),
        }
    }

    pub fn metadata(&self) -> &IndexMetadata {
        match self {
            IndexHandle::InMemory(index) => &index.metadata,
            IndexHandle::Streaming(index) => index.metadata(),
        }
    }

    pub fn as_in_memory(&self) -> Option<&Arc<ReadOnlyIndex>> {
        match self {
            IndexHandle::InMemory(index) => Some(index),
            _ => None,
        }
    }

    pub fn as_streaming(&self) -> Option<&Arc<StreamingIndex>> {
        match self {
            IndexHandle::Streaming(index) => Some(index),
            _ => None,
        }
    }

    pub fn read_codes_range(&self, start: i64, len: i64) -> Result<Vec<i64>> {
        match self {
            IndexHandle::InMemory(index) => index.read_codes_range(start, len),
            IndexHandle::Streaming(index) => index.read_codes_range(start, len),
        }
    }

    pub fn read_residuals_range(&self, start: i64, len: i64) -> Result<Vec<u8>> {
        match self {
            IndexHandle::InMemory(index) => index.read_residuals_range(start, len),
            IndexHandle::Streaming(index) => index.read_residuals_range(start, len),
        }
    }

    pub fn read_centroid_chunk(
        &self,
        centroid_id: CentroidId,
        start: i64,
        len: i64,
    ) -> Result<(Vec<i64>, Vec<u8>)> {
        match self {
            IndexHandle::InMemory(index) => index.read_centroid_chunk(start, len),
            IndexHandle::Streaming(index) => index.read_centroid_chunk(centroid_id, start, len),
        }
    }
}
