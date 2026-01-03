// Search module - Core search and retrieval functionality

pub mod centroid_selector;
pub mod decompressor;
pub mod loader;
pub mod merger;
pub mod scorer;
use std::sync::Arc;

// Re-export main types for convenience
pub use centroid_selector::CentroidSelector;
pub use decompressor::CentroidDecompressor;
pub use loader::IndexLoader;
pub use merger::ResultMerger;
pub use scorer::WARPScorer;

use anyhow::Result;

use crate::utils::types::{Query, ReadOnlyIndex, SearchConfig, SearchResult};

/// Main search interface combining all components
pub struct Searcher {
    scorer: WARPScorer,
}

impl Searcher {
    /// Create a new searcher with loaded index
    pub fn new(index: &Arc<ReadOnlyIndex>, config: &SearchConfig) -> Result<Self> {
        // Initialize the WARP scorer which integrates all phase 1 components
        let scorer = WARPScorer::new(index, config.clone())?;

        Ok(Self { scorer })
    }

    /// Search for top-k passages given a query
    pub fn search(&self, query: Query) -> Result<Vec<SearchResult>> {
        // Use the WARPScorer which handles the entire batch
        self.scorer.rank(&query)
    }
}
