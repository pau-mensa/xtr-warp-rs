// Search module - Core search and retrieval functionality

pub mod centroid_selector;
pub mod decompressor;
pub mod loader;
pub mod merger;
pub mod scorer;
use std::collections::HashSet;
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
    pub fn new(
        index: &Arc<ReadOnlyIndex>,
        config: &SearchConfig,
        deleted_pids: Arc<HashSet<i64>>,
    ) -> Result<Self> {
        let scorer = WARPScorer::new(index, config.clone(), deleted_pids)?;
        Ok(Self { scorer })
    }

    /// Search for top-k passages given a query
    pub fn search(&self, query: Query) -> Result<Vec<SearchResult>> {
        self.scorer.rank(&query)
    }
}
