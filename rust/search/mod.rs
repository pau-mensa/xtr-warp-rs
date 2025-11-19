// Search module - Core search and retrieval functionality
//
// XTR-WARP Implementation Reference:
// - Main searcher: xtr-warp/warp/searcher.py (Searcher class)
// - Engine searcher: xtr-warp/warp/engine/searcher.py (WARPSearcher class)
// - Index storage: xtr-warp/warp/engine/search/index_storage.py (IndexScorerWARP)
// - Search pipeline: Combines centroid selection → decompression → scoring → ranking
//
// Fast-PLAID Rust Reference:
// - Search implementation: fast-plaid/rust/search/search.rs (search function, lines 342-490)
// - Index structure: fast-plaid/rust/search/load.rs (LoadedIndex struct)
// - Multi-stage search: IVF probing → approximate scoring → re-ranking → top-k selection
//
// Key differences from original:
// - XTR-WARP separates Searcher and WARPSearcher, we combine into one
// - Fast-PLAID uses strided tensors for efficient memory access
// - XTR-WARP supports both batched and unbatched search modes
// - We integrate WARP's bounded selection with fast-plaid's search pipeline

pub mod centroid_selector;
pub mod decompressor;
pub mod loader;
pub mod merger;
pub mod scorer;

// Re-export main types for convenience
pub use centroid_selector::CentroidSelector;
pub use decompressor::CentroidDecompressor;
pub use loader::{IndexLoader, IndexStats};
pub use merger::{ResultMerger, ScoreCombination};
pub use scorer::WARPScorer;

use anyhow::Result;

use crate::utils::types::{LoadedIndex, Query, SearchConfig, SearchResult};

/// Main search interface combining all components
pub struct Searcher {
    index: std::sync::Arc<LoadedIndex>,
    scorer: WARPScorer,
    config: SearchConfig,
}

impl Searcher {
    /// Create a new searcher with loaded index
    pub fn new(index: LoadedIndex, config: &SearchConfig) -> Result<Self> {
        // Initialize the WARP scorer which integrates all phase 1 components
        let index_arc = std::sync::Arc::new(index);
        let scorer = WARPScorer::new(index_arc.clone(), config.clone())?;

        Ok(Self {
            index: index_arc,
            scorer,
            config: config.clone(),
        })
    }

    /// Search for top-k passages given a query
    pub fn search(&self, query: Query, k: Option<usize>) -> Result<Vec<SearchResult>> {
        let k = k.unwrap_or(self.config.k);

        // Use the WARPScorer which handles the entire batch
        self.scorer.rank(&query, k)
    }
}
