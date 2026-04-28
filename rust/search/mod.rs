// Search module - Core search and retrieval functionality

pub mod centroid_selector;
pub mod decompressor;
pub mod loader;
pub mod merger;
mod candidate_assembly;
pub mod sharded_scorer;

// Re-export main types for convenience
pub use centroid_selector::CentroidSelector;
pub use decompressor::CentroidDecompressor;
pub use loader::IndexLoader;
pub use merger::ResultMerger;
pub use sharded_scorer::ShardedScorer;
