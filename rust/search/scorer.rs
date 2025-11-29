use anyhow::Result;
use std::sync::Arc;

use crate::search::centroid_selector::CentroidSelector;
use crate::search::decompressor::CentroidDecompressor;
use crate::search::merger::{MergerConfig, ResultMerger, ScoreCombination};
use crate::utils::types::{
    parse_device, parse_dtype, LoadedIndex, Query, SearchConfig, SearchResult,
};

/// Main scorer struct that handles WARP scoring operations
/// This integrates the phase 1 components (CentroidSelector, CentroidDecompressor)
/// with the ranking pipeline
pub struct WARPScorer {
    /// Shared reference to the loaded index
    index: Arc<LoadedIndex>,

    /// Centroid selector component from phase 1
    centroid_selector: CentroidSelector,

    /// Decompressor component from phase 1
    decompressor: CentroidDecompressor,

    /// Result merger for combining scores
    merger: ResultMerger,

    /// Configuration
    config: SearchConfig,
}

impl WARPScorer {
    pub fn new(index: Arc<LoadedIndex>, config: SearchConfig) -> Result<Self> {
        // Initialize centroid selector from phase 1
        let centroid_selector = CentroidSelector::new(
            &config,
            index.metadata.num_embeddings,
            index.metadata.num_centroids,
        );

        let device = parse_device(&config.device)?;
        let dtype = parse_dtype(&config.dtype)?;

        // Initialize decompressor from phase 1
        let decompressor = CentroidDecompressor::new(
            index.metadata.nbits,
            index.metadata.dim,
            device,
            dtype,
            config.parallel,
        )?;

        // Initialize merger
        let max_candidates = config.max_candidates.unwrap_or(256);
        let merger_config = MergerConfig {
            max_candidates: max_candidates,
            use_parallel: config.parallel,
            num_threads: config.num_threads.unwrap_or(1),
            combination_strategy: ScoreCombination::Sum,
            device: device,
        };
        let merger = ResultMerger::new(merger_config);

        Ok(Self {
            index,
            centroid_selector,
            decompressor,
            merger,
            config,
        })
    }

    /// Main ranking function that scores and ranks passages for a batch of queries
    pub fn rank(
        &self,
        query: &Query, // [batch, num_tokens, dim]
    ) -> Result<Vec<SearchResult>> {
        use tch::no_grad;

        let k = self.config.k;

        no_grad(|| {
            let batch_size = query.embeddings.size()[0] as usize;
            let mut results = Vec::with_capacity(batch_size);

            // Compute all centroid scores at once for efficiency
            // [batch, num_tokens, dim] @ [num_centroids, dim].T = [batch, num_tokens, num_centroids]
            let all_centroid_scores = query
                .embeddings
                .matmul(&self.index.centroids.transpose(0, 1));

            // Process each query independently
            for b in 0..batch_size {
                // Extract single query tensors
                let query_embeddings = query.embeddings.narrow(0, b as i64, 1).squeeze_dim(0);
                let centroid_scores = all_centroid_scores.narrow(0, b as i64, 1).squeeze_dim(0);
                // Create query mask
                let query_mask = query_embeddings.ne(0).any_dim(1, false);

                // Select centroids for this query using pre-computed scores
                let selected = self.centroid_selector.select_centroids(
                    &query_mask,
                    &centroid_scores,
                    &self.index.sizes_compacted,
                    self.index.kdummy_centroid,
                    k,
                )?;

                // Decompress selected centroids
                let decompressed = self.decompressor.decompress_centroids(
                    &selected.centroid_ids.to_kind(tch::Kind::Int64),
                    &selected.scores,
                    &self.index,
                    &query_embeddings,
                    self.config.nprobe as usize,
                )?;

                // Merge candidate scores
                //println!("Scores PRE {:?}", decompressed.scores.i(0));
                //println!("Scores sizes {:?}", decompressed.scores.size());
                let (pids, scores) = self.merger.merge_candidate_scores(
                    &decompressed.capacities,
                    &decompressed.sizes,
                    &decompressed.passage_ids,
                    &decompressed.scores,
                    &selected.mse_estimate,
                    self.config.nprobe as usize,
                    k,
                )?;
                //println!("Scores POST {:?}", scores[0]);
                //println!("Scores POST Size {:?}", scores.len());

                // Build result for this query
                /*results.push(SearchResult {
                    passage_ids: pids,
                    scores: scores,
                    query_id: (b + 1) as usize,
                });*/
                results.push(SearchResult {
                    passage_ids: pids[..k.min(pids.len())].to_vec(),
                    scores: scores[..k.min(scores.len())].to_vec(),
                    query_id: (b + 1) as usize,
                });
            }

            Ok(results)
        })
    }
}
