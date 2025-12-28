use anyhow::Result;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;

use crate::search::centroid_selector::CentroidSelector;
use crate::search::decompressor::CentroidDecompressor;
use crate::search::merger::{MergerConfig, ResultMerger};
use crate::utils::types::{
    parse_device, parse_dtype, Query, ReadOnlyIndex, SearchConfig, SearchResult,
};

/// Main scorer struct that handles WARP scoring operations
/// This integrates the phase 1 components (CentroidSelector, CentroidDecompressor)
/// with the ranking pipeline
pub struct WARPScorer {
    /// Shared reference to the loaded index
    index: Arc<ReadOnlyIndex>,

    /// Centroid selector component from phase 1
    centroid_selector: CentroidSelector,

    /// Decompressor component from phase 1
    decompressor: CentroidDecompressor,

    /// Result merger for combining scores
    merger: ResultMerger,

    /// Configuration
    config: SearchConfig,

    /// Shared rayon thread pool respecting user-provided num_threads
    thread_pool: Arc<ThreadPool>,

    /// Device to perform the scoring on
    device: tch::Device,
}

impl WARPScorer {
    pub fn new(index: &Arc<ReadOnlyIndex>, config: SearchConfig) -> Result<Self> {
        // Initialize centroid selector from phase 1
        let device = parse_device(&config.device)?;
        let centroid_selector = CentroidSelector::new(
            &config,
            index.metadata.num_embeddings,
            index.metadata.num_centroids,
        );
        let dtype = parse_dtype(&config.dtype)?;

        // Build a dedicated rayon pool to reuse across all parallel sections
        let num_threads = config
            .num_threads
            .unwrap_or_else(rayon::current_num_threads)
            .max(1);
        let thread_pool = Arc::new(ThreadPoolBuilder::new().num_threads(num_threads).build()?);

        // Initialize decompressor from phase 1
        let decompressor = CentroidDecompressor::new(
            index.metadata.nbits,
            index.metadata.dim,
            device,
            dtype,
            Arc::clone(&thread_pool),
        )?;

        // Initialize merger
        let max_candidates = config.max_candidates.unwrap_or(256);
        let merger_config = MergerConfig {
            max_candidates: max_candidates,
            num_threads: config.num_threads.unwrap_or(1),
            device: device,
        };
        let merger = ResultMerger::new(merger_config);

        Ok(Self {
            index: Arc::clone(index),
            centroid_selector,
            decompressor,
            merger,
            config,
            thread_pool,
            device,
        })
    }

    /// Main ranking function that scores and ranks passages for a batch of queries
    pub fn rank(
        &self,
        query: &Query, // [batch, num_tokens, dim]
    ) -> Result<Vec<SearchResult>> {
        let k = self.config.k;
        let batch_size = query.embeddings.size()[0] as usize;

        let use_parallel = self.thread_pool.current_num_threads() > 1 && batch_size > 1;

        // Pre-split queries so each thread owns its tensor (Tensor is Send but not Sync)
        let queries: Vec<tch::Tensor> = (0..batch_size)
            .map(|b| query.embeddings.narrow(0, b as i64, 1).squeeze_dim(0))
            .collect();

        let centroid_selector = self.centroid_selector.clone();
        let decompressor = self.decompressor.clone();
        let merger = self.merger.clone();
        let index = Arc::clone(&self.index);
        let nprobe = self.config.nprobe as usize;

        let process_query =
            move |b: usize, query_embeddings: tch::Tensor| -> Result<SearchResult> {
                let _guard = tch::no_grad_guard();

                let query_mask = query_embeddings.ne(0).any_dim(1, false);

                // Compute centroid scores on selector device
                let centroid_scores = query_embeddings.matmul(&index.centroids.transpose(0, 1));

                // Select centroids for this query using pre-computed scores
                let selected = centroid_selector.select_centroids(
                    &query_mask.to_device(self.device),
                    &centroid_scores,
                    &index.sizes_compacted,
                    index.kdummy_centroid,
                    k,
                )?;

                // Decompress selected centroids
                let decompressed = decompressor.decompress_centroids(
                    &selected.centroid_ids.to_kind(tch::Kind::Int64),
                    &selected.scores,
                    &index,
                    &query_embeddings,
                    nprobe,
                )?;

                // Merge candidate scores
                let (pids, scores) = merger.merge_candidate_scores(
                    &decompressed.capacities,
                    &decompressed.sizes,
                    &decompressed.passage_ids,
                    &decompressed.scores,
                    &selected.mse_estimate,
                    nprobe,
                    k,
                )?;

                Ok(SearchResult {
                    passage_ids: pids[..k.min(pids.len())].to_vec(),
                    scores: scores[..k.min(scores.len())].to_vec(),
                    query_id: (b + 1) as usize,
                })
            };

        if !use_parallel {
            let mut results = Vec::with_capacity(batch_size);
            for (idx, q) in queries.into_iter().enumerate() {
                results.push(process_query(idx, q)?);
            }
            Ok(results)
        } else {
            self.thread_pool.install(|| {
                queries
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, q)| process_query(idx, q))
                    .collect()
            })
        }
    }
}
