use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::search::centroid_selector::CentroidSelector;
use crate::search::decompressor::CentroidDecompressor;
use crate::search::merger::{MergerConfig, ResultMerger};
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

    selector_device: tch::Device,
    decompress_device: tch::Device,
    selector_centroids: tch::Tensor,
    selector_sizes: tch::Tensor,
}

impl WARPScorer {
    pub fn new(index: &Arc<LoadedIndex>, config: SearchConfig) -> Result<Self> {
        // Initialize centroid selector from phase 1
        let selector_device = parse_device(&config.selector_device)?;
        let decompress_device = parse_device(&config.decompress_device)?;
        let centroid_selector = CentroidSelector::new(
            &config,
            index.metadata.num_embeddings,
            index.metadata.num_centroids,
        );

        let device = parse_device(&config.decompress_device)?;
        let dtype = parse_dtype(&config.dtype)?;

        // Initialize decompressor from phase 1
        let decompressor = CentroidDecompressor::new(
            index.metadata.nbits,
            index.metadata.dim,
            device,
            dtype,
            // keep inner parallelism disabled for now
            if config.enable_inner_parallelism.unwrap_or(false) {
                config.num_threads.unwrap_or(1usize)
            } else {
                1usize
            },
        )?;

        // Initialize merger
        let max_candidates = config.max_candidates.unwrap_or(256);
        let merger_config = MergerConfig {
            max_candidates: max_candidates,
            num_threads: config.num_threads.unwrap_or(1),
            device: device,
        };
        let merger = ResultMerger::new(merger_config);

        let selector_centroids = if selector_device == decompress_device {
            index.centroids.shallow_clone()
        } else {
            index.centroids.to_device(selector_device)
        };
        let selector_sizes = if selector_device == decompress_device {
            index.sizes_compacted.shallow_clone()
        } else {
            index.sizes_compacted.to_device(selector_device)
        };

        Ok(Self {
            index: Arc::clone(index),
            centroid_selector,
            decompressor,
            merger,
            config,
            selector_device,
            decompress_device,
            selector_centroids,
            selector_sizes,
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
            println!("Query size is {:?}", query.embeddings.size());
            let process_query = |b: usize| -> Result<SearchResult> {
                // Extract single query tensors
                let query_embeddings = query.embeddings.narrow(0, b as i64, 1).squeeze_dim(0);
                let query_mask = query_embeddings.ne(0).any_dim(1, false);

                // Prepare query on selector device
                let query_for_selector = if self.selector_device == query_embeddings.device() {
                    query_embeddings.shallow_clone()
                } else {
                    query_embeddings.to_device(self.selector_device)
                };
                //let start = Instant::now();
                // Compute centroid scores on selector device
                let centroid_scores =
                    query_for_selector.matmul(&self.selector_centroids.transpose(0, 1));
                //let duration = start.elapsed();
                //println!("MATMUL time is {:?}", duration);

                //let start2 = Instant::now();

                // Select centroids for this query using pre-computed scores
                let mut selected = self.centroid_selector.select_centroids(
                    &query_mask.to_device(self.selector_device),
                    &centroid_scores,
                    &self.selector_sizes,
                    self.index.kdummy_centroid,
                    k,
                )?;
                //let duration2 = start2.elapsed();
                //println!("Centroids selector time is {:?}", duration2);

                // Move selection back to decompress device if needed
                if self.selector_device != self.decompress_device {
                    selected.centroid_ids = selected.centroid_ids.to_device(self.decompress_device);
                    selected.scores = selected.scores.to_device(self.decompress_device);
                    selected.mse_estimate = selected.mse_estimate.to_device(self.decompress_device);
                }

                // Prepare query for decompression
                let query_for_decompress = if self.decompress_device == query_embeddings.device() {
                    query_embeddings
                } else {
                    query_embeddings.to_device(self.decompress_device)
                };

                // Decompress selected centroids
                //let start3 = Instant::now();
                let decompressed = self.decompressor.decompress_centroids(
                    &selected.centroid_ids.to_kind(tch::Kind::Int64),
                    &selected.scores,
                    &self.index,
                    &query_for_decompress,
                    self.config.nprobe as usize,
                )?;
                //let duration3 = start3.elapsed();
                //println!("Decompression time {:?}", duration3);

                // Merge candidate scores
                //let start4 = Instant::now();
                let (pids, scores) = self.merger.merge_candidate_scores(
                    &decompressed.capacities,
                    &decompressed.sizes,
                    &decompressed.passage_ids,
                    &decompressed.scores,
                    &selected.mse_estimate,
                    self.config.nprobe as usize,
                    k,
                )?;
                //let duration4 = start4.elapsed();
                //println!("Merging time is {:?}", duration4);

                Ok(SearchResult {
                    passage_ids: pids[..k.min(pids.len())].to_vec(),
                    scores: scores[..k.min(scores.len())].to_vec(),
                    query_id: (b + 1) as usize,
                })
            };

            let mut results = Vec::with_capacity(batch_size);
            for b in 0..batch_size {
                results.push(process_query(b)?);
            }
            Ok(results)
        })
    }
}
