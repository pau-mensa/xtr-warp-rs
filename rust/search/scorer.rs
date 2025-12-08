use anyhow::Result;
use rayon::prelude::*;
use std::sync::Arc;

use crate::search::centroid_selector::CentroidSelector;
use crate::search::decompressor::CentroidDecompressor;
use crate::search::merger::{MergerConfig, ResultMerger};
use crate::utils::types::{
    parse_device, parse_dtype, Query, ReadOnlyIndex, ReadOnlyTensor, SearchConfig, SearchResult,
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

    selector_device: tch::Device,
    decompress_device: tch::Device,
    selector_centroids: ReadOnlyTensor,
    selector_sizes: ReadOnlyTensor,
}

impl WARPScorer {
    pub fn new(index: &Arc<ReadOnlyIndex>, config: SearchConfig) -> Result<Self> {
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
            selector_centroids: ReadOnlyTensor(selector_centroids),
            selector_sizes: ReadOnlyTensor(selector_sizes),
        })
    }

    /// Main ranking function that scores and ranks passages for a batch of queries
    pub fn rank(
        &self,
        query: &Query, // [batch, num_tokens, dim]
    ) -> Result<Vec<SearchResult>> {
        let k = self.config.k;
        let batch_size = query.embeddings.size()[0] as usize;
        println!("Query size is {:?}", query.embeddings.size());

        let max_threads = self
            .config
            .num_threads
            .unwrap_or_else(rayon::current_num_threads)
            .max(1);
        let device_mode = self.config.device_mode.to_lowercase();
        let enable_batch_parallelism = self.config.enable_batch_parallelism.unwrap_or(false)
            && max_threads > 1
            && batch_size > 1
            && (device_mode == "cpu" || device_mode == "hybrid");

        let allow_inner_parallelism = self.config.enable_inner_parallelism.unwrap_or(false);

        let outer_threads = if enable_batch_parallelism {
            batch_size.min(max_threads)
        } else {
            1
        };
        let inner_threads = if !allow_inner_parallelism {
            1
        } else if enable_batch_parallelism && outer_threads >= max_threads {
            1
        } else if enable_batch_parallelism {
            (max_threads / outer_threads).max(1)
        } else {
            max_threads
        };

        // Pre-split queries so each thread owns its tensor (Tensor is Send but not Sync)
        let queries: Vec<tch::Tensor> = (0..batch_size)
            .map(|b| query.embeddings.narrow(0, b as i64, 1).squeeze_dim(0))
            .collect();

        let selector_centroids = self.selector_centroids.clone();
        let selector_sizes = self.selector_sizes.clone();
        let centroid_selector = self.centroid_selector.clone();
        let decompressor = self.decompressor.clone();
        let merger = self.merger.clone();
        let index = Arc::clone(&self.index);
        let selector_device = self.selector_device;
        let decompress_device = self.decompress_device;
        let nprobe = self.config.nprobe as usize;

        let process_query = move |b: usize,
                                  query_embeddings: tch::Tensor|
              -> Result<SearchResult> {
            let _guard = tch::no_grad_guard();

            let query_mask = query_embeddings.ne(0).any_dim(1, false);

            // Prepare query on selector device
            let query_for_selector = if selector_device == query_embeddings.device() {
                query_embeddings.shallow_clone()
            } else {
                query_embeddings.to_device(selector_device)
            };

            // Compute centroid scores on selector device
            let centroid_scores = query_for_selector.matmul(&selector_centroids.transpose(0, 1));

            // Select centroids for this query using pre-computed scores
            let mut selected = centroid_selector.select_centroids(
                &query_mask.to_device(selector_device),
                &centroid_scores,
                &selector_sizes,
                index.kdummy_centroid,
                k,
            )?;

            // Move selection back to decompress device if needed
            if selector_device != decompress_device {
                selected.centroid_ids = selected.centroid_ids.to_device(decompress_device);
                selected.scores = selected.scores.to_device(decompress_device);
                selected.mse_estimate = selected.mse_estimate.to_device(decompress_device);
            }

            // Prepare query for decompression
            let query_for_decompress = if decompress_device == query_embeddings.device() {
                query_embeddings
            } else {
                query_embeddings.to_device(decompress_device)
            };

            // Decompress selected centroids
            let decompressed = decompressor.decompress_centroids(
                &selected.centroid_ids.to_kind(tch::Kind::Int64),
                &selected.scores,
                &index,
                &query_for_decompress,
                nprobe,
                inner_threads,
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

        if outer_threads == 1 {
            let mut results = Vec::with_capacity(batch_size);
            for (idx, q) in queries.into_iter().enumerate() {
                results.push(process_query(idx, q)?);
            }
            Ok(results)
        } else {
            // Custom pool so we don't oversubscribe beyond the heuristic
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(outer_threads)
                .build()
                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

            let results: Result<Vec<SearchResult>> = pool.install(|| {
                queries
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, q)| process_query(idx, q))
                    .collect()
            });

            results
        }
    }
}
