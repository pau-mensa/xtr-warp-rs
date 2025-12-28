use anyhow::Result;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use tch::{Device, IndexOp};

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

    /// Batch size for cuda processing
    batch_size: i64,
}

impl WARPScorer {
    pub fn new(index: &Arc<ReadOnlyIndex>, config: SearchConfig) -> Result<Self> {
        // Initialize centroid selector from phase 1
        let device = parse_device(&config.device)?;
        let batch_size = config.batch_size;
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
            batch_size,
        })
    }

    /// Main ranking function that scores and ranks passages for a batch of queries
    pub fn rank(
        &self,
        query: &Query, // [batch, num_tokens, dim]
    ) -> Result<Vec<SearchResult>> {
        let k = self.config.k;
        let n_queries = query.embeddings.size()[0] as usize;

        let use_parallel = self.thread_pool.current_num_threads() > 1 && n_queries > 1;

        let centroid_selector = self.centroid_selector.clone();
        let decompressor = self.decompressor.clone();
        let merger = self.merger.clone();
        let nprobe = self.config.nprobe as usize;

        let centroid_selector_cuda = self.centroid_selector.clone();
        let decompressor_cuda = self.decompressor.clone();
        let merger_cuda = self.merger.clone();
        let nprobe_cuda = self.config.nprobe as usize;

        let process_query = move |b: usize,
                                  query_embeddings: tch::Tensor,
                                  index: &Arc<ReadOnlyIndex>|
              -> Result<SearchResult> {
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

        let process_query_cuda = move |full_centroid_scores: tch::Tensor,
                                       queries: tch::Tensor,
                                       full_query_mask: tch::Tensor,
                                       index: &Arc<ReadOnlyIndex>|
              -> Result<Vec<SearchResult>> {
            let _guard = tch::no_grad_guard();
            let batch = *full_centroid_scores.size().get(0).unwrap() as i64;
            let mut results: Vec<SearchResult> = Vec::with_capacity(batch as usize);

            for b in 0..batch {
                let centroid_scores = full_centroid_scores.i(b);
                let query_mask = full_query_mask.i(b);
                let query_embeddings = queries.i(b);
                // Select centroids for this query using pre-computed scores
                let selected = centroid_selector_cuda.select_centroids(
                    &query_mask.to_device(self.device),
                    &centroid_scores,
                    &index.sizes_compacted,
                    index.kdummy_centroid,
                    k,
                )?;

                // Decompress selected centroids
                let decompressed = decompressor_cuda.decompress_centroids(
                    &selected.centroid_ids.to_kind(tch::Kind::Int64),
                    &selected.scores,
                    &index,
                    &query_embeddings,
                    nprobe_cuda,
                )?;

                // Merge candidate scores
                let (pids, scores) = merger_cuda.merge_candidate_scores(
                    &decompressed.capacities,
                    &decompressed.sizes,
                    &decompressed.passage_ids,
                    &decompressed.scores,
                    &selected.mse_estimate,
                    nprobe_cuda,
                    k,
                )?;

                results.push(SearchResult {
                    passage_ids: pids[..k.min(pids.len())].to_vec(),
                    scores: scores[..k.min(scores.len())].to_vec(),
                    query_id: (b + 1) as usize,
                });
            }
            Ok(results)
        };

        let queries = if use_parallel {
            (0..n_queries)
                .map(|b| query.embeddings.narrow(0, b as i64, 1).squeeze_dim(0))
                .collect::<Vec<tch::Tensor>>()
        } else {
            Vec::new() // dummy to avoid compilation error
        };

        match self.device {
            Device::Cpu => {
                if !use_parallel {
                    let mut results = Vec::with_capacity(n_queries);
                    for idx in 0..n_queries {
                        results.push(process_query(
                            idx,
                            query.embeddings.i(idx as i64),
                            &self.index,
                        )?);
                    }
                    Ok(results)
                } else {
                    self.thread_pool.install(|| {
                        queries
                            .into_par_iter()
                            .enumerate()
                            .map(|(idx, q)| process_query(idx, q, &self.index))
                            .collect()
                    })
                }
            },
            _ => {
                if !use_parallel {
                    let mut results: Vec<SearchResult> = Vec::with_capacity(n_queries);
                    for c in (0..n_queries).step_by(self.batch_size as usize) {
                        let batch_queries = query.embeddings.narrow(
                            0,
                            c as i64,
                            self.batch_size.min((n_queries - c).try_into().unwrap()),
                        );
                        let centroid_scores = tch::Tensor::einsum(
                            "btd,cd->btc",
                            &[&batch_queries, &self.index.centroids],
                            None::<&[i64]>,
                        );
                        let query_mask = batch_queries.ne(0).any_dim(2, false);
                        let search_result: Vec<SearchResult> = process_query_cuda(
                            centroid_scores,
                            batch_queries,
                            query_mask,
                            &self.index,
                        )?;
                        results.extend(search_result);
                    }
                    Ok(results)
                } else {
                    // This is inefficient and should not be used
                    self.thread_pool.install(|| {
                        queries
                            .into_par_iter()
                            .enumerate()
                            .map(|(idx, q)| process_query(idx, q, &self.index))
                            .collect()
                    })
                }
            },
        }
    }
}
