use anyhow::Result;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use tch::{Device, IndexOp, Kind, Tensor};

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

    /// Shared rayon thread pool
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

    /// Process a single query
    #[inline]
    fn process_query(
        &self,
        query_idx: usize,
        query_embeddings: Tensor,
        centroid_scores: Tensor,
        query_mask: Tensor,
        k: usize,
    ) -> Result<SearchResult> {
        let selected = self.centroid_selector.select_centroids(
            &query_mask.to_device(self.device),
            &centroid_scores,
            &self.index.sizes_compacted,
            self.index.kdummy_centroid,
            k,
        )?;
        let decompressed = self.decompressor.decompress_centroids(
            &selected.centroid_ids.to_kind(Kind::Int64),
            &selected.scores,
            &self.index,
            &query_embeddings,
            self.config.nprobe as usize,
        )?;
        let (pids, scores) = self.merger.merge_candidate_scores(
            &decompressed.capacities,
            &decompressed.sizes,
            &decompressed.passage_ids,
            &decompressed.scores,
            &selected.mse_estimate,
            self.config.nprobe as usize,
            k,
        )?;

        Ok(SearchResult {
            passage_ids: pids[..k.min(pids.len())].to_vec(),
            scores: scores[..k.min(scores.len())].to_vec(),
            query_id: query_idx + 1,
        })
    }

    /// Main ranking function that scores and ranks passages for a batch of queries
    pub fn rank(
        &self,
        query: &Query, // [batch, num_tokens, dim]
    ) -> Result<Vec<SearchResult>> {
        let _guard = tch::no_grad_guard();

        let k = self.config.k;
        let n_queries = query.embeddings.size()[0] as usize;
        let use_parallel = self.thread_pool.current_num_threads() > 1 && n_queries > 1;
        // Need to wrap it to ensure it can be shared in the parallel path
        let masks: ReadOnlyTensor = ReadOnlyTensor(query.embeddings.ne(0).any_dim(2, false));

        if use_parallel {
            if self.device != Device::Cpu {
                anyhow::bail!("Parallel processing is not supported on an accelerator. Set num_threads=1 if you want to use an accelerator or set device='cpu' if you want to use parallel processing.");
            }
            // parallel cpu
            let queries: Vec<ReadOnlyTensor> = (0..n_queries)
                .map(|b| ReadOnlyTensor(query.embeddings.select(0, b as i64)))
                .collect();

            let centroid_selector = self.centroid_selector.clone();
            let decompressor = self.decompressor.clone();
            let merger = self.merger.clone();
            let index = Arc::clone(&self.index);
            let nprobe = self.config.nprobe as usize;
            let device = self.device;

            self.thread_pool.install(move || {
                queries
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, query_embeddings)| {
                        let query_mask = masks.i(idx as i64);
                        let centroid_scores =
                            query_embeddings.matmul(&index.centroids.transpose(0, 1));

                        let selected = centroid_selector.select_centroids(
                            &query_mask.to_device(device),
                            &centroid_scores,
                            &index.sizes_compacted,
                            index.kdummy_centroid,
                            k,
                        )?;

                        let decompressed = decompressor.decompress_centroids(
                            &selected.centroid_ids.to_kind(Kind::Int64),
                            &selected.scores,
                            &index,
                            &query_embeddings,
                            nprobe,
                        )?;

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
                            query_id: idx + 1,
                        })
                    })
                    .collect()
            })
        } else {
            // accelerator path (optimized for cuda)
            let mut results = Vec::with_capacity(n_queries);

            for c in (0..n_queries).step_by(self.batch_size as usize) {
                let batch_size = self.batch_size.min((n_queries - c) as i64);
                let batch_queries = query.embeddings.narrow(0, c as i64, batch_size);
                let batch_mask = masks.narrow(0, c as i64, batch_size);

                let centroid_scores = Tensor::einsum(
                    "btd,cd->btc",
                    &[&batch_queries, &self.index.centroids],
                    None::<&[i64]>,
                );

                for b in 0..batch_size {
                    // TODO this could be vectorized, with some effort
                    let result = self.process_query(
                        c + b as usize,
                        batch_queries.i(b),
                        centroid_scores.i(b),
                        batch_mask.i(b),
                        k,
                    )?;
                    results.push(result);
                }
            }
            Ok(results)
        }
    }
}
