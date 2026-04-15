use anyhow::Result;
use indicatif::ProgressBar;
use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use tch::{Device, IndexOp, Kind, Tensor};

use crate::utils::maybe_progress;

use crate::search::centroid_selector::CentroidSelector;
use crate::search::decompressor::CentroidDecompressor;
use crate::search::merger::{MergerConfig, ResultMerger};
use crate::utils::types::{
    parse_dtype, Query, ReadOnlyShardedIndex, ReadOnlyTensor, SearchConfig, SearchResult,
};

/// Scorer for sharded indices. For single-shard indices, queries flow through
/// `process_query` (centroid select → decompress → merge) with no overhead.
/// For multi-shard, `rank_multi_shard` processes shard-by-shard across query
/// batches: CUDA shards first (async), then CPU shards (overlap), then
/// assembly + merge.
pub struct ShardedScorer {
    index: Arc<ReadOnlyShardedIndex>,
    centroid_selector: CentroidSelector,
    /// One decompressor per shard, configured for the shard's device.
    decompressors: Vec<CentroidDecompressor>,
    merger: ResultMerger,
    config: SearchConfig,
    thread_pool: Arc<ThreadPool>,
    scoring_device: Device,
    batch_size: i64,
    /// sizes_compacted pre-moved to scoring_device (avoids per-query transfer).
    sizes_on_scoring_device: Tensor,
}

impl ShardedScorer {
    pub fn new(index: &Arc<ReadOnlyShardedIndex>, mut config: SearchConfig) -> Result<Self> {
        let shared = &index.shared;
        let num_centroids = shared.metadata.num_centroids as u32;
        if config.nprobe > num_centroids {
            config.nprobe = num_centroids;
        }
        if config.bound > num_centroids as usize {
            config.bound = num_centroids as usize;
        }

        let scoring_device = shared.scoring_device;
        let batch_size = config.batch_size;
        let dtype = parse_dtype(&config.dtype)?;

        let centroid_selector = CentroidSelector::new(
            &config,
            shared.metadata.num_embeddings as usize,
            shared.metadata.num_centroids,
        );

        let num_threads = config
            .num_threads
            .unwrap_or_else(rayon::current_num_threads)
            .max(1);
        let thread_pool = Arc::new(ThreadPoolBuilder::new().num_threads(num_threads).build()?);

        // One decompressor per shard, each configured for the shard's device
        let decompressors: Vec<CentroidDecompressor> = index
            .shards
            .iter()
            .map(|_shard| {
                CentroidDecompressor::new(
                    shared.metadata.nbits,
                    shared.metadata.dim,
                    dtype,
                    Arc::clone(&thread_pool),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let max_candidates = config.max_candidates.unwrap_or(256);
        // Merger device must match the decompressor format:
        // - CUDA decompressors produce unsorted/non-deduped data → needs CUDA merger
        // - CPU decompressors produce sorted/deduped data → works with CPU merger
        // Using scoring_device ensures CUDA merger when any accelerator is present.
        let merger_config = MergerConfig {
            max_candidates,
            num_threads: config.num_threads.unwrap_or(1),
            device: scoring_device,
        };
        let merger = ResultMerger::new(merger_config);

        let sizes_on_scoring_device = shared
            .sizes_compacted
            .to_device(scoring_device);

        Ok(Self {
            index: Arc::clone(index),
            centroid_selector,
            decompressors,
            merger,
            config,
            thread_pool: thread_pool,
            scoring_device,
            batch_size,
            sizes_on_scoring_device,
        })
    }

    /// Process a single query: centroid selection → decompress → merge.
    /// Only used for single-shard indices; multi-shard is handled by
    /// `rank_multi_shard` at the batch level.
    #[inline]
    fn process_query(
        &self,
        query_idx: usize,
        query_embeddings: Tensor,
        centroid_scores: Tensor,
        query_mask: Tensor,
        k: usize,
        subset: Option<&[i64]>,
    ) -> Result<SearchResult> {
        let shared = &self.index.shared;
        let shard = &self.index.shards[0];
        let nprobe = self.config.nprobe as usize;

        let selected = self.centroid_selector.select_centroids(
            &query_mask.to_device(self.scoring_device),
            &centroid_scores,
            &self.sizes_on_scoring_device,
            shared.kdummy_centroid,
            k,
        )?;

        let cids = selected.centroid_ids.to_kind(Kind::Int64);

        let decompressed = self.decompressors[0].decompress_centroids_for_shard(
            &cids,
            &selected.scores,
            shard,
            &shared.bucket_weights,
            &query_embeddings,
            nprobe,
            subset,
            None,
        )?;

        let (pids, scores) = self.merger.merge_candidate_scores(
            &decompressed.capacities,
            &decompressed.sizes,
            &decompressed.passage_ids,
            &decompressed.scores,
            &selected.mse_estimate,
            nprobe,
            k,
        )?;

        Ok(SearchResult {
            passage_ids: pids,
            scores,
            query_id: query_idx + 1,
        })
    }

    /// Batch-level multi-shard processing.
    ///
    /// Instead of processing each query end-to-end (select → decompress all
    /// shards → assemble → merge), processes shard-by-shard across the batch:
    ///
    /// 1. Centroid selection + cell partitioning for all queries
    /// 2. CUDA shard decompression for all queries (async, no D2H sync)
    /// 3. CPU shard decompression for all queries (CUDA runs in background)
    /// 4. Assembly + merge for all queries (D2H here, but CUDA work is done)
    fn rank_multi_shard(
        &self,
        query: &Query,
        masks: &ReadOnlyTensor,
        subsets: Option<&[Vec<i64>]>,
        k: usize,
        bar: &ProgressBar,
    ) -> Result<Vec<SearchResult>> {
        let shared = &self.index.shared;
        let nprobe = self.config.nprobe as usize;
        let n_queries = query.embeddings.size()[0] as usize;
        let num_shards = self.index.shards.len();
        let merge_device = self.scoring_device;

        let mut all_results = Vec::with_capacity(n_queries);

        for c in (0..n_queries).step_by(self.batch_size as usize) {
            let batch_len = self.batch_size.min((n_queries - c) as i64) as usize;
            let batch_queries = query.embeddings.narrow(0, c as i64, batch_len as i64);
            let batch_mask = masks.narrow(0, c as i64, batch_len as i64);
            let centroid_scores = Tensor::einsum(
                "btd,cd->btc",
                &[&batch_queries, &shared.centroids],
                None::<&[i64]>,
            );

            // ── Phase 1: Centroid selection + cell partitioning ─────
            // Centroid IDs, scores, and token indices stored as Vecs so
            // they can be shared across rayon threads in Phase 2.
            let mut mse_estimates: Vec<Option<Tensor>> = Vec::with_capacity(batch_len);
            let mut ids_vecs: Vec<Option<Vec<i64>>> = Vec::with_capacity(batch_len);
            let mut scores_vecs: Vec<Option<Vec<f32>>> = Vec::with_capacity(batch_len);
            let mut token_indices_vecs: Vec<Option<Vec<i64>>> = Vec::with_capacity(batch_len);
            let mut shard_cells: Vec<Option<Vec<Vec<usize>>>> =
                Vec::with_capacity(batch_len);
            let mut num_cells: Vec<usize> = vec![0; batch_len];
            let mut batch_results: Vec<Option<SearchResult>> = vec![None; batch_len];

            for b in 0..batch_len {
                let query_idx = c + b;
                let subset: Option<&[i64]> = match subsets {
                    None => None,
                    Some(lists) if lists.len() == 1 => Some(&lists[0]),
                    Some(lists) => Some(&lists[query_idx]),
                };

                if matches!(subset, Some(s) if s.is_empty()) {
                    batch_results[b] = Some(SearchResult {
                        passage_ids: vec![],
                        scores: vec![],
                        query_id: query_idx + 1,
                    });
                    bar.inc(1);
                    mse_estimates.push(None);
                    ids_vecs.push(None);
                    scores_vecs.push(None);
                    token_indices_vecs.push(None);
                    shard_cells.push(None);
                    continue;
                }

                let selected = self.centroid_selector.select_centroids(
                    &batch_mask.i(b as i64).to_device(self.scoring_device),
                    &centroid_scores.i(b as i64),
                    &self.sizes_on_scoring_device,
                    shared.kdummy_centroid,
                    k,
                )?;

                let centroid_ids = selected.centroid_ids.to_kind(Kind::Int64);
                let nc = centroid_ids.size()[0] as usize;
                let ids_vec: Vec<i64> = centroid_ids.to_device(Device::Cpu).try_into()?;
                let scores_vec: Vec<f32> =
                    selected.scores.to_device(Device::Cpu).try_into()?;
                let tok_vec: Vec<i64> = (0..nc).map(|i| (i / nprobe) as i64).collect();

                let mut per_shard: Vec<Vec<usize>> = vec![Vec::new(); num_shards];
                for (cell_idx, &cid) in ids_vec.iter().enumerate() {
                    for (si, shard) in self.index.shards.iter().enumerate() {
                        if (cid as usize) >= shard.centroid_start
                            && (cid as usize) < shard.centroid_end
                        {
                            per_shard[si].push(cell_idx);
                            break;
                        }
                    }
                }

                mse_estimates.push(Some(selected.mse_estimate));
                ids_vecs.push(Some(ids_vec));
                scores_vecs.push(Some(scores_vec));
                token_indices_vecs.push(Some(tok_vec));
                shard_cells.push(Some(per_shard));
                num_cells[b] = nc;
            }

            // ── Phase 2: Per-shard decompression ────────────────────
            // CUDA shards: sequential (async kernel launches, no D2H sync).
            // CPU shards: parallel across queries via rayon — much better
            // thread utilization than parallelizing across cells within a
            // single query (32 threads × 64 queries vs 32 threads × 4 cells).
            enum ShardOut {
                Device(Tensor, Tensor, Tensor, Tensor, Tensor),
                Vecs(Vec<i64>, Vec<i32>, Vec<i64>, Vec<i64>, Vec<f32>),
            }
            let mut shard_outs: Vec<Vec<Option<ShardOut>>> = (0..batch_len)
                .map(|_| (0..num_shards).map(|_| None).collect())
                .collect();

            for si in 0..num_shards {
                let shard = &self.index.shards[si];

                if shard.device.is_cuda() {
                    // CUDA: sequential — kernel launches are async so
                    // pipelining across queries happens on the device.
                    for b in 0..batch_len {
                        let cells = match &shard_cells[b] {
                            Some(sc) => &sc[si],
                            None => continue,
                        };
                        if cells.is_empty() {
                            continue;
                        }

                        let ids_vec = ids_vecs[b].as_ref().unwrap();
                        let sc_vec = scores_vecs[b].as_ref().unwrap();
                        let tok_vec = token_indices_vecs[b].as_ref().unwrap();

                        let s_cids_v: Vec<i64> =
                            cells.iter().map(|&i| ids_vec[i]).collect();
                        let s_scores_v: Vec<f32> =
                            cells.iter().map(|&i| sc_vec[i]).collect();
                        let s_tok_v: Vec<i64> =
                            cells.iter().map(|&i| tok_vec[i]).collect();

                        let s_cids =
                            Tensor::from_slice(&s_cids_v).to_device(shard.device);
                        let s_cscores =
                            Tensor::from_slice(&s_scores_v).to_device(shard.device);
                        let s_query =
                            batch_queries.i(b as i64).to_device(shard.device);
                        let token_indices =
                            Tensor::from_slice(&s_tok_v).to_device(shard.device);

                        let query_idx = c + b;
                        let subset: Option<&[i64]> = match subsets {
                            None => None,
                            Some(lists) if lists.len() == 1 => Some(&lists[0]),
                            Some(lists) => Some(&lists[query_idx]),
                        };

                        let d = self.decompressors[si]
                            .decompress_centroids_for_shard(
                                &s_cids,
                                &s_cscores,
                                shard,
                                &shared.bucket_weights,
                                &s_query,
                                nprobe,
                                subset,
                                Some(&token_indices),
                            )?;

                        shard_outs[b][si] = Some(ShardOut::Device(
                            d.capacities,
                            d.sizes,
                            d.offsets,
                            d.passage_ids,
                            d.scores,
                        ));
                    }
                } else {
                    // CPU: parallel across queries — rayon work-stealing
                    // naturally serializes the per-query cell loop when all
                    // threads are busy at the query level.
                    let batch_queries_ro =
                        ReadOnlyTensor(batch_queries.shallow_clone());
                    let bw_ro =
                        ReadOnlyTensor(shared.bucket_weights.shallow_clone());
                    let decompressor = self.decompressors[si].clone();

                    let cpu_results: Vec<Result<(usize, ShardOut)>> =
                        self.thread_pool.install(|| {
                            (0..batch_len)
                                .into_par_iter()
                                .filter_map(|b| {
                                    let cells = match &shard_cells[b] {
                                        Some(sc) => &sc[si],
                                        None => return None,
                                    };
                                    if cells.is_empty() {
                                        return None;
                                    }
                                    Some((|| -> Result<(usize, ShardOut)> {
                                        let ids_vec =
                                            ids_vecs[b].as_ref().unwrap();
                                        let sc_v =
                                            scores_vecs[b].as_ref().unwrap();
                                        let tok_v =
                                            token_indices_vecs[b].as_ref().unwrap();

                                        let s_cids_v: Vec<i64> =
                                            cells.iter().map(|&i| ids_vec[i]).collect();
                                        let s_scores_v: Vec<f32> =
                                            cells.iter().map(|&i| sc_v[i]).collect();
                                        let s_tok_v: Vec<i64> =
                                            cells.iter().map(|&i| tok_v[i]).collect();

                                        let s_cids =
                                            Tensor::from_slice(&s_cids_v);
                                        let s_cscores =
                                            Tensor::from_slice(&s_scores_v);
                                        let s_query = batch_queries_ro
                                            .i(b as i64)
                                            .to_device(shard.device);
                                        let token_indices =
                                            Tensor::from_slice(&s_tok_v);

                                        let query_idx = c + b;
                                        let subset: Option<&[i64]> = match subsets
                                        {
                                            None => None,
                                            Some(lists) if lists.len() == 1 => {
                                                Some(&lists[0])
                                            }
                                            Some(lists) => Some(&lists[query_idx]),
                                        };

                                        let d = decompressor
                                            .decompress_centroids_for_shard(
                                                &s_cids,
                                                &s_cscores,
                                                shard,
                                                &bw_ro,
                                                &s_query,
                                                nprobe,
                                                subset,
                                                Some(&token_indices),
                                            )?;

                                        Ok((
                                            b,
                                            ShardOut::Vecs(
                                                d.capacities.try_into()?,
                                                d.sizes
                                                    .to_kind(Kind::Int)
                                                    .try_into()?,
                                                d.offsets.try_into()?,
                                                d.passage_ids
                                                    .to_kind(Kind::Int64)
                                                    .try_into()?,
                                                d.scores
                                                    .to_kind(Kind::Float)
                                                    .try_into()?,
                                            ),
                                        ))
                                    })())
                                })
                                .collect()
                        });

                    for result in cpu_results {
                        let (b, out) = result?;
                        shard_outs[b][si] = Some(out);
                    }
                }
            }

            // ── Phase 3: Assembly + merge ───────────────────────────
            for b in 0..batch_len {
                if batch_results[b].is_some() {
                    continue;
                }
                let query_idx = c + b;
                let nc = num_cells[b];
                let sc = shard_cells[b].as_ref().unwrap();

                let mut g_cap = vec![0i64; nc];
                let mut g_sizes = vec![0i32; nc];
                let mut g_pids: Vec<Vec<i64>> = vec![Vec::new(); nc];
                let mut g_scores: Vec<Vec<f32>> = vec![Vec::new(); nc];

                for si in 0..num_shards {
                    let cells = &sc[si];
                    if cells.is_empty() {
                        continue;
                    }
                    let (cv, sv, ov, pv, scv) = match shard_outs[b][si].take() {
                        Some(ShardOut::Device(cap, sz, off, pid, sc)) => {
                            let cv: Vec<i64> =
                                cap.to_device(Device::Cpu).try_into()?;
                            let sv: Vec<i32> = sz
                                .to_device(Device::Cpu)
                                .to_kind(Kind::Int)
                                .try_into()?;
                            let ov: Vec<i64> =
                                off.to_device(Device::Cpu).try_into()?;
                            let pv: Vec<i64> = pid
                                .to_device(Device::Cpu)
                                .to_kind(Kind::Int64)
                                .try_into()?;
                            let scv: Vec<f32> = sc
                                .to_device(Device::Cpu)
                                .to_kind(Kind::Float)
                                .try_into()?;
                            (cv, sv, ov, pv, scv)
                        }
                        Some(ShardOut::Vecs(cv, sv, ov, pv, scv)) => {
                            (cv, sv, ov, pv, scv)
                        }
                        None => continue,
                    };

                    for (local, &gc) in cells.iter().enumerate() {
                        g_cap[gc] = cv[local];
                        g_sizes[gc] = sv[local];
                        let s = ov[local] as usize;
                        let e = ov[local + 1] as usize;
                        g_pids[gc] = pv[s..e].to_vec();
                        g_scores[gc] = scv[s..e].to_vec();
                    }
                }

                let mut flat_pids: Vec<i64> = Vec::new();
                let mut flat_scores: Vec<f32> = Vec::new();
                for i in 0..nc {
                    flat_pids.extend(&g_pids[i]);
                    flat_scores.extend(&g_scores[i]);
                }

                let cap_t = Tensor::from_slice(&g_cap)
                    .to_kind(Kind::Int64)
                    .to_device(merge_device);
                let sz_t = Tensor::from_slice(&g_sizes)
                    .to_kind(Kind::Int)
                    .to_device(merge_device);
                let pid_t = Tensor::from_slice(&flat_pids)
                    .to_kind(Kind::Int64)
                    .to_device(merge_device);
                let sc_t = Tensor::from_slice(&flat_scores)
                    .to_kind(Kind::Float)
                    .to_device(merge_device);

                let mse = mse_estimates[b].as_ref().unwrap();
                let (pids, scores) = self.merger.merge_candidate_scores(
                    &cap_t,
                    &sz_t,
                    &pid_t,
                    &sc_t,
                    &mse.to_device(merge_device),
                    nprobe,
                    k,
                )?;

                batch_results[b] = Some(SearchResult {
                    passage_ids: pids,
                    scores,
                    query_id: query_idx + 1,
                });
                bar.inc(1);
            }

            for r in batch_results {
                all_results.push(r.unwrap());
            }
        }

        Ok(all_results)
    }

    /// Rank all queries in a batch through the sharded pipeline.
    pub fn rank(
        &self,
        query: &Query,
        subsets: Option<&[Vec<i64>]>,
        show_progress: bool,
    ) -> Result<Vec<SearchResult>> {
        let _guard = tch::no_grad_guard();

        let k = self.config.k;
        let n_queries = query.embeddings.size()[0] as usize;
        let masks = ReadOnlyTensor(query.embeddings.ne(0).any_dim(2, false));
        let shared = &self.index.shared;

        let bar = maybe_progress(show_progress, n_queries as u64, "Searching");

        // Multi-shard: batch-level shard processing.
        // Processes CUDA shards for all queries first (async kernel launches),
        // then CPU shards (while CUDA runs in background), avoiding per-query
        // CUDA pipeline syncs.
        if self.index.shards.len() > 1 {
            let results = self.rank_multi_shard(query, &masks, subsets, k, &bar)?;
            bar.finish_and_clear();
            return Ok(results);
        }

        if self.scoring_device == Device::Cpu {
            // CPU scoring: rayon parallel across queries (single-shard only,
            // multi-shard returned early above).
            let queries: Vec<ReadOnlyTensor> = (0..n_queries)
                .map(|b| ReadOnlyTensor(query.embeddings.select(0, b as i64)))
                .collect();

            let centroid_selector = self.centroid_selector.clone();
            let decompressor = self.decompressors[0].clone();
            let merger = self.merger.clone();
            let index = Arc::clone(&self.index);
            let nprobe = self.config.nprobe as usize;
            let scoring_device = self.scoring_device;
            let bar_clone = bar.clone();

            let results = self.thread_pool.install(move || {
                queries
                    .into_par_iter()
                    .enumerate()
                    .map(|(idx, query_embeddings)| {
                        let subset: Option<&[i64]> = match subsets {
                            None => None,
                            Some(lists) if lists.len() == 1 => Some(&lists[0]),
                            Some(lists) => Some(&lists[idx]),
                        };

                        if matches!(subset, Some(s) if s.is_empty()) {
                            bar_clone.inc(1);
                            return Ok(SearchResult {
                                passage_ids: vec![],
                                scores: vec![],
                                query_id: idx + 1,
                            });
                        }

                        let centroid_scores = query_embeddings
                            .matmul(&index.shared.centroids.transpose(0, 1));
                        let query_mask = masks.i(idx as i64);

                        let selected = centroid_selector.select_centroids(
                            &query_mask.to_device(scoring_device),
                            &centroid_scores,
                            &index.shared.sizes_compacted,
                            index.shared.kdummy_centroid,
                            k,
                        )?;

                        let cids = selected.centroid_ids.to_kind(Kind::Int64);
                        let shard = &index.shards[0];

                        let decompressed =
                            decompressor.decompress_centroids_for_shard(
                                &cids,
                                &selected.scores,
                                shard,
                                &index.shared.bucket_weights,
                                &query_embeddings,
                                nprobe,
                                subset,
                                None,
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

                        bar_clone.inc(1);

                        Ok(SearchResult {
                            passage_ids: pids,
                            scores,
                            query_id: idx + 1,
                        })
                    })
                    .collect()
            });

            bar.finish_and_clear();
            results
        } else {
            // Accelerator scoring path: batch centroid matmul
            let mut results = Vec::with_capacity(n_queries);

            for c in (0..n_queries).step_by(self.batch_size as usize) {
                let batch_size = self.batch_size.min((n_queries - c) as i64);
                let batch_queries = query.embeddings.narrow(0, c as i64, batch_size);
                let batch_mask = masks.narrow(0, c as i64, batch_size);

                let centroid_scores = Tensor::einsum(
                    "btd,cd->btc",
                    &[&batch_queries, &shared.centroids],
                    None::<&[i64]>,
                );

                for b in 0..batch_size {
                    let query_idx = c + b as usize;
                    let subset: Option<&[i64]> = match subsets {
                        None => None,
                        Some(lists) if lists.len() == 1 => Some(&lists[0]),
                        Some(lists) => Some(&lists[query_idx]),
                    };

                    if matches!(subset, Some(s) if s.is_empty()) {
                        results.push(SearchResult {
                            passage_ids: vec![],
                            scores: vec![],
                            query_id: query_idx + 1,
                        });
                        bar.inc(1);
                        continue;
                    }

                    let result = self.process_query(
                        query_idx,
                        batch_queries.i(b),
                        centroid_scores.i(b),
                        batch_mask.i(b),
                        k,
                        subset,
                    )?;
                    results.push(result);
                    bar.inc(1);
                }
            }

            bar.finish_and_clear();
            Ok(results)
        }
    }
}
