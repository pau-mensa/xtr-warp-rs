use anyhow::{anyhow, Result};
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

/// Per-shard decompression output. CUDA shards keep tensors on device to
/// avoid a D2H sync; CPU shards already have the data as Vecs.
#[allow(clippy::large_enum_variant)]
enum ShardOut {
    Device(Tensor, Tensor, Tensor, Tensor, Tensor),
    Vecs(Vec<i64>, Vec<i32>, Vec<i64>, Vec<i64>, Vec<f32>),
}

/// Per-batch data produced by `prepare_batch` (einsum + Phase 1 +
/// Phase 2 CUDA launches). Consumed by `process_cpu_shards` (fills in
/// the CPU entries of `shard_outs`) and then by `run_phase3`.
struct PreparedBatch {
    c: usize,
    batch_len: usize,
    /// Wrapped in ReadOnlyTensor so it can cross thread boundaries in the
    /// CPU shard par_iter (tch::Tensor is !Sync).
    batch_queries: ReadOnlyTensor,
    mse_estimates: Vec<Option<Tensor>>,
    ids_vecs: Vec<Option<Vec<i64>>>,
    scores_vecs: Vec<Option<Vec<f32>>>,
    token_indices_vecs: Vec<Option<Vec<i64>>>,
    shard_cells: Vec<Option<Vec<Vec<usize>>>>,
    num_cells: Vec<usize>,
    /// Pre-populated with empty results for queries with empty subsets.
    batch_results: Vec<Option<SearchResult>>,
    /// Per (query, shard) decompress output. CUDA entries filled by
    /// `prepare_batch`; CPU entries filled after `process_cpu_shards`.
    shard_outs: Vec<Vec<Option<ShardOut>>>,
}

/// Scorer for sharded indices. For single-shard indices, queries flow through
/// `process_query` (centroid select → decompress → merge) with no overhead.
/// For multi-shard, `rank_multi_shard` pipelines two batches: while batch N's
/// CPU shard decompression runs on the rayon pool, the main thread prepares
/// batch N+1 (einsum + Phase 1 + Phase 2 CUDA launches) on GPU.
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

        let sizes_on_scoring_device = shared.sizes_compacted.to_device(scoring_device);

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

    /// Prepare one batch: einsum + Phase 1 (centroid selection + cell
    /// partitioning) + Phase 2 CUDA launches. Drops `centroid_scores` and
    /// `batch_mask` before returning to free GPU memory for the next batch's
    /// pipelined prep.
    fn prepare_batch(
        &self,
        query: &Query,
        masks: &ReadOnlyTensor,
        subsets: Option<&[Vec<i64>]>,
        c: usize,
        k: usize,
        bar: &ProgressBar,
    ) -> Result<PreparedBatch> {
        let shared = &self.index.shared;
        let nprobe = self.config.nprobe as usize;
        let n_queries = query.embeddings.size()[0] as usize;
        let num_shards = self.index.shards.len();
        let batch_size = self.batch_size as usize;
        let batch_len = batch_size.min(n_queries - c);

        let batch_queries = query.embeddings.narrow(0, c as i64, batch_len as i64);
        let batch_mask = masks.narrow(0, c as i64, batch_len as i64);
        let centroid_scores = Tensor::einsum(
            "btd,cd->btc",
            &[&batch_queries, &shared.centroids],
            None::<&[i64]>,
        );

        // Centroid selection + cell partitioning
        let mut mse_estimates: Vec<Option<Tensor>> = Vec::with_capacity(batch_len);
        let mut ids_vecs: Vec<Option<Vec<i64>>> = Vec::with_capacity(batch_len);
        let mut scores_vecs: Vec<Option<Vec<f32>>> = Vec::with_capacity(batch_len);
        let mut token_indices_vecs: Vec<Option<Vec<i64>>> = Vec::with_capacity(batch_len);
        let mut shard_cells: Vec<Option<Vec<Vec<usize>>>> = Vec::with_capacity(batch_len);
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
            let scores_vec: Vec<f32> = selected.scores.to_device(Device::Cpu).try_into()?;
            let tok_vec: Vec<i64> = (0..nc).map(|i| (i / nprobe) as i64).collect();

            let mut per_shard: Vec<Vec<usize>> = vec![Vec::new(); num_shards];
            for (cell_idx, &cid) in ids_vec.iter().enumerate() {
                for (si, shard) in self.index.shards.iter().enumerate() {
                    if (cid as usize) >= shard.centroid_start && (cid as usize) < shard.centroid_end
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

        // CUDA launches
        // CUDA kernels are async: launches return immediately, kernels
        // execute in the background while the main thread moves on.
        let mut shard_outs: Vec<Vec<Option<ShardOut>>> = (0..batch_len)
            .map(|_| (0..num_shards).map(|_| None).collect())
            .collect();

        for si in 0..num_shards {
            let shard = &self.index.shards[si];
            if !shard.device.is_cuda() {
                continue; // CPU shards are handled by process_cpu_shards
            }

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

                let s_cids_v: Vec<i64> = cells.iter().map(|&i| ids_vec[i]).collect();
                let s_scores_v: Vec<f32> = cells.iter().map(|&i| sc_vec[i]).collect();
                let s_tok_v: Vec<i64> = cells.iter().map(|&i| tok_vec[i]).collect();

                let s_cids = Tensor::from_slice(&s_cids_v).to_device(shard.device);
                let s_cscores = Tensor::from_slice(&s_scores_v).to_device(shard.device);
                let s_query = batch_queries.i(b as i64).to_device(shard.device);
                let token_indices = Tensor::from_slice(&s_tok_v).to_device(shard.device);

                let query_idx = c + b;
                let subset: Option<&[i64]> = match subsets {
                    None => None,
                    Some(lists) if lists.len() == 1 => Some(&lists[0]),
                    Some(lists) => Some(&lists[query_idx]),
                };

                let d = self.decompressors[si].decompress_centroids_for_shard(
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

            // Batched D2H for this CUDA shard: one `cat` + one D2H per
            // column, then split back to per-query ShardOut::Vecs. Replaces
            // 5 × batch_len per-query D2Hs in Phase 3 Pass A1 with 4 big
            // transfers here. Ordering is stable (cat preserves input
            // order), so Phase 3 reads identical data to before.
            //
            // `offsets` is not transferred — we rebuild it CPU-side from
            // `sizes` (offsets[i+1] = offsets[i] + sizes[i]), which is
            // the invariant the decompressor maintains.
            let mut active_bs: Vec<usize> = Vec::new();
            let mut caps_refs: Vec<Tensor> = Vec::new();
            let mut sizes_refs: Vec<Tensor> = Vec::new();
            let mut pids_refs: Vec<Tensor> = Vec::new();
            let mut scores_refs: Vec<Tensor> = Vec::new();
            let mut nc_per: Vec<usize> = Vec::new();
            let mut nt_per: Vec<usize> = Vec::new();

            for b in 0..batch_len {
                if let Some(ShardOut::Device(cap, sz, _off, pid, sc)) = shard_outs[b][si].take() {
                    let nc = cap.size()[0] as usize;
                    let nt = pid.size()[0] as usize;
                    active_bs.push(b);
                    nc_per.push(nc);
                    nt_per.push(nt);
                    caps_refs.push(cap);
                    sizes_refs.push(sz);
                    pids_refs.push(pid);
                    scores_refs.push(sc);
                }
            }

            if active_bs.is_empty() {
                continue;
            }

            let caps_ref_view: Vec<&Tensor> = caps_refs.iter().collect();
            let sizes_ref_view: Vec<&Tensor> = sizes_refs.iter().collect();
            let pids_ref_view: Vec<&Tensor> = pids_refs.iter().collect();
            let scores_ref_view: Vec<&Tensor> = scores_refs.iter().collect();

            let caps_cat = Tensor::cat(&caps_ref_view, 0);
            let sizes_cat = Tensor::cat(&sizes_ref_view, 0).to_kind(Kind::Int);
            let pids_cat = Tensor::cat(&pids_ref_view, 0).to_kind(Kind::Int64);
            let scores_cat = Tensor::cat(&scores_ref_view, 0).to_kind(Kind::Float);

            let caps_vec: Vec<i64> = caps_cat.to_device(Device::Cpu).try_into()?;
            let sizes_vec: Vec<i32> = sizes_cat.to_device(Device::Cpu).try_into()?;
            let pids_vec: Vec<i64> = pids_cat.to_device(Device::Cpu).try_into()?;
            let scores_vec: Vec<f32> = scores_cat.to_device(Device::Cpu).try_into()?;

            let mut cell_off = 0usize;
            let mut cand_off = 0usize;
            for (i, &b) in active_bs.iter().enumerate() {
                let nc = nc_per[i];
                let nt = nt_per[i];

                let caps_slice = caps_vec[cell_off..cell_off + nc].to_vec();
                let sizes_slice = sizes_vec[cell_off..cell_off + nc].to_vec();
                let pids_slice = pids_vec[cand_off..cand_off + nt].to_vec();
                let scores_slice = scores_vec[cand_off..cand_off + nt].to_vec();

                // Reconstruct per-query offsets from sizes (0, cumsum).
                let mut offsets_slice = vec![0i64; nc + 1];
                for j in 0..nc {
                    offsets_slice[j + 1] = offsets_slice[j] + sizes_slice[j] as i64;
                }

                shard_outs[b][si] = Some(ShardOut::Vecs(
                    caps_slice,
                    sizes_slice,
                    offsets_slice,
                    pids_slice,
                    scores_slice,
                ));

                cell_off += nc;
                cand_off += nt;
            }
        }

        // centroid_scores and batch_mask go out of scope here — GPU memory
        // for the large [B, T, C] einsum output is freed before the next
        // batch's prep allocates its own.
        drop(centroid_scores);
        drop(batch_mask);

        Ok(PreparedBatch {
            c,
            batch_len,
            batch_queries: ReadOnlyTensor(batch_queries),
            mse_estimates,
            ids_vecs,
            scores_vecs,
            token_indices_vecs,
            shard_cells,
            num_cells,
            batch_results,
            shard_outs,
        })
    }

    /// Batch-level multi-shard processing with cross-batch pipelining.
    ///
    /// Timeline per steady-state iteration:
    /// ```text
    ///   main:  [prep N+1 (GPU: ein+P1+P2c)]       [Phase 3 N]
    ///   pool:                [P2_cpu N (rayon)]
    /// ```
    /// While batch N's CPU-shard decompression saturates the rayon pool,
    /// the main thread calls `prepare_batch(N+1)` — the GPU is mostly idle
    /// during P2_cpu, so its einsum + Phase 1 + Phase 2 CUDA launches slot
    /// neatly into that window. Phase 3 for batch N runs after the scope,
    /// once both N+1's prep and N's CPU work have completed.
    fn rank_multi_shard(
        &self,
        query: &Query,
        masks: &ReadOnlyTensor,
        subsets: Option<&[Vec<i64>]>,
        k: usize,
        bar: &ProgressBar,
    ) -> Result<Vec<SearchResult>> {
        let n_queries = query.embeddings.size()[0] as usize;
        let mut all_results = Vec::with_capacity(n_queries);
        if n_queries == 0 {
            return Ok(all_results);
        }

        let merge_device = self.scoring_device;
        let nprobe = self.config.nprobe as usize;
        let num_shards = self.index.shards.len();

        // Kick off the first batch. Subsequent batches are prepared inside
        // the scope (pipelined with the previous batch's CPU work).
        let mut prepared: Option<PreparedBatch> =
            Some(self.prepare_batch(query, masks, subsets, 0, k, bar)?);

        while let Some(current) = prepared.take() {
            let next_c = current.c + current.batch_len;
            let has_next = next_c < n_queries;

            // Channel for returning CPU shard results from the spawned task.
            type CpuOut = Result<Vec<Vec<(usize, ShardOut)>>>;
            let (cpu_tx, cpu_rx) = std::sync::mpsc::sync_channel::<CpuOut>(1);
            let mut next_prep_result: Option<Result<PreparedBatch>> = None;

            // in_place_scope runs the op on the calling thread, so we can
            // freely use &self here. The spawn body moves to a pool thread,
            // so it only captures Send values.
            self.thread_pool.in_place_scope(|s| {
                // Capture Send-safe handles for the spawn closure.
                let shards = &self.index.shards;
                let decompressors = self.decompressors.clone();
                let bw = ReadOnlyTensor(self.index.shared.bucket_weights.shallow_clone());
                let shard_cells = &current.shard_cells;
                let ids_vecs = &current.ids_vecs;
                let scores_vecs = &current.scores_vecs;
                let token_indices_vecs = &current.token_indices_vecs;
                let batch_queries = &current.batch_queries;
                let batch_len = current.batch_len;
                let c = current.c;

                s.spawn(move |_| {
                    let result: CpuOut = (|| {
                        let mut per_shard: Vec<Vec<(usize, ShardOut)>> =
                            (0..num_shards).map(|_| Vec::new()).collect();

                        for si in 0..num_shards {
                            let shard = &shards[si];
                            if shard.device.is_cuda() {
                                continue;
                            }
                            let decompressor = &decompressors[si];

                            let shard_results: Vec<Result<(usize, ShardOut)>> = (0..batch_len)
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
                                        let ids_vec = ids_vecs[b].as_ref().unwrap();
                                        let sc_v = scores_vecs[b].as_ref().unwrap();
                                        let tok_v = token_indices_vecs[b].as_ref().unwrap();

                                        let s_cids_v: Vec<i64> =
                                            cells.iter().map(|&i| ids_vec[i]).collect();
                                        let s_scores_v: Vec<f32> =
                                            cells.iter().map(|&i| sc_v[i]).collect();
                                        let s_tok_v: Vec<i64> =
                                            cells.iter().map(|&i| tok_v[i]).collect();

                                        let s_cids = Tensor::from_slice(&s_cids_v);
                                        let s_cscores = Tensor::from_slice(&s_scores_v);
                                        let s_query =
                                            batch_queries.i(b as i64).to_device(shard.device);
                                        let token_indices = Tensor::from_slice(&s_tok_v);

                                        let query_idx = c + b;
                                        let subset: Option<&[i64]> = match subsets {
                                            None => None,
                                            Some(lists) if lists.len() == 1 => Some(&lists[0]),
                                            Some(lists) => Some(&lists[query_idx]),
                                        };

                                        let d = decompressor.decompress_centroids_for_shard(
                                            &s_cids,
                                            &s_cscores,
                                            shard,
                                            &bw,
                                            &s_query,
                                            nprobe,
                                            subset,
                                            Some(&token_indices),
                                        )?;

                                        Ok((
                                            b,
                                            ShardOut::Vecs(
                                                d.capacities.try_into()?,
                                                d.sizes.to_kind(Kind::Int).try_into()?,
                                                d.offsets.try_into()?,
                                                d.passage_ids.to_kind(Kind::Int64).try_into()?,
                                                d.scores.to_kind(Kind::Float).try_into()?,
                                            ),
                                        ))
                                    })())
                                })
                                .collect();

                            for res in shard_results {
                                let (b, out) = res?;
                                per_shard[si].push((b, out));
                            }
                        }
                        Ok(per_shard)
                    })();
                    let _ = cpu_tx.send(result);
                });

                // Main thread: while P2_cpu runs on the pool, prep the next
                // batch (einsum + Phase 1 + Phase 2 CUDA launches).
                if has_next {
                    next_prep_result =
                        Some(self.prepare_batch(query, masks, subsets, next_c, k, bar));
                }
            });

            // Scope has joined: both the CPU work and the next prep are done.
            let cpu_per_shard = cpu_rx
                .recv()
                .map_err(|_| anyhow!("CPU shard worker disconnected"))??;

            // Merge CPU shard results into current.shard_outs.
            let mut current = current;
            for (si, cpu_outs) in cpu_per_shard.into_iter().enumerate() {
                for (b, out) in cpu_outs {
                    current.shard_outs[b][si] = Some(out);
                }
            }

            // Stage next iteration's input.
            prepared = match next_prep_result {
                Some(Ok(p)) => Some(p),
                Some(Err(e)) => return Err(e),
                None => None,
            };

            // Assembly + (deferred-D2H) merge
            // Four-pass structure when merge_device is CUDA:
            //   A1: per-query CPU assembly into *flat master arrays*
            //       (caps/sizes/pids/scores concatenated across queries),
            //       with per-query offsets recorded.
            //   A2: ONE H2D per column for the whole batch (~4 transfers
            //       total instead of 4 × batch_len per-query transfers).
            //   A3: per-query narrow() views into the big GPU tensors +
            //       launch per-query merger with deferred D2H.
            //   B:  D2H the deferred GPU top-k results per query.
            // Per-query merger inputs are byte-identical to before — just
            // sliced from a big H2D'd tensor instead of many small ones.
            // No computation change → retrieval metrics unchanged.
            let PreparedBatch {
                c,
                batch_len,
                mse_estimates,
                shard_cells,
                num_cells,
                mut batch_results,
                mut shard_outs,
                ..
            } = current;

            let use_defer_d2h = merge_device.is_cuda();

            // Pass A1a: parallel per-query CPU assembly
            // After Lever C, shard_outs contains only ShardOut::Vecs for
            // CUDA shards (D2H already done in prepare_batch). Each query's
            // assembly is now pure CPU work — independent, embarrassingly
            // parallel across the rayon pool. We collect per-query flat
            // Vecs; a quick serial pass afterwards concatenates them into
            // master arrays (preserving batch order for the merger).
            type QueryFlats = (Vec<i64>, Vec<i32>, Vec<i64>, Vec<f32>);
            let per_query: Vec<Option<QueryFlats>> = self.thread_pool.install(|| {
                shard_outs
                    .par_iter_mut()
                    .enumerate()
                    .map(|(b, shard_outs_b)| -> Option<QueryFlats> {
                        if batch_results[b].is_some() {
                            return None;
                        }
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
                            let (cv, sv, ov, pv, scv) = match shard_outs_b[si].take() {
                                // CUDA-shard Device variants were converted
                                // to Vecs in prepare_batch — this arm is a
                                // safety net only.
                                Some(ShardOut::Device(cap, sz, off, pid, sc)) => {
                                    let cv: Vec<i64> =
                                        cap.to_device(Device::Cpu).try_into().ok()?;
                                    let sv: Vec<i32> = sz
                                        .to_device(Device::Cpu)
                                        .to_kind(Kind::Int)
                                        .try_into()
                                        .ok()?;
                                    let ov: Vec<i64> =
                                        off.to_device(Device::Cpu).try_into().ok()?;
                                    let pv: Vec<i64> = pid
                                        .to_device(Device::Cpu)
                                        .to_kind(Kind::Int64)
                                        .try_into()
                                        .ok()?;
                                    let scv: Vec<f32> = sc
                                        .to_device(Device::Cpu)
                                        .to_kind(Kind::Float)
                                        .try_into()
                                        .ok()?;
                                    (cv, sv, ov, pv, scv)
                                },
                                Some(ShardOut::Vecs(cv, sv, ov, pv, scv)) => (cv, sv, ov, pv, scv),
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

                        Some((g_cap, g_sizes, flat_pids, flat_scores))
                    })
                    .collect()
            });

            // Pass A1b: serial merge into master arrays
            // Fast: just Vec::extend in batch order. Preserves the
            // per-query order that the merger sees.
            let mut active: Vec<usize> = Vec::with_capacity(batch_len);
            let mut all_caps: Vec<i64> = Vec::new();
            let mut all_sizes: Vec<i32> = Vec::new();
            let mut all_pids: Vec<i64> = Vec::new();
            let mut all_scores: Vec<f32> = Vec::new();
            let mut cell_offsets: Vec<usize> = vec![0];
            let mut cand_offsets: Vec<usize> = vec![0];

            for (b, result) in per_query.into_iter().enumerate() {
                if let Some((g_cap, g_sizes, flat_pids, flat_scores)) = result {
                    all_caps.extend_from_slice(&g_cap);
                    all_sizes.extend_from_slice(&g_sizes);
                    all_pids.extend(flat_pids);
                    all_scores.extend(flat_scores);
                    cell_offsets.push(all_caps.len());
                    cand_offsets.push(all_pids.len());
                    active.push(b);
                }
            }

            // Pass A results (CUDA path only): GPU tensors to D2H in pass B.
            let mut gpu_topk: Vec<Option<(Tensor, Tensor)>> = if use_defer_d2h {
                (0..batch_len).map(|_| None).collect()
            } else {
                Vec::new()
            };

            if !active.is_empty() {
                // Pass A2: batched H2D
                // Empty-array `from_slice` → empty tensors, handled by
                // `active.is_empty()` guard above.
                let caps_all_t = Tensor::from_slice(&all_caps)
                    .to_kind(Kind::Int64)
                    .to_device(merge_device);
                let sizes_all_t = Tensor::from_slice(&all_sizes)
                    .to_kind(Kind::Int)
                    .to_device(merge_device);
                let pids_all_t = Tensor::from_slice(&all_pids)
                    .to_kind(Kind::Int64)
                    .to_device(merge_device);
                let scores_all_t = Tensor::from_slice(&all_scores)
                    .to_kind(Kind::Float)
                    .to_device(merge_device);

                // Pass A3: per-query slice + merger launch
                for (ai, &b) in active.iter().enumerate() {
                    let cell_start = cell_offsets[ai] as i64;
                    let cell_len = (cell_offsets[ai + 1] - cell_offsets[ai]) as i64;
                    let cand_start = cand_offsets[ai] as i64;
                    let cand_len = (cand_offsets[ai + 1] - cand_offsets[ai]) as i64;

                    let cap_t = caps_all_t.narrow(0, cell_start, cell_len);
                    let sz_t = sizes_all_t.narrow(0, cell_start, cell_len);
                    let pid_t = pids_all_t.narrow(0, cand_start, cand_len);
                    let sc_t = scores_all_t.narrow(0, cand_start, cand_len);

                    let mse = mse_estimates[b].as_ref().unwrap();
                    let mse_dev = mse.to_device(merge_device);

                    if use_defer_d2h {
                        let (pids_gpu, scores_gpu) = self.merger.merge_candidate_scores_defer_d2h(
                            &sz_t, &pid_t, &sc_t, &mse_dev, nprobe, k,
                        )?;
                        gpu_topk[b] = Some((pids_gpu, scores_gpu));
                    } else {
                        let (pids, scores) = self.merger.merge_candidate_scores(
                            &cap_t, &sz_t, &pid_t, &sc_t, &mse_dev, nprobe, k,
                        )?;
                        batch_results[b] = Some(SearchResult {
                            passage_ids: pids,
                            scores,
                            query_id: c + b + 1,
                        });
                        bar.inc(1);
                    }
                }
            }

            // Pass B: D2H the deferred GPU results (CUDA path only).
            if use_defer_d2h {
                for b in 0..batch_len {
                    if batch_results[b].is_some() {
                        continue;
                    }
                    let (pids_gpu, scores_gpu) = gpu_topk[b].take().unwrap();
                    let pids: Vec<i64> = pids_gpu.to_device(Device::Cpu).try_into()?;
                    let scores: Vec<f32> = scores_gpu
                        .to_kind(Kind::Float)
                        .to_device(Device::Cpu)
                        .try_into()?;
                    batch_results[b] = Some(SearchResult {
                        passage_ids: pids,
                        scores,
                        query_id: c + b + 1,
                    });
                    bar.inc(1);
                }
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

        // Multi-shard: batch-level shard processing with pipelining.
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

                        let centroid_scores =
                            query_embeddings.matmul(&index.shared.centroids.transpose(0, 1));
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

                        let decompressed = decompressor.decompress_centroids_for_shard(
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
