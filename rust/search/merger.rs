use anyhow::Result;
use std::collections::BTreeMap;
use std::ptr;
use tch::{no_grad, Device, Kind, Tensor};
use torch_sys::C_tensor;

use crate::utils::types::{PassageId, Score};

//TODO num tokens can't be hardcoded to 32

extern "C" {
    fn xtr_warp_gpu_merge(
        candidate_sizes: *const C_tensor,
        candidate_pids: *const C_tensor,
        candidate_scores: *const C_tensor,
        mse_estimates: *const C_tensor,
        nprobe: i64,
        k: i64,
        max_candidates: i64,
        out_pids: *mut *mut C_tensor,
        out_scores: *mut *mut C_tensor,
    );
}

/// Configuration for the merger
#[derive(Debug, Clone)]
pub struct MergerConfig {
    /// Maximum number of candidates to keep during merging
    pub max_candidates: usize,

    /// Number of threads for parallel operations
    pub num_threads: usize,

    /// Device to use
    pub device: Device,
}

/// Represents a view into strided candidate data
#[derive(Clone)]
pub struct AnnotatedStrideView {
    /// Passage IDs
    pub pids: Vec<PassageId>,

    /// Scores
    pub scores: Vec<Score>,

    /// Actual size of valid data
    pub size: usize,
}

impl AnnotatedStrideView {
    /// Create a new stride view with given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        AnnotatedStrideView {
            pids: vec![0; capacity],
            scores: vec![0.0; capacity],
            size: 0,
        }
    }

    /// Create from existing data
    pub fn from_data(pids: Vec<PassageId>, scores: Vec<Score>, size: usize) -> Self {
        AnnotatedStrideView { pids, scores, size }
    }
}

/// Combiner for max-reduction of token-level scores from different clusters
struct ReduceMaxCombiner;

/// Combiner for sum-reduction with MSE correction
struct ReduceSumMseCombiner {
    lhs_mse: f32,
    rhs_mse: f32,
}

impl ReduceSumMseCombiner {
    fn new(lhs_mse: f32, rhs_mse: f32) -> Self {
        ReduceSumMseCombiner { lhs_mse, rhs_mse }
    }
}

/// Main merger struct for combining results from multiple sources
#[derive(Clone)]
pub struct ResultMerger {
    config: MergerConfig,
}

impl ResultMerger {
    /// Creates a new result merger with the given configuration
    pub fn new(config: MergerConfig) -> Self {
        ResultMerger { config }
    }

    /// Merges two candidate strides with specific combiner logic
    fn merge_candidate_strides_with_combiner<C>(
        stride1: &AnnotatedStrideView,
        stride2: &AnnotatedStrideView,
        result: &mut AnnotatedStrideView,
        combiner: &C,
    ) where
        C: Combiner,
    {
        let c1_size = stride1.size;
        let c2_size = stride2.size;
        let mut result_size = 0;
        let mut i1 = 0;
        let mut i2 = 0;

        // Ensure result has enough capacity
        if result.pids.len() < c1_size + c2_size {
            result.pids.resize(c1_size + c2_size, 0);
            result.scores.resize(c1_size + c2_size, 0.0);
        }

        while i1 < c1_size && i2 < c2_size {
            let key1 = stride1.pids[i1];
            let key2 = stride2.pids[i2];
            result.pids[result_size] = key1.min(key2);

            if key1 == key2 {
                result.scores[result_size] =
                    combiner.combine(stride1.scores[i1], stride2.scores[i2]);
                i1 += 1;
                i2 += 1;
            } else if key1 < key2 {
                result.scores[result_size] = combiner.lhs(stride1.scores[i1]);
                i1 += 1;
            } else {
                result.scores[result_size] = combiner.rhs(stride2.scores[i2]);
                i2 += 1;
            }
            result_size += 1;
        }

        // Copy remaining elements from stride1
        while i1 < c1_size {
            result.pids[result_size] = stride1.pids[i1];
            result.scores[result_size] = combiner.lhs(stride1.scores[i1]);
            i1 += 1;
            result_size += 1;
        }

        // Copy remaining elements from stride2
        while i2 < c2_size {
            result.pids[result_size] = stride2.pids[i2];
            result.scores[result_size] = combiner.rhs(stride2.scores[i2]);
            i2 += 1;
            result_size += 1;
        }

        result.size = result_size;
    }

    /// Copies a candidate stride to another
    fn copy_candidate_stride(source: &AnnotatedStrideView, destination: &mut AnnotatedStrideView) {
        let size = source.size;
        if destination.pids.len() < size {
            destination.pids.resize(size, 0);
            destination.scores.resize(size, 0.0);
        }
        destination.size = size;
        destination.pids[..size].copy_from_slice(&source.pids[..size]);
        destination.scores[..size].copy_from_slice(&source.scores[..size]);
    }

    /// Merge the `nprobe` candidate lists associated with a specific token index
    pub fn merge_candidates_nprobe(
        views: &mut Vec<AnnotatedStrideView>,
        views_buffer: &mut Vec<AnnotatedStrideView>,
        nprobe: usize,
        query_token_idx: usize,
    ) -> usize {
        let mut num_iterations = 0;
        let begin = query_token_idx * nprobe;
        let mut buf1 = views;
        let mut buf2 = views_buffer;
        let combiner = ReduceMaxCombiner;

        let mut step_size = 1;
        while step_size < nprobe {
            for lhs in (0..nprobe).step_by(step_size * 2) {
                let rhs = lhs + step_size;
                if rhs < nprobe {
                    Self::merge_candidate_strides_with_combiner(
                        &buf1[begin + lhs],
                        &buf1[begin + rhs],
                        &mut buf2[begin + lhs],
                        &combiner,
                    );
                } else {
                    // No merge partner, copy as-is
                    Self::copy_candidate_stride(&buf1[begin + lhs], &mut buf2[begin + lhs]);
                }
            }
            // Swap buffers
            std::mem::swap(&mut buf1, &mut buf2);
            step_size <<= 1;
            num_iterations += 1;
        }

        num_iterations
    }

    /// Merge the 32 strides of token-level scores into a single stride of document-level scores
    pub fn merge_candidates_tokens(
        views: &mut Vec<AnnotatedStrideView>,
        views_buffer: &mut Vec<AnnotatedStrideView>,
        nprobe: usize,
        mse_estimates: &[f32],
    ) {
        const NUM_TOKENS: usize = 32;

        // Compute MSE prefix sums
        let mut mse_prefix = vec![0.0; NUM_TOKENS + 1];
        for i in 0..NUM_TOKENS {
            mse_prefix[i + 1] = mse_prefix[i] + mse_estimates.get(i).unwrap_or(&0.0);
        }

        let mut step_size = 1;
        while step_size < NUM_TOKENS {
            for lhs in (0..NUM_TOKENS).step_by(step_size * 2) {
                let rhs = lhs + step_size;
                if rhs < NUM_TOKENS {
                    // Calculate MSE values using prefix sums
                    let lhs_mse = mse_prefix[rhs] - mse_prefix[lhs];
                    let rhs_mse = mse_prefix[(rhs + step_size).min(NUM_TOKENS)] - mse_prefix[rhs];

                    let combiner = ReduceSumMseCombiner::new(lhs_mse, rhs_mse);
                    Self::merge_candidate_strides_with_combiner(
                        &views[lhs * nprobe],
                        &views[rhs * nprobe],
                        &mut views_buffer[lhs * nprobe],
                        &combiner,
                    );
                } else {
                    Self::copy_candidate_stride(
                        &views[lhs * nprobe],
                        &mut views_buffer[lhs * nprobe],
                    );
                }
            }
            std::mem::swap(views, views_buffer);
            step_size <<= 1;
        }
    }

    /// Partial sort results to get top-k candidates
    fn partial_sort_results(stride: &AnnotatedStrideView, num_results: usize) -> Vec<usize> {
        let size = stride.size;
        let mut pid_idx: Vec<usize> = (0..size).collect();

        let scores = &stride.scores;
        pid_idx[..num_results.min(size)].select_nth_unstable_by(
            num_results.min(size) - 1,
            |&idx1, &idx2| {
                let score1 = scores[idx1];
                let score2 = scores[idx2];
                // Sort descending by score, with tie-breaking on index
                score2
                    .partial_cmp(&score1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| idx1.cmp(&idx2))
            },
        );

        // Sort the top-k elements
        pid_idx[..num_results.min(size)].sort_unstable_by(|&idx1, &idx2| {
            let score1 = scores[idx1];
            let score2 = scores[idx2];
            score2
                .partial_cmp(&score1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| idx1.cmp(&idx2))
        });

        pid_idx
    }

    /// Merges candidate scores from multiple centroids/sources
    pub fn merge_candidate_scores(
        &self,
        capacities: &Tensor,
        candidate_sizes: &Tensor,
        candidate_pids: &Tensor,
        candidate_scores: &Tensor,
        candidate_token_indices: Option<&Tensor>,
        mse_estimates: &Tensor,
        nprobe: usize,
        k: usize,
    ) -> Result<(Vec<PassageId>, Vec<Score>)> {
        // If tensors are on CUDA and the GPU merge kernel is available, use it.
        if self.config.device.is_cuda() {
            if let Ok((pids, scores)) = self.merge_candidate_scores_cuda(
                candidate_sizes,
                candidate_pids,
                candidate_scores,
                candidate_token_indices,
                mse_estimates,
                nprobe,
                k,
                self.config.max_candidates,
            ) {
                return Ok((pids, scores));
            }
        }

        no_grad(|| {
            let num_cells = capacities.size()[0] as usize;
            let num_candidates = candidate_pids.size()[0] as usize;

            // convert tensors to vectors for creating stride views
            // we used to have .to_kind conversions here, bring them back if needed
            let sizes_vec: Vec<i32> = candidate_sizes.try_into()?;
            let pids_vec: Vec<PassageId> = candidate_pids.try_into()?;
            let scores_vec: Vec<Score> = candidate_scores.try_into()?;
            let mse_vec: Vec<f32> = mse_estimates.try_into()?;

            // Create strided views into the data (each view represents one centroid's candidates)
            let mut views = Vec::new();
            let mut offset = 0;
            let mut total_dedup_size = 0usize;
            for i in 0..num_cells {
                let size = sizes_vec[i] as usize;

                // Use size instead of capacity since the data is deduplicated and compacted
                let end = (offset + size).min(num_candidates);
                let mut pid_score_map: BTreeMap<PassageId, Score> = BTreeMap::new();

                for idx in offset..end {
                    let pid = pids_vec[idx];
                    let score = scores_vec[idx];
                    pid_score_map
                        .entry(pid)
                        .and_modify(|existing| {
                            if score > *existing {
                                *existing = score;
                            }
                        })
                        .or_insert(score);
                }

                let dedup_size = pid_score_map.len();
                total_dedup_size += dedup_size;

                let mut dedup_pids = Vec::with_capacity(dedup_size);
                let mut dedup_scores = Vec::with_capacity(dedup_size);
                for (pid, score) in pid_score_map {
                    dedup_pids.push(pid);
                    dedup_scores.push(score);
                }

                views.push(AnnotatedStrideView::from_data(
                    dedup_pids,
                    dedup_scores,
                    dedup_size,
                ));
                offset += size;
            }

            // Create buffer views for merging
            // We need to create views that can handle the worst-case merge scenario
            // When merging in a tree-like fashion, the maximum size at any level
            // is the sum of all individual sizes
            let max_merged_size = total_dedup_size.max(1);
            let mut views_buffer = Vec::new();
            for _ in 0..num_cells {
                // Each buffer view needs enough capacity to hold the fully merged result
                // in the worst case (no deduplication during merge)
                views_buffer.push(AnnotatedStrideView::with_capacity(max_merged_size));
            }

            // Merge candidates for each token
            let mut last_num_iterations = 0;
            for query_token_idx in 0..32 {
                // TODO: Add early stopping here for queries with fewer than 32 tokens
                last_num_iterations = Self::merge_candidates_nprobe(
                    &mut views,
                    &mut views_buffer,
                    nprobe,
                    query_token_idx,
                );
            }

            // If we performed an odd number of iterations, the scratch buffer contains the result
            if last_num_iterations % 2 != 0 {
                std::mem::swap(&mut views, &mut views_buffer);
            }

            // Merge token-level scores into document-level scores
            Self::merge_candidates_tokens(&mut views, &mut views_buffer, nprobe, &mse_vec);

            // Get top-k results from the first stride (which contains the final merged results)
            let budget = self.config.max_candidates.min(views[0].size).max(k);
            let top_idx = Self::partial_sort_results(&views[0], budget);

            // Extract the top-k PIDs and scores
            let mut result_pids = vec![0i64; budget];
            let mut result_scores = vec![0.0f32; budget];
            let limit = top_idx.len();

            for i in 0..budget {
                if i >= limit {
                    break;
                }
                let idx = top_idx[i];
                // let k_idx = &top_idx[..k.min(top_idx.len())];
                result_pids[i] = views[0].pids[idx];
                result_scores[i] = views[0].scores[idx];
            }

            Ok((result_pids, result_scores))
        })
    }

    fn merge_candidate_scores_cuda(
        &self,
        candidate_sizes: &Tensor,
        candidate_pids: &Tensor,
        candidate_scores: &Tensor,
        candidate_token_indices: Option<&Tensor>,
        mse_estimates: &Tensor,
        nprobe: usize,
        k: usize,
        max_candidates: usize,
    ) -> Result<(Vec<PassageId>, Vec<Score>)> {
        let kNumTokens = 32i64;
        if candidate_pids.numel() == 0 {
            let empty_pid: Vec<i64> = Vec::new();
            let empty_scores: Vec<f32> = Vec::new();
            return Ok((empty_pid, empty_scores));
        }

        let device = candidate_pids.device();
        let sizes = candidate_sizes.shallow_clone();
        let pids = candidate_pids.shallow_clone();
        let scores = candidate_scores.shallow_clone();
        let mse_estimates = mse_estimates.shallow_clone();

        let candidate_tokens = match candidate_token_indices {
            Some(t) => t.shallow_clone(),
            None => {
                // Token index per cell, repeated per candidate
                let num_cells = sizes.size()[0];
                let mut token_indices = Tensor::arange(num_cells, (Kind::Int64, device));
                token_indices = token_indices.divide_scalar_mode(nprobe as i64, "trunc");
                Tensor::repeat_interleave_self_tensor(&token_indices, &sizes, 0, None)
            },
        };

        // Flatten token+pid into a combined id to compactly deduplicate
        let combined_ids = pids * kNumTokens + candidate_tokens;
        let sort_result = combined_ids.sort(0, /*descending=*/ false);
        let sorted_ids = sort_result.0;
        let sort_idx = sort_result.1;
        let sorted_scores = scores.index_select(0, &sort_idx);

        // Unique ids and inverse for max reduction
        let unique_result = sorted_ids.unique_consecutive(
            /*return_inverse=*/ true, /*return_counts=*/ false, 0,
        );
        let unique_ids = unique_result.0;
        let inverse = unique_result.1;
        let max_init = Tensor::full(
            unique_ids.size(),
            f64::NEG_INFINITY,
            (sorted_scores.kind(), sorted_scores.device()),
        );
        let max_per_id = Tensor::index_reduce(
            &max_init,
            0,
            &inverse,
            &sorted_scores,
            "amax",
            /*include_self=*/ true,
        );

        // Split combined id back into pid and token
        let pid = unique_ids
            .divide_scalar_mode(kNumTokens, "trunc")
            .to_kind(Kind::Int64);
        let token = (unique_ids - pid.shallow_clone() * kNumTokens).to_kind(Kind::Int64);

        // Prepare MSE vector (pad to kNumTokens if needed)
        let mut mse = mse_estimates.shallow_clone();
        if mse.size()[0] < kNumTokens {
            let pad = Tensor::zeros(&[kNumTokens - mse.size()[0]], (mse.kind(), mse.device()));
            mse = Tensor::cat(&[mse, pad], 0);
        }
        mse = mse.narrow(0, 0, kNumTokens);

        let sum_mse = mse.sum(None);
        let mse_for_tokens = mse.index_select(0, &token);
        let delta = max_per_id - mse_for_tokens;

        // Reduce by pid: since keys were sorted, pid is non-decreasing
        let pid_result = &pid.unique_consecutive(
            /*return_inverse=*/ true, /*return_counts=*/ true, 0,
        );
        let unique_pids = pid_result.0.shallow_clone();
        let pid_counts = pid_result.2.to_kind(Kind::Int64);

        // Deterministic per-PID sum using cumulative sums to avoid nondeterministic atomics.
        // Exclusive prefix to get exact per-pid sums.
        let delta_cumsum = delta.cumsum(0, delta.kind()).contiguous();
        let mut prefix = Tensor::zeros(
            &[delta_cumsum.size()[0] + 1],
            (delta_cumsum.kind(), delta_cumsum.device()),
        );
        prefix
            .narrow(0, 1, delta_cumsum.size()[0])
            .copy_(&delta_cumsum);

        let end_indices = pid_counts.cumsum(0, Kind::Int64) - 1;
        let start_indices = end_indices.shallow_clone() - pid_counts + 1;
        let sums_at_end = prefix.index_select(0, &(end_indices + 1));
        let sums_before = prefix.index_select(0, &start_indices);
        let deltas_per_pid = sums_at_end - sums_before;
        let totals = deltas_per_pid + sum_mse;

        if totals.numel() == 0 {
            let empty_pid: Vec<i64> = Vec::new();
            let empty_scores_out: Vec<f32> = Vec::new();
            return Ok((empty_pid, empty_scores_out));
        }

        let totals_size = totals.size();
        let num_candidates = totals_size.get(0).unwrap();
        let budget = k.max(max_candidates.min(*num_candidates as usize));
        let take_k = (budget as i64).min(*num_candidates);

        let topk = totals.topk(take_k, 0, /*largest=*/ true, /*sorted=*/ true);
        let top_scores: Vec<f32> = topk.0.to_device(Device::Cpu).try_into()?;
        let top_indices = topk.1;
        let top_pids: Vec<i64> = unique_pids
            .index_select(0, &top_indices)
            .to_device(Device::Cpu)
            .try_into()?;

        return Ok((top_pids, top_scores));
    }

    fn merge_candidate_scores_cuda_cpp(
        candidate_sizes: &Tensor,
        candidate_pids: &Tensor,
        candidate_scores: &Tensor,
        mse_estimates: &Tensor,
        nprobe: usize,
        k: usize,
        max_candidates: usize,
    ) -> Result<(Vec<PassageId>, Vec<Score>)> {
        unsafe {
            let mut out_pids: *mut C_tensor = ptr::null_mut();
            let mut out_scores: *mut C_tensor = ptr::null_mut();

            xtr_warp_gpu_merge(
                candidate_sizes.as_ptr() as *const C_tensor,
                candidate_pids.as_ptr() as *const C_tensor,
                candidate_scores.as_ptr() as *const C_tensor,
                mse_estimates.as_ptr() as *const C_tensor,
                nprobe as i64,
                k as i64,
                max_candidates as i64,
                &mut out_pids,
                &mut out_scores,
            );

            if out_pids.is_null() || out_scores.is_null() {
                anyhow::bail!("xtr_warp_gpu_merge returned null outputs");
            }

            let top_pids = Tensor::from_ptr(out_pids);
            let top_scores = Tensor::from_ptr(out_scores);
            let pids: Vec<i64> = top_pids.to_device(Device::Cpu).try_into()?;
            let scores: Vec<f32> = top_scores.to_device(Device::Cpu).try_into()?;
            Ok((pids, scores))
        }
    }
}

/// Trait for combiners
trait Combiner {
    fn combine(&self, lhs: f32, rhs: f32) -> f32;
    fn lhs(&self, lhs: f32) -> f32;
    fn rhs(&self, rhs: f32) -> f32;
}

impl Combiner for ReduceMaxCombiner {
    fn combine(&self, lhs: f32, rhs: f32) -> f32 {
        lhs.max(rhs)
    }

    fn lhs(&self, lhs: f32) -> f32 {
        lhs
    }

    fn rhs(&self, rhs: f32) -> f32 {
        rhs
    }
}

impl Combiner for ReduceSumMseCombiner {
    fn combine(&self, lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }

    fn lhs(&self, lhs: f32) -> f32 {
        lhs + self.rhs_mse
    }

    fn rhs(&self, rhs: f32) -> f32 {
        self.lhs_mse + rhs
    }
}
