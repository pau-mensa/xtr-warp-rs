use anyhow::Result;
use tch::{no_grad, Device, Kind, Tensor};

use crate::utils::types::{PassageId, Score};

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
        num_tokens: usize,
    ) {
        // Compute MSE prefix sums
        let mut mse_prefix = vec![0.0; num_tokens + 1];
        for i in 0..num_tokens {
            mse_prefix[i + 1] = mse_prefix[i] + mse_estimates.get(i).unwrap_or(&0.0);
        }

        let mut step_size = 1;
        while step_size < num_tokens {
            for lhs in (0..num_tokens).step_by(step_size * 2) {
                let rhs = lhs + step_size;
                if rhs < num_tokens {
                    // Calculate MSE values using prefix sums
                    let lhs_mse = mse_prefix[rhs] - mse_prefix[lhs];
                    let rhs_mse = mse_prefix[(rhs + step_size).min(num_tokens)] - mse_prefix[rhs];

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
        let effective = num_results.min(size);
        if effective == 0 {
            return vec![];
        }
        let mut pid_idx: Vec<usize> = (0..size).collect();

        let scores = &stride.scores;
        pid_idx.select_nth_unstable_by(effective - 1, |&idx1, &idx2| {
            let score1 = scores[idx1];
            let score2 = scores[idx2];
            // Sort descending by score, with tie-breaking on index
            score2
                .partial_cmp(&score1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| idx1.cmp(&idx2))
        });

        // Sort the top-k elements
        pid_idx[..effective].sort_unstable_by(|&idx1, &idx2| {
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
        mse_estimates: &Tensor,
        nprobe: usize,
        k: usize,
    ) -> Result<(Vec<PassageId>, Vec<Score>)> {
        // use cuda if possible
        if self.config.device.is_cuda() {
            if let Ok((top_pids_t, top_scores_t)) = self.merge_candidate_scores_cuda(
                candidate_sizes,
                candidate_pids,
                candidate_scores,
                mse_estimates,
                nprobe,
                k,
            ) {
                let top_scores: Vec<f32> = top_scores_t
                    .to_kind(Kind::Float)
                    .to_device(Device::Cpu)
                    .try_into()?;
                let top_pids: Vec<i64> = top_pids_t.to_device(Device::Cpu).try_into()?;
                return Ok((top_pids, top_scores));
            }
        }

        no_grad(|| {
            let num_cells = capacities.size()[0] as usize;
            let num_candidates = candidate_pids.size()[0] as usize;

            // convert tensors to vectors for creating stride views
            let sizes_vec: Vec<i32> = candidate_sizes.try_into()?;
            let pids_vec: Vec<PassageId> = candidate_pids.try_into()?;
            let scores_vec: Vec<Score> = candidate_scores.try_into()?;
            let mse_vec: Vec<f32> = mse_estimates.try_into()?;

            // Create strided views into the data (each view represents one centroid's candidates)
            // Data arrives already sorted by pid and deduped from the decompressor
            let mut views = Vec::new();
            let mut offset = 0;
            for i in 0..num_cells {
                let size = sizes_vec[i] as usize;
                let end = (offset + size).min(num_candidates);

                let cell_pids = pids_vec[offset..end].to_vec();
                let cell_scores = scores_vec[offset..end].to_vec();

                views.push(AnnotatedStrideView::from_data(cell_pids, cell_scores, size));
                offset += size;
            }

            // Create buffer views for merging
            // We need to create views that can handle the worst-case merge scenario
            // When merging in a tree-like fashion, the maximum size at any level
            // is the sum of all individual sizes
            let mut views_buffer = Vec::new();
            for _ in 0..num_cells {
                views_buffer.push(AnnotatedStrideView::with_capacity(0));
            }

            // Merge candidates for each token
            let mut last_num_iterations = 0;
            // IMPORTANT: the original implementation uses a hardcoded constant (32)
            // this destroys retrieval metrics for longer queries, so we infer the number of tokens
            let num_tokens = (num_cells + nprobe - 1) / nprobe;
            for query_token_idx in 0..num_tokens {
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
            Self::merge_candidates_tokens(
                &mut views,
                &mut views_buffer,
                nprobe,
                &mse_vec,
                num_tokens,
            );

            // Get top-k results from the first stride (which contains the final merged results)
            let budget = self.config.max_candidates.min(views[0].size).max(k);
            let top_idx = Self::partial_sort_results(&views[0], budget);

            // Extract the top-k PIDs and scores
            let limit = top_idx.len();
            let mut result_pids = Vec::with_capacity(limit);
            let mut result_scores = Vec::with_capacity(limit);

            for &idx in &top_idx {
                result_pids.push(views[0].pids[idx]);
                result_scores.push(views[0].scores[idx]);
            }

            Ok((result_pids, result_scores))
        })
    }

    /// CUDA merger. Returns results as GPU tensors — callers that need a
    /// `Vec<PassageId>, Vec<Score>` should D2H themselves (see
    /// `merge_candidate_scores`'s CUDA branch). Keeping the output on-device
    /// lets `sharded_scorer` launch many queries' merges back-to-back and
    /// drain with a single D2H at the end.
    ///
    /// Sync-free implementation: zero host-device syncs inside. All output
    /// shapes are `[N]` where `N = candidate_pids.size()[0]` is known
    /// from input, so we never have to read `.size()` on a dynamic-shape
    /// tensor. Over-allocated slots get sentinel values (-inf for scores,
    /// -1 for pids) and are masked out before the final topk.
    ///
    /// Reduction semantics match a conventional `unique_consecutive +
    /// index_reduce` pipeline:
    ///   - same sort keys → same reduction order
    ///   - `scatter_reduce_(amax)` is equivalent to `index_reduce(amax)`
    ///     (max is commutative + associative, bit-identical across orderings)
    ///   - per-pid sum uses `prefix[end+1] - prefix[start]` on the same
    ///     elements in the same order; padding is clamped from -inf to 0
    ///     before cumsum so prefixes through the padding region are 0 and
    ///     don't introduce catastrophic cancellation for real groups.
    ///
    /// Designed to stack with the multi-stream scheduling in
    /// `sharded_scorer::rank_multi_shard` (Pass A3): the extra small
    /// kernels this path introduces relative to a traditional merger run
    /// concurrently across the stream pool, which is what makes the
    /// sync-free approach viable here.
    pub(crate) fn merge_candidate_scores_cuda(
        &self,
        candidate_sizes: &Tensor,
        candidate_pids: &Tensor,
        candidate_scores: &Tensor,
        mse_estimates: &Tensor,
        nprobe: usize,
        k: usize,
    ) -> Result<(Tensor, Tensor)> {
        let max_candidates = self.config.max_candidates;
        let device = candidate_pids.device();

        let n = candidate_pids.size()[0];
        if n == 0 {
            let empty_pids = Tensor::zeros(&[0], (Kind::Int64, device));
            let empty_scores = Tensor::zeros(&[0], (Kind::Float, device));
            return Ok((empty_pids, empty_scores));
        }

        let sizes = candidate_sizes.shallow_clone();
        let pids = candidate_pids.shallow_clone();
        let scores = candidate_scores.shallow_clone();
        let score_kind = scores.kind();

        // Token index per cell, repeated per candidate. Pass `Some(n)` so
        // `repeat_interleave_self_tensor` doesn't have to infer the output
        // size (which would itself force a sync).
        let num_cells = sizes.size()[0];
        let num_tokens = (num_cells + (nprobe as i64) - 1) / (nprobe as i64);
        let mut token_indices = Tensor::arange(num_cells, (Kind::Int64, device));
        token_indices = token_indices.divide_scalar_mode(nprobe as i64, "trunc");
        let candidate_tokens =
            Tensor::repeat_interleave_self_tensor(&token_indices, &sizes, 0, Some(n));

        // Flatten (pid, token) into one int64 key.
        let combined_ids = pids.shallow_clone() * num_tokens + &candidate_tokens;
        let sort_result = combined_ids.sort(0, /*descending=*/ false);
        let sorted_ids = sort_result.0;
        let sort_idx = sort_result.1;
        let sorted_scores = scores.index_select(0, &sort_idx);
        let sorted_pids_by_combined = pids.index_select(0, &sort_idx);
        let sorted_tokens_by_combined = candidate_tokens.index_select(0, &sort_idx);

        // --- Phase 1: dedup (pid, token) → max score (sync-free) ---
        //
        // group_ids[i] = cumulative count of "changes" in sorted_ids up to i.
        // Elements of sorted_ids that are equal get the same group_id.
        let sorted_lhs = sorted_ids.slice(0, 1, n, 1);
        let sorted_rhs = sorted_ids.slice(0, 0, n - 1, 1);
        let changes = sorted_lhs.ne_tensor(&sorted_rhs).to_kind(Kind::Int64);
        let changes_cumsum = changes.cumsum(0, Kind::Int64);
        let zero_head = Tensor::zeros(&[1], (Kind::Int64, device));
        let group_ids = Tensor::cat(&[zero_head.shallow_clone(), changes_cumsum], 0);

        // Max per group into a pre-allocated [N] buffer filled with -inf.
        let mut max_per_group =
            Tensor::full(&[n], f64::NEG_INFINITY, (score_kind, device));
        let _ = max_per_group.internal_scatter_reduce_(
            0,
            &group_ids,
            &sorted_scores,
            "amax",
            /*include_self=*/ true,
        );

        // Scatter pid/token per group. All elements in a group have the
        // same pid+token (that's the group definition), so last-write-wins
        // is correct.
        let mut group_pid = Tensor::full(&[n], -1i64, (Kind::Int64, device));
        let _ = group_pid.scatter_(0, &group_ids, &sorted_pids_by_combined);
        let mut group_token = Tensor::full(&[n], 0i64, (Kind::Int64, device));
        let _ = group_token.scatter_(0, &group_ids, &sorted_tokens_by_combined);

        // --- Phase 2: MSE correction ---
        let mut mse = mse_estimates.shallow_clone();
        if mse.size()[0] < num_tokens {
            let pad = Tensor::zeros(&[num_tokens - mse.size()[0]], (mse.kind(), mse.device()));
            mse = Tensor::cat(&[mse, pad], 0);
        }
        mse = mse.narrow(0, 0, num_tokens);
        let sum_mse = mse.sum(None);

        // Clamp tokens to [0, num_tokens-1] to keep index_select safe for
        // the zero-filled padding entries in group_token.
        let safe_tokens = group_token.clamp(0, num_tokens - 1);
        let mse_for_groups = mse.index_select(0, &safe_tokens);
        let delta = &max_per_group - mse_for_groups;
        // For padding entries, max_per_group = -inf, so delta = -inf.

        // --- Phase 3: per-pid sum via sort(pid) + shift-compare + cumsum ---
        //
        // Ascending sort: -1 padding floats to the front, real pids follow
        // in ascending order. pid_group 0 captures all -1 padding (later
        // masked out).
        let pid_sort = group_pid.argsort(0, /*descending=*/ false);
        let sorted_gpid = group_pid.index_select(0, &pid_sort);
        let sorted_delta = delta.index_select(0, &pid_sort);

        // Replace -inf deltas with 0 before cumsum. Real pid groups' prefix
        // diffs only involve the real portion of the array; padding
        // contributes 0 and doesn't affect numerics for real groups.
        let safe_delta = sorted_delta.clamp_min(0.0f64);
        let delta_cumsum = safe_delta.cumsum(0, score_kind).contiguous();
        let prefix_zero = Tensor::zeros(&[1], (score_kind, device));
        let prefix = Tensor::cat(&[prefix_zero, delta_cumsum], 0);

        // Pid-group boundaries via shift-compare.
        let pid_lhs = sorted_gpid.slice(0, 1, n, 1);
        let pid_rhs = sorted_gpid.slice(0, 0, n - 1, 1);
        let pid_changes = pid_lhs.ne_tensor(&pid_rhs).to_kind(Kind::Int64);
        let pid_changes_cs = pid_changes.cumsum(0, Kind::Int64);
        let pid_group_ids = Tensor::cat(&[zero_head.shallow_clone(), pid_changes_cs], 0);

        // Start/end index per pid group via scatter_reduce with min/max.
        let positions = Tensor::arange(n, (Kind::Int64, device));
        let mut pid_starts = Tensor::full(&[n], n, (Kind::Int64, device));
        let _ = pid_starts.internal_scatter_reduce_(0, &pid_group_ids, &positions, "amin", true);
        let mut pid_ends = Tensor::full(&[n], 0i64, (Kind::Int64, device));
        let _ = pid_ends.internal_scatter_reduce_(0, &pid_group_ids, &positions, "amax", true);

        let start_idx = pid_starts.clamp_max(n);
        let end_idx_plus_one = (pid_ends + 1).clamp_max(n);
        let sums_at_end = prefix.index_select(0, &end_idx_plus_one);
        let sums_before = prefix.index_select(0, &start_idx);
        let per_pid_delta = sums_at_end - sums_before;
        let totals = per_pid_delta + sum_mse;

        // unique_pids_buf[g] = pid label for pid-group g. Default -1 for
        // over-allocated / padding slots. Used to mask invalid totals.
        let mut unique_pids_buf = Tensor::full(&[n], -1i64, (Kind::Int64, device));
        let _ = unique_pids_buf.scatter_(0, &pid_group_ids, &sorted_gpid);

        // Mask: groups whose pid is -1 (both the padding group at position
        // 0 AND over-allocated slots that never got scattered) have totals
        // we must exclude from topk.
        let invalid_mask = unique_pids_buf.eq(-1);
        let totals = totals.masked_fill(&invalid_mask, f64::NEG_INFINITY);

        // --- Phase 4: topk ---
        // Budget and take_k computed from N (known), not from num_unique.
        // topk over a [N] buffer with -inf-padded invalids returns the real
        // top-k provided enough valid entries exist.
        let budget = k.max(max_candidates.min(n as usize));
        let take_k = (budget as i64).min(n);

        let topk = totals.topk(take_k, 0, /*largest=*/ true, /*sorted=*/ true);
        let top_scores_gpu = topk.0;
        let top_indices = topk.1;
        let top_pids_gpu = unique_pids_buf.index_select(0, &top_indices);

        Ok((top_pids_gpu, top_scores_gpu))
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
