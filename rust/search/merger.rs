use anyhow::Result;
use ndarray::Array1;
use std::cmp::Ordering;
use std::sync::Arc;
use tch::{no_grad, Device, Kind, Tensor};

use crate::utils::types::{PassageId, Score};

/// Represents a scored candidate document
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    /// Passage/document ID
    pub pid: PassageId,

    /// Score for this candidate
    pub score: Score,

    /// Source centroid index (for debugging/tracing)
    pub centroid_id: Option<u32>,

    /// Token-level scores (if available)
    pub token_scores: Option<Vec<Score>>,
}

impl PartialEq for ScoredCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.pid == other.pid
    }
}

impl Eq for ScoredCandidate {}

impl PartialOrd for ScoredCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for max-heap behavior
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for ScoredCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Configuration for the merger
#[derive(Debug, Clone)]
pub struct MergerConfig {
    /// Maximum number of candidates to keep during merging
    pub max_candidates: usize,

    /// Whether to use parallel merging
    pub use_parallel: bool,

    /// Number of threads for parallel operations
    pub num_threads: usize,

    /// Strategy for combining scores
    pub combination_strategy: ScoreCombination,

    /// Device to use
    pub device: Device,
}

/// Strategy for combining scores from multiple sources
#[derive(Clone)]
pub enum ScoreCombination {
    /// Take maximum score across all occurrences
    Max,

    /// Sum scores across all occurrences
    Sum,

    /// Average scores across all occurrences
    Average,

    /// Custom combination function
    Custom(Arc<dyn Fn(&[f32]) -> f32 + Send + Sync>),
}

impl std::fmt::Debug for ScoreCombination {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Max => write!(f, "Max"),
            Self::Sum => write!(f, "Sum"),
            Self::Average => write!(f, "Average"),
            Self::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
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

impl ReduceMaxCombiner {
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

/// Combiner for sum-reduction with MSE correction
struct ReduceSumMseCombiner {
    lhs_mse: f32,
    rhs_mse: f32,
}

impl ReduceSumMseCombiner {
    fn new(lhs_mse: f32, rhs_mse: f32) -> Self {
        ReduceSumMseCombiner { lhs_mse, rhs_mse }
    }

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

/// Represents a stride of candidates from a single source
pub struct CandidateStride {
    /// Passage IDs in this stride
    pub pids: Array1<PassageId>,

    /// Scores for each PID
    pub scores: Array1<Score>,

    /// Actual size (may be less than capacity)
    pub size: usize,

    /// Capacity of this stride
    pub capacity: usize,
}

/// Main merger struct for combining results from multiple sources
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

        // If odd number of iterations, swap back so result is in original buffer
        if num_iterations % 2 != 0 {
            for i in 0..nprobe {
                Self::copy_candidate_stride(&buf2[begin + i], &mut buf1[begin + i]);
            }
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
        mse_estimates: &Tensor,
        nprobe: usize,
        k: usize,
    ) -> Result<(Vec<PassageId>, Vec<Score>)> {
        no_grad(|| {
            let num_cells = capacities.size()[0] as usize;
            let num_candidates = candidate_pids.size()[0] as usize;

            // Convert tensors to vectors for creating stride views
            let sizes_vec: Vec<i32> = candidate_sizes.to_kind(Kind::Int).try_into()?;
            let pids_vec: Vec<PassageId> = candidate_pids.to_kind(Kind::Int64).try_into()?;
            let scores_vec: Vec<Score> = candidate_scores.to_kind(Kind::Float).try_into()?;
            let mse_vec: Vec<f32> = mse_estimates.to_kind(Kind::Float).try_into()?;

            let total_size: i32 = sizes_vec.iter().sum();

            // Create strided views into the data (each view represents one centroid's candidates)
            let mut views = Vec::new();
            let mut offset = 0;
            for i in 0..num_cells {
                let size = sizes_vec[i] as usize;

                // Use size instead of capacity since the data is deduplicated and compacted
                let end = (offset + size).min(num_candidates);
                let pids_slice = pids_vec[offset..end].to_vec();
                let scores_slice = scores_vec[offset..end].to_vec();

                views.push(AnnotatedStrideView::from_data(
                    pids_slice,
                    scores_slice,
                    size,
                ));
                offset += size;
            }

            // Create buffer views for merging
            // We need to create views that can handle the worst-case merge scenario
            // When merging in a tree-like fashion, the maximum size at any level
            // is the sum of all individual sizes
            let max_merged_size = total_size as usize;
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
            let num_results = views[0].size.min(k);
            let pid_idx = Self::partial_sort_results(&views[0], num_results);

            // Extract the top-k PIDs and scores
            let mut result_pids = vec![0i64; num_results];
            let mut result_scores = vec![0.0f32; num_results];

            for i in 0..num_results {
                let idx = pid_idx[i];
                result_pids[i] = views[0].pids[idx];
                result_scores[i] = views[0].scores[idx];
            }

            // Convert back to tensors TODO might not be necessary
            //let candidate_pids = Tensor::from_slice(&result_pids).to_device(self.config.device);
            //let candidate_scores = Tensor::from_slice(&result_scores).to_device(self.config.device);

            Ok((result_pids, result_scores))
        })
    }

    /// Merges two sorted candidate strides
    /// Implementation reference: xtr-warp/warp/engine/search/merge_candidate_scores.cpp::merge_candidate_strides
    pub fn merge_candidate_strides(
        &self,
        stride1: &CandidateStride,
        stride2: &CandidateStride,
        mse1: Option<Score>,
        mse2: Option<Score>,
    ) -> Result<CandidateStride> {
        // Convert CandidateStride to AnnotatedStrideView
        let view1 = AnnotatedStrideView::from_data(
            stride1.pids.to_vec(),
            stride1.scores.to_vec(),
            stride1.size,
        );

        let view2 = AnnotatedStrideView::from_data(
            stride2.pids.to_vec(),
            stride2.scores.to_vec(),
            stride2.size,
        );

        // Create result view with capacity for worst case (sum of both sizes)
        let capacity = stride1.size + stride2.size;
        let mut result_view = AnnotatedStrideView::with_capacity(capacity);

        // Apply MSE if provided and use appropriate combiner
        match (mse1, mse2) {
            (Some(mse1_val), Some(mse2_val)) => {
                let combiner = ReduceSumMseCombiner::new(mse1_val, mse2_val);
                Self::merge_candidate_strides_with_combiner(
                    &view1,
                    &view2,
                    &mut result_view,
                    &combiner,
                );
            },
            _ => {
                // Use the configured combination strategy
                match &self.config.combination_strategy {
                    ScoreCombination::Max => {
                        let combiner = ReduceMaxCombiner;
                        Self::merge_candidate_strides_with_combiner(
                            &view1,
                            &view2,
                            &mut result_view,
                            &combiner,
                        );
                    },
                    ScoreCombination::Sum => {
                        let combiner = SimpleCombiner::new(|lhs, rhs| lhs + rhs);
                        Self::merge_candidate_strides_with_combiner(
                            &view1,
                            &view2,
                            &mut result_view,
                            &combiner,
                        );
                    },
                    ScoreCombination::Average => {
                        let combiner = SimpleCombiner::new(|lhs, rhs| (lhs + rhs) / 2.0);
                        Self::merge_candidate_strides_with_combiner(
                            &view1,
                            &view2,
                            &mut result_view,
                            &combiner,
                        );
                    },
                    ScoreCombination::Custom(f) => {
                        let f_clone = f.clone();
                        let combiner = SimpleCombiner::new(move |lhs, rhs| f_clone(&[lhs, rhs]));
                        Self::merge_candidate_strides_with_combiner(
                            &view1,
                            &view2,
                            &mut result_view,
                            &combiner,
                        );
                    },
                }
            },
        }

        // Convert result back to CandidateStride
        let result_pids = Array1::from_vec(
            result_view.pids[..result_view.size]
                .iter()
                .map(|&p| p as PassageId)
                .collect(),
        );
        let result_scores = Array1::from_vec(result_view.scores[..result_view.size].to_vec());

        Ok(CandidateStride {
            pids: result_pids,
            scores: result_scores,
            size: result_view.size,
            capacity,
        })
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

/// Simple combiner that just passes through single values
struct SimpleCombiner<F> {
    combine_fn: F,
}

impl<F> SimpleCombiner<F>
where
    F: Fn(f32, f32) -> f32,
{
    fn new(combine_fn: F) -> Self {
        SimpleCombiner { combine_fn }
    }
}

impl<F> Combiner for SimpleCombiner<F>
where
    F: Fn(f32, f32) -> f32,
{
    fn combine(&self, lhs: f32, rhs: f32) -> f32 {
        (self.combine_fn)(lhs, rhs)
    }

    fn lhs(&self, lhs: f32) -> f32 {
        lhs
    }

    fn rhs(&self, rhs: f32) -> f32 {
        rhs
    }
}
