use anyhow::Result;
use tch::{no_grad, Device, Kind, Tensor};

use crate::utils::types::{
    parse_device, parse_dtype, SearchConfig, SelectedCentroids, TPrimePolicy,
};

/// WARP centroid selection for efficient top-k retrieval
#[derive(Clone)]
pub struct CentroidSelector {
    nprobe: u32,
    t_prime_policy: TPrimePolicy,
    bound: usize,
    device: Device,
    dtype: Kind,
    centroid_score_threshold: f32,
}

impl CentroidSelector {
    /// Create a new centroid selector with the given configuration
    pub fn new(config: &SearchConfig, num_embeddings: usize, num_centroids: usize) -> Self {
        let nprobe = config.nprobe;
        let device = parse_device(&config.device).expect("Invalid device string");
        let dtype = parse_dtype(&config.dtype).expect("Invalid dtype string");

        let t_prime_policy = match config.t_prime {
            Some(value) => TPrimePolicy::Fixed(value),
            None if num_centroids <= (1 << 16) => {
                let estimate =
                    ((8.0 * num_embeddings as f64).sqrt() / 1000.0).floor() as usize * 1000;
                let adjusted = if estimate == 0 { 1000 } else { estimate };
                TPrimePolicy::Fixed(adjusted)
            },
            None => TPrimePolicy::Max,
        };

        let bound = if config.bound == 0 { 128 } else { config.bound };

        Self {
            nprobe,
            t_prime_policy,
            bound,
            device,
            dtype,
            centroid_score_threshold: config.centroid_score_threshold.unwrap_or(0.4),
        }
    }

    pub fn select_centroids(
        &self,
        query_mask: &Tensor,      // [num_tokens]
        centroid_scores: &Tensor, // [num_tokens, num_centroids]
        sizes_compacted: &Tensor, // [num_centroids]
        kdummy_centroid: i64,
        k: usize,
    ) -> Result<SelectedCentroids> {
        no_grad(|| {
            let num_query_tokens = query_mask.size()[0];
            let num_centroids = centroid_scores.size()[1] as i64;

            let mut nprobe = self.nprobe as i64;
            if nprobe > num_centroids {
                eprintln!(
                    "[CentroidSelector] Warning: requested nprobe {} exceeds number of centroids {}; clamping to {}.",
                    nprobe, num_centroids, num_centroids
                );
                nprobe = num_centroids;
            }

            let mut bound = self.bound.min(num_centroids as usize) as i64;
            if bound < nprobe {
                bound = nprobe;
            }

            let bound_i64 = bound;
            let nprobe_i64 = nprobe;

            let mut cells =
                Tensor::zeros(&[num_query_tokens, nprobe_i64], (Kind::Int, self.device));
            let mut scores =
                Tensor::zeros(&[num_query_tokens, nprobe_i64], (self.dtype, self.device));
            let mut mse = Tensor::zeros(&[num_query_tokens], (self.dtype, self.device));

            // We only want to treat the non-zero tokens
            // Note that this skips tensors padded with 0, not with the pad token
            let mask_indices = query_mask.to_kind(Kind::Bool).nonzero().squeeze_dim(-1);
            let active_tokens = mask_indices.size()[0];

            if active_tokens > 0 && bound > 0 {
                let active_scores = centroid_scores.index_select(0, &mask_indices);

                // Compute max score for each token and create threshold mask
                let (max_scores, _) = active_scores.max_dim(1, true);
                let threshold_tensor = Tensor::full(
                    &[1],
                    self.centroid_score_threshold as f64,
                    (self.dtype, max_scores.device()),
                );
                let above_threshold_mask = max_scores.squeeze_dim(1).ge_tensor(&threshold_tensor);

                // Get indices of tokens that pass the threshold
                let threshold_indices = above_threshold_mask.nonzero().squeeze_dim(-1);
                let filtered_tokens = threshold_indices.size()[0];

                if filtered_tokens > 0 {
                    // Filter to only tokens above threshold
                    let filtered_scores = active_scores.index_select(0, &threshold_indices);
                    let filtered_mask_indices = mask_indices.index_select(0, &threshold_indices);

                    let (top_values, top_indices) = filtered_scores.topk(bound_i64, 1, true, true);

                    let top_cells = top_indices.slice(1, 0, nprobe_i64, 1);
                    let top_scores = top_values.slice(1, 0, nprobe_i64, 1);

                    let sizes_selected = sizes_compacted
                        .index_select(0, &top_indices.view([-1]))
                        .view([filtered_tokens, bound_i64]);

                    let cumsum_sizes = sizes_selected.cumsum(1, Kind::Int64);
                    let t_prime_val = self.t_prime_policy.value(k) as i64;
                    let t_prime_tensor =
                        Tensor::full(&[1], t_prime_val, (Kind::Int64, cumsum_sizes.device()));
                    let less_than_t_prime = cumsum_sizes.lt_tensor(&t_prime_tensor);
                    let clamp_limit = if bound_i64 > 0 { bound_i64 - 1 } else { 0 };
                    let sum_dims = [1i64];
                    let mse_indices = less_than_t_prime.to_kind(Kind::Int64).sum_dim_intlist(
                        &sum_dims[..],
                        false,
                        Kind::Int64,
                    );
                    let mse_indices = if clamp_limit >= 0 {
                        mse_indices
                            .to_kind(self.dtype)
                            .clamp_max(clamp_limit as f64)
                            .to_kind(Kind::Int64)
                    } else {
                        mse_indices
                    };
                    let mse_selected = top_values
                        .gather(1, &mse_indices.unsqueeze(1), false)
                        .squeeze_dim(1);

                    let _ =
                        cells.index_copy_(0, &filtered_mask_indices, &top_cells.to_kind(Kind::Int));
                    let _ = scores.index_copy_(0, &filtered_mask_indices, &top_scores);
                    let _ = mse.index_copy_(0, &filtered_mask_indices, &mse_selected);
                }
                // Tokens below threshold will remain as zeros and get replaced by dummy centroid
            }

            let cells_flat = cells.flatten(0, -1).contiguous();
            let scores_flat = scores.flatten(0, -1).contiguous();

            // Replace cells with zero scores with kdummy_centroid
            // This handles masked query tokens
            let zero_mask = scores_flat.eq(0.0);
            let kdummy_tensor = Tensor::full_like(&cells_flat, kdummy_centroid as i64);
            let cells_final = cells_flat.where_self(&zero_mask.logical_not(), &kdummy_tensor);

            Ok(SelectedCentroids {
                centroid_ids: cells_final,
                scores: scores_flat,
                mse_estimate: mse,
            })
        })
    }
}
