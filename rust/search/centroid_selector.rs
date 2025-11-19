use anyhow::Result;
use tch::{no_grad, Device, Kind, Tensor};

use crate::utils::types::{SearchConfig, SelectedCentroids, TPrimePolicy};

/// WARP centroid selection for efficient top-k retrieval
pub struct CentroidSelector {
    nprobe: u32,
    t_prime_policy: TPrimePolicy,
    bound: usize,
    device: Device,
}

impl CentroidSelector {
    /// Create a new centroid selector with the given configuration
    pub fn new(
        config: &SearchConfig,
        num_embeddings: usize,
        num_centroids: usize,
        device: Device,
    ) -> Self {
        let nprobe = config.nprobe;

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
        }
    }

    /// Select top centroids using pre-computed scores
    pub fn select_centroids_with_scores(
        &self,
        query_mask: &Tensor,      // [num_tokens]
        centroid_scores: &Tensor, // [num_tokens, num_centroids]
        sizes_compacted: &Tensor, // [num_centroids]
        kdummy_centroid: i64,
        k: usize,
    ) -> Result<SelectedCentroids> {
        no_grad(|| {
            let num_query_tokens = i64::try_from(query_mask.sum(Kind::Int64))? as usize;
            let num_centroids = centroid_scores.size()[1] as usize;

            let cells = Tensor::zeros(
                &[num_query_tokens as i64, self.nprobe as i64],
                (Kind::Int, self.device),
            );
            let scores = Tensor::zeros(
                &[num_query_tokens as i64, self.nprobe as i64],
                (Kind::Float, self.device),
            );
            let mse = Tensor::zeros(&[num_query_tokens as i64], (Kind::Float, self.device));

            let centroid_scores_vec: Vec<f32> = centroid_scores.flatten(0, -1).try_into()?;
            let sizes_vec: Vec<i64> = Vec::try_from(sizes_compacted)?;

            for i in 0..num_query_tokens {
                let offset = i * num_centroids;
                let query_scores = &centroid_scores_vec[offset..offset + num_centroids];

                let mut centroid_idx: Vec<usize> = (0..num_centroids).collect();

                centroid_idx.select_nth_unstable_by(self.bound, |&a, &b| {
                    query_scores[b]
                        .partial_cmp(&query_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });

                let top_centroids = &centroid_idx[..self.nprobe as usize];
                for (j, &centroid_id) in top_centroids.iter().enumerate() {
                    let _ = cells.get(i as i64).get(j as i64).fill_(centroid_id as i64);
                    let _ = scores
                        .get(i as i64)
                        .get(j as i64)
                        .fill_(query_scores[centroid_id] as f64);
                }

                let mut cumsum: i64 = 0;
                let mut idx = 0;

                while cumsum < self.t_prime_policy.value(k) as i64 && idx < self.bound {
                    cumsum += sizes_vec[centroid_idx[idx]];
                    idx += 1;
                }

                let mse_value = if idx == 0 {
                    0.0
                } else {
                    query_scores[centroid_idx[idx - 1]] as f64
                };
                let _ = mse.get(i as i64).fill_(mse_value);
            }

            // Flatten cells and scores
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

    /// Select top centroids for a single query using WARP algorithm
    pub fn select_centroids(
        &self,
        query: &Tensor,           // [num_tokens]
        centroids: &Tensor,       // [num_centroids, dim]
        sizes_compacted: &Tensor, // [num_centroids]
        kdummy_centroid: i64,     // Index of dummy centroid
        k: usize,
    ) -> Result<SelectedCentroids> {
        no_grad(|| {
            let query_mask = query.ne(0).any_dim(1, false);
            let num_query_tokens = i64::try_from(query_mask.sum(Kind::Int64))? as usize;
            let num_centroids = centroids.size()[0] as usize;

            let cells = Tensor::zeros(&[32i64, self.nprobe as i64], (Kind::Int, self.device));
            let scores = Tensor::zeros(&[32i64, self.nprobe as i64], (Kind::Float, self.device));
            let mse = Tensor::zeros(&[32i64], (Kind::Float, self.device));

            let centroid_scores = query.matmul(&centroids.transpose(0, 1));
            let centroid_scores_vec: Vec<f32> = centroid_scores.flatten(0, -1).try_into()?;
            let sizes_vec: Vec<i64> = Vec::try_from(sizes_compacted)?;

            for i in 0..num_query_tokens {
                let offset = i * num_centroids;
                let query_scores = &centroid_scores_vec[offset..offset + num_centroids];

                let mut centroid_idx: Vec<usize> = (0..num_centroids).collect();

                centroid_idx.select_nth_unstable_by(self.bound, |&a, &b| {
                    query_scores[b]
                        .partial_cmp(&query_scores[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.cmp(&b))
                });

                let top_centroids = &centroid_idx[..self.nprobe as usize];
                for (j, &centroid_id) in top_centroids.iter().enumerate() {
                    let _ = cells.get(i as i64).get(j as i64).fill_(centroid_id as i64);
                    let _ = scores
                        .get(i as i64)
                        .get(j as i64)
                        .fill_(query_scores[centroid_id] as f64);
                }

                let mut cumsum: i64 = 0;
                let mut idx = 0;

                while cumsum < self.t_prime_policy.value(k) as i64 && idx < self.bound {
                    cumsum += sizes_vec[centroid_idx[idx]];
                    idx += 1;
                }

                let mse_value = if idx == 0 {
                    0.0
                } else {
                    query_scores[centroid_idx[idx - 1]] as f64
                };
                let _ = mse.get(i as i64).fill_(mse_value);
            }

            // Flatten cells and scores
            let cells_flat = cells.flatten(0, -1).contiguous();
            let scores_flat = scores.flatten(0, -1).contiguous();

            // Handle masked query tokens
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
