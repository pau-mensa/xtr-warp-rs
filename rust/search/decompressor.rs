use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tch::{kind, Device, Kind, Tensor};

use crate::utils::types::{DecompressedCentroidsOutput, LoadedIndex};

/// Centroid decompressor for efficient residual decompression
pub struct CentroidDecompressor {
    nbits: u8,
    device: Device,
    use_parallel: bool,
    dim: usize,
}

impl CentroidDecompressor {
    /// Create a new centroid decompressor
    pub fn new(nbits: u8, dim: usize, device: Device, use_parallel: bool) -> Result<Self> {
        if nbits != 2 && nbits != 4 {
            return Err(anyhow!("nbits must be 2 or 4, got {}", nbits));
        }

        Ok(Self {
            nbits,
            device,
            use_parallel,
            dim,
        })
    }

    /// Decompress multiple centroids for a query
    pub fn decompress_centroids(
        &self,
        centroid_ids: &Tensor,
        centroid_scores: &Tensor,
        index: &Arc<LoadedIndex>,
        query_embeddings: &Tensor, // [num_tokens, dim]
        nprobe: usize,
    ) -> Result<DecompressedCentroidsOutput> {
        let centroid_ids = centroid_ids.to_kind(Kind::Int64);
        let num_centroids = centroid_ids.size()[0] as i64;

        let num_total_centroids = index.offsets_compacted.size()[0] - 1;
        let max_centroid_id = centroid_ids.max().int64_value(&[]);

        if max_centroid_id >= num_total_centroids {
            return Err(anyhow::anyhow!(
                "Centroid ID {} is out of bounds (max valid ID is {})",
                max_centroid_id,
                num_total_centroids - 1
            ));
        }

        // Get offsets for all centroids - these are i64 tensors
        let begins = index.offsets_compacted.index_select(0, &centroid_ids);
        let ends = index.offsets_compacted.index_select(0, &(centroid_ids + 1));
        let capacities = &ends - &begins; // Number of embeddings per centroid (before deduplication)

        // Build indices for all embeddings across all centroids
        let mut all_indices_vec = Vec::new();
        for i in 0..num_centroids {
            let begin = begins.int64_value(&[i]);
            let end = ends.int64_value(&[i]);
            let capacity = end - begin;

            if capacity > 0 {
                // Create range [begin, begin+1, ..., end-1]
                let range = Tensor::arange(capacity, (Kind::Int64, self.device)) + begin;
                all_indices_vec.push(range);
            }
        }

        // Concatenate all indices into a single tensor
        let all_indices = Tensor::cat(&all_indices_vec, 0);

        // Get passage IDs for all embeddings
        let all_codes = index.codes_compacted.index_select(0, &all_indices);

        // Decompress all residuals in batch
        let all_residuals = self.decompress_residuals_batch(
            &all_indices,
            query_embeddings,
            &index.bucket_weights,
            &begins,
            &capacities,
            nprobe,
            &index,
        )?;

        // Compute final scores
        let all_scores =
            self.compute_final_scores(&centroid_scores, &all_residuals, &begins, &capacities)?;

        // Deduplicate passages within each centroid
        let dedup_result =
            self.deduplicate_passages_per_centroid(&all_codes, &all_scores, &begins, &capacities)?;

        Ok(DecompressedCentroidsOutput {
            capacities,
            sizes: dedup_result.3, // Actual sizes after deduplication
            passage_ids: dedup_result.0,
            scores: dedup_result.1,
            offsets: dedup_result.2,
        })
    }

    /// Decompress residuals for all centroids in batch
    /// Returns a tensor of residual scores aligned with all_indices
    fn decompress_residuals_batch(
        &self,
        all_indices: &Tensor, // [total_embeddings] - global indices into residuals
        query_embeddings: &Tensor, // [num_tokens, dim] - query embeddings
        bucket_weights: &Tensor, // [dim, num_buckets_per_dim] - bucket weights
        begins: &Tensor,      // [num_centroids] - start indices
        capacities: &Tensor, // [num_centroids] - number of embeddings per centroid (before deduplication)
        nprobe: usize,
        index: &LoadedIndex,
    ) -> Result<Tensor> {
        let total_embeddings = all_indices.size()[0] as usize;
        let num_centroids = begins.size()[0] as usize;

        // Prepare output tensor
        let mut residual_scores = vec![0.0f32; total_embeddings];

        let mut global_idx = 0;
        for centroid_idx in 0..num_centroids {
            let capacity = capacities.int64_value(&[centroid_idx as i64]) as usize;
            if capacity == 0 {
                continue;
            }

            let begin = begins.int64_value(&[centroid_idx as i64]);
            let token_idx = centroid_idx / nprobe; // Which query token this centroid belongs to

            // Process each embedding in this centroid
            for local_idx in 0..capacity {
                let residual_idx = begin + local_idx as i64;

                let score = if self.nbits == 2 {
                    self.decompress_single_residual_2bit(
                        residual_idx, // Pass the row index directly
                        query_embeddings,
                        bucket_weights,
                        token_idx,
                        &index.residuals_compacted,
                    )?
                } else {
                    self.decompress_single_residual_4bit(
                        residual_idx, // Pass the row index directly
                        query_embeddings,
                        bucket_weights,
                        token_idx,
                        &index.residuals_compacted,
                    )?
                };

                residual_scores[global_idx] = score;
                global_idx += 1;
            }
        }

        Ok(Tensor::from_slice(&residual_scores).to_device(self.device))
    }

    /// Compute final scores by adding centroid scores to residuals
    fn compute_final_scores(
        &self,
        centroid_scores: &Tensor,
        residual_scores: &Tensor,
        begins: &Tensor,
        capacities: &Tensor,
    ) -> Result<Tensor> {
        let total_embeddings = residual_scores.size()[0] as usize;
        let num_centroids = begins.size()[0] as usize;

        let mut final_scores = vec![0.0f32; total_embeddings];

        let mut global_idx = 0;
        for centroid_idx in 0..num_centroids {
            let capacity = capacities.int64_value(&[centroid_idx as i64]) as usize;
            if capacity == 0 {
                continue;
            }

            let centroid_score = centroid_scores.double_value(&[centroid_idx as i64]) as f32;

            for _ in 0..capacity {
                let residual = residual_scores.double_value(&[global_idx as i64]) as f32;
                final_scores[global_idx] = centroid_score + residual;
                global_idx += 1;
            }
        }

        Ok(Tensor::from_slice(&final_scores).to_device(self.device))
    }

    /// Decompress a single 2-bit encoded residual
    fn decompress_single_residual_2bit(
        &self,
        residual_idx: i64, // Changed from residual_offset to residual_idx (row index)
        query_embeddings: &Tensor, // [num_tokens, dim]
        bucket_weights: &Tensor, // [dim, num_buckets_per_dim]
        token_idx: usize,
        residuals_compacted: &Tensor,
    ) -> Result<f32> {
        let packed_dim = self.dim / 4; // 4 values per byte for 2-bit encoding
        let mut score = 0.0f32;

        for packed_idx in 0..packed_dim {
            // Get the packed byte - residuals are stored as uint8 in a 2D tensor [N, packed_dim]
            let col_idx = packed_idx as i64;
            let packed_val = if residual_idx < residuals_compacted.size()[0]
                && col_idx < residuals_compacted.size()[1]
            {
                // Access the 2D tensor properly: [residual_idx, col_idx]
                let val_i64 = residuals_compacted.int64_value(&[residual_idx, col_idx]);
                (val_i64 & 0xFF) as u8
            } else {
                0u8 // Padding for out-of-bounds access
            };

            // Unpack 4 2-bit values
            let unpacked_0 = (packed_val & 0xC0) >> 6;
            let unpacked_1 = (packed_val & 0x30) >> 4;
            let unpacked_2 = (packed_val & 0x0C) >> 2;
            let unpacked_3 = packed_val & 0x03;

            let unpacked_idx_0 = packed_idx * 4;
            let unpacked_idx_1 = unpacked_idx_0 + 1;
            let unpacked_idx_2 = unpacked_idx_0 + 2;
            let unpacked_idx_3 = unpacked_idx_0 + 3;

            // Get query values for these dimensions
            let q0 =
                query_embeddings.double_value(&[token_idx as i64, unpacked_idx_0 as i64]) as f32;
            let q1 =
                query_embeddings.double_value(&[token_idx as i64, unpacked_idx_1 as i64]) as f32;
            let q2 =
                query_embeddings.double_value(&[token_idx as i64, unpacked_idx_2 as i64]) as f32;
            let q3 =
                query_embeddings.double_value(&[token_idx as i64, unpacked_idx_3 as i64]) as f32;

            // Get bucket weights for each dimension and bucket value
            let w0 =
                bucket_weights.double_value(&[unpacked_idx_0 as i64, unpacked_0 as i64]) as f32;
            let w1 =
                bucket_weights.double_value(&[unpacked_idx_1 as i64, unpacked_1 as i64]) as f32;
            let w2 =
                bucket_weights.double_value(&[unpacked_idx_2 as i64, unpacked_2 as i64]) as f32;
            let w3 =
                bucket_weights.double_value(&[unpacked_idx_3 as i64, unpacked_3 as i64]) as f32;

            // Accumulate scores
            score += q0 * w0;
            score += q1 * w1;
            score += q2 * w2;
            score += q3 * w3;
        }

        Ok(score)
    }

    /// Decompress a single 4-bit encoded residual
    fn decompress_single_residual_4bit(
        &self,
        residual_idx: i64, // Changed from residual_offset to residual_idx (row index)
        query_embeddings: &Tensor, // [num_tokens, dim]
        bucket_weights: &Tensor, // [dim, num_buckets_per_dim]
        token_idx: usize,
        residuals_compacted: &Tensor,
    ) -> Result<f32> {
        let packed_dim = self.dim / 2; // 2 values per byte for 4-bit encoding
        let mut score = 0.0f32;

        for packed_idx in 0..packed_dim {
            // Get the packed byte from the 2D tensor
            let col_idx = packed_idx as i64;
            let packed_val = if residual_idx < residuals_compacted.size()[0]
                && col_idx < residuals_compacted.size()[1]
            {
                // Access the 2D tensor properly: [residual_idx, col_idx]
                let val_i64 = residuals_compacted.int64_value(&[residual_idx, col_idx]);
                (val_i64 & 0xFF) as u8
            } else {
                0u8
            };

            // Unpack 2 4-bit values
            let unpacked_0 = (packed_val & 0xF0) >> 4;
            let unpacked_1 = packed_val & 0x0F;

            let unpacked_idx_0 = packed_idx * 2;
            let unpacked_idx_1 = unpacked_idx_0 + 1;

            // Get query values for these dimensions
            let q0 =
                query_embeddings.double_value(&[token_idx as i64, unpacked_idx_0 as i64]) as f32;
            let q1 =
                query_embeddings.double_value(&[token_idx as i64, unpacked_idx_1 as i64]) as f32;

            // Get bucket weights for each dimension and bucket value
            let w0 =
                bucket_weights.double_value(&[unpacked_idx_0 as i64, unpacked_0 as i64]) as f32;
            let w1 =
                bucket_weights.double_value(&[unpacked_idx_1 as i64, unpacked_1 as i64]) as f32;

            // Accumulate scores
            score += q0 * w0;
            score += q1 * w1;
        }

        Ok(score)
    }

    /// Deduplicate passages within each centroid, keeping maximum scores
    /// Returns (passage_ids, scores, offsets) where offsets allow slicing per centroid
    fn deduplicate_passages_per_centroid(
        &self,
        passage_ids: &Tensor,
        scores: &Tensor,
        begins: &Tensor,
        capacities: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let num_centroids = begins.size()[0] as usize;

        // Collect deduplicated results per centroid
        let mut all_pids = Vec::new();
        let mut all_scores = Vec::new();
        let mut offsets = vec![0i64];
        let mut deduplicated_sizes = Vec::new();

        let mut global_idx = 0i64;

        for centroid_idx in 0..num_centroids {
            let capacity = capacities.int64_value(&[centroid_idx as i64]);

            if capacity == 0 {
                offsets.push(*offsets.last().unwrap());
                deduplicated_sizes.push(0i32);
                continue;
            }

            // Use HashMap for deduplication within this centroid
            let mut max_scores: HashMap<i64, f32> = HashMap::new();

            for _ in 0..capacity {
                let pid = passage_ids.int64_value(&[global_idx]);
                let score = scores.double_value(&[global_idx]) as f32;

                max_scores
                    .entry(pid)
                    .and_modify(|s| *s = s.max(score))
                    .or_insert(score);

                global_idx += 1;
            }

            // Sort by passage ID for deterministic output
            let mut centroid_results: Vec<(i64, f32)> = max_scores.into_iter().collect();
            centroid_results.sort_by_key(|&(pid, _)| pid);

            // Add to global results
            let centroid_dedup_size = centroid_results.len() as i32;
            for (pid, score) in centroid_results {
                all_pids.push(pid);
                all_scores.push(score);
            }

            deduplicated_sizes.push(centroid_dedup_size);
            offsets.push(all_pids.len() as i64);
        }

        // Convert to tensors
        let pids_tensor = Tensor::from_slice(&all_pids)
            .to_device(self.device)
            .to_kind(Kind::Int64);
        let scores_tensor = Tensor::from_slice(&all_scores)
            .to_device(self.device)
            .to_kind(Kind::Float);
        let offsets_tensor = Tensor::from_slice(&offsets)
            .to_device(self.device)
            .to_kind(Kind::Int64);
        let sizes_tensor = Tensor::from_slice(&deduplicated_sizes)
            .to_device(self.device)
            .to_kind(Kind::Int);

        Ok((pids_tensor, scores_tensor, offsets_tensor, sizes_tensor))
    }
}

#[cfg(test)]
mod tests {}
