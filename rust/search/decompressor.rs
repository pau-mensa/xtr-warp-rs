use anyhow::{anyhow, Result};
use rayon::prelude::*;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

use crate::utils::types::{DecompressedCentroidsOutput, ReadOnlyIndex};

/// Centroid decompressor for efficient residual decompression
#[derive(Clone)]
pub struct CentroidDecompressor {
    nbits: u8,
    device: Device,
    dtype: Kind,
    num_threads: usize,
    dim: usize,
    reversed_bit_map: [u8; 256],
}

impl CentroidDecompressor {
    /// Create a new centroid decompressor
    pub fn new(
        nbits: u8,
        dim: usize,
        device: Device,
        dtype: Kind,
        num_threads: usize,
    ) -> Result<Self> {
        if nbits != 2 && nbits != 4 {
            return Err(anyhow!("nbits must be 2 or 4, got {}", nbits));
        }

        let reversed_bit_map = Self::build_reversed_bit_map(nbits);

        Ok(Self {
            nbits,
            device,
            dtype,
            num_threads,
            dim,
            reversed_bit_map,
        })
    }

    fn build_reversed_bit_map(nbits: u8) -> [u8; 256] {
        let mut reversed = [0u8; 256];
        let nbits_mask = (1 << nbits) - 1;
        for byte_val in 0..256u32 {
            let mut reversed_bits = 0u32;
            let mut bit_pos = 8;
            while bit_pos >= nbits {
                let segment = (byte_val >> (bit_pos - nbits)) & nbits_mask;
                let mut reversed_segment = 0u32;
                for k in 0..nbits {
                    if (segment & (1 << k)) != 0 {
                        reversed_segment |= 1 << (nbits - 1 - k);
                    }
                }
                reversed_bits |= reversed_segment;
                if bit_pos > nbits {
                    reversed_bits <<= nbits;
                }
                bit_pos -= nbits;
            }
            reversed[byte_val as usize] = (reversed_bits & 0xFF) as u8;
        }
        reversed
    }

    pub fn decompress_centroids(
        &self,
        centroid_ids: &Tensor,
        centroid_scores: &Tensor,
        index: &Arc<ReadOnlyIndex>,
        query_embeddings: &Tensor, // [num_tokens, dim]
        nprobe: usize,
        num_threads: usize,
    ) -> Result<DecompressedCentroidsOutput> {
        let centroid_ids = centroid_ids.to_kind(Kind::Int64);
        let num_cells = centroid_ids.size()[0] as usize;

        let num_total_centroids = index.offsets_compacted.size()[0] - 1;
        let max_centroid_id = centroid_ids.max().int64_value(&[]);

        if max_centroid_id >= num_total_centroids {
            return Err(anyhow!(
                "Centroid ID {} is out of bounds (max valid ID is {})",
                max_centroid_id,
                num_total_centroids - 1
            ));
        }

        // Gather begin/end offsets and capacities for every requested centroid
        let begins = index.offsets_compacted.index_select(0, &centroid_ids);
        let ends = index.offsets_compacted.index_select(0, &(centroid_ids + 1));
        let capacities = &ends - &begins;

        // Early exit when there is nothing to process
        if num_cells == 0 {
            let empty = Tensor::zeros(&[0], (Kind::Int, self.device));
            return Ok(DecompressedCentroidsOutput {
                capacities,
                sizes: empty.to_kind(Kind::Int),
                passage_ids: Tensor::zeros(&[0], (Kind::Int64, self.device)),
                scores: Tensor::zeros(&[0], (self.dtype, self.device)),
                offsets: Tensor::zeros(&[1], (Kind::Int64, self.device)),
            });
        }

        anyhow::ensure!(nprobe > 0, "nprobe must be greater than zero");

        let query_embeddings = query_embeddings.to_kind(self.dtype);
        anyhow::ensure!(
            query_embeddings.size()[1] == self.dim as i64,
            "Query embedding dim ({}) does not match index dim ({})",
            query_embeddings.size()[1],
            self.dim
        );

        let num_tokens = query_embeddings.size()[0] as usize;
        anyhow::ensure!(
            num_tokens > 0,
            "Expected at least one query token for decompression"
        );

        let bucket_weights = index.bucket_weights.to_kind(self.dtype);
        let vt_bucket_scores =
            (query_embeddings.unsqueeze(2) * &bucket_weights.unsqueeze(0)).contiguous();

        let bucket_scores_flat: Vec<f32> = vt_bucket_scores.flatten(0, -1).try_into()?;
        let centroid_scores_vec: Vec<f32> = centroid_scores.flatten(0, -1).try_into()?;

        anyhow::ensure!(
            centroid_scores_vec.len() == num_cells,
            "Centroid score count ({}) does not match number of cells ({})",
            centroid_scores_vec.len(),
            num_cells
        );

        let total_capacity = capacities.sum(Kind::Int64).int64_value(&[]).max(0) as usize;

        let mut candidate_sizes = vec![0i32; num_cells];
        let mut candidate_pids = Vec::with_capacity(total_capacity);
        let mut candidate_scores = Vec::with_capacity(total_capacity);
        let mut offsets = Vec::with_capacity(num_cells + 1);
        offsets.push(0i64);

        let num_buckets = 1usize << (self.nbits as usize);
        let bucket_dim_shift = self.nbits as usize;
        let bucket_score_stride = self.dim * num_buckets;
        let packed_vals_per_byte = 8usize / self.nbits as usize;
        let residual_bytes_per_embedding = self.dim / packed_vals_per_byte;

        // we need to prefetch all data to cpu, to avoid costly gpu -> cpu transfers
        let capacities_vec: Vec<i64> = capacities.shallow_clone().try_into()?;
        let begins_vec: Vec<i64> = begins.try_into()?;

        // collect all indices
        let mut all_indices = Vec::new();
        let mut cell_ranges = Vec::new();

        for cell_idx in 0..num_cells {
            let capacity = capacities_vec[cell_idx] as usize;
            if capacity == 0 {
                cell_ranges.push((0, 0));
                continue;
            }

            let begin = begins_vec[cell_idx];
            let start_idx = all_indices.len();
            for inner_idx in 0..capacity {
                all_indices.push(begin + inner_idx as i64);
            }
            let end_idx = all_indices.len();
            cell_ranges.push((start_idx, end_idx));
        }

        let (all_pids_vec, all_residuals_data) = if !all_indices.is_empty() {
            let indices_tensor =
                Tensor::from_slice(&all_indices).to_device(index.codes_compacted.device());
            let batch_pids = index.codes_compacted.index_select(0, &indices_tensor);
            let batch_residuals = index.residuals_compacted.index_select(0, &indices_tensor);

            // transfer to cpu
            let pids_vec: Vec<i64> = batch_pids.try_into()?;
            let residuals_flat: Vec<u8> = batch_residuals
                .to_kind(Kind::Uint8)
                .contiguous()
                .view([-1])
                .try_into()?;
            (pids_vec, residuals_flat)
        } else {
            (Vec::new(), Vec::new())
        };

        let worker_threads = if num_threads == 0 {
            self.num_threads
        } else {
            num_threads
        };

        let use_parallel = worker_threads > 1; // && nprobe >= 4 && num_cells >= 8;

        if use_parallel {
            // Use scoped thread pool to respect num_threads parameter
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(worker_threads)
                .build();

            let cell_results: Vec<_> = match pool {
                Ok(pool) => pool.install(|| {
                    (0..num_cells)
                        .into_par_iter()
                        .map(|cell_idx| {
                            self.process_cell(
                                cell_idx,
                                &capacities_vec,
                                &centroid_scores_vec,
                                nprobe,
                                num_tokens,
                                bucket_score_stride,
                                &bucket_scores_flat,
                                &cell_ranges,
                                &all_pids_vec,
                                &all_residuals_data,
                                residual_bytes_per_embedding,
                                bucket_dim_shift,
                            )
                        })
                        .collect()
                }),
                Err(_) => {
                    // Fallback to global thread pool if custom pool creation fails
                    (0..num_cells)
                        .into_par_iter()
                        .map(|cell_idx| {
                            self.process_cell(
                                cell_idx,
                                &capacities_vec,
                                &centroid_scores_vec,
                                nprobe,
                                num_tokens,
                                bucket_score_stride,
                                &bucket_scores_flat,
                                &cell_ranges,
                                &all_pids_vec,
                                &all_residuals_data,
                                residual_bytes_per_embedding,
                                bucket_dim_shift,
                            )
                        })
                        .collect()
                },
            };

            // rebuild the results
            offsets.clear();
            offsets.push(0i64);
            candidate_pids.clear();
            candidate_scores.clear();

            for (cell_idx, (local_pids, local_scores, size)) in cell_results.into_iter().enumerate()
            {
                candidate_sizes[cell_idx] = size;
                candidate_pids.extend(local_pids);
                candidate_scores.extend(local_scores);
                let next_offset = offsets.last().copied().unwrap_or(0) + size as i64;
                offsets.push(next_offset);
            }
        } else {
            // Sequential processing
            for cell_idx in 0..num_cells {
                let capacity = capacities_vec[cell_idx] as usize;
                if capacity == 0 {
                    offsets.push(*offsets.last().unwrap());
                    continue;
                }
                let centroid_score = centroid_scores_vec[cell_idx];

                let token_idx = (cell_idx / nprobe).min(num_tokens - 1);
                let bucket_scores_offset = token_idx * bucket_score_stride;
                let token_bucket_scores = &bucket_scores_flat
                    [bucket_scores_offset..bucket_scores_offset + bucket_score_stride];
                let mut size = 0i32;
                let mut prev_pid: Option<i64> = None;

                let (start_idx, end_idx) = cell_ranges[cell_idx];

                for local_idx in 0..(end_idx - start_idx) {
                    let global_idx = start_idx + local_idx;
                    let pid = all_pids_vec[global_idx];

                    // extract residual bytes for the embedding
                    let residual_start = global_idx * residual_bytes_per_embedding;
                    let residual_end = residual_start + residual_bytes_per_embedding;
                    let residual_bytes = &all_residuals_data[residual_start..residual_end];

                    let residual_score = if self.nbits == 2 {
                        Self::decompress_residual_2bit(
                            &residual_bytes,
                            &self.reversed_bit_map,
                            token_bucket_scores,
                            bucket_dim_shift,
                        )
                    } else {
                        Self::decompress_residual_4bit(
                            &residual_bytes,
                            &self.reversed_bit_map,
                            token_bucket_scores,
                            bucket_dim_shift,
                        )
                    };

                    let total_score = centroid_score + residual_score;

                    match prev_pid {
                        Some(prev) if prev == pid => {
                            let last_idx = candidate_scores.len() - 1;
                            if total_score > candidate_scores[last_idx] {
                                candidate_scores[last_idx] = total_score;
                            }
                        },
                        _ => {
                            candidate_pids.push(pid);
                            candidate_scores.push(total_score);
                            size += 1;
                            prev_pid = Some(pid);
                        },
                    }
                }

                // ensure size reflects whether any entry was written
                if size > 0 || prev_pid.is_some() {
                    candidate_sizes[cell_idx] = size;
                }

                let next_offset = offsets.last().copied().unwrap_or(0) + size as i64;
                offsets.push(next_offset);
            }
        }

        let sizes_tensor = Tensor::from_slice(&candidate_sizes)
            .to_device(self.device)
            .to_kind(Kind::Int);
        let pids_tensor = Tensor::from_slice(&candidate_pids)
            .to_device(self.device)
            .to_kind(Kind::Int64);
        let scores_tensor = Tensor::from_slice(&candidate_scores)
            .to_device(self.device)
            .to_kind(self.dtype);
        let offsets_tensor = Tensor::from_slice(&offsets)
            .to_device(self.device)
            .to_kind(Kind::Int64);

        Ok(DecompressedCentroidsOutput {
            capacities,
            sizes: sizes_tensor,
            passage_ids: pids_tensor,
            scores: scores_tensor,
            offsets: offsets_tensor,
        })
    }

    /// This is the original sequential implementation
    pub fn old_decompress_centroids(
        &self,
        centroid_ids: &Tensor,
        centroid_scores: &Tensor,
        index: &Arc<ReadOnlyIndex>,
        query_embeddings: &Tensor, // [num_tokens, dim]
        nprobe: usize,
    ) -> Result<DecompressedCentroidsOutput> {
        let centroid_ids = centroid_ids.to_kind(Kind::Int64);
        let num_cells = centroid_ids.size()[0] as usize;

        let num_total_centroids = index.offsets_compacted.size()[0] - 1;
        let max_centroid_id = centroid_ids.max().int64_value(&[]);

        if max_centroid_id >= num_total_centroids {
            return Err(anyhow!(
                "Centroid ID {} is out of bounds (max valid ID is {})",
                max_centroid_id,
                num_total_centroids - 1
            ));
        }

        // Gather begin/end offsets and capacities for every requested centroid
        let begins = index.offsets_compacted.index_select(0, &centroid_ids);
        let ends = index.offsets_compacted.index_select(0, &(centroid_ids + 1));
        let capacities = &ends - &begins;

        // Early exit when there is nothing to process
        if num_cells == 0 {
            let empty = Tensor::zeros(&[0], (Kind::Int, self.device));
            return Ok(DecompressedCentroidsOutput {
                capacities,
                sizes: empty.to_kind(Kind::Int),
                passage_ids: Tensor::zeros(&[0], (Kind::Int64, self.device)),
                scores: Tensor::zeros(&[0], (self.dtype, self.device)),
                offsets: Tensor::zeros(&[1], (Kind::Int64, self.device)),
            });
        }

        anyhow::ensure!(nprobe > 0, "nprobe must be greater than zero");

        let query_embeddings = query_embeddings.to_kind(self.dtype);
        anyhow::ensure!(
            query_embeddings.size()[1] == self.dim as i64,
            "Query embedding dim ({}) does not match index dim ({})",
            query_embeddings.size()[1],
            self.dim
        );

        let num_tokens = query_embeddings.size()[0] as usize;
        anyhow::ensure!(
            num_tokens > 0,
            "Expected at least one query token for decompression"
        );

        let bucket_weights = index.bucket_weights.to_kind(self.dtype);
        let vt_bucket_scores =
            (query_embeddings.unsqueeze(2) * &bucket_weights.unsqueeze(0)).contiguous();

        let bucket_scores_flat: Vec<f32> = vt_bucket_scores.flatten(0, -1).try_into()?;
        let centroid_scores_vec: Vec<f32> = centroid_scores.flatten(0, -1).try_into()?;

        anyhow::ensure!(
            centroid_scores_vec.len() == num_cells,
            "Centroid score count ({}) does not match number of cells ({})",
            centroid_scores_vec.len(),
            num_cells
        );

        let total_capacity = capacities.sum(Kind::Int64).int64_value(&[]).max(0) as usize;

        let mut candidate_sizes = vec![0i32; num_cells];
        let mut candidate_pids = Vec::with_capacity(total_capacity);
        let mut candidate_scores = Vec::with_capacity(total_capacity);
        let mut offsets = Vec::with_capacity(num_cells + 1);
        offsets.push(0i64);

        let num_buckets = 1usize << (self.nbits as usize);
        let bucket_dim_shift = self.nbits as usize;
        let bucket_score_stride = self.dim * num_buckets;
        let packed_vals_per_byte = 8usize / self.nbits as usize;
        let residual_bytes_per_embedding = self.dim / packed_vals_per_byte;

        // we need to prefetch all data to cpu, to avoid costly gpu -> cpu transfers
        let capacities_vec: Vec<i64> = capacities.shallow_clone().try_into()?;
        let begins_vec: Vec<i64> = begins.try_into()?;

        // collect all indices
        let mut all_indices = Vec::new();
        let mut cell_ranges = Vec::new();

        for cell_idx in 0..num_cells {
            let capacity = capacities_vec[cell_idx] as usize;
            if capacity == 0 {
                cell_ranges.push((0, 0));
                continue;
            }

            let begin = begins_vec[cell_idx];
            let start_idx = all_indices.len();
            for inner_idx in 0..capacity {
                all_indices.push(begin + inner_idx as i64);
            }
            let end_idx = all_indices.len();
            cell_ranges.push((start_idx, end_idx));
        }

        let (all_pids_vec, all_residuals_data) = if !all_indices.is_empty() {
            let indices_tensor =
                Tensor::from_slice(&all_indices).to_device(index.codes_compacted.device());
            let batch_pids = index.codes_compacted.index_select(0, &indices_tensor);
            let batch_residuals = index.residuals_compacted.index_select(0, &indices_tensor);

            // transfer to cpu
            let pids_vec: Vec<i64> = batch_pids.try_into()?;
            let residuals_flat: Vec<u8> = batch_residuals
                .to_kind(Kind::Uint8)
                .contiguous()
                .view([-1])
                .try_into()?;
            (pids_vec, residuals_flat)
        } else {
            (Vec::new(), Vec::new())
        };

        for cell_idx in 0..num_cells {
            let capacity = capacities_vec[cell_idx] as usize;
            if capacity == 0 {
                offsets.push(*offsets.last().unwrap());
                continue;
            }
            let centroid_score = centroid_scores_vec[cell_idx];

            let token_idx = (cell_idx / nprobe).min(num_tokens - 1);
            let bucket_scores_offset = token_idx * bucket_score_stride;
            let token_bucket_scores = &bucket_scores_flat
                [bucket_scores_offset..bucket_scores_offset + bucket_score_stride];

            let mut size = 0i32;
            let mut prev_pid: Option<i64> = None;

            let (start_idx, end_idx) = cell_ranges[cell_idx];

            for local_idx in 0..(end_idx - start_idx) {
                let global_idx = start_idx + local_idx;
                let pid = all_pids_vec[global_idx];

                // extract residual bytes for the embedding
                let residual_start = global_idx * residual_bytes_per_embedding;
                let residual_end = residual_start + residual_bytes_per_embedding;
                let residual_bytes = &all_residuals_data[residual_start..residual_end];

                let residual_score = if self.nbits == 2 {
                    Self::decompress_residual_2bit(
                        &residual_bytes,
                        &self.reversed_bit_map,
                        token_bucket_scores,
                        bucket_dim_shift,
                    )
                } else {
                    Self::decompress_residual_4bit(
                        &residual_bytes,
                        &self.reversed_bit_map,
                        token_bucket_scores,
                        bucket_dim_shift,
                    )
                };

                let total_score = centroid_score + residual_score;

                match prev_pid {
                    Some(prev) if prev == pid => {
                        let last_idx = candidate_scores.len() - 1;
                        if total_score > candidate_scores[last_idx] {
                            candidate_scores[last_idx] = total_score;
                        }
                    },
                    _ => {
                        candidate_pids.push(pid);
                        candidate_scores.push(total_score);
                        size += 1;
                        prev_pid = Some(pid);
                    },
                }
            }

            // Ensure size reflects whether any entry was written
            if size > 0 || prev_pid.is_some() {
                candidate_sizes[cell_idx] = size;
            }

            let next_offset = offsets.last().copied().unwrap_or(0) + size as i64;
            offsets.push(next_offset);
        }

        let sizes_tensor = Tensor::from_slice(&candidate_sizes)
            .to_device(self.device)
            .to_kind(Kind::Int);
        let pids_tensor = Tensor::from_slice(&candidate_pids)
            .to_device(self.device)
            .to_kind(Kind::Int64);
        let scores_tensor = Tensor::from_slice(&candidate_scores)
            .to_device(self.device)
            .to_kind(self.dtype);
        let offsets_tensor = Tensor::from_slice(&offsets)
            .to_device(self.device)
            .to_kind(Kind::Int64);

        Ok(DecompressedCentroidsOutput {
            capacities,
            sizes: sizes_tensor,
            passage_ids: pids_tensor,
            scores: scores_tensor,
            offsets: offsets_tensor,
        })
    }

    // Helper function to process a single cell
    fn process_cell(
        &self,
        cell_idx: usize,
        capacities_vec: &[i64],
        centroid_scores_vec: &[f32],
        nprobe: usize,
        num_tokens: usize,
        bucket_score_stride: usize,
        bucket_scores_flat: &[f32],
        cell_ranges: &[(usize, usize)],
        all_pids_vec: &[i64],
        all_residuals_data: &[u8],
        residual_bytes_per_embedding: usize,
        bucket_dim_shift: usize,
    ) -> (Vec<i64>, Vec<f32>, i32) {
        let capacity = capacities_vec[cell_idx] as usize;
        if capacity == 0 {
            return (vec![], vec![], 0i32);
        }

        let centroid_score = centroid_scores_vec[cell_idx];
        let token_idx = (cell_idx / nprobe).min(num_tokens - 1);
        let bucket_scores_offset = token_idx * bucket_score_stride;
        let token_bucket_scores =
            &bucket_scores_flat[bucket_scores_offset..bucket_scores_offset + bucket_score_stride];

        let mut local_pids = Vec::with_capacity(capacity);
        let mut local_scores = Vec::with_capacity(capacity);
        let mut size = 0i32;
        let mut prev_pid: Option<i64> = None;

        let (start_idx, end_idx) = cell_ranges[cell_idx];

        for local_idx in 0..(end_idx - start_idx) {
            let global_idx = start_idx + local_idx;
            let pid = all_pids_vec[global_idx];

            // extract residual bytes for the embedding
            let residual_start = global_idx * residual_bytes_per_embedding;
            let residual_end = residual_start + residual_bytes_per_embedding;
            let residual_bytes = &all_residuals_data[residual_start..residual_end];

            let residual_score = if self.nbits == 2 {
                Self::decompress_residual_2bit(
                    &residual_bytes,
                    &self.reversed_bit_map,
                    token_bucket_scores,
                    bucket_dim_shift,
                )
            } else {
                Self::decompress_residual_4bit(
                    &residual_bytes,
                    &self.reversed_bit_map,
                    token_bucket_scores,
                    bucket_dim_shift,
                )
            };

            let total_score = centroid_score + residual_score;

            match prev_pid {
                Some(prev) if prev == pid => {
                    let last_idx = local_scores.len() - 1;
                    if total_score > local_scores[last_idx] {
                        local_scores[last_idx] = total_score;
                    }
                },
                _ => {
                    local_pids.push(pid);
                    local_scores.push(total_score);
                    size += 1;
                    prev_pid = Some(pid);
                },
            }
        }

        (local_pids, local_scores, size)
    }

    fn decompress_residual_2bit(
        residual: &[u8],
        reversed_bit_map: &[u8; 256],
        bucket_scores: &[f32],
        bucket_dim_shift: usize,
    ) -> f32 {
        let mut score = 0.0f32;
        for (packed_idx, &packed_val) in residual.iter().enumerate() {
            let packed_val = reversed_bit_map[packed_val as usize];
            let unpacked_idx_0 = packed_idx << 2;
            let unpacked_idx_1 = unpacked_idx_0 + 1;
            let unpacked_idx_2 = unpacked_idx_0 + 2;
            let unpacked_idx_3 = unpacked_idx_0 + 3;

            let unpacked_0 = (packed_val >> 6) as usize;
            let unpacked_1 = ((packed_val >> 4) & 0x03) as usize;
            let unpacked_2 = ((packed_val >> 2) & 0x03) as usize;
            let unpacked_3 = (packed_val & 0x03) as usize;

            let idx0 = (unpacked_idx_0 << bucket_dim_shift) | unpacked_0;
            let idx1 = (unpacked_idx_1 << bucket_dim_shift) | unpacked_1;
            let idx2 = (unpacked_idx_2 << bucket_dim_shift) | unpacked_2;
            let idx3 = (unpacked_idx_3 << bucket_dim_shift) | unpacked_3;

            score += bucket_scores[idx0]
                + bucket_scores[idx1]
                + bucket_scores[idx2]
                + bucket_scores[idx3];
        }
        score
    }

    fn decompress_residual_4bit(
        residual: &[u8],
        reversed_bit_map: &[u8; 256],
        bucket_scores: &[f32],
        bucket_dim_shift: usize,
    ) -> f32 {
        let mut score = 0.0f32;
        for (packed_idx, &packed_val) in residual.iter().enumerate() {
            let packed_val = reversed_bit_map[packed_val as usize];
            let unpacked_idx_0 = packed_idx << 1;
            let unpacked_idx_1 = unpacked_idx_0 + 1;

            let unpacked_0 = (packed_val >> 4) as usize;
            let unpacked_1 = (packed_val & 0x0F) as usize;

            let idx0 = (unpacked_idx_0 << bucket_dim_shift) | unpacked_0;
            let idx1 = (unpacked_idx_1 << bucket_dim_shift) | unpacked_1;

            score += bucket_scores[idx0] + bucket_scores[idx1];
        }
        score
    }
}
