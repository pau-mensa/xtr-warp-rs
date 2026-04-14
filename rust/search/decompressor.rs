use anyhow::{anyhow, Result};
use rayon::{prelude::*, ThreadPool};
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

use crate::utils::types::{
    DecompressSource, DecompressedCentroidsOutput, IndexShard, PassageBitset, ReadOnlyIndex,
    ShardSource,
};

/// Centroid decompressor for efficient residual decompression
#[derive(Clone)]
pub struct CentroidDecompressor {
    nbits: u8,
    dtype: Kind,
    dim: usize,
    reversed_bit_map: [u8; 256],
    thread_pool: Arc<ThreadPool>,
}

impl CentroidDecompressor {
    /// Create a new centroid decompressor
    pub fn new(nbits: u8, dim: usize, dtype: Kind, thread_pool: Arc<ThreadPool>) -> Result<Self> {
        if nbits != 2 && nbits != 4 {
            return Err(anyhow!("nbits must be 2 or 4, got {}", nbits));
        }

        let reversed_bit_map = Self::build_reversed_bit_map(nbits);

        Ok(Self {
            nbits,
            dtype,
            dim,
            reversed_bit_map,
            thread_pool,
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

    /// Decompress centroids for a non-sharded (whole) index.
    pub fn decompress_centroids(
        &self,
        centroid_ids: &Tensor,
        centroid_scores: &Tensor,
        index: &Arc<ReadOnlyIndex>,
        query_embeddings: &Tensor,
        nprobe: usize,
        subset: Option<&[i64]>,
    ) -> Result<DecompressedCentroidsOutput> {
        self.decompress(
            centroid_ids,
            centroid_scores,
            index.as_ref(),
            query_embeddings,
            nprobe,
            subset,
            None,
        )
    }

    /// Decompress centroids for a single shard. Global centroid IDs are
    /// translated to shard-local IDs before indexing into the shard's tensors.
    ///
    /// `per_cell_tokens` maps each local cell position to its query token
    /// index. When `None`, the decompressor computes `cell_idx / nprobe`
    /// (correct for single-shard case where local cell == global cell).
    pub fn decompress_centroids_for_shard(
        &self,
        centroid_ids: &Tensor,
        centroid_scores: &Tensor,
        shard: &IndexShard,
        bucket_weights: &Tensor,
        query_embeddings: &Tensor,
        nprobe: usize,
        subset: Option<&[i64]>,
        per_cell_tokens: Option<&Tensor>,
    ) -> Result<DecompressedCentroidsOutput> {
        let source = ShardSource {
            shard,
            bucket_weights,
        };
        self.decompress(
            centroid_ids,
            centroid_scores,
            &source,
            query_embeddings,
            nprobe,
            subset,
            per_cell_tokens,
        )
    }

    /// Core decompression entry point, generic over the data source.
    fn decompress<S: DecompressSource>(
        &self,
        centroid_ids: &Tensor,
        centroid_scores: &Tensor,
        source: &S,
        query_embeddings: &Tensor,
        nprobe: usize,
        subset: Option<&[i64]>,
        per_cell_tokens: Option<&Tensor>,
    ) -> Result<DecompressedCentroidsOutput> {
        let centroid_ids = centroid_ids.to_kind(Kind::Int64);
        let num_cells = centroid_ids.size()[0] as usize;
        let device = source.device();

        // Empty result for zero cells
        if num_cells == 0 {
            let empty = Tensor::zeros(&[0], (Kind::Int, device));
            return Ok(DecompressedCentroidsOutput {
                capacities: Tensor::zeros(&[0], (Kind::Int64, device)),
                sizes: empty,
                passage_ids: Tensor::zeros(&[0], (Kind::Int64, device)),
                scores: Tensor::zeros(&[0], (self.dtype, device)),
                offsets: Tensor::zeros(&[1], (Kind::Int64, device)),
            });
        }

        // Translate global centroid IDs to source-local IDs
        let local_ids = source.localize_centroid_ids(&centroid_ids);

        // Bounds check
        let num_source_centroids = source.offsets_compacted().size()[0] - 1;
        let max_centroid_id = local_ids.max().int64_value(&[]);
        if max_centroid_id >= num_source_centroids {
            return Err(anyhow!(
                "Centroid ID {} is out of bounds (max valid ID is {})",
                max_centroid_id,
                num_source_centroids - 1
            ));
        }

        // Gather begin/end offsets and capacities
        let begins = source.offsets_compacted().index_select(0, &local_ids);
        let ends = source
            .offsets_compacted()
            .index_select(0, &(&local_ids + 1));
        let capacities = &ends - &begins;

        anyhow::ensure!(nprobe > 0, "nprobe must be greater than zero");

        let query_embeddings = query_embeddings.to_kind(self.dtype);
        anyhow::ensure!(
            query_embeddings.size()[1] == self.dim as i64,
            "Query embedding dim ({}) does not match index dim ({})",
            query_embeddings.size()[1],
            self.dim
        );

        // Resolve bucket_weights to correct device/dtype once
        let bw = source.bucket_weights();
        let bucket_weights = if bw.device() == device && bw.kind() == self.dtype {
            bw.shallow_clone()
        } else {
            bw.to_device(device).to_kind(self.dtype)
        };

        if device.is_cuda() {
            return self.decompress_cuda(
                &begins,
                &capacities,
                centroid_scores,
                source,
                &bucket_weights,
                &query_embeddings,
                nprobe,
                subset,
                per_cell_tokens,
            );
        }

        let subset_bitset = subset.map(PassageBitset::new);

        let vt_bucket_scores =
            (query_embeddings.unsqueeze(2) * bucket_weights.unsqueeze(0)).contiguous();

        let bucket_scores_flat: Vec<f32> = vt_bucket_scores.flatten(0, -1).try_into()?;
        let centroid_scores_vec: Vec<f32> = centroid_scores.flatten(0, -1).try_into()?;

        anyhow::ensure!(
            centroid_scores_vec.len() == num_cells,
            "Centroid score count ({}) does not match number of cells ({})",
            centroid_scores_vec.len(),
            num_cells
        );

        let capacities_vec: Vec<i64> = capacities.shallow_clone().try_into()?;
        let begins_vec: Vec<i64> = begins.try_into()?;

        let num_tokens = query_embeddings.size()[0] as usize;
        anyhow::ensure!(
            num_tokens > 0,
            "Expected at least one query token for decompression"
        );

        let num_buckets = 1usize << (self.nbits as usize);
        let bucket_dim_shift = self.nbits as usize;
        let bucket_score_stride = self.dim * num_buckets;
        let packed_vals_per_byte = 8usize / self.nbits as usize;
        let residual_bytes_per_embedding = self.dim / packed_vals_per_byte;

        let total_capacity = capacities_vec.iter().sum::<i64>().max(0) as usize;

        // Convert per-cell token indices (None → derive from cell_idx / nprobe)
        let per_cell_tokens_vec: Option<Vec<i64>> = per_cell_tokens
            .map(|t| t.to_device(Device::Cpu).try_into())
            .transpose()?;

        let mut candidate_sizes = vec![0i32; num_cells];
        let mut candidate_pids = Vec::with_capacity(total_capacity);
        let mut candidate_scores = Vec::with_capacity(total_capacity);
        let mut offsets = Vec::with_capacity(num_cells + 1);
        offsets.push(0i64);

        let use_parallel = self.thread_pool.current_num_threads() > 1 && num_cells > 1;

        if use_parallel {
            let subset_bitset_ref = subset_bitset.as_ref();
            let tokens_ref = per_cell_tokens_vec.as_ref();
            let cell_results: Vec<_> = self.thread_pool.install(|| {
                (0..num_cells)
                    .into_par_iter()
                    .map(|cell_idx| {
                        let token_idx = tokens_ref
                            .map_or(cell_idx / nprobe, |v| v[cell_idx] as usize)
                            .min(num_tokens - 1);
                        self.process_cell_impl(
                            token_idx,
                            cell_idx,
                            &capacities_vec,
                            &begins_vec,
                            &centroid_scores_vec,
                            bucket_score_stride,
                            &bucket_scores_flat,
                            source,
                            residual_bytes_per_embedding,
                            bucket_dim_shift,
                            subset_bitset_ref,
                        )
                    })
                    .collect()
            });

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
            for cell_idx in 0..num_cells {
                let token_idx = per_cell_tokens_vec
                    .as_ref()
                    .map_or(cell_idx / nprobe, |v| v[cell_idx] as usize)
                    .min(num_tokens - 1);

                let (local_pids, local_scores, size) = self.process_cell_impl(
                    token_idx,
                    cell_idx,
                    &capacities_vec,
                    &begins_vec,
                    &centroid_scores_vec,
                    bucket_score_stride,
                    &bucket_scores_flat,
                    source,
                    residual_bytes_per_embedding,
                    bucket_dim_shift,
                    subset_bitset.as_ref(),
                );

                candidate_sizes[cell_idx] = size;
                candidate_pids.extend(local_pids);
                candidate_scores.extend(local_scores);
                let next_offset = offsets.last().copied().unwrap_or(0) + size as i64;
                offsets.push(next_offset);
            }
        }

        let sizes_tensor = Tensor::from_slice(&candidate_sizes)
            .to_device(device)
            .to_kind(Kind::Int);
        let pids_tensor = Tensor::from_slice(&candidate_pids)
            .to_device(device)
            .to_kind(Kind::Int64);
        let scores_tensor = Tensor::from_slice(&candidate_scores)
            .to_device(device)
            .to_kind(self.dtype);
        let offsets_tensor = Tensor::from_slice(&offsets)
            .to_device(device)
            .to_kind(Kind::Int64);

        Ok(DecompressedCentroidsOutput {
            capacities,
            sizes: sizes_tensor,
            passage_ids: pids_tensor,
            scores: scores_tensor,
            offsets: offsets_tensor,
        })
    }

    /// CUDA decompression path, generic over the data source.
    fn decompress_cuda<S: DecompressSource>(
        &self,
        begins: &Tensor,
        capacities: &Tensor,
        centroid_scores: &Tensor,
        source: &S,
        bucket_weights: &Tensor,
        query_embeddings: &Tensor,
        nprobe: usize,
        subset: Option<&[i64]>,
        per_cell_tokens: Option<&Tensor>,
    ) -> Result<DecompressedCentroidsOutput> {
        let device = source.device();
        anyhow::ensure!(
            device.is_cuda(),
            "CUDA decompression requested but source is on {:?}",
            device
        );
        anyhow::ensure!(nprobe > 0, "nprobe must be greater than zero");

        let capacities_i64 = capacities.to_kind(Kind::Int64);
        let num_cells = capacities_i64.size()[0];
        let total_capacity = capacities_i64.sum(Kind::Int64).int64_value(&[]).max(0);

        let mut sizes = capacities_i64.to_kind(Kind::Int);

        let end_offsets = capacities_i64.cumsum(0, Kind::Int64);
        let mut offsets = Tensor::zeros(&[num_cells + 1], (Kind::Int64, device));
        offsets
            .narrow(0, 1, num_cells)
            .copy_(&end_offsets.contiguous());

        if total_capacity == 0 {
            return Ok(DecompressedCentroidsOutput {
                capacities: capacities.shallow_clone(),
                sizes,
                passage_ids: Tensor::zeros(&[0], (Kind::Int64, device)),
                scores: Tensor::zeros(&[0], (Kind::Float, device)),
                offsets,
            });
        }

        let start_offsets = &end_offsets - &capacities_i64;
        let ranges = Tensor::arange(total_capacity, (Kind::Int64, device));

        let cell_ids = Tensor::arange(num_cells, (Kind::Int64, device));
        let candidate_cells =
            cell_ids.repeat_interleave_self_tensor(&capacities_i64, 0, Some(total_capacity));

        let candidate_cell_starts =
            start_offsets.repeat_interleave_self_tensor(&capacities_i64, 0, Some(total_capacity));
        let candidate_begins = begins.to_kind(Kind::Int64).repeat_interleave_self_tensor(
            &capacities_i64,
            0,
            Some(total_capacity),
        );

        let intra = &ranges - &candidate_cell_starts;
        let embedding_indices = &candidate_begins + &intra;

        let mut passage_ids = source
            .pids_compacted()
            .index_select(0, &embedding_indices)
            .to_kind(Kind::Int64);

        // Apply subset filter before expensive residual retrieval
        let (embedding_indices, candidate_cells, total_capacity) = if let Some(subset_ids) = subset
        {
            let subset_tensor = Tensor::from_slice(subset_ids)
                .to_device(device)
                .to_kind(Kind::Int64);
            let max_pid = passage_ids.max().int64_value(&[]);
            let max_subset = subset_tensor.max().int64_value(&[]);
            let lookup_size = max_pid.max(max_subset) + 1;
            let mut lookup = Tensor::zeros(&[lookup_size], (Kind::Bool, device));
            let _ = lookup.index_fill_(0, &subset_tensor, 1);
            let mask = lookup.index_select(0, &passage_ids);
            let valid_indices = mask.nonzero().squeeze_dim(-1);

            if valid_indices.numel() == 0 {
                return Ok(DecompressedCentroidsOutput {
                    capacities: capacities.shallow_clone(),
                    sizes,
                    passage_ids: Tensor::zeros(&[0], (Kind::Int64, device)),
                    scores: Tensor::zeros(&[0], (Kind::Float, device)),
                    offsets,
                });
            }

            passage_ids = passage_ids.index_select(0, &valid_indices);
            let embedding_indices = embedding_indices.index_select(0, &valid_indices);
            let candidate_cells = candidate_cells.index_select(0, &valid_indices);
            let total_capacity = valid_indices.size()[0];

            // Rebuild sizes and offsets to reflect the filtered data so that
            // downstream consumers index into the flat arrays correctly.
            let mut filtered_counts = Tensor::zeros(&[num_cells], (Kind::Int64, device));
            let ones = Tensor::ones(&[total_capacity], (Kind::Int64, device));
            let _ = filtered_counts.scatter_add_(0, &candidate_cells, &ones);
            sizes = filtered_counts.to_kind(Kind::Int);
            offsets = Tensor::zeros(&[num_cells + 1], (Kind::Int64, device));
            offsets
                .narrow(0, 1, num_cells)
                .copy_(&filtered_counts.cumsum(0, Kind::Int64).contiguous());

            (embedding_indices, candidate_cells, total_capacity)
        } else {
            (embedding_indices, candidate_cells, total_capacity)
        };

        let residuals = source
            .residuals_compacted()
            .index_select(0, &embedding_indices)
            .to_kind(Kind::Uint8);

        let packed_vals_per_byte = (8u8 / self.nbits) as i64;
        let packed_dim = residuals.size()[1];
        let dim = query_embeddings.size()[1];
        anyhow::ensure!(
            packed_dim * packed_vals_per_byte == dim,
            "Residual shape mismatch: packed_dim={} implies dim={}, but query dim={}",
            packed_dim,
            packed_dim * packed_vals_per_byte,
            dim
        );

        // Reverse bit order within each n-bit segment.
        let residuals = if self.nbits == 2 {
            let odd_bits = residuals
                .bitwise_and(0xAA)
                .bitwise_right_shift_tensor_scalar(1);
            let even_bits = residuals
                .bitwise_and(0x55)
                .bitwise_left_shift_tensor_scalar(1);
            odd_bits.bitwise_or_tensor(&even_bits)
        } else {
            // nbits == 4
            let swapped = {
                let odd_bits = residuals
                    .bitwise_and(0xAA)
                    .bitwise_right_shift_tensor_scalar(1);
                let even_bits = residuals
                    .bitwise_and(0x55)
                    .bitwise_left_shift_tensor_scalar(1);
                odd_bits.bitwise_or_tensor(&even_bits)
            };
            let hi_pairs = swapped
                .bitwise_and(0xCC)
                .bitwise_right_shift_tensor_scalar(2);
            let lo_pairs = swapped
                .bitwise_and(0x33)
                .bitwise_left_shift_tensor_scalar(2);
            hi_pairs.bitwise_or_tensor(&lo_pairs)
        };

        let codes = if self.nbits == 2 {
            let c0 = residuals.bitwise_right_shift_tensor_scalar(6);
            let c1 = residuals
                .bitwise_right_shift_tensor_scalar(4)
                .bitwise_and(0x03);
            let c2 = residuals
                .bitwise_right_shift_tensor_scalar(2)
                .bitwise_and(0x03);
            let c3 = residuals.bitwise_and(0x03);
            Tensor::stack(&[c0, c1, c2, c3], -1).view([total_capacity, dim])
        } else {
            // nbits == 4
            let hi = residuals.bitwise_right_shift_tensor_scalar(4);
            let lo = residuals.bitwise_and(0x0F);
            Tensor::stack(&[hi, lo], -1).view([total_capacity, dim])
        };

        // Map each candidate to its query token index.
        // When per_cell_tokens is provided, it maps cell positions directly to
        // query tokens. When None, derive from cell_idx / nprobe.
        let token_indices = match per_cell_tokens {
            Some(ti) => ti.to_device(device).index_select(0, &candidate_cells),
            None => candidate_cells.divide_scalar_mode(nprobe as i64, "trunc"),
        };

        let bucket_weights_f = bucket_weights.to_kind(Kind::Float);
        let query = query_embeddings.to_kind(Kind::Float);

        let query_per_candidate = query.index_select(0, &token_indices);
        let codes_flat = codes.to_kind(Kind::Int).view([-1]);
        let weights_flat = bucket_weights_f.index_select(0, &codes_flat);
        let weights = weights_flat.view([total_capacity, dim]);

        let residual_scores = Tensor::einsum(
            "td,td->t",
            &[&query_per_candidate, &weights],
            None::<&[i64]>,
        );

        let centroid_scores_f = centroid_scores.to_kind(Kind::Float);
        let centroid_per_candidate = centroid_scores_f.index_select(0, &candidate_cells);
        let scores = centroid_per_candidate + residual_scores;

        Ok(DecompressedCentroidsOutput {
            capacities: capacities.shallow_clone(),
            sizes,
            passage_ids,
            scores,
            offsets,
        })
    }

    /// Process a single cell on CPU, generic over the data source.
    ///
    /// `token_idx` is the precomputed query token index for this cell.
    /// `data_cell_idx` is the index into capacities/begins/scores arrays.
    fn process_cell_impl<S: DecompressSource>(
        &self,
        token_idx: usize,
        data_cell_idx: usize,
        capacities_vec: &[i64],
        begins_vec: &[i64],
        centroid_scores_vec: &[f32],
        bucket_score_stride: usize,
        bucket_scores_flat: &[f32],
        source: &S,
        residual_bytes_per_embedding: usize,
        bucket_dim_shift: usize,
        subset_bitset: Option<&PassageBitset>,
    ) -> (Vec<i64>, Vec<f32>, i32) {
        let capacity = capacities_vec[data_cell_idx] as usize;
        if capacity == 0 {
            return (vec![], vec![], 0i32);
        }

        let begin = begins_vec[data_cell_idx];

        // Use narrow for zero-copy views into compacted data
        let local_pids_raw: Vec<i64> = source
            .pids_compacted()
            .narrow(0, begin, capacity as i64)
            .try_into()
            .unwrap_or_default();
        let local_residuals_raw: Vec<u8> = source
            .residuals_compacted()
            .narrow(0, begin, capacity as i64)
            .to_kind(Kind::Uint8)
            .contiguous()
            .view([-1])
            .try_into()
            .unwrap_or_default();

        let centroid_score = centroid_scores_vec[data_cell_idx];
        let bucket_scores_offset = token_idx * bucket_score_stride;
        let token_bucket_scores =
            &bucket_scores_flat[bucket_scores_offset..bucket_scores_offset + bucket_score_stride];

        // Score all embeddings in this cell
        let mut scored: Vec<(i64, f32)> = Vec::with_capacity(capacity);
        for i in 0..capacity {
            let pid = local_pids_raw[i];
            if let Some(bitset) = subset_bitset {
                if !bitset.contains(pid) {
                    continue;
                }
            }
            let residual_start = i * residual_bytes_per_embedding;
            let residual_end = residual_start + residual_bytes_per_embedding;
            let residual_bytes = &local_residuals_raw[residual_start..residual_end];

            let residual_score = if self.nbits == 2 {
                Self::decompress_residual_2bit(
                    residual_bytes,
                    &self.reversed_bit_map,
                    token_bucket_scores,
                    bucket_dim_shift,
                )
            } else {
                Self::decompress_residual_4bit(
                    residual_bytes,
                    &self.reversed_bit_map,
                    token_bucket_scores,
                    bucket_dim_shift,
                )
            };

            scored.push((pid, centroid_score + residual_score));
        }

        // Sort by pid for dedup and downstream merge compatibility
        scored.sort_unstable_by_key(|&(pid, _)| pid);

        // Dedup adjacent entries with same pid, keeping max score
        let mut dedup_pids = Vec::with_capacity(capacity);
        let mut dedup_scores = Vec::with_capacity(capacity);

        for &(pid, score) in &scored {
            if let Some(&last_pid) = dedup_pids.last() {
                if last_pid == pid {
                    let last_idx = dedup_scores.len() - 1;
                    if score > dedup_scores[last_idx] {
                        dedup_scores[last_idx] = score;
                    }
                    continue;
                }
            }
            dedup_pids.push(pid);
            dedup_scores.push(score);
        }

        let size = dedup_pids.len() as i32;
        (dedup_pids, dedup_scores, size)
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
