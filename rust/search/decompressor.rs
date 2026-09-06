use anyhow::{anyhow, Result};
use rayon::{prelude::*, ThreadPool};
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

use crate::utils::types::{DecompressedCentroidsOutput, IndexShard, PassageBitset};

/// Centroid decompressor for efficient residual decompression
#[derive(Clone)]
pub struct CentroidDecompressor {
    nbits: u8,
    dim: usize,
    reversed_bit_map: [u8; 256],
    thread_pool: Arc<ThreadPool>,
}

impl CentroidDecompressor {
    /// Create a new centroid decompressor
    pub fn new(nbits: u8, dim: usize, thread_pool: Arc<ThreadPool>) -> Result<Self> {
        if nbits != 2 && nbits != 4 {
            return Err(anyhow!("nbits must be 2 or 4, got {}", nbits));
        }

        let reversed_bit_map = Self::build_reversed_bit_map(nbits);

        Ok(Self {
            nbits,
            dim,
            reversed_bit_map,
            thread_pool,
        })
    }

    pub fn build_reversed_bit_map(nbits: u8) -> [u8; 256] {
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

    /// Build quantized int8 lookup table with per-token symmetric scaling.
    ///
    /// The bit reversal is baked into the table so the scoring loop can
    /// index directly with raw nibble/code values, skipping reversed_bit_map.
    /// See: https://chaochunhsu.github.io/blog/slow-half-of-plaid/
    pub fn build_int8_lut(
        bucket_scores_flat: &[f32],
        num_tokens: usize,
        dim: usize,
        num_buckets: usize,
        nbits: u8,
    ) -> (Vec<i8>, Vec<f32>) {
        let stride = dim * num_buckets;

        let code_rev: Vec<u8> = (0..num_buckets)
            .map(|code| {
                let mut r = 0u8;
                for bit in 0..nbits {
                    if (code as u8) & (1 << bit) != 0 {
                        r |= 1 << (nbits - 1 - bit);
                    }
                }
                r
            })
            .collect();

        let mut weights = vec![0i8; num_tokens * stride];
        let mut scales = vec![0.0f32; num_tokens];

        for token in 0..num_tokens {
            let offset = token * stride;
            let token_scores = &bucket_scores_flat[offset..offset + stride];

            let abs_max = token_scores
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            let scale = if abs_max > 1e-10 {
                abs_max / 127.0
            } else {
                1.0
            };
            let inv_scale = 1.0 / scale;
            scales[token] = scale;

            for d in 0..dim {
                for raw_code in 0..num_buckets {
                    let reversed_code = code_rev[raw_code] as usize;
                    let f32_val = token_scores[d * num_buckets + reversed_code];
                    let quantized =
                        (f32_val * inv_scale).round().clamp(-127.0, 127.0) as i8;
                    weights[offset + d * num_buckets + raw_code] = quantized;
                }
            }
        }

        (weights, scales)
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
        let centroid_ids = centroid_ids.to_kind(Kind::Int64);
        let num_cells = centroid_ids.size()[0] as usize;
        let device = shard.device;

        // Empty result for zero cells
        if num_cells == 0 {
            let empty = Tensor::zeros(&[0], (Kind::Int, device));
            return Ok(DecompressedCentroidsOutput {
                capacities: Tensor::zeros(&[0], (Kind::Int64, device)),
                sizes: empty,
                passage_ids: Tensor::zeros(&[0], (Kind::Int64, device)),
                scores: Tensor::zeros(&[0], (Kind::Float, device)),
                offsets: Tensor::zeros(&[1], (Kind::Int64, device)),
            });
        }

        // Translate global centroid IDs to shard-local IDs
        let local_ids = shard.localize_centroid_ids(&centroid_ids);

        // Bounds check
        let num_source_centroids = shard.offsets_compacted.size()[0] - 1;
        let max_centroid_id = local_ids.max().int64_value(&[]);
        if max_centroid_id >= num_source_centroids {
            return Err(anyhow!(
                "Centroid ID {} is out of bounds (max valid ID is {})",
                max_centroid_id,
                num_source_centroids - 1
            ));
        }

        // Gather begin/end offsets and capacities
        let begins = shard.offsets_compacted.index_select(0, &local_ids);
        let ends = shard.offsets_compacted.index_select(0, &(&local_ids + 1));
        let capacities = &ends - &begins;

        anyhow::ensure!(nprobe > 0, "nprobe must be greater than zero");

        anyhow::ensure!(
            query_embeddings.size()[1] == self.dim as i64,
            "Query embedding dim ({}) does not match index dim ({})",
            query_embeddings.size()[1],
            self.dim
        );

        // Resolve bucket_weights to correct device once
        let bucket_weights = if bucket_weights.device() == device {
            bucket_weights.shallow_clone()
        } else {
            bucket_weights.to_device(device)
        };

        if device.is_cuda() {
            return self.decompress_cuda(
                &begins,
                &capacities,
                centroid_scores,
                shard,
                &bucket_weights,
                query_embeddings,
                nprobe,
                subset,
                per_cell_tokens,
            );
        }

        let subset_bitset = subset.map(PassageBitset::new);

        // CPU: always compute bucket scores in Float32.
        // x86 CPUs lack native FP16 ALU so Half tensor ops are emulated
        // and very slow. The inner scoring loop works with f32 anyway.
        let query_f32 = query_embeddings.to_kind(Kind::Float);
        let bw_f32 = bucket_weights.to_kind(Kind::Float);
        let vt_bucket_scores =
            (query_f32.unsqueeze(2) * bw_f32.unsqueeze(0)).contiguous();

        let bucket_scores_flat: Vec<f32> = vt_bucket_scores.flatten(0, -1).try_into()?;
        let centroid_scores_vec: Vec<f32> = centroid_scores.to_kind(Kind::Float).flatten(0, -1).try_into()?;

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

        // Validate the on-disk residual row width against the metadata dim
        // before scoring anything. `decompress_cuda` already does this; the
        // CPU path did not, so an index whose residuals.npy disagrees with
        // metadata.json (built at a different dim, or truncated) reached the
        // scoring loop and produced wrong scores rather than an error.
        // Checking here, once per call, keeps the per-cell scorers free of
        // shape concerns.
        let packed_dim = shard.residuals_compacted.size()[1];
        anyhow::ensure!(
            packed_dim as usize * packed_vals_per_byte == self.dim,
            "Residual shape mismatch: packed_dim={} implies dim={}, but index dim={}",
            packed_dim,
            packed_dim as usize * packed_vals_per_byte,
            self.dim
        );

        let use_int8 = std::env::var("XTR_WARP_INT8").map_or(true, |v| v != "0");
        let (int8_lut, int8_scales) = if use_int8 {
            let (w, s) = Self::build_int8_lut(
                &bucket_scores_flat,
                num_tokens,
                self.dim,
                num_buckets,
                self.nbits,
            );
            (Some(w), Some(s))
        } else {
            (None, None)
        };

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
            let int8_lut_ref = int8_lut.as_deref();
            let int8_scales_ref = int8_scales.as_deref();
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
                            shard,
                            residual_bytes_per_embedding,
                            bucket_dim_shift,
                            subset_bitset_ref,
                            int8_lut_ref,
                            int8_scales_ref,
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
                    shard,
                    residual_bytes_per_embedding,
                    bucket_dim_shift,
                    subset_bitset.as_ref(),
                    int8_lut.as_deref(),
                    int8_scales.as_deref(),
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
            .to_device(device);
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

    /// CUDA decompression path.
    fn decompress_cuda(
        &self,
        begins: &Tensor,
        capacities: &Tensor,
        centroid_scores: &Tensor,
        shard: &IndexShard,
        bucket_weights: &Tensor,
        query_embeddings: &Tensor,
        nprobe: usize,
        subset: Option<&[i64]>,
        per_cell_tokens: Option<&Tensor>,
    ) -> Result<DecompressedCentroidsOutput> {
        let device = shard.device;
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

        let mut passage_ids = shard
            .pids_compacted
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

        let residuals = shard
            .residuals_compacted
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

    /// Process a single cell on CPU.
    ///
    /// `token_idx` is the precomputed query token index for this cell.
    /// `data_cell_idx` is the index into capacities/begins/scores arrays.
    fn process_cell_impl(
        &self,
        token_idx: usize,
        data_cell_idx: usize,
        capacities_vec: &[i64],
        begins_vec: &[i64],
        centroid_scores_vec: &[f32],
        bucket_score_stride: usize,
        bucket_scores_flat: &[f32],
        shard: &IndexShard,
        residual_bytes_per_embedding: usize,
        bucket_dim_shift: usize,
        subset_bitset: Option<&PassageBitset>,
        int8_lut: Option<&[i8]>,
        int8_scales: Option<&[f32]>,
    ) -> (Vec<i64>, Vec<f32>, i32) {
        let capacity = capacities_vec[data_cell_idx] as usize;
        if capacity == 0 {
            return (vec![], vec![], 0i32);
        }

        let begin = begins_vec[data_cell_idx];

        // Use narrow for zero-copy views into compacted data
        let local_pids_raw: Vec<i64> = shard
            .pids_compacted
            .narrow(0, begin, capacity as i64)
            .try_into()
            .unwrap_or_default();
        let local_residuals_raw: Vec<u8> = shard
            .residuals_compacted
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

        // Pre-compute all residual scores when SIMD batch scoring applies
        // (int8 enabled, 4-bit codes). The batch kernel scores 16 documents
        // per tbl/pshufb instruction instead of one scalar lookup each.
        let batch_i32: Option<Vec<i32>> =
            if let (Some(lut), Some(_)) = (int8_lut, int8_scales) {
                if self.nbits == 4 {
                    let off = token_idx * bucket_score_stride;
                    Some(crate::search::int8_simd::score_batch_4bit(
                        &local_residuals_raw,
                        capacity,
                        residual_bytes_per_embedding,
                        &lut[off..off + bucket_score_stride],
                    ))
                } else {
                    None
                }
            } else {
                None
            };

        // Score all embeddings in this cell
        let mut scored: Vec<(i64, f32)> = Vec::with_capacity(capacity);
        for i in 0..capacity {
            let pid = local_pids_raw[i];
            if let Some(bitset) = subset_bitset {
                if !bitset.contains(pid) {
                    continue;
                }
            }

            let residual_score = if let Some(ref batch) = batch_i32 {
                batch[i] as f32 * int8_scales.unwrap()[token_idx]
            } else {
                let residual_start = i * residual_bytes_per_embedding;
                let residual_end = residual_start + residual_bytes_per_embedding;
                let residual_bytes =
                    &local_residuals_raw[residual_start..residual_end];

                if let (Some(lut), Some(scales)) = (int8_lut, int8_scales) {
                    let off = token_idx * bucket_score_stride;
                    let token_lut = &lut[off..off + bucket_score_stride];
                    // Route on nbits explicitly. Today this branch is only
                    // reachable with nbits == 2 (4-bit is handled by the batch
                    // above), but calling the 2-bit scorer on 4-bit data reads
                    // the wrong LUT entries *without* going out of bounds, so
                    // it would silently return garbage rather than panic if
                    // the batch path ever gains a guard.
                    let raw_sum = if self.nbits == 4 {
                        Self::score_residual_4bit_int8(residual_bytes, token_lut)
                    } else {
                        Self::score_residual_2bit_int8(residual_bytes, token_lut)
                    };
                    raw_sum as f32 * scales[token_idx]
                } else if self.nbits == 2 {
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
                }
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

    pub fn decompress_residual_2bit(
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

    pub fn decompress_residual_4bit(
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

    /// Int8 scoring for 4-bit residuals. Bit reversal is pre-baked into the
    /// LUT, so the hot loop is two table lookups and two i32 adds per byte.
    #[inline]
    pub fn score_residual_4bit_int8(residual: &[u8], lut: &[i8]) -> i32 {
        let mut sum: i32 = 0;
        for (i, &packed) in residual.iter().enumerate() {
            let d0 = i << 1;
            let d1 = d0 + 1;
            let hi = (packed >> 4) as usize;
            let lo = (packed & 0x0F) as usize;
            sum += lut[(d0 << 4) | hi] as i32;
            sum += lut[(d1 << 4) | lo] as i32;
        }
        sum
    }

    /// Int8 scoring for 2-bit residuals.
    #[inline]
    pub fn score_residual_2bit_int8(residual: &[u8], lut: &[i8]) -> i32 {
        let mut sum: i32 = 0;
        for (i, &packed) in residual.iter().enumerate() {
            let d0 = i << 2;
            let c0 = (packed >> 6) as usize;
            let c1 = ((packed >> 4) & 0x03) as usize;
            let c2 = ((packed >> 2) & 0x03) as usize;
            let c3 = (packed & 0x03) as usize;
            sum += lut[(d0 << 2) | c0] as i32;
            sum += lut[((d0 + 1) << 2) | c1] as i32;
            sum += lut[((d0 + 2) << 2) | c2] as i32;
            sum += lut[((d0 + 3) << 2) | c3] as i32;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_4bit_matches_f32() {
        let dim = 128;
        let num_buckets = 16usize;
        let num_tokens = 1;
        let stride = dim * num_buckets;

        let bucket_scores: Vec<f32> = (0..num_tokens * stride)
            .map(|i| (i as f32 * 0.7123).sin() * 0.5)
            .collect();

        let reversed_bit_map = CentroidDecompressor::build_reversed_bit_map(4);
        let (i8_lut, scales) =
            CentroidDecompressor::build_int8_lut(&bucket_scores, num_tokens, dim, num_buckets, 4);

        let residual: Vec<u8> = (0..dim / 2).map(|i| (i * 37 + 13) as u8).collect();

        let f32_score = CentroidDecompressor::decompress_residual_4bit(
            &residual,
            &reversed_bit_map,
            &bucket_scores,
            4,
        );

        let i8_sum = CentroidDecompressor::score_residual_4bit_int8(&residual, &i8_lut);
        let i8_score = i8_sum as f32 * scales[0];

        let abs_err = (f32_score - i8_score).abs();
        let rel_err = abs_err / f32_score.abs().max(1e-6);
        assert!(
            rel_err < 0.02,
            "f32={f32_score:.6} i8={i8_score:.6} rel_err={rel_err:.4}"
        );
    }

    #[test]
    fn int8_2bit_matches_f32() {
        let dim = 128;
        let num_buckets = 4usize;
        let num_tokens = 1;
        let stride = dim * num_buckets;

        let bucket_scores: Vec<f32> = (0..num_tokens * stride)
            .map(|i| (i as f32 * 0.3917).cos() * 0.4)
            .collect();

        let reversed_bit_map = CentroidDecompressor::build_reversed_bit_map(2);
        let (i8_lut, scales) =
            CentroidDecompressor::build_int8_lut(&bucket_scores, num_tokens, dim, num_buckets, 2);

        let residual: Vec<u8> = (0..dim / 4).map(|i| (i * 53 + 7) as u8).collect();

        let f32_score = CentroidDecompressor::decompress_residual_2bit(
            &residual,
            &reversed_bit_map,
            &bucket_scores,
            2,
        );

        let i8_sum = CentroidDecompressor::score_residual_2bit_int8(&residual, &i8_lut);
        let i8_score = i8_sum as f32 * scales[0];

        let abs_err = (f32_score - i8_score).abs();
        let rel_err = abs_err / f32_score.abs().max(1e-6);
        assert!(
            rel_err < 0.02,
            "f32={f32_score:.6} i8={i8_score:.6} rel_err={rel_err:.4}"
        );
    }
}
