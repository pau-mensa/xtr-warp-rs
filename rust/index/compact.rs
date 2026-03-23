use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::utils::types::CompactStats;

/// Per-centroid data built from a subset of chunks (used by incremental merge).
pub struct PartialCompacted {
    /// Passage IDs per centroid (sorted within each centroid).
    pub centroid_pids: Vec<Vec<i64>>,
    /// Flattened residual bytes per centroid.
    pub centroid_residuals: Vec<Vec<u8>>,
    /// Number of embeddings per centroid.
    pub centroid_counts: Vec<i64>,
    /// Total active passage count across these chunks.
    pub num_active_passages: usize,
    /// Total embedding count across these chunks.
    pub total_embeddings: i64,
}

/// Build the IVF (inverted file) from compacted codes and write to disk.
/// Returns the number of unique passage IDs across all centroids.
fn build_ivf_from_compacted(
    codes_compacted: &Tensor,
    sizes_vec: &[i64],
    index_path: &Path,
    device: Device,
) -> Result<usize> {
    let mut ivf_list: Vec<Tensor> = Vec::with_capacity(sizes_vec.len());
    let mut ivf_lens_vec: Vec<i64> = Vec::with_capacity(sizes_vec.len());
    let mut all_unique_pids: HashSet<i64> = HashSet::new();

    let mut offset: i64 = 0;
    for &size in sizes_vec {
        if size == 0 {
            ivf_lens_vec.push(0);
            continue;
        }
        let segment = codes_compacted.narrow(0, offset, size);
        let (unique_pids, _, _) = segment.unique_dim(0, true, false, false);
        ivf_lens_vec.push(unique_pids.size()[0]);
        let pids_vec: Vec<i64> = unique_pids
            .to_device(Device::Cpu)
            .try_into()?;
        all_unique_pids.extend(&pids_vec);
        ivf_list.push(unique_pids);
        offset += size;
    }

    let ivf = if ivf_list.is_empty() {
        Tensor::zeros(&[0], (Kind::Int64, device))
    } else {
        Tensor::cat(&ivf_list, 0)
    };
    let ivf_lens = Tensor::from_slice(&ivf_lens_vec)
        .to_device(device)
        .to_kind(Kind::Int64);

    ivf.to_device(Device::Cpu)
        .write_npy(index_path.join("ivf.npy"))?;
    ivf_lens
        .to_device(Device::Cpu)
        .write_npy(index_path.join("ivf_lengths.npy"))?;

    Ok(all_unique_pids.len())
}

/// Compact all chunks into the compacted layout, excluding tombstoned passage IDs.
///
/// Two-pass counting sort: first counts per-centroid embeddings (excluding deleted),
/// then places surviving embeddings into the compacted layout.
/// Pass `&HashSet::new()` for `deleted_pids` when no filtering is needed.
pub fn compact_index(
    index_path: &Path,
    num_chunks: usize,
    num_centroids: usize,
    embedding_dim: usize,
    nbits: usize,
    device: Device,
    deleted_pids: &HashSet<i64>,
) -> Result<CompactStats> {
    // ── Pass 1: count embeddings per centroid, excluding deleted ──
    let mut centroid_counts = vec![0i64; num_centroids];
    let mut total_filtered = 0i64;
    let mut active_pids_set: HashSet<i64> = HashSet::new();

    for chunk_idx in 0..num_chunks {
        let codes = Tensor::read_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64);
        let doclens = Tensor::read_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64);
        let doclens_vec: Vec<i64> = doclens.try_into()?;

        let pids_base: Vec<i64> = Tensor::read_npy(index_path.join(format!("{}.passage_ids.npy", chunk_idx)))?
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64)
            .try_into()?;

        let codes_vec: Vec<i64> = codes.try_into()?;
        let mut emb_offset = 0usize;
        for (doc_idx, &doc_len) in doclens_vec.iter().enumerate() {
            let pid = pids_base[doc_idx];
            if deleted_pids.contains(&pid) {
                emb_offset += doc_len as usize;
                continue;
            }
            active_pids_set.insert(pid);
            for _ in 0..doc_len as usize {
                let centroid_id = codes_vec[emb_offset] as usize;
                centroid_counts[centroid_id] += 1;
                total_filtered += 1;
                emb_offset += 1;
            }
        }
    }

    // Build offsets
    let mut offsets_vec = vec![0i64; num_centroids + 1];
    for i in 0..num_centroids {
        offsets_vec[i + 1] = offsets_vec[i] + centroid_counts[i];
    }

    // Write sizes
    let sizes_tensor = Tensor::from_slice(&centroid_counts);
    sizes_tensor.write_npy(index_path.join("sizes.compacted.npy"))?;

    // ── Pass 2: place non-deleted embeddings into compacted arrays ──
    let residual_dim = (embedding_dim * nbits) / 8;
    let compacted_residuals =
        Tensor::zeros(&[total_filtered, residual_dim as i64], (Kind::Uint8, device));
    let compacted_codes = Tensor::zeros(&[total_filtered], (Kind::Int64, device));

    let mut write_offsets = offsets_vec[..num_centroids].to_vec();

    for chunk_idx in 0..num_chunks {
        let codes = Tensor::read_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?
            .to_device(device);
        let residuals =
            Tensor::read_npy(index_path.join(format!("{}.residuals.npy", chunk_idx)))?
                .to_device(device);
        let doclens = Tensor::read_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?
            .to_device(device)
            .to_kind(Kind::Int64);
        let pids_base = Tensor::read_npy(index_path.join(format!("{}.passage_ids.npy", chunk_idx)))?
            .to_device(device)
            .to_kind(Kind::Int64);

        let chunk_total = codes.size()[0];
        let pids = Tensor::repeat_interleave_self_tensor(
            &pids_base,
            &doclens,
            0,
            Some(chunk_total),
        );

        // Build keep indices (non-deleted)
        let pids_vec: Vec<i64> = pids.to_device(Device::Cpu).try_into()?;
        let keep_indices: Vec<i64> = (0..chunk_total)
            .filter(|&i| !deleted_pids.contains(&pids_vec[i as usize]))
            .collect();

        if keep_indices.is_empty() {
            continue;
        }

        let keep_tensor = Tensor::from_slice(&keep_indices).to_device(device);
        let filtered_codes = codes.index_select(0, &keep_tensor);
        let filtered_residuals = residuals.index_select(0, &keep_tensor);
        let filtered_pids = pids.index_select(0, &keep_tensor);

        // Sort by centroid for counting-sort placement
        let sort_result = filtered_codes.sort(0, false);
        let sorted_codes = sort_result.0;
        let sorted_idx = sort_result.1;
        let sorted_residuals = filtered_residuals.index_select(0, &sorted_idx);
        let sorted_pids = filtered_pids.index_select(0, &sorted_idx);

        let chunk_counts = sorted_codes.bincount::<Tensor>(None, num_centroids as i64);
        let chunk_counts_vec: Vec<i64> = chunk_counts.to_device(Device::Cpu).try_into()?;

        let mut local_offset: i64 = 0;
        for (centroid_id, &count) in chunk_counts_vec.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let write_pos = write_offsets[centroid_id];
            compacted_residuals
                .narrow(0, write_pos, count)
                .copy_(&sorted_residuals.narrow(0, local_offset, count));
            compacted_codes
                .narrow(0, write_pos, count)
                .copy_(&sorted_pids.narrow(0, local_offset, count));
            write_offsets[centroid_id] += count;
            local_offset += count;
        }
    }

    // Write compacted data
    compacted_residuals
        .to_device(Device::Cpu)
        .write_npy(index_path.join("residuals.compacted.npy"))?;
    compacted_codes
        .to_device(Device::Cpu)
        .write_npy(index_path.join("codes.compacted.npy"))?;

    let offsets_tensor = Tensor::from_slice(&offsets_vec).to_device(device);
    offsets_tensor
        .to_device(Device::Cpu)
        .write_npy(index_path.join("offsets.compacted.npy"))?;

    let _num_unique = build_ivf_from_compacted(&compacted_codes, &centroid_counts, index_path, device)?;

    Ok(CompactStats {
        total_embeddings: total_filtered,
        num_active_passages: active_pids_set.len(),
    })
}

// ── Incremental merge helpers ──

/// Build per-centroid data from a range of chunks.
///
/// Only processes chunks `[start_chunk, end_chunk)`, skipping any
/// tombstoned passage IDs.  When `include_only_pids` is `Some`, only
/// embeddings whose passage ID is in the set are included (used when
/// coalescing to avoid double-counting old data in a merged chunk).
pub fn build_partial_compacted(
    index_path: &Path,
    start_chunk: usize,
    end_chunk: usize,
    num_centroids: usize,
    residual_dim: usize,
    deleted_pids: &HashSet<i64>,
    include_only_pids: Option<&HashSet<i64>>,
) -> Result<PartialCompacted> {
    let mut centroid_pids: Vec<Vec<i64>> = vec![Vec::new(); num_centroids];
    let mut centroid_residuals: Vec<Vec<u8>> = vec![Vec::new(); num_centroids];
    let mut centroid_counts = vec![0i64; num_centroids];
    let mut active_pids_set: HashSet<i64> = HashSet::new();
    let mut total_embeddings: i64 = 0;

    for chunk_idx in start_chunk..end_chunk {
        let codes = Tensor::read_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64);
        let residuals = Tensor::read_npy(index_path.join(format!("{}.residuals.npy", chunk_idx)))?
            .to_device(Device::Cpu);
        let doclens = Tensor::read_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64);
        let doclens_vec: Vec<i64> = doclens.try_into()?;

        let pids_vec: Vec<i64> = Tensor::read_npy(index_path.join(format!("{}.passage_ids.npy", chunk_idx)))?
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64)
            .try_into()?;

        let codes_vec: Vec<i64> = codes.try_into()?;
        // Get residuals as raw bytes — flatten [total_embs, residual_dim] -> [total_embs * residual_dim]
        let res_flat: Vec<u8> = Vec::<u8>::try_from(
            residuals.to_kind(Kind::Uint8).contiguous().flatten(0, -1),
        )?;

        let mut emb_offset = 0usize;
        for (doc_idx, &doc_len) in doclens_vec.iter().enumerate() {
            let pid = pids_vec[doc_idx];
            let skip = deleted_pids.contains(&pid)
                || include_only_pids.map_or(false, |set| !set.contains(&pid));
            if skip {
                emb_offset += doc_len as usize;
                continue;
            }
            active_pids_set.insert(pid);
            for _ in 0..doc_len as usize {
                let centroid_id = codes_vec[emb_offset] as usize;
                centroid_pids[centroid_id].push(pid);
                let byte_start = emb_offset * residual_dim;
                let byte_end = byte_start + residual_dim;
                centroid_residuals[centroid_id].extend_from_slice(&res_flat[byte_start..byte_end]);
                centroid_counts[centroid_id] += 1;
                total_embeddings += 1;
                emb_offset += 1;
            }
        }
    }

    Ok(PartialCompacted {
        centroid_pids,
        centroid_residuals,
        centroid_counts,
        num_active_passages: active_pids_set.len(),
        total_embeddings,
    })
}

/// Incrementally merge new data into existing compacted structures.
///
/// For centroids that are untouched (no new data), copies contiguous ranges
/// from the old compacted arrays.  For modified centroids, copies old data
/// and appends new data.
pub fn merge_compacted_incremental(
    index_path: &Path,
    partial: &PartialCompacted,
    num_centroids: usize,
    residual_dim: usize,
    device: Device,
) -> Result<CompactStats> {
    // Load existing compacted structures
    let old_sizes_path = index_path.join("sizes.compacted.npy");
    let old_sizes_tensor = Tensor::read_npy(&old_sizes_path)?
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64);
    let old_sizes: Vec<i64> = old_sizes_tensor.try_into()?;

    let old_codes = Tensor::read_npy(index_path.join("codes.compacted.npy"))?
        .to_device(device)
        .to_kind(Kind::Int64);
    let old_residuals = Tensor::read_npy(index_path.join("residuals.compacted.npy"))?
        .to_device(device);

    // Compute old offsets
    let mut old_offsets = vec![0i64; num_centroids + 1];
    for i in 0..num_centroids {
        old_offsets[i + 1] = old_offsets[i] + old_sizes[i];
    }
    let _old_total = old_offsets[num_centroids];

    // Compute new sizes and offsets
    let new_sizes: Vec<i64> = old_sizes
        .iter()
        .zip(&partial.centroid_counts)
        .map(|(&o, &n)| o + n)
        .collect();

    let mut new_offsets = vec![0i64; num_centroids + 1];
    for i in 0..num_centroids {
        new_offsets[i + 1] = new_offsets[i] + new_sizes[i];
    }
    let new_total = new_offsets[num_centroids];

    // Allocate new compacted arrays
    let new_codes = Tensor::zeros(&[new_total], (Kind::Int64, device));
    let new_residuals =
        Tensor::zeros(&[new_total, residual_dim as i64], (Kind::Uint8, device));

    // Walk centroids and merge
    let mut c = 0usize;
    while c < num_centroids {
        if partial.centroid_counts[c] == 0 {
            // Fast path: find contiguous range of unchanged centroids
            let range_start = c;
            while c < num_centroids && partial.centroid_counts[c] == 0 {
                c += 1;
            }
            // Copy entire contiguous slice at once
            let old_start = old_offsets[range_start];
            let old_len = old_offsets[c] - old_start;
            let dst_start = new_offsets[range_start];
            if old_len > 0 {
                new_codes
                    .narrow(0, dst_start, old_len)
                    .copy_(&old_codes.narrow(0, old_start, old_len));
                new_residuals
                    .narrow(0, dst_start, old_len)
                    .copy_(&old_residuals.narrow(0, old_start, old_len));
            }
        } else {
            // Modified partition: copy old data + append new
            let old_start = old_offsets[c];
            let old_count = old_sizes[c];
            let dst_start = new_offsets[c];

            if old_count > 0 {
                new_codes
                    .narrow(0, dst_start, old_count)
                    .copy_(&old_codes.narrow(0, old_start, old_count));
                new_residuals
                    .narrow(0, dst_start, old_count)
                    .copy_(&old_residuals.narrow(0, old_start, old_count));
            }

            // Append new data
            let new_count = partial.centroid_counts[c];
            let append_start = dst_start + old_count;

            let pids_tensor =
                Tensor::from_slice(&partial.centroid_pids[c]).to_device(device);
            new_codes
                .narrow(0, append_start, new_count)
                .copy_(&pids_tensor);

            let res_tensor = Tensor::from_slice(&partial.centroid_residuals[c])
                .to_device(device)
                .reshape([new_count, residual_dim as i64]);
            new_residuals
                .narrow(0, append_start, new_count)
                .copy_(&res_tensor);

            c += 1;
        }
    }

    // Write compacted files
    let new_sizes_tensor = Tensor::from_slice(&new_sizes);
    new_sizes_tensor.write_npy(index_path.join("sizes.compacted.npy"))?;

    new_codes
        .to_device(Device::Cpu)
        .write_npy(index_path.join("codes.compacted.npy"))?;
    new_residuals
        .to_device(Device::Cpu)
        .write_npy(index_path.join("residuals.compacted.npy"))?;

    let new_offsets_tensor = Tensor::from_slice(&new_offsets).to_device(device);
    new_offsets_tensor
        .to_device(Device::Cpu)
        .write_npy(index_path.join("offsets.compacted.npy"))?;

    // Rebuild IVF from new compacted codes
    let num_active_passages =
        build_ivf_from_compacted(&new_codes.to_device(device), &new_sizes, index_path, device)?;

    Ok(CompactStats {
        total_embeddings: new_total,
        num_active_passages,
    })
}

/// Remove entries for specific PIDs from the existing compacted structures,
/// then merge new data in. Used by `update_in_index` to avoid a full recompaction.
///
/// Cost: O(existing_compacted + new_embeddings) — reads old compacted once,
/// filters, appends new data. Much cheaper than re-scanning all chunks.
pub fn remove_and_merge_compacted(
    index_path: &Path,
    partial: &PartialCompacted,
    pids_to_remove: &HashSet<i64>,
    num_centroids: usize,
    residual_dim: usize,
    device: Device,
) -> Result<CompactStats> {
    let old_sizes_tensor = Tensor::read_npy(index_path.join("sizes.compacted.npy"))?
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64);
    let old_sizes: Vec<i64> = old_sizes_tensor.try_into()?;

    let old_codes = Tensor::read_npy(index_path.join("codes.compacted.npy"))?
        .to_device(device)
        .to_kind(Kind::Int64);
    let old_residuals = Tensor::read_npy(index_path.join("residuals.compacted.npy"))?
        .to_device(device);

    let mut old_offsets = vec![0i64; num_centroids + 1];
    for i in 0..num_centroids {
        old_offsets[i + 1] = old_offsets[i] + old_sizes[i];
    }

    // Single pass over old_codes_vec: build global keep_indices and per-centroid filtered sizes
    let old_codes_vec: Vec<i64> = old_codes.to_device(Device::Cpu).try_into()?;

    let mut keep_indices: Vec<i64> = Vec::with_capacity(old_codes_vec.len());
    let mut filtered_sizes = vec![0i64; num_centroids];
    for c in 0..num_centroids {
        let start = old_offsets[c] as usize;
        let end = old_offsets[c + 1] as usize;
        for (local_i, &pid) in old_codes_vec[start..end].iter().enumerate() {
            if !pids_to_remove.contains(&pid) {
                keep_indices.push((start + local_i) as i64);
                filtered_sizes[c] += 1;
            }
        }
    }

    // Two index_select calls for all kept entries (instead of per-centroid)
    let kept_codes;
    let kept_residuals;
    if !keep_indices.is_empty() {
        let keep_tensor = Tensor::from_slice(&keep_indices).to_device(device);
        kept_codes = old_codes.index_select(0, &keep_tensor);
        kept_residuals = old_residuals.index_select(0, &keep_tensor);
    } else {
        kept_codes = Tensor::zeros(&[0], (Kind::Int64, device));
        kept_residuals = Tensor::zeros(&[0, residual_dim as i64], (Kind::Uint8, device));
    }
    drop(old_codes);
    drop(old_residuals);

    // Compute new sizes and offsets (filtered old + new from partial)
    let new_sizes: Vec<i64> = filtered_sizes
        .iter()
        .zip(&partial.centroid_counts)
        .map(|(&f, &n)| f + n)
        .collect();

    let mut new_offsets = vec![0i64; num_centroids + 1];
    for i in 0..num_centroids {
        new_offsets[i + 1] = new_offsets[i] + new_sizes[i];
    }
    let new_total = new_offsets[num_centroids];

    let new_codes = Tensor::zeros(&[new_total], (Kind::Int64, device));
    let new_residuals = Tensor::zeros(&[new_total, residual_dim as i64], (Kind::Uint8, device));

    // Copy filtered old data — already in centroid order since keep_indices
    // were built in centroid order. Then append new data per centroid.
    let mut filtered_offsets = vec![0i64; num_centroids + 1];
    for i in 0..num_centroids {
        filtered_offsets[i + 1] = filtered_offsets[i] + filtered_sizes[i];
    }

    for c in 0..num_centroids {
        let dst_start = new_offsets[c];

        // Copy filtered old entries for this centroid
        let filt_count = filtered_sizes[c];
        if filt_count > 0 {
            let src_start = filtered_offsets[c];
            new_codes
                .narrow(0, dst_start, filt_count)
                .copy_(&kept_codes.narrow(0, src_start, filt_count));
            new_residuals
                .narrow(0, dst_start, filt_count)
                .copy_(&kept_residuals.narrow(0, src_start, filt_count));
        }

        // Append new data from partial
        let new_count = partial.centroid_counts[c];
        if new_count > 0 {
            let append_start = dst_start + filt_count;

            let pids_tensor =
                Tensor::from_slice(&partial.centroid_pids[c]).to_device(device);
            new_codes
                .narrow(0, append_start, new_count)
                .copy_(&pids_tensor);

            let res_tensor = Tensor::from_slice(&partial.centroid_residuals[c])
                .to_device(device)
                .reshape([new_count, residual_dim as i64]);
            new_residuals
                .narrow(0, append_start, new_count)
                .copy_(&res_tensor);
        }
    }

    // Write compacted files
    Tensor::from_slice(&new_sizes).write_npy(index_path.join("sizes.compacted.npy"))?;
    new_codes
        .to_device(Device::Cpu)
        .write_npy(index_path.join("codes.compacted.npy"))?;
    new_residuals
        .to_device(Device::Cpu)
        .write_npy(index_path.join("residuals.compacted.npy"))?;
    Tensor::from_slice(&new_offsets)
        .to_device(Device::Cpu)
        .write_npy(index_path.join("offsets.compacted.npy"))?;

    let num_active_passages =
        build_ivf_from_compacted(&new_codes.to_device(device), &new_sizes, index_path, device)?;

    Ok(CompactStats {
        total_embeddings: new_total,
        num_active_passages,
    })
}

/// Remove centroids with zero embeddings from the codebook and chunk files.
///
/// Determines which centroids are actually used by scanning chunk codes
/// directly (not relying on compacted sizes, which may be stale).
/// Returns the new centroid count, or `None` if no centroids were pruned.
pub fn prune_empty_centroids(
    index_path: &Path,
    num_chunks: usize,
    device: Device,
) -> Result<Option<usize>> {
    let centroids = Tensor::read_npy(index_path.join("centroids.npy"))?
        .to_device(Device::Cpu);
    let old_count = centroids.size()[0] as usize;

    // Scan chunk codes to find which centroids are actually used
    let mut used = vec![false; old_count];
    for chunk_idx in 0..num_chunks {
        let codes_path = index_path.join(format!("{}.codes.npy", chunk_idx));
        if !codes_path.exists() {
            continue;
        }
        let codes: Vec<i64> = Tensor::read_npy(&codes_path)?
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64)
            .try_into()?;
        for &c in &codes {
            used[c as usize] = true;
        }
    }

    let empty_count = used.iter().filter(|&&u| !u).count();
    if empty_count == 0 {
        return Ok(None);
    }

    // Build old->new renumbering map
    let mut old_to_new = vec![-1i64; old_count];
    let mut keep_indices: Vec<i64> = Vec::new();
    let mut new_id = 0i64;
    for (old_id, &is_used) in used.iter().enumerate() {
        if is_used {
            old_to_new[old_id] = new_id;
            keep_indices.push(old_id as i64);
            new_id += 1;
        }
    }
    let new_num_centroids = new_id as usize;

    // Rewrite centroids.npy
    let keep_tensor = Tensor::from_slice(&keep_indices);
    let new_centroids = centroids.index_select(0, &keep_tensor);
    new_centroids.write_npy(index_path.join("centroids.npy"))?;

    // Rewrite chunk codes with renumbered centroid IDs
    let remap_tensor = Tensor::from_slice(&old_to_new).to_device(device);
    for chunk_idx in 0..num_chunks {
        let codes_path = index_path.join(format!("{}.codes.npy", chunk_idx));
        if !codes_path.exists() {
            continue;
        }
        let codes = Tensor::read_npy(&codes_path)?
            .to_device(device)
            .to_kind(Kind::Int64);
        let new_codes = remap_tensor.index_select(0, &codes);
        new_codes
            .to_device(Device::Cpu)
            .write_npy(&codes_path)?;
    }

    Ok(Some(new_num_centroids))
}

/// Recompute codec statistics from current index data.
///
/// Samples up to 50 000 embeddings, decompresses their quantized residuals
/// via bucket weights, and recomputes:
/// - `cluster_threshold.npy`: 75th percentile of residual L2 norms
/// - `avg_residual.npy`: per-dimension mean absolute residual
pub fn recalibrate_threshold(
    index_path: &Path,
    num_chunks: usize,
    nbits: u8,
    dim: usize,
    device: Device,
) -> Result<()> {
    let bucket_weights_path = index_path.join("bucket_weights.npy");
    if !bucket_weights_path.exists() {
        return Ok(());
    }
    let bucket_weights = Tensor::read_npy(&bucket_weights_path)?
        .to_device(device)
        .to_kind(Kind::Float);

    let max_sample = 50_000i64;
    let mut all_approx: Vec<Tensor> = Vec::new();
    let mut total_sampled = 0i64;

    for chunk_idx in 0..num_chunks {
        if total_sampled >= max_sample {
            break;
        }

        let res_path = index_path.join(format!("{}.residuals.npy", chunk_idx));
        if !res_path.exists() {
            continue;
        }
        let packed = Tensor::read_npy(&res_path)?.to_device(device);
        let n = packed.size()[0];
        let take = (max_sample - total_sampled).min(n);
        let sample = packed.narrow(0, 0, take);

        let bucket_indices = unpack_residuals(&sample, nbits, dim as i64);
        let approx = bucket_weights
            .index_select(0, &bucket_indices.flatten(0, -1).to_kind(Kind::Int64))
            .reshape([take, dim as i64]);
        all_approx.push(approx.to_device(Device::Cpu));
        total_sampled += take;
    }

    if all_approx.is_empty() {
        return Ok(());
    }

    let all_residuals = Tensor::cat(&all_approx, 0); // [total_sampled, dim]

    // Cluster threshold: 75th percentile of L2 norms
    let norms = all_residuals.norm_scalaropt_dim(2, &[1], false);
    let n = norms.size()[0];
    let k = ((0.75 * n as f64).ceil() as i64).max(1).min(n);
    let (threshold, _) = norms.kthvalue(k, 0, false);
    threshold
        .to_device(Device::Cpu)
        .write_npy(index_path.join("cluster_threshold.npy"))?;

    // Average residual: per-dimension mean absolute value
    let avg_residual = all_residuals
        .abs()
        .mean_dim(Some(&[0i64][..]), false, Kind::Float);
    avg_residual
        .to_device(Device::Cpu)
        .write_npy(index_path.join("avg_residual.npy"))?;

    Ok(())
}

/// Unpack quantized residuals from packed bytes back to bucket indices.
///
/// Input: `[N, dim * nbits / 8]` uint8 tensor.
/// Output: `[N, dim]` int64 tensor with values in `[0, 2^nbits)`.
fn unpack_residuals(packed: &Tensor, nbits: u8, dim: i64) -> Tensor {
    let n = packed.size()[0];
    let device = packed.device();

    // Expand each byte into 8 bits
    let shifts = Tensor::from_slice(&[7i64, 6, 5, 4, 3, 2, 1, 0]).to_device(device);
    let bits = packed
        .to_kind(Kind::Int64)
        .unsqueeze(-1)
        .bitwise_right_shift(&shifts)
        .bitwise_and_tensor(&Tensor::ones(&[1], (Kind::Int64, device)));
    // bits: [N, bytes_per_row, 8]

    let total_bits = dim * nbits as i64;
    let bits_flat = bits.reshape([n, total_bits]);
    // bits_flat: [N, dim * nbits]

    // Group into nbits-wide chunks and combine: bucket = sum(bit_k * 2^k)
    let bits_grouped = bits_flat.reshape([n, dim, nbits as i64]);
    let powers: Vec<i64> = (0..nbits).map(|i| 1i64 << i).collect();
    let powers_tensor = Tensor::from_slice(&powers).to_device(device);
    (&bits_grouped * &powers_tensor).sum_dim_intlist(-1, false, Kind::Int64)
}
