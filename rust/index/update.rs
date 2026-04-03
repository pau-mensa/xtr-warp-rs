use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;
use tch::{Device, IndexOp, Kind, Tensor};

use crate::index::compact::{
    build_partial_compacted, compact_index, merge_compacted_incremental,
    prune_empty_centroids, recalibrate_threshold, remove_and_merge_compacted,
};
use crate::index::delete::{clear_tombstones, load_tombstones, save_tombstones};
use crate::index::encode::{
    encode_chunks, encode_chunks_with_norms, read_chunk_data, ChunkData, CHUNK_SIZE,
};
use crate::index::source::EmbeddingSource;
use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::{AddResult, CompactStats, IndexMetadata, IndexPlan};

/// Threshold below which we append to the last chunk instead of creating a new one.
const COALESCE_THRESHOLD: usize = 2_000;

// ── Public operations ──

/// Add new documents to an existing index.
///
/// Encodes new embeddings as new chunk files, then incrementally merges them
/// into the compacted search structures. Call `compact_standalone` later to
/// physically reclaim space from any prior deletes.
pub fn add_to_index(
    embeddings: &mut dyn EmbeddingSource,
    index_path: &Path,
    device: Device,
    show_progress: bool,
) -> Result<AddResult> {
    let meta = IndexMetadata::load(index_path)?;
    let next_pid = meta.next_passage_id;

    let codec = load_codec_from_disk(index_path, meta.nbits, device)?;

    let num_new_docs = embeddings.num_docs();
    let new_passage_ids: Vec<i64> = (next_pid..next_pid + num_new_docs as i64).collect();

    // Check if we should coalesce into the last chunk
    let (start_chunk_idx, coalesce) = should_coalesce(index_path, &meta)?;

    // If coalescing, read old chunk data into memory before encode overwrites it
    let old_chunk = if coalesce {
        Some(read_chunk_data(index_path, start_chunk_idx)?)
    } else {
        None
    };

    let new_num_chunks = (num_new_docs as f64 / CHUNK_SIZE as f64).ceil().max(1.0) as usize;
    let plan = IndexPlan {
        n_docs: num_new_docs,
        num_chunks: new_num_chunks,
        avg_doc_len: 0.0,
        est_total_embs: 0,
        nbits: meta.nbits,
    };

    let encode_result = encode_chunks_with_norms(
        &plan,
        embeddings,
        &codec.centroids,
        &codec,
        index_path,
        device,
        meta.dim as u32,
        Some(&new_passage_ids),
        start_chunk_idx,
        show_progress,
    )?;

    // If coalescing, prepend old data to the first new chunk
    if let Some(old) = old_chunk {
        prepend_chunk_data(index_path, start_chunk_idx, &old)?;
    }

    let total_chunks = start_chunk_idx + encode_result.chunk_stats.len();
    let residual_dim = (meta.dim * meta.nbits as usize) / 8;

    // Incremental merge: only process NEW embeddings, merge into existing compacted.
    // When coalescing, the chunk at start_chunk_idx contains both old and new data,
    // so we filter to only include the new passage IDs.
    let new_pids_set: HashSet<i64> = new_passage_ids.iter().copied().collect();
    let partial = build_partial_compacted(
        index_path,
        start_chunk_idx,
        total_chunks,
        meta.num_centroids,
        residual_dim,
        &HashSet::new(), // no tombstones to exclude from new data
        Some(&new_pids_set),
    )?;

    let stats = merge_compacted_incremental(
        index_path,
        &partial,
        meta.num_centroids,
        residual_dim,
        device,
    )?;

    save_metadata_from_stats(index_path, &meta, &stats, total_chunks, next_pid + num_new_docs as i64)?;

    Ok(AddResult {
        new_passage_ids,
        residual_norms: encode_result.residual_norms.unwrap_or_default(),
        embedding_dim: meta.dim,
    })
}

/// Update documents in-place: new embeddings, same passage IDs.
///
/// Marks old data as deleted, encodes new data with the original IDs,
/// then incrementally merges the new chunks into the compacted structures.
/// The old chunk data carrying those IDs becomes dead weight until the next
/// `compact_standalone` call.
///
/// Cost: O(new_embeddings) — only the new chunks are processed.
pub fn update_in_index(
    passage_ids: &[i64],
    embeddings: &mut dyn EmbeddingSource,
    index_path: &Path,
    device: Device,
    show_progress: bool,
) -> Result<()> {
    anyhow::ensure!(
        passage_ids.len() == embeddings.num_docs(),
        "passage_ids length ({}) must match number of documents ({})",
        passage_ids.len(),
        embeddings.num_docs()
    );

    let meta = IndexMetadata::load(index_path)?;

    // Mark old passage IDs as deleted so search/compaction excludes them
    crate::index::delete::delete_from_index(passage_ids, index_path)?;

    let codec = load_codec_from_disk(index_path, meta.nbits, device)?;

    let num_new_docs = embeddings.num_docs();
    let new_num_chunks = (num_new_docs as f64 / CHUNK_SIZE as f64).ceil().max(1.0) as usize;
    let plan = IndexPlan {
        n_docs: num_new_docs,
        num_chunks: new_num_chunks,
        avg_doc_len: 0.0,
        est_total_embs: 0,
        nbits: meta.nbits,
    };

    let encode_result = encode_chunks(
        &plan,
        embeddings,
        &codec.centroids,
        &codec,
        index_path,
        device,
        meta.dim as u32,
        Some(passage_ids),
        meta.num_chunks,
        show_progress,
    )?;

    // Remove updated PIDs from tombstones — fresh data lives in the new chunks.
    // Any other tombstones (from prior deletes) are preserved.
    {
        let mut tombstones = load_tombstones(index_path)?;
        for &pid in passage_ids {
            tombstones.remove(&pid);
        }
        save_tombstones(&tombstones, index_path)?;
    }

    // Incremental merge: only process the NEW chunks, same as add_to_index.
    // The old chunks still contain stale data for the updated PIDs, but those
    // PIDs are tombstoned so they're excluded from the compacted structures.
    let total_chunks = meta.num_chunks + encode_result.chunk_stats.len();
    let residual_dim = (meta.dim * meta.nbits as usize) / 8;

    let updated_pids_set: HashSet<i64> = passage_ids.iter().copied().collect();
    let partial = build_partial_compacted(
        index_path,
        meta.num_chunks,   // start from the first new chunk
        total_chunks,
        meta.num_centroids,
        residual_dim,
        &HashSet::new(),
        Some(&updated_pids_set),
    )?;

    // Remove old entries for updated PIDs from compacted, then merge new ones in.
    // We do this by rebuilding: remove old PIDs + merge new data.
    let stats = remove_and_merge_compacted(
        index_path,
        &partial,
        &updated_pids_set,
        meta.num_centroids,
        residual_dim,
        device,
    )?;

    // next_passage_id stays the same: we're replacing, not appending
    save_metadata_from_stats(index_path, &meta, &stats, total_chunks, meta.next_passage_id)?;

    Ok(())
}

/// Rebuild the compacted index excluding tombstoned passages.
///
/// Rewrites chunk files to remove deleted data, prunes empty centroids,
/// rebuilds compacted structures, and recalibrates the cluster threshold.
/// Single-pass: no redundant recompaction.
pub fn compact_standalone(index_path: &Path, device: Device, show_progress: bool) -> Result<()> {
    let meta = IndexMetadata::load(index_path)?;
    let tombstones = load_tombstones(index_path)?;

    // Step 1: Rewrite chunks to physically remove deleted data
    let num_chunks = if !tombstones.is_empty() {
        rewrite_chunks_filtered(index_path, meta.num_chunks, &tombstones)?;
        clear_tombstones(index_path)?;
        count_chunks(index_path)
    } else {
        meta.num_chunks
    };

    // Step 2: Prune empty centroids (renumbers codes in chunk files)
    let num_centroids =
        if let Some(new_count) = prune_empty_centroids(index_path, num_chunks, device)? {
            new_count
        } else {
            meta.num_centroids
        };

    // Step 3: Rebuild compacted structures from clean chunks (single pass)
    let stats = compact_index(
        index_path,
        num_chunks,
        num_centroids,
        meta.dim,
        meta.nbits as usize,
        device,
        &HashSet::new(), // chunks are already clean
        show_progress,
    )?;

    // Step 4: Recalibrate cluster threshold from current data
    recalibrate_threshold(index_path, num_chunks, meta.nbits, meta.dim, device)?;

    // Step 5: Update metadata
    let updated = IndexMetadata {
        num_chunks,
        num_centroids,
        num_embeddings: stats.total_embeddings,
        num_passages: stats.num_active_passages,
        avg_doclen: if stats.num_active_passages > 0 {
            stats.total_embeddings as f64 / stats.num_active_passages as f64
        } else {
            0.0
        },
        next_passage_id: meta.next_passage_id,
        nbits: meta.nbits,
        num_partitions: meta.num_partitions,
        dim: meta.dim,
        created_at: meta.created_at,
    };
    updated.save(index_path)?;

    Ok(())
}

/// Append new centroids to the codebook and extend compacted structures.
///
/// Called from Python after K-means on outlier embeddings produces new centroids.
pub fn append_centroids(index_path: &Path, new_centroids: &Tensor) -> Result<()> {
    let k_new = new_centroids.size()[0];
    if k_new == 0 {
        return Ok(());
    }

    // Append to centroids.npy
    let old_centroids = Tensor::read_npy(index_path.join("centroids.npy"))?
        .to_device(Device::Cpu);
    let combined = Tensor::cat(
        &[old_centroids, new_centroids.to_device(Device::Cpu).to_kind(Kind::Half)],
        0,
    );
    combined.write_npy(index_path.join("centroids.npy"))?;

    // Extend sizes.compacted with zeros
    let old_sizes = Tensor::read_npy(index_path.join("sizes.compacted.npy"))?
        .to_device(Device::Cpu);
    let ext = Tensor::zeros(&[k_new], (old_sizes.kind(), Device::Cpu));
    Tensor::cat(&[old_sizes, ext], 0)
        .write_npy(index_path.join("sizes.compacted.npy"))?;

    // Extend offsets.compacted: new entries equal to old total
    let old_offsets = Tensor::read_npy(index_path.join("offsets.compacted.npy"))?
        .to_device(Device::Cpu);
    let total = old_offsets.i(-1).int64_value(&[]);
    let ext_offsets = Tensor::full(&[k_new], total, (old_offsets.kind(), Device::Cpu));
    Tensor::cat(&[old_offsets, ext_offsets], 0)
        .write_npy(index_path.join("offsets.compacted.npy"))?;

    // Update metadata
    let mut meta = IndexMetadata::load(index_path)?;
    meta.num_centroids += k_new as usize;
    meta.save(index_path)?;

    Ok(())
}

// ── Helpers ──

/// Decide whether the new data should be coalesced into the last existing chunk.
fn should_coalesce(index_path: &Path, meta: &IndexMetadata) -> Result<(usize, bool)> {
    if meta.num_chunks == 0 {
        return Ok((0, false));
    }
    let last_idx = meta.num_chunks - 1;
    let meta_path = index_path.join(format!("{}.metadata.json", last_idx));
    if !meta_path.exists() {
        return Ok((meta.num_chunks, false));
    }
    let f = std::fs::File::open(&meta_path)?;
    let chunk_meta: serde_json::Value =
        serde_json::from_reader(std::io::BufReader::new(f))?;
    let num_passages = chunk_meta
        .get("num_passages")
        .and_then(|v| v.as_u64())
        .unwrap_or(COALESCE_THRESHOLD as u64 + 1) as usize;
    if num_passages < COALESCE_THRESHOLD {
        Ok((last_idx, true))
    } else {
        Ok((meta.num_chunks, false))
    }
}

/// Prepend old chunk data to the chunk that `encode_chunks` just wrote.
fn prepend_chunk_data(index_path: &Path, chunk_idx: usize, old: &ChunkData) -> Result<()> {
    // Read new data (just written by encode_chunks)
    let new = read_chunk_data(index_path, chunk_idx)?;

    // Concatenate: old first, then new
    let old_num_embs = old.codes.size()[0];
    let new_num_embs = new.codes.size()[0];
    Tensor::cat(&[old.codes.shallow_clone(), new.codes.shallow_clone()], 0)
        .write_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?;
    Tensor::cat(&[old.residuals.shallow_clone(), new.residuals.shallow_clone()], 0)
        .write_npy(index_path.join(format!("{}.residuals.npy", chunk_idx)))?;

    let combined_doclens: Vec<i64> = old.doclens.iter().chain(&new.doclens).copied().collect();
    Tensor::from_slice(&combined_doclens)
        .write_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?;

    let combined_pids: Vec<i64> = old.pids.iter().chain(&new.pids).copied().collect();
    Tensor::from_slice(&combined_pids)
        .write_npy(index_path.join(format!("{}.passage_ids.npy", chunk_idx)))?;

    // Update chunk metadata
    let total_embs = old_num_embs + new_num_embs;
    let chunk_meta = serde_json::json!({
        "num_passages": combined_doclens.len(),
        "num_embeddings": total_embs,
        "embedding_offset": 0,
    });
    let meta_f = std::fs::File::create(
        index_path.join(format!("{}.metadata.json", chunk_idx)),
    )?;
    serde_json::to_writer(std::io::BufWriter::new(meta_f), &chunk_meta)?;

    Ok(())
}

/// Count how many contiguous chunk files exist starting from index 0.
fn count_chunks(index_path: &Path) -> usize {
    (0..)
        .take_while(|i| index_path.join(format!("{}.codes.npy", i)).exists())
        .count()
}

/// Update metadata from compaction stats.
fn save_metadata_from_stats(
    index_path: &Path,
    meta: &IndexMetadata,
    stats: &CompactStats,
    num_chunks: usize,
    next_pid: i64,
) -> Result<()> {
    let updated = IndexMetadata {
        num_chunks,
        nbits: meta.nbits,
        num_partitions: meta.num_partitions,
        num_embeddings: stats.total_embeddings,
        avg_doclen: if stats.num_active_passages > 0 {
            stats.total_embeddings as f64 / stats.num_active_passages as f64
        } else {
            0.0
        },
        num_passages: stats.num_active_passages,
        next_passage_id: next_pid,
        num_centroids: meta.num_centroids,
        dim: meta.dim,
        created_at: meta.created_at.clone(),
    };
    updated.save(index_path)
}

/// Rewrite chunk files, removing deleted passages and eliminating empty chunks.
///
/// Streams one chunk at a time: read → filter → write temp → drop.
/// Peak memory is O(max_chunk_size) instead of O(total_index_size).
/// All work is done on CPU — this is pure data shuffling, not compute.
fn rewrite_chunks_filtered(
    index_path: &Path,
    num_chunks: usize,
    deleted_pids: &HashSet<i64>,
) -> Result<()> {
    use serde_json::json;
    use std::io::BufWriter;

    let chunk_files = |idx: usize| -> [String; 5] {
        [
            format!("{}.codes.npy", idx),
            format!("{}.residuals.npy", idx),
            format!("doclens.{}.npy", idx),
            format!("{}.passage_ids.npy", idx),
            format!("{}.metadata.json", idx),
        ]
    };

    // Pass 1: scan each chunk to determine which survive and build a mapping.
    let mut chunk_mapping: Vec<(usize, Vec<i64>, Vec<i64>, Vec<i64>)> = Vec::new();

    for chunk_idx in 0..num_chunks {
        let doclens_tensor =
            Tensor::read_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?
                .to_device(Device::Cpu)
                .to_kind(Kind::Int64);
        let doclens_vec: Vec<i64> = doclens_tensor.try_into()?;

        let pids_vec: Vec<i64> = Tensor::read_npy(index_path.join(format!("{}.passage_ids.npy", chunk_idx)))?
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64)
            .try_into()?;

        let mut keep_doclens = Vec::new();
        let mut keep_pids = Vec::new();
        let mut keep_emb_indices: Vec<i64> = Vec::new();
        let mut emb_offset = 0i64;

        for (doc_idx, &doc_len) in doclens_vec.iter().enumerate() {
            let pid = pids_vec[doc_idx];
            if deleted_pids.contains(&pid) {
                emb_offset += doc_len;
                continue;
            }
            keep_pids.push(pid);
            keep_doclens.push(doc_len);
            for i in 0..doc_len {
                keep_emb_indices.push(emb_offset + i);
            }
            emb_offset += doc_len;
        }

        if !keep_doclens.is_empty() {
            chunk_mapping.push((chunk_idx, keep_doclens, keep_pids, keep_emb_indices));
        }
    }

    let new_count = chunk_mapping.len();

    // Pass 2: stream each surviving chunk — load heavy data, filter, write temp, drop.
    let mut passage_offset: usize = 0;
    let mut emb_offset: usize = 0;

    for (new_idx, (old_idx, keep_doclens, keep_pids, keep_emb_indices)) in
        chunk_mapping.iter().enumerate()
    {
        let codes = Tensor::read_npy(index_path.join(format!("{}.codes.npy", old_idx)))?
            .to_device(Device::Cpu);
        let residuals = Tensor::read_npy(index_path.join(format!("{}.residuals.npy", old_idx)))?
            .to_device(Device::Cpu);

        let idx_tensor = Tensor::from_slice(keep_emb_indices);
        let filtered_codes = codes.index_select(0, &idx_tensor);
        let filtered_residuals = residuals.index_select(0, &idx_tensor);
        drop(codes);
        drop(residuals);

        filtered_codes
            .write_npy(index_path.join(format!("{}.codes.npy.tmp", new_idx)))?;
        filtered_residuals
            .write_npy(index_path.join(format!("{}.residuals.npy.tmp", new_idx)))?;

        Tensor::from_slice(keep_doclens)
            .write_npy(index_path.join(format!("doclens.{}.npy.tmp", new_idx)))?;

        Tensor::from_slice(keep_pids)
            .write_npy(index_path.join(format!("{}.passage_ids.npy.tmp", new_idx)))?;

        let num_embs = filtered_codes.size()[0] as usize;
        let chk_meta = json!({
            "passage_offset": passage_offset,
            "num_passages": keep_doclens.len(),
            "num_embeddings": num_embs,
            "embedding_offset": emb_offset,
        });
        let meta_f =
            std::fs::File::create(index_path.join(format!("{}.metadata.json.tmp", new_idx)))?;
        serde_json::to_writer(BufWriter::new(meta_f), &chk_meta)?;

        passage_offset += keep_doclens.len();
        emb_offset += num_embs;
    }

    // Atomic rename temp → final
    for new_idx in 0..new_count {
        for name in &chunk_files(new_idx) {
            let tmp = index_path.join(format!("{}.tmp", name));
            if tmp.exists() {
                std::fs::rename(&tmp, index_path.join(name))?;
            }
        }
    }

    // Delete leftover old chunk files
    for chunk_idx in new_count..num_chunks {
        for name in &chunk_files(chunk_idx) {
            let p = index_path.join(name);
            if p.exists() {
                std::fs::remove_file(&p)?;
            }
        }
    }

    Ok(())
}

/// Load the residual codec from existing index files on disk.
fn load_codec_from_disk(index_path: &Path, nbits: u8, device: Device) -> Result<ResidualCodec> {
    let centroids = Tensor::read_npy(index_path.join("centroids.npy"))?
        .to_device(device)
        .to_kind(Kind::Half);
    let avg_residual = Tensor::read_npy(index_path.join("avg_residual.npy"))?.to_device(device);
    let bucket_cutoffs = Tensor::read_npy(index_path.join("bucket_cutoffs.npy"))?.to_device(device);
    let bucket_weights = Tensor::read_npy(index_path.join("bucket_weights.npy"))?.to_device(device);

    ResidualCodec::load(
        nbits,
        centroids,
        avg_residual,
        Some(bucket_cutoffs),
        Some(bucket_weights),
        device,
    )
}
