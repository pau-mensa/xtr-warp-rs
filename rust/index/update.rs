use anyhow::Result;
use std::path::Path;
use tch::{Device, IndexOp, Kind, Tensor};

use crate::index::compact::{
    build_partial_compacted, compact_index_filtered, merge_compacted_incremental,
    prune_empty_centroids,
};
use crate::index::delete::{clear_tombstones, load_tombstones, save_tombstones};
use crate::index::encode::{encode_chunks, encode_chunks_with_norms, CHUNK_SIZE};
use crate::index::source::EmbeddingSource;
use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::{AddResult, IndexMetadata, IndexPlan};

// ── Public operations ──

/// Add new documents to an existing index.
///
/// Encodes new embeddings as new chunk files, then rebuilds the compacted
/// search structures excluding tombstoned passage IDs.  Chunk files and
/// tombstones are left in place — call `compact_standalone` to clean them up.
/// Threshold below which we append to the last chunk instead of creating a new one.
const COALESCE_THRESHOLD: usize = 2_000;

pub fn add_to_index(
    embeddings: &mut dyn EmbeddingSource,
    index_path: &Path,
    device: Device,
) -> Result<AddResult> {
    let meta = IndexMetadata::load(index_path)?;
    let next_pid = meta.next_pid();

    let codec = load_codec_from_disk(index_path, meta.nbits, device)?;
    let centroids = Tensor::read_npy(index_path.join("centroids.npy"))?.to_device(device);

    // Assign new passage IDs: next_pid .. next_pid + num_new_docs
    let num_new_docs = embeddings.num_docs();
    let new_passage_ids: Vec<i64> = (next_pid..next_pid + num_new_docs as i64).collect();

    // Check if we should coalesce into the last chunk
    let (start_chunk_idx, coalesce) = should_coalesce(index_path, &meta)?;

    let new_num_chunks = (num_new_docs as f64 / CHUNK_SIZE as f64).ceil().max(1.0) as usize;
    let plan = IndexPlan {
        n_docs: num_new_docs,
        num_chunks: new_num_chunks,
        avg_doc_len: 0.0,
        est_total_embs: 0,
        nbits: meta.nbits,
    };

    // If coalescing, backup old chunk files before encode overwrites them
    if coalesce {
        for (pattern, rename_pattern) in &[
            (format!("{}.codes.npy", start_chunk_idx), format!("{}.codes.npy.old", start_chunk_idx)),
            (format!("{}.residuals.npy", start_chunk_idx), format!("{}.residuals.npy.old", start_chunk_idx)),
            (format!("doclens.{}.npy", start_chunk_idx), format!("doclens.{}.npy.old", start_chunk_idx)),
            (format!("{}.passage_ids.npy", start_chunk_idx), format!("{}.passage_ids.npy.old", start_chunk_idx)),
            (format!("{}.metadata.json", start_chunk_idx), format!("{}.metadata.json.old", start_chunk_idx)),
        ] {
            let src = index_path.join(pattern);
            if src.exists() {
                std::fs::rename(&src, index_path.join(rename_pattern))?;
            }
        }
    }

    let encode_result = encode_chunks_with_norms(
        &plan,
        embeddings,
        &centroids,
        &codec,
        index_path,
        device,
        meta.dim as u32,
        Some(&new_passage_ids),
        start_chunk_idx,
    )?;

    // If coalescing, prepend old last-chunk data to the first new chunk
    if coalesce {
        coalesce_with_last_chunk(index_path, start_chunk_idx)?;
    }

    let total_chunks = start_chunk_idx + encode_result.chunk_stats.len();
    let residual_dim = (meta.dim * meta.nbits as usize) / 8;

    // Incremental merge: only process NEW embeddings, merge into existing compacted.
    // When coalescing, the chunk at start_chunk_idx contains both old and new data,
    // so we filter to only include the new passage IDs.
    let new_pids_set: std::collections::HashSet<i64> =
        new_passage_ids.iter().copied().collect();
    let partial = build_partial_compacted(
        index_path,
        start_chunk_idx,
        total_chunks,
        meta.num_centroids,
        residual_dim,
        &std::collections::HashSet::new(), // no tombstones to exclude from new data
        Some(&new_pids_set),               // only include new PIDs
    )?;

    let stats = merge_compacted_incremental(
        index_path,
        &partial,
        meta.num_centroids,
        residual_dim,
        device,
    )?;

    // Update metadata
    let avg_doclen = if stats.num_active_passages > 0 {
        stats.total_embeddings as f64 / stats.num_active_passages as f64
    } else {
        0.0
    };
    let updated = IndexMetadata {
        num_chunks: total_chunks,
        nbits: meta.nbits,
        num_partitions: meta.num_partitions,
        num_embeddings: stats.total_embeddings,
        avg_doclen,
        num_passages: stats.num_active_passages,
        next_passage_id: Some(next_pid + num_new_docs as i64),
        num_centroids: meta.num_centroids,
        dim: meta.dim,
        created_at: meta.created_at.clone(),
    };
    updated.save(index_path)?;

    Ok(AddResult {
        new_passage_ids,
        residual_norms: encode_result.residual_norms.unwrap_or_default(),
        embedding_dim: meta.dim,
    })
}

/// Update documents in-place: new embeddings, same passage IDs.
///
/// Marks old data as deleted, encodes new data with the original IDs,
/// then recompacts so the old chunk data is excluded but the new chunk data
/// (carrying the same IDs) is included.
pub fn update_in_index(
    passage_ids: &[i64],
    embeddings: &mut dyn EmbeddingSource,
    index_path: &Path,
    device: Device,
) -> Result<()> {
    anyhow::ensure!(
        passage_ids.len() == embeddings.num_docs(),
        "passage_ids length ({}) must match number of documents ({})",
        passage_ids.len(),
        embeddings.num_docs()
    );

    let meta = IndexMetadata::load(index_path)?;

    // Mark old passage IDs as deleted so compaction excludes them from old chunks
    crate::index::delete::delete_from_index(passage_ids, index_path)?;

    let codec = load_codec_from_disk(index_path, meta.nbits, device)?;
    let centroids = Tensor::read_npy(index_path.join("centroids.npy"))?.to_device(device);

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
        &centroids,
        &codec,
        index_path,
        device,
        meta.dim as u32,
        Some(passage_ids), // same IDs
        meta.num_chunks,
    )?;

    // Remove updated PIDs from tombstones, fresh data in the new chunks.
    // Any other tombstones (from prior deletes) are preserved.
    {
        let mut tombstones = load_tombstones(index_path)?;
        for &pid in passage_ids {
            tombstones.remove(&pid);
        }
        save_tombstones(&tombstones, index_path)?;
    }

    let total_chunks = meta.num_chunks + encode_result.chunk_stats.len();
    // next_passage_id stays the same: we're replacing, not appending
    compact_and_update_metadata(index_path, &meta, total_chunks, None, device, false)?;

    Ok(())
}

/// Rebuild the compacted index excluding tombstoned passages, without adding new data.
///
/// Also prunes empty centroids (those with zero embeddings after filtering
/// tombstones) and rebuilds compacted structures with the reduced codebook.
pub fn compact_standalone(index_path: &Path, device: Device) -> Result<()> {
    let meta = IndexMetadata::load(index_path)?;
    compact_and_update_metadata(index_path, &meta, meta.num_chunks, None, device, true)?;

    // Prune empty centroids: remove from codebook, renumber chunk codes, recompact
    let meta = IndexMetadata::load(index_path)?;
    if let Some(new_num_centroids) = prune_empty_centroids(
        index_path,
        meta.num_chunks,
        device,
    )? {
        let mut meta = IndexMetadata::load(index_path)?;
        meta.num_centroids = new_num_centroids;
        meta.save(index_path)?;
        // Recompact with the reduced centroid set (no tombstones — already cleared)
        compact_and_update_metadata(index_path, &meta, meta.num_chunks, None, device, false)?;
    }

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

    // Extend offsets.compacted: old offsets + k_new entries equal to old total
    let old_offsets = Tensor::read_npy(index_path.join("offsets.compacted.npy"))?
        .to_device(Device::Cpu);
    let total = old_offsets.i(-1).int64_value(&[]);
    let ext_offsets = Tensor::full(&[k_new], total, (old_offsets.kind(), Device::Cpu));
    Tensor::cat(&[old_offsets, ext_offsets], 0)
        .write_npy(index_path.join("offsets.compacted.npy"))?;

    // Extend ivf_lengths with zeros
    let ivf_lengths_path = index_path.join("ivf_lengths.npy");
    if ivf_lengths_path.exists() {
        let old_lens = Tensor::read_npy(&ivf_lengths_path)?.to_device(Device::Cpu);
        let ext_lens = Tensor::zeros(&[k_new], (old_lens.kind(), Device::Cpu));
        Tensor::cat(&[old_lens, ext_lens], 0).write_npy(&ivf_lengths_path)?;
    }

    // Update metadata
    let mut meta = IndexMetadata::load(index_path)?;
    meta.num_centroids += k_new as usize;
    meta.save(index_path)?;

    Ok(())
}

// ── Shared helpers ──

/// Decide whether the new data should be coalesced into the last existing chunk.
/// Returns `(start_chunk_idx, should_coalesce)`.
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

/// Prepend old chunk data to the newly-written chunk at `chunk_idx`.
///
/// `encode_chunks` has already written new files at `{chunk_idx}.codes.npy` etc.
/// We read the old data (saved before encoding overwrote it — but since encode
/// uses the same index, the old files have been overwritten). So we need a
/// different approach: before `encode_chunks`, we backup old files, then after
/// encoding we concatenate.
///
/// **Revised approach**: `encode_chunks` has overwritten the chunk files.  We
/// rely on the fact that the old chunk data was already read and saved *before*
/// `encode_chunks` was called.  Instead, this function reads `.old` backup files
/// created by `add_to_index` before the encode call.
///
/// Actually, the simplest approach: `encode_chunks` writes to `start_chunk_idx`.
/// If coalescing, the old chunk at that index is overwritten.  We save backups
/// BEFORE calling encode_chunks, then concatenate here.
fn coalesce_with_last_chunk(index_path: &Path, chunk_idx: usize) -> Result<()> {
    let old_codes_path = index_path.join(format!("{}.codes.npy.old", chunk_idx));
    if !old_codes_path.exists() {
        return Ok(());
    }

    // Load old data
    let old_codes =
        Tensor::read_npy(&old_codes_path)?.to_device(tch::Device::Cpu);
    let old_residuals = Tensor::read_npy(
        index_path.join(format!("{}.residuals.npy.old", chunk_idx)),
    )?
    .to_device(tch::Device::Cpu);
    let old_doclens =
        Tensor::read_npy(index_path.join(format!("doclens.{}.npy.old", chunk_idx)))?
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Int64);
    let old_pids = Tensor::read_npy(
        index_path.join(format!("{}.passage_ids.npy.old", chunk_idx)),
    )?
    .to_device(tch::Device::Cpu)
    .to_kind(Kind::Int64);

    // Load new data (just written by encode_chunks)
    let new_codes =
        Tensor::read_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?
            .to_device(tch::Device::Cpu);
    let new_residuals =
        Tensor::read_npy(index_path.join(format!("{}.residuals.npy", chunk_idx)))?
            .to_device(tch::Device::Cpu);
    let new_doclens =
        Tensor::read_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Int64);
    let new_pids =
        Tensor::read_npy(index_path.join(format!("{}.passage_ids.npy", chunk_idx)))?
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Int64);

    // Concatenate: old first, then new
    Tensor::cat(&[old_codes, new_codes], 0)
        .write_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?;
    Tensor::cat(&[old_residuals, new_residuals], 0)
        .write_npy(index_path.join(format!("{}.residuals.npy", chunk_idx)))?;
    Tensor::cat(&[old_doclens, new_doclens], 0)
        .write_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?;
    Tensor::cat(&[old_pids, new_pids], 0)
        .write_npy(index_path.join(format!("{}.passage_ids.npy", chunk_idx)))?;

    // Update chunk metadata
    let combined_codes = Tensor::read_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?;
    let combined_doclens: Vec<i64> = Tensor::read_npy(
        index_path.join(format!("doclens.{}.npy", chunk_idx)),
    )?
    .to_device(tch::Device::Cpu)
    .to_kind(Kind::Int64)
    .try_into()?;

    let chunk_meta = serde_json::json!({
        "num_passages": combined_doclens.len(),
        "num_embeddings": combined_codes.size()[0],
        "embedding_offset": 0, // will be recomputed by compaction
    });
    let meta_f = std::fs::File::create(
        index_path.join(format!("{}.metadata.json", chunk_idx)),
    )?;
    serde_json::to_writer(std::io::BufWriter::new(meta_f), &chunk_meta)?;

    // Cleanup backup files
    for ext in &["codes.npy.old", "residuals.npy.old", "passage_ids.npy.old"] {
        let p = index_path.join(format!("{}.{}", chunk_idx, ext));
        if p.exists() {
            std::fs::remove_file(&p)?;
        }
    }
    let dl_old = index_path.join(format!("doclens.{}.npy.old", chunk_idx));
    if dl_old.exists() {
        std::fs::remove_file(&dl_old)?;
    }
    // Also clean up old json doclens if present
    let dl_json_old = index_path.join(format!("doclens.{}.json.old", chunk_idx));
    if dl_json_old.exists() {
        std::fs::remove_file(&dl_json_old)?;
    }
    let meta_old = index_path.join(format!("{}.metadata.json.old", chunk_idx));
    if meta_old.exists() {
        std::fs::remove_file(&meta_old)?;
    }

    Ok(())
}

/// Run filtered compaction, rewrite chunk files to contain only active data,
/// update metadata, and clear tombstones.
///
/// After this call the chunk files are clean. No deleted PIDs remain in any
/// chunk, so tombstones can be safely removed.
///
/// `new_next_pid`: if `Some`, overwrite `next_passage_id` (used by add).
///                 if `None`, keep the existing value (used by update / compact).
fn compact_and_update_metadata(
    index_path: &Path,
    meta: &IndexMetadata,
    total_chunks: usize,
    new_next_pid: Option<i64>,
    device: Device,
    rewrite_chunks: bool,
) -> Result<()> {
    let deleted_pids = load_tombstones(index_path)?;
    let stats = compact_index_filtered(
        index_path,
        total_chunks,
        meta.num_centroids,
        meta.dim,
        meta.nbits as usize,
        device,
        &deleted_pids,
    )?;

    // Only rewrite chunk files and clear tombstones during explicit compact.
    let chunks_rewritten = rewrite_chunks && !deleted_pids.is_empty();
    if chunks_rewritten {
        rewrite_chunks_filtered(index_path, total_chunks, &deleted_pids, device)?;
        clear_tombstones(index_path)?;
    }

    let avg_doclen = if stats.num_active_passages > 0 {
        stats.total_embeddings as f64 / stats.num_active_passages as f64
    } else {
        0.0
    };

    let new_num_chunks = if chunks_rewritten {
        // After rewrite, count how many chunks remain
        (0..)
            .take_while(|i| index_path.join(format!("{}.codes.npy", i)).exists())
            .count()
    } else {
        total_chunks
    };

    let updated = IndexMetadata {
        num_chunks: new_num_chunks,
        nbits: meta.nbits,
        num_partitions: meta.num_partitions,
        num_embeddings: stats.total_embeddings,
        avg_doclen,
        num_passages: stats.num_active_passages,
        next_passage_id: Some(new_next_pid.unwrap_or(meta.next_pid())),
        num_centroids: meta.num_centroids,
        dim: meta.dim,
        created_at: meta.created_at.clone(),
    };
    updated.save(index_path)?;
    Ok(())
}

/// Rewrite chunk files, removing deleted passages and eliminating empty chunks.
///
/// Streams one chunk at a time: read → filter → write temp → drop.
/// Peak memory is O(max_chunk_size) instead of O(total_index_size).
fn rewrite_chunks_filtered(
    index_path: &Path,
    num_chunks: usize,
    deleted_pids: &std::collections::HashSet<i64>,
    device: Device,
) -> Result<()> {
    use serde_json::json;
    use std::io::BufWriter;

    let chunk_files = |idx: usize| -> [String; 6] {
        [
            format!("{}.codes.npy", idx),
            format!("{}.residuals.npy", idx),
            format!("doclens.{}.npy", idx),
            format!("doclens.{}.json", idx),
            format!("{}.passage_ids.npy", idx),
            format!("{}.metadata.json", idx),
        ]
    };

    // Pass 1: scan each chunk to determine which survive and build a mapping.
    // We only load lightweight doclens + pids here, no codes/residuals.
    let mut chunk_mapping: Vec<(usize, Vec<i64>, Vec<i64>, Vec<i64>)> = Vec::new(); // (old_idx, keep_doclens, keep_pids, keep_emb_indices)
    let mut positional_pid_counter: i64 = 0;

    for chunk_idx in 0..num_chunks {
        let doclens_tensor =
            Tensor::read_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?
                .to_device(Device::Cpu)
                .to_kind(Kind::Int64);
        let doclens_vec: Vec<i64> = doclens_tensor.try_into()?;

        let pids_path = index_path.join(format!("{}.passage_ids.npy", chunk_idx));
        let pids_vec: Vec<i64> = if pids_path.exists() {
            Tensor::read_npy(&pids_path)?
                .to_device(Device::Cpu)
                .to_kind(Kind::Int64)
                .try_into()?
        } else {
            let start = positional_pid_counter;
            (start..start + doclens_vec.len() as i64).collect()
        };
        positional_pid_counter += doclens_vec.len() as i64;

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
            .to_device(device);
        let residuals = Tensor::read_npy(index_path.join(format!("{}.residuals.npy", old_idx)))?
            .to_device(device);

        let idx_tensor = Tensor::from_slice(keep_emb_indices).to_device(device);
        let filtered_codes = codes.index_select(0, &idx_tensor).to_device(Device::Cpu);
        let filtered_residuals = residuals.index_select(0, &idx_tensor).to_device(Device::Cpu);
        // Drop originals immediately
        drop(codes);
        drop(residuals);

        filtered_codes
            .write_npy(index_path.join(format!("{}.codes.npy.tmp", new_idx)))?;
        filtered_residuals
            .write_npy(index_path.join(format!("{}.residuals.npy.tmp", new_idx)))?;

        Tensor::from_slice(keep_doclens)
            .write_npy(index_path.join(format!("doclens.{}.npy.tmp", new_idx)))?;

        let dl_file =
            std::fs::File::create(index_path.join(format!("doclens.{}.json.tmp", new_idx)))?;
        serde_json::to_writer(BufWriter::new(dl_file), keep_doclens)?;

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

    // Step 3: Atomic rename temp → final
    for new_idx in 0..new_count {
        for name in &chunk_files(new_idx) {
            let tmp = index_path.join(format!("{}.tmp", name));
            if tmp.exists() {
                std::fs::rename(&tmp, index_path.join(name))?;
            }
        }
    }

    // Step 4: Delete leftover old chunk files
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
