use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::index::compact::compact_index_filtered;
use crate::index::delete::{clear_tombstones, load_tombstones, save_tombstones};
use crate::index::encode::{encode_chunks, CHUNK_SIZE};
use crate::index::source::EmbeddingSource;
use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::{AddResult, IndexPlan};

// ── On-disk metadata ──

/// Complete on-disk representation of `metadata.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMetadata {
    pub num_chunks: usize,
    pub nbits: u8,
    pub num_partitions: i64,
    pub num_embeddings: i64,
    pub avg_doclen: f64,
    pub num_passages: usize,
    /// Watermark for the next passage ID to assign.
    /// Defaults to `num_passages` for indexes created before this field existed.
    pub next_passage_id: Option<i64>,
    pub num_centroids: usize,
    pub dim: usize,
    pub created_at: String,
}

impl DiskMetadata {
    pub fn load(index_path: &Path) -> Result<Self> {
        let path = index_path.join("metadata.json");
        let file =
            File::open(&path).with_context(|| format!("Failed to open {}", path.display()))?;
        let meta: DiskMetadata = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("Failed to parse {}", path.display()))?;
        Ok(meta)
    }

    pub fn save(&self, index_path: &Path) -> Result<()> {
        let path = index_path.join("metadata.json");
        let file =
            File::create(&path).with_context(|| format!("Failed to create {}", path.display()))?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)?;
        Ok(())
    }

    /// Effective next passage ID
    pub fn next_pid(&self) -> i64 {
        self.next_passage_id.unwrap_or(self.num_passages as i64)
    }
}

// ── Public operations ──

/// Add new documents to an existing index.
///
/// Encodes new embeddings as new chunk files, then rebuilds the compacted
/// search structures excluding tombstoned passage IDs.  Chunk files and
/// tombstones are left in place — call `compact_standalone` to clean them up.
pub fn add_to_index(
    embeddings: &mut dyn EmbeddingSource,
    index_path: &Path,
    device: Device,
) -> Result<AddResult> {
    let meta = DiskMetadata::load(index_path)?;
    let next_pid = meta.next_pid();

    let codec = load_codec_from_disk(index_path, meta.nbits, device)?;
    let centroids = Tensor::read_npy(index_path.join("centroids.npy"))?.to_device(device);

    // Assign new passage IDs: next_pid .. next_pid + num_new_docs
    let num_new_docs = embeddings.num_docs();
    let new_passage_ids: Vec<i64> = (next_pid..next_pid + num_new_docs as i64).collect();

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
        Some(&new_passage_ids),
        meta.num_chunks,
    )?;

    let total_chunks = meta.num_chunks + encode_result.chunk_stats.len();
    compact_and_update_metadata(
        index_path,
        &meta,
        total_chunks,
        Some(next_pid + num_new_docs as i64),
        device,
        false,
    )?;

    Ok(AddResult { new_passage_ids })
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

    let meta = DiskMetadata::load(index_path)?;

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
pub fn compact_standalone(index_path: &Path, device: Device) -> Result<()> {
    let meta = DiskMetadata::load(index_path)?;
    compact_and_update_metadata(index_path, &meta, meta.num_chunks, None, device, true)?;
    Ok(())
}

// ── Shared helpers ──

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
    meta: &DiskMetadata,
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

    let updated = DiskMetadata {
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
/// Reads each existing chunk, filters out passages whose IDs are in
/// `deleted_pids`, writes surviving data back as contiguous chunks
/// numbered 0..N, and deletes any leftover old chunk files.
fn rewrite_chunks_filtered(
    index_path: &Path,
    num_chunks: usize,
    deleted_pids: &std::collections::HashSet<i64>,
    device: Device,
) -> Result<()> {
    use serde_json::json;
    use std::io::BufWriter;

    // Collect filtered chunk data into memory first so we can renumber from 0.
    struct FilteredChunk {
        codes: Tensor,
        residuals: Tensor,
        doclens: Vec<i64>,
        passage_ids: Vec<i64>,
    }

    let mut filtered_chunks: Vec<FilteredChunk> = Vec::new();

    for chunk_idx in 0..num_chunks {
        let codes = Tensor::read_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?
            .to_device(device);
        let residuals = Tensor::read_npy(index_path.join(format!("{}.residuals.npy", chunk_idx)))?
            .to_device(device);
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
            // Backward compat: derive from position
            let start: i64 = filtered_chunks
                .iter()
                .flat_map(|c| c.passage_ids.iter())
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            (start..start + doclens_vec.len() as i64).collect()
        };

        // Filter: keep only passages not in deleted_pids
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

        if keep_doclens.is_empty() {
            continue; // skip entire chunk deleted
        }

        let idx_tensor = Tensor::from_slice(&keep_emb_indices).to_device(device);
        filtered_chunks.push(FilteredChunk {
            codes: codes.index_select(0, &idx_tensor),
            residuals: residuals.index_select(0, &idx_tensor),
            doclens: keep_doclens,
            passage_ids: keep_pids,
        });
    }

    let new_count = filtered_chunks.len();
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

    // Step 1: Write new chunks to temp files
    let mut passage_offset: usize = 0;
    let mut emb_offset: usize = 0;

    for (new_idx, chunk) in filtered_chunks.iter().enumerate() {
        chunk
            .codes
            .to_device(Device::Cpu)
            .write_npy(index_path.join(format!("{}.codes.npy.tmp", new_idx)))?;
        chunk
            .residuals
            .to_device(Device::Cpu)
            .write_npy(index_path.join(format!("{}.residuals.npy.tmp", new_idx)))?;

        Tensor::from_slice(&chunk.doclens)
            .write_npy(index_path.join(format!("doclens.{}.npy.tmp", new_idx)))?;

        let dl_file =
            std::fs::File::create(index_path.join(format!("doclens.{}.json.tmp", new_idx)))?;
        serde_json::to_writer(BufWriter::new(dl_file), &chunk.doclens)?;

        Tensor::from_slice(&chunk.passage_ids)
            .write_npy(index_path.join(format!("{}.passage_ids.npy.tmp", new_idx)))?;

        let num_embs = chunk.codes.size()[0] as usize;
        let chk_meta = json!({
            "passage_offset": passage_offset,
            "num_passages": chunk.doclens.len(),
            "num_embeddings": num_embs,
            "embedding_offset": emb_offset,
        });
        let meta_f =
            std::fs::File::create(index_path.join(format!("{}.metadata.json.tmp", new_idx)))?;
        serde_json::to_writer(BufWriter::new(meta_f), &chk_meta)?;

        passage_offset += chunk.doclens.len();
        emb_offset += num_embs;
    }

    // Step 2: Atomic rename temp → final (replaces old files for indices 0..new_count)
    for new_idx in 0..new_count {
        for name in &chunk_files(new_idx) {
            std::fs::rename(
                index_path.join(format!("{}.tmp", name)),
                index_path.join(name),
            )?;
        }
    }

    // Step 3: Delete leftover old chunk files (indices new_count..num_chunks)
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
