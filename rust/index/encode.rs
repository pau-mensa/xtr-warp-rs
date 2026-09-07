use anyhow::Result;
use serde_json::json;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::utils::maybe_progress;

/// Data from a single chunk, used for coalescing during add.
pub struct ChunkData {
    pub codes: Tensor,
    pub residuals: Tensor,
    pub doclens: Vec<i64>,
    pub pids: Vec<i64>,
}

/// Read the lightweight chunk files (codes, residuals, doclens, passage IDs).
pub fn read_chunk_data(index_path: &Path, chunk_idx: usize) -> Result<ChunkData> {
    let codes = Tensor::read_npy(index_path.join(format!("{}.codes.npy", chunk_idx)))?
        .to_device(tch::Device::Cpu);
    let residuals = Tensor::read_npy(index_path.join(format!("{}.residuals.npy", chunk_idx)))?
        .to_device(tch::Device::Cpu);
    let doclens: Vec<i64> =
        Tensor::read_npy(index_path.join(format!("doclens.{}.npy", chunk_idx)))?
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Int64)
            .try_into()?;
    let pids: Vec<i64> =
        Tensor::read_npy(index_path.join(format!("{}.passage_ids.npy", chunk_idx)))?
            .to_device(tch::Device::Cpu)
            .to_kind(Kind::Int64)
            .try_into()?;
    Ok(ChunkData {
        codes,
        residuals,
        doclens,
        pids,
    })
}

use crate::index::source::EmbeddingSource;
use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::IndexPlan;

pub const CHUNK_SIZE: usize = 256;
pub const EMB_BATCH_SIZE: i64 = 1 << 15;
pub const CODE_BATCH_SIZE: i64 = 1 << 20;

const BIT_WEIGHTS: [i64; 8] = [128, 64, 32, 16, 8, 4, 2, 1];

pub struct EncodeResult {
    pub chunk_stats: Vec<ChunkStats>,
    pub total_embeddings: i64,
    pub global_centroid_counts: Tensor,
    /// Per-embedding L2 residual norms (only populated when `collect_norms` is true).
    pub residual_norms: Option<Vec<f32>>,
}

pub struct ChunkStats {
    pub embedding_offset: usize,
    pub num_embeddings: usize,
}

pub fn encode_chunks(
    plan: &IndexPlan,
    source: &mut dyn EmbeddingSource,
    centroids: &Tensor,
    codec: &ResidualCodec,
    index_path: &Path,
    device: Device,
    embedding_dim: u32,
    passage_ids: Option<&[i64]>,
    start_chunk_idx: usize,
    chunk_size: usize,
    compression_batch_size: i64,
    resume: bool,
    show_progress: bool,
) -> Result<EncodeResult> {
    encode_chunks_inner(
        plan,
        source,
        centroids,
        codec,
        index_path,
        device,
        embedding_dim,
        passage_ids,
        start_chunk_idx,
        chunk_size,
        compression_batch_size,
        resume,
        false,
        show_progress,
    )
}

/// Like `encode_chunks` but also returns per-embedding residual norms.
pub fn encode_chunks_with_norms(
    plan: &IndexPlan,
    source: &mut dyn EmbeddingSource,
    centroids: &Tensor,
    codec: &ResidualCodec,
    index_path: &Path,
    device: Device,
    embedding_dim: u32,
    passage_ids: Option<&[i64]>,
    start_chunk_idx: usize,
    chunk_size: usize,
    compression_batch_size: i64,
    resume: bool,
    show_progress: bool,
) -> Result<EncodeResult> {
    encode_chunks_inner(
        plan,
        source,
        centroids,
        codec,
        index_path,
        device,
        embedding_dim,
        passage_ids,
        start_chunk_idx,
        chunk_size,
        compression_batch_size,
        resume,
        true,
        show_progress,
    )
}

fn encode_chunks_inner(
    plan: &IndexPlan,
    source: &mut dyn EmbeddingSource,
    centroids: &Tensor,
    codec: &ResidualCodec,
    index_path: &Path,
    device: Device,
    embedding_dim: u32,
    passage_ids: Option<&[i64]>,
    start_chunk_idx: usize,
    chunk_size: usize,
    compression_batch_size: i64,
    resume: bool,
    collect_norms: bool,
    show_progress: bool,
) -> Result<EncodeResult> {
    if let Some(pids) = passage_ids {
        anyhow::ensure!(
            pids.len() == source.num_docs(),
            "passage_ids length ({}) must match source num_docs ({})",
            pids.len(),
            source.num_docs()
        );
    }

    let num_centroids = centroids.size()[0] as usize;
    let mut chunk_stats = Vec::with_capacity(plan.num_chunks);
    let mut current_emb_offset: usize = 0;
    let mut total_embeddings: i64 = 0;
    anyhow::ensure!(chunk_size > 0, "indexing_chunk_size must be positive");
    anyhow::ensure!(
        compression_batch_size > 0,
        "compression_batch_size must be positive"
    );
    let mut global_counts = Tensor::zeros(&[num_centroids as i64], (Kind::Int64, Device::Cpu));
    let mut passage_offset: usize = 0;
    let mut all_norms: Vec<f32> = Vec::new();

    let completed_chunks = if resume {
        restore_completed_prefix(
            index_path,
            start_chunk_idx,
            plan.num_chunks,
            &mut chunk_stats,
            &mut passage_offset,
            &mut current_emb_offset,
            &mut total_embeddings,
        )?
    } else {
        0
    };

    let bar = maybe_progress(show_progress, plan.num_chunks as u64, "Encoding chunks");
    bar.set_position(completed_chunks as u64);

    let chunk_iter = source.chunk_iter_from(chunk_size, passage_offset)?;
    for (local_chk_idx, chunk) in chunk_iter.enumerate() {
        let chk_idx = start_chunk_idx + completed_chunks + local_chk_idx;
        let chunk = chunk?;
        let chk_doclens = chunk.doclens;
        let chk_embs_vec = chunk.embeddings;
        // Keep the complete source chunk on CPU. Only one bounded token batch
        // and one distance/residual working set live on the accelerator.
        let chk_embs_tensor = Tensor::cat(&chk_embs_vec, 0).to_kind(Kind::Half);
        total_embeddings += chk_embs_tensor.size()[0];

        let mut chk_codes_list: Vec<Tensor> = Vec::new();
        let mut chk_res_list: Vec<Tensor> = Vec::new();

        for emb_batch_cpu in chk_embs_tensor.split(compression_batch_size, 0) {
            let emb_batch = emb_batch_cpu.to_device(device);
            let code_batch = compress_into_codes(&emb_batch, &codec.centroids);
            chk_codes_list.push(code_batch.to_device(Device::Cpu));

            let mut recon_centroids_batches: Vec<Tensor> = Vec::new();
            for sub_code_batch in code_batch.split(CODE_BATCH_SIZE, 0) {
                recon_centroids_batches.push(codec.centroids.index_select(0, &sub_code_batch));
            }
            let recon_centroids = Tensor::cat(&recon_centroids_batches, 0);

            let mut res_batch = &emb_batch - &recon_centroids;

            if collect_norms {
                let norms = res_batch
                    .to_kind(Kind::Float)
                    .norm_scalaropt_dim(2, &[1], false)
                    .to_device(Device::Cpu);
                let norms_vec: Vec<f32> = norms.try_into()?;
                all_norms.extend(norms_vec);
            }

            let bucket_cutoffs = codec.bucket_cutoffs.as_ref().unwrap().contiguous();
            res_batch = Tensor::bucketize(&res_batch, &bucket_cutoffs, true, false);

            let mut res_shape = res_batch.size();
            res_shape.push(plan.nbits as i64);
            res_batch = res_batch.unsqueeze(-1).expand(&res_shape, false);
            res_batch = res_batch.bitwise_right_shift(&codec.arange_bits);
            let ones = Tensor::ones_like(&res_batch).to_device(device);
            res_batch = res_batch.bitwise_and_tensor(&ones);

            let res_flat = res_batch.flatten(0, -1);

            let res_packed = packbits(&res_flat);

            let shape = [
                res_batch.size()[0],
                (embedding_dim as i64) / 8 * (plan.nbits as i64),
            ];
            chk_res_list.push(res_packed.reshape(&shape).to_device(Device::Cpu));
        }

        let chk_codes = Tensor::cat(&chk_codes_list, 0);
        let chk_residuals = Tensor::cat(&chk_res_list, 0);
        let chunk_num_embeddings = chk_codes.size()[0] as usize;

        let chunk_counts = chk_codes.bincount::<Tensor>(None, num_centroids as i64);
        global_counts = &global_counts + &chunk_counts;

        let chk_codes_fpath = index_path.join(&format!("{}.codes.npy", chk_idx));
        chk_codes
            .to_device(Device::Cpu)
            .write_npy(&chk_codes_fpath)?;

        let chk_res_fpath = index_path.join(&format!("{}.residuals.npy", chk_idx));
        chk_residuals
            .to_device(Device::Cpu)
            .write_npy(&chk_res_fpath)?;

        let chk_doclens_fpath = index_path.join(format!("doclens.{}.npy", chk_idx));
        Tensor::from_slice(&chk_doclens).write_npy(chk_doclens_fpath)?;

        // Write explicit passage IDs for this chunk
        let chunk_pids: Vec<i64> = if let Some(pids) = passage_ids {
            pids[passage_offset..passage_offset + chk_doclens.len()].to_vec()
        } else {
            (passage_offset as i64..(passage_offset + chk_doclens.len()) as i64).collect()
        };
        let chunk_pids_fpath = index_path.join(format!("{}.passage_ids.npy", chk_idx));
        Tensor::from_slice(&chunk_pids).write_npy(&chunk_pids_fpath)?;

        let chk_meta = json!({
            "passage_offset": passage_offset,
            "num_passages": chk_doclens.len(),
            "num_embeddings": chunk_num_embeddings,
            "embedding_offset": current_emb_offset,
        });
        // The metadata file marks the chunk as a complete checkpoint, so it is
        // written last and atomically (temp + rename): a crash mid-write must
        // not leave a truncated file that looks like a finished chunk.
        let chk_meta_fpath = index_path.join(format!("{}.metadata.json", chk_idx));
        let chk_meta_tmp_fpath = index_path.join(format!("{}.metadata.json.tmp", chk_idx));
        {
            let mut buf_writer_meta = BufWriter::new(File::create(&chk_meta_tmp_fpath)?);
            serde_json::to_writer(&mut buf_writer_meta, &chk_meta)?;
            buf_writer_meta.flush()?;
            buf_writer_meta.get_ref().sync_all()?;
        }
        std::fs::rename(&chk_meta_tmp_fpath, &chk_meta_fpath)?;

        chunk_stats.push(ChunkStats {
            embedding_offset: current_emb_offset,
            num_embeddings: chunk_num_embeddings,
        });
        current_emb_offset += chunk_num_embeddings;
        passage_offset += chk_doclens.len();

        bar.inc(1);
        if show_progress
            && ((completed_chunks + local_chk_idx + 1) % 10 == 0
                || completed_chunks + local_chk_idx + 1 == plan.num_chunks)
        {
            eprintln!(
                "Encoded chunk {}/{} ({} token embeddings total)",
                completed_chunks + local_chk_idx + 1,
                plan.num_chunks,
                total_embeddings
            );
        }
    }

    bar.finish_and_clear();

    Ok(EncodeResult {
        chunk_stats,
        total_embeddings,
        global_centroid_counts: global_counts,
        residual_norms: if collect_norms { Some(all_norms) } else { None },
    })
}

#[derive(serde::Deserialize)]
struct StoredChunkMetadata {
    passage_offset: usize,
    num_passages: usize,
    num_embeddings: usize,
    embedding_offset: usize,
}

#[allow(clippy::too_many_arguments)]
fn restore_completed_prefix(
    index_path: &Path,
    start_chunk_idx: usize,
    num_chunks: usize,
    chunk_stats: &mut Vec<ChunkStats>,
    passage_offset: &mut usize,
    embedding_offset: &mut usize,
    total_embeddings: &mut i64,
) -> Result<usize> {
    let mut completed = 0usize;
    for local_idx in 0..num_chunks {
        let chunk_idx = start_chunk_idx + local_idx;
        let required = [
            index_path.join(format!("{}.codes.npy", chunk_idx)),
            index_path.join(format!("{}.residuals.npy", chunk_idx)),
            index_path.join(format!("doclens.{}.npy", chunk_idx)),
            index_path.join(format!("{}.passage_ids.npy", chunk_idx)),
            index_path.join(format!("{}.metadata.json", chunk_idx)),
        ];
        if required.iter().any(|path| !path.is_file()) {
            break;
        }
        // An unreadable metadata file means the chunk never finished; stop the
        // completed prefix here and re-encode from this chunk on.
        let metadata: StoredChunkMetadata =
            match serde_json::from_reader(BufReader::new(File::open(&required[4])?)) {
                Ok(metadata) => metadata,
                Err(_) => break,
            };
        anyhow::ensure!(
            metadata.passage_offset == *passage_offset
                && metadata.embedding_offset == *embedding_offset,
            "chunk {} is not a contiguous resume checkpoint",
            chunk_idx
        );
        chunk_stats.push(ChunkStats {
            embedding_offset: metadata.embedding_offset,
            num_embeddings: metadata.num_embeddings,
        });
        *passage_offset += metadata.num_passages;
        *embedding_offset += metadata.num_embeddings;
        *total_embeddings += metadata.num_embeddings as i64;
        completed += 1;
    }
    Ok(completed)
}

pub fn compress_into_codes(embs: &Tensor, centroids: &Tensor) -> Tensor {
    let embs = embs.to_kind(Kind::Half);
    let centroids = centroids.to_kind(Kind::Half);
    let mut codes = Vec::new();
    let batch_sz = (1 << 29) / centroids.size()[0] as i64;
    for mut emb_batch in embs.split(batch_sz, 0) {
        codes.push(centroids.matmul(&emb_batch.t_()).argmax(0, false));
    }
    Tensor::cat(&codes, 0)
}

pub fn packbits(res: &Tensor) -> Tensor {
    let bits_mat = res.reshape(&[-1, 8]);
    let weights = Tensor::from_slice(&BIT_WEIGHTS)
        .to_device(res.device())
        .to_kind(Kind::Float);
    let packed = bits_mat
        .to_kind(Kind::Float)
        .matmul(&weights)
        .to_kind(Kind::Uint8);
    packed
}
