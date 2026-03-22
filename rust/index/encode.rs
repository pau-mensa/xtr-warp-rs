use anyhow::Result;
use serde_json::json;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tch::{Device, Kind, Tensor};

use crate::index::source::EmbeddingSource;
use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::IndexPlan;

pub const CHUNK_SIZE: usize = 25_000;
pub const EMB_BATCH_SIZE: i64 = 1 << 18;
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
) -> Result<EncodeResult> {
    encode_chunks_inner(plan, source, centroids, codec, index_path, device, embedding_dim, passage_ids, start_chunk_idx, false)
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
) -> Result<EncodeResult> {
    encode_chunks_inner(plan, source, centroids, codec, index_path, device, embedding_dim, passage_ids, start_chunk_idx, true)
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
    collect_norms: bool,
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
    let mut global_counts = Tensor::zeros(&[num_centroids as i64], (Kind::Int64, device));
    let mut passage_offset: usize = 0;
    let mut all_norms: Vec<f32> = Vec::new();

    let chunk_iter = source.chunk_iter(CHUNK_SIZE)?;
    for (local_chk_idx, chunk) in chunk_iter.enumerate() {
        let chk_idx = start_chunk_idx + local_chk_idx;
        let chunk = chunk?;
        let chk_doclens = chunk.doclens;
        let chk_embs_vec = chunk.embeddings;
        let chk_embs_tensor = Tensor::cat(&chk_embs_vec, 0)
            .to_kind(Kind::Half)
            .to_device(device);
        total_embeddings += chk_embs_tensor.size()[0];

        let mut chk_codes_list: Vec<Tensor> = Vec::new();
        let mut chk_res_list: Vec<Tensor> = Vec::new();

        for emb_batch in chk_embs_tensor.split(EMB_BATCH_SIZE, 0) {
            let code_batch = compress_into_codes(&emb_batch, &codec.centroids);
            chk_codes_list.push(code_batch.shallow_clone());

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
            chk_res_list.push(res_packed.reshape(&shape));
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

        let chk_doclens_path = index_path.join(format!("doclens.{}.json", chk_idx));
        let dl_file = File::create(chk_doclens_path)?;
        let buf_writer = BufWriter::new(dl_file);
        serde_json::to_writer(buf_writer, &chk_doclens)?;

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
        let chk_meta_fpath = index_path.join(format!("{}.metadata.json", chk_idx));
        let meta_f_w = File::create(chk_meta_fpath)?;
        let buf_writer_meta = BufWriter::new(meta_f_w);
        serde_json::to_writer(buf_writer_meta, &chk_meta)?;

        chunk_stats.push(ChunkStats {
            embedding_offset: current_emb_offset,
            num_embeddings: chunk_num_embeddings,
        });
        current_emb_offset += chunk_num_embeddings;
        passage_offset += chk_doclens.len();
    }

    Ok(EncodeResult {
        chunk_stats,
        total_embeddings,
        global_centroid_counts: global_counts,
        residual_norms: if collect_norms { Some(all_norms) } else { None },
    })
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
