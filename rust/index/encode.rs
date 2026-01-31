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

pub struct Phase2Out {
    pub chunk_stats: Vec<ChunkStats>,
    pub total_embeddings: i64,
    pub global_centroid_counts: Tensor,
}

pub struct ChunkStats {
    pub embedding_offset: usize,
    pub num_embeddings: usize,
}

pub fn encode_phase_two(
    plan: &IndexPlan,
    source: &mut dyn EmbeddingSource,
    centroids: &Tensor,
    codec: &ResidualCodec,
    index_path: &Path,
    device: Device,
    embedding_dim: u32,
) -> Result<Phase2Out> {
    let num_centroids = centroids.size()[0] as usize;
    let mut chunk_stats = Vec::with_capacity(plan.num_chunks);
    let mut current_emb_offset: usize = 0;
    let mut total_embeddings: i64 = 0;
    let mut global_counts = Tensor::zeros(&[num_centroids as i64], (Kind::Int64, device));
    let mut passage_offset: usize = 0;

    let chunk_iter = source.chunk_iter(CHUNK_SIZE)?;
    for (chk_idx, chunk) in chunk_iter.enumerate() {
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

    Ok(Phase2Out {
        chunk_stats,
        total_embeddings,
        global_centroid_counts: global_counts,
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
