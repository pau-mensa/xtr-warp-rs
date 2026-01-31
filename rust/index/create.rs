use anyhow::{anyhow, bail, Result};
use chrono::Utc;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use super::{compact, encode, ivf, source::EmbeddingSource};
use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::{IndexConfig, IndexPlan};

/// Creates a new WARP index from a collection of document embeddings.
/// Result containing the index metadata on success
pub fn create_index(
    config: &IndexConfig,
    embeddings_source: &mut dyn EmbeddingSource,
    centroids: Tensor,
    seed: Option<u64>,
) -> Result<()> {
    // Create the index directory if it doesn't exist
    std::fs::create_dir_all(&config.index_path)?;

    let (index_plan, sample_pids, sampled_embeddings) =
        plan_and_sample(config, embeddings_source, seed)?;

    let plan_fpath = config.index_path.join("plan.json");
    let plan_data = json!({ "nbits": index_plan.nbits, "num_chunks": index_plan.num_chunks });
    let mut plan_file = File::create(plan_fpath)?;
    writeln!(plan_file, "{}", serde_json::to_string_pretty(&plan_data)?)?;

    let path_str = config
        .index_path
        .as_path()
        .to_str()
        .expect("index_path is not valid UTF-8");

    let pids_fpath = Path::new(&path_str).join("pids.npy");
    Tensor::from_slice(&sample_pids).write_npy(&pids_fpath)?;

    // Train residual codec using sampled embeddings
    let codec = train_residual_codec(
        &sampled_embeddings,
        &centroids,
        config.nbits,
        config.embedding_dim,
        config.device,
        &path_str,
    )?;

    let phase2_out = encode::encode_phase_two(
        &index_plan,
        embeddings_source,
        &centroids,
        &codec,
        &config.index_path,
        config.device,
        config.embedding_dim,
    )?;
    phase3(config, &index_plan, &phase2_out, &centroids)?;

    Ok(())
}

fn plan_and_sample(
    config: &IndexConfig,
    source: &mut dyn EmbeddingSource,
    seed: Option<u64>,
) -> Result<(IndexPlan, Vec<i64>, Tensor)> {
    let n_docs = source.num_docs();
    if n_docs == 0 {
        bail!("No embeddings provided");
    }
    let num_chunks =
        (n_docs as f64 / (encode::CHUNK_SIZE as f64).min(1.0 + n_docs as f64)).ceil() as usize;

    let total_doc_len = if source.get_doc(0).is_some() {
        let mut total: i64 = 0;
        for idx in 0..n_docs {
            let doc = source
                .get_doc(idx)
                .ok_or_else(|| anyhow!("Missing embedding at index {}", idx))?;
            total += doc.size()[0];
        }
        total
    } else {
        let mut total: i64 = 0;
        let chunk_iter = source.chunk_iter(encode::CHUNK_SIZE)?;
        for chunk in chunk_iter {
            total += chunk.doclens.iter().sum::<i64>();
        }
        total
    };

    let avg_doc_len = total_doc_len as f64 / n_docs as f64;
    let mut est_total_embs_f64 = (n_docs as f64) * avg_doc_len;
    est_total_embs_f64 = (16.0 * est_total_embs_f64.sqrt()).log2().floor();
    let est_total_embs = 2f64.powf(est_total_embs_f64) as i64;

    let index_plan = IndexPlan {
        n_docs,
        num_chunks,
        avg_doc_len,
        est_total_embs,
        nbits: config.nbits,
    };

    let mut rng = if let Some(seed_value) = seed {
        Box::new(StdRng::seed_from_u64(seed_value)) as Box<dyn RngCore>
    } else {
        Box::new(rand::rng()) as Box<dyn RngCore>
    };
    let (sample_pids, sampled_embeddings) =
        sample_embeddings(source, n_docs, config.embedding_dim, &mut *rng, config.device)?;

    Ok((index_plan, sample_pids, sampled_embeddings))
}

fn sample_embeddings(
    source: &mut dyn EmbeddingSource,
    n_docs: usize,
    embedding_dim: u32,
    rng: &mut dyn RngCore,
    device: Device,
) -> Result<(Vec<i64>, Tensor)> {
    let sample_k_float = 16.0 * (120.0 * n_docs as f64).sqrt();
    let k = (1.0 + sample_k_float).min(n_docs as f64) as usize;
    if k == 0 {
        let empty = Tensor::zeros(&[0, embedding_dim as i64], (Kind::Half, device));
        return Ok((Vec::new(), empty));
    }

    if source.get_doc(0).is_some() {
        let mut passage_indices: Vec<i64> = (0..n_docs as i64).collect();
        passage_indices.shuffle(rng);
        let sample_pids: Vec<i64> = passage_indices.into_iter().take(k).collect();

        let mut sample_tensors_vec: Vec<&Tensor> = Vec::with_capacity(k);
        for &pid in &sample_pids {
            let doc = source
                .get_doc(pid as usize)
                .ok_or_else(|| anyhow!("Missing embedding at index {}", pid))?;
            sample_tensors_vec.push(doc);
        }

        let sampled_embeddings = Tensor::cat(&sample_tensors_vec, 0)
            .to_kind(Kind::Half)
            .to_device(device);
        return Ok((sample_pids, sampled_embeddings));
    }

    let mut sample_tensors: Vec<Tensor> = Vec::with_capacity(k);
    let mut sample_pids: Vec<i64> = Vec::with_capacity(k);
    let mut seen: i64 = 0;
    let mut doc_offset: i64 = 0;
    let chunk_iter = source.chunk_iter(encode::CHUNK_SIZE)?;
    for chunk in chunk_iter {
        for doc in &chunk.embeddings {
            if (seen as usize) < k {
                sample_tensors.push(doc.shallow_clone());
                sample_pids.push(doc_offset);
            } else {
                let j = (rng.next_u64() % (seen as u64 + 1)) as usize;
                if j < k {
                    sample_tensors[j] = doc.shallow_clone();
                    sample_pids[j] = doc_offset;
                }
            }
            seen += 1;
            doc_offset += 1;
        }
    }

    let sample_refs: Vec<&Tensor> = sample_tensors.iter().collect();
    let sampled_embeddings = Tensor::cat(&sample_refs, 0)
        .to_kind(Kind::Half)
        .to_device(device);

    Ok((sample_pids, sampled_embeddings))
}

fn finalize_ivf_and_compact(
    config: &IndexConfig,
    plan: &IndexPlan,
    phase2_out: &encode::Phase2Out,
    centroids: &Tensor,
) -> Result<()> {
    let final_meta_fpath = config.index_path.join("metadata.json");
    let final_avg_doclen = if plan.n_docs > 0 {
        phase2_out.total_embeddings as f64 / plan.n_docs as f64
    } else {
        0.0
    };

    let final_meta_json = json!({
        "num_chunks": plan.num_chunks,
        "nbits": plan.nbits,
        "num_partitions": plan.est_total_embs,
        "num_embeddings": phase2_out.total_embeddings,
        "avg_doclen": final_avg_doclen,
        "num_passages": plan.n_docs,
        "num_centroids": centroids.size()[0] as usize,
        "dim": config.embedding_dim,
        "created_at": Utc::now().to_rfc3339(),
        "index_version": "xtr-warp-1.0"
    });

    let meta_file = File::create(final_meta_fpath)?;
    let writer = BufWriter::new(meta_file);
    serde_json::to_writer_pretty(writer, &final_meta_json)?;

    compact::compact_index_counting_sort(
        &config.index_path,
        plan.num_chunks,
        centroids.size()[0] as usize,
        config.embedding_dim as usize,
        plan.nbits as usize,
        config.device,
        &phase2_out.global_centroid_counts,
    )?;

    Ok(())
}

#[allow(dead_code)]
fn phase_two(
    plan: &IndexPlan,
    source: &mut dyn EmbeddingSource,
    centroids: &Tensor,
    codec: &ResidualCodec,
    index_path: &Path,
    device: Device,
    embedding_dim: u32,
) -> Result<encode::Phase2Out> {
    encode::encode_phase_two(plan, source, centroids, codec, index_path, device, embedding_dim)
}

pub struct Phase3Out {}

#[allow(dead_code)]
fn phase3(
    config: &IndexConfig,
    plan: &IndexPlan,
    phase2_out: &encode::Phase2Out,
    centroids: &Tensor,
) -> Result<Phase3Out> {
    finalize_ivf_and_compact(config, plan, phase2_out, centroids)?;
    Ok(Phase3Out {})
}

#[allow(dead_code)]
fn legacy_full_pipeline(
    config: &IndexConfig,
    n_docs: usize,
    num_chunks: usize,
    est_total_embs: i64,
    document_embeddings: &Vec<Tensor>,
    codec: &ResidualCodec,
    path_str: &str,
    centroids: &Tensor,
) -> Result<()> {
    let chunk_stats = compress_into_residuals(
        n_docs,
        num_chunks,
        config.nbits,
        config.embedding_dim as i64,
        document_embeddings,
        config.device,
        path_str,
        codec,
    )?;

    let chk_emb_offsets: Vec<usize> = chunk_stats.iter().map(|s| s.embedding_offset).collect();
    let total_num_embs = chunk_stats
        .last()
        .map(|s| s.embedding_offset + s.num_embeddings)
        .unwrap_or(0);
    ivf::build_ivf(
        &chk_emb_offsets,
        est_total_embs,
        total_num_embs,
        num_chunks,
        path_str,
        config.device,
    )?;

    let final_meta_fpath = Path::new(&config.index_path).join("metadata.json");
    let final_avg_doclen = if n_docs > 0 {
        total_num_embs as f64 / n_docs as f64
    } else {
        0.0
    };

    let final_meta_json = json!({
        "num_chunks": num_chunks,
        "nbits": config.nbits,
        "num_partitions": est_total_embs,
        "num_embeddings": total_num_embs,
        "avg_doclen": final_avg_doclen
    });

    let meta_file = File::create(final_meta_fpath)?;
    let writer = BufWriter::new(meta_file);
    serde_json::to_writer_pretty(writer, &final_meta_json)?;

    compact::compact_index(
        &config.index_path,
        num_chunks,
        centroids.size()[0] as usize,
        config.embedding_dim as usize,
        config.nbits as usize,
        config.device,
    )?;

    Ok(())
}

/// Trains the residual codec for quantization.
/// # Returns
/// Trained residual codec
fn train_residual_codec(
    sample_embeddings: &Tensor,
    centroids: &Tensor,
    nbits: u8,
    embedding_dim: u32,
    device: Device,
    index_path: &str,
) -> Result<ResidualCodec> {
    let total_samples = sample_embeddings.size()[0] as f64;
    let heldout_sz = (0.05 * total_samples).min(50_000f64).round() as i64;
    let sample_splits =
        sample_embeddings.split_with_sizes(&[total_samples as i64 - heldout_sz, heldout_sz], 0);

    let heldout_samples = sample_splits[1].shallow_clone();

    let centroids_half = centroids.to_kind(Kind::Half);
    let initial_codec = ResidualCodec::load(
        nbits,
        centroids_half.copy(),
        Tensor::zeros(&[embedding_dim as i64], (Kind::Float, device)),
        None,
        None,
        device,
    )?;

    let heldout_codes = encode::compress_into_codes(&heldout_samples, &initial_codec.centroids);

    let mut recon_embs_vec = Vec::new();
    for code_batch_idxs in heldout_codes.split(encode::CODE_BATCH_SIZE, 0) {
        recon_embs_vec.push(initial_codec.centroids.index_select(0, &code_batch_idxs));
    }
    let heldout_recon_embs = Tensor::cat(&recon_embs_vec, 0);

    let heldout_res_raw = (&heldout_samples - &heldout_recon_embs).to_kind(Kind::Float);
    let avg_res_per_dim = heldout_res_raw
        .abs()
        .mean_dim(Some(&[0i64][..]), false, Kind::Float)
        .to_device(device);

    let n_options = 2_i32.pow(nbits as u32);
    let quantiles_base =
        Tensor::arange_start(0, n_options.into(), (Kind::Float, device)) * (1.0 / n_options as f64);
    let cutoff_quantiles = quantiles_base.narrow(0, 1, n_options as i64 - 1);
    let weight_quantiles = &quantiles_base + (0.5 / n_options as f64);

    let heldout_res_flat = heldout_res_raw.flatten(0, -1); // Flatten all residuals
    let b_cutoffs = heldout_res_flat.quantile(&cutoff_quantiles, None, false, "linear"); // Results in [num_quantiles]

    let b_weights = heldout_res_flat.quantile(&weight_quantiles, None, false, "linear"); // Results in [num_quantiles]

    let final_codec = ResidualCodec::load(
        nbits,
        initial_codec.centroids.copy(), // TODO could this be improved by setting the avg_res_per_dim, b_cutoffs, b_weights so we don't have to copy the centroids tensor?
        avg_res_per_dim,
        Some(b_cutoffs.copy()),
        Some(b_weights.copy()),
        device,
    )?;

    let centroids_fpath = Path::new(&index_path).join("centroids.npy");
    final_codec
        .centroids
        .to_device(Device::Cpu)
        .write_npy(&centroids_fpath)?;

    let cutoffs_fpath = Path::new(&index_path).join("bucket_cutoffs.npy");
    b_cutoffs.to_device(Device::Cpu).write_npy(&cutoffs_fpath)?;

    let weights_fpath = Path::new(&index_path).join("bucket_weights.npy");
    b_weights.to_device(Device::Cpu).write_npy(&weights_fpath)?;

    let avg_res_fpath = Path::new(&index_path).join("avg_residual.npy");
    final_codec
        .avg_residual
        .to_device(Device::Cpu)
        .write_npy(&avg_res_fpath)?;

    Ok(final_codec)
}

// Legacy encoder kept for reference; Phase 2 lives in `encode::encode_phase_two`.
pub fn compress_into_residuals(
    n_docs: usize,
    n_chunks: usize,
    nbits: u8,
    embedding_dim: i64,
    document_embeddings: &Vec<Tensor>,
    device: Device,
    index_path: &str,
    codec: &ResidualCodec,
) -> Result<Vec<encode::ChunkStats>> {
    let proc_chunk_sz = encode::CHUNK_SIZE.min(1 + n_docs);
    let mut chunk_stats = Vec::with_capacity(n_chunks);
    let mut current_emb_offset: usize = 0;
    let doclens: Vec<i64> = document_embeddings.iter().map(|e| e.size()[0]).collect();
    let mut _total_embeddings: i64 = 0;
    let mut _global_centroid_counts: i64 = 0;

    for chk_idx in 0..n_chunks {
        let chk_offset = chk_idx * proc_chunk_sz;
        let chk_end_offset = (chk_offset + proc_chunk_sz).min(n_docs);

        let chk_embs_vec: Vec<Tensor> = document_embeddings[chk_offset..chk_end_offset]
            .iter()
            .map(|t| t.to_kind(Kind::Half))
            .collect();
        // let chk_doclens: Vec<i64> = chk_embs_vec.iter().map(|e| e.size()[0]).collect();
        let chk_doclens: Vec<i64> = doclens[chk_offset..chk_end_offset].to_vec();
        let chk_embs_tensor = Tensor::cat(&chk_embs_vec, 0)
            .to_kind(Kind::Half)
            .to_device(device);
        _total_embeddings += chk_embs_tensor.size()[0];

        let mut chk_codes_list: Vec<Tensor> = Vec::new();
        let mut chk_res_list: Vec<Tensor> = Vec::new();

        for emb_batch in chk_embs_tensor.split(encode::EMB_BATCH_SIZE, 0) {
            let code_batch = encode::compress_into_codes(&emb_batch, &codec.centroids);
            chk_codes_list.push(code_batch.shallow_clone());

            let mut recon_centroids_batches: Vec<Tensor> = Vec::new();
            for sub_code_batch in code_batch.split(encode::CODE_BATCH_SIZE, 0) {
                recon_centroids_batches.push(codec.centroids.index_select(0, &sub_code_batch));
            }
            let recon_centroids = Tensor::cat(&recon_centroids_batches, 0);

            let mut res_batch = &emb_batch - &recon_centroids;

            let bucket_cutoffs = codec.bucket_cutoffs.as_ref().unwrap().contiguous();
            res_batch = Tensor::bucketize(&res_batch, &bucket_cutoffs, true, false);

            let mut res_shape = res_batch.size();
            res_shape.push(nbits as i64);
            res_batch = res_batch.unsqueeze(-1).expand(&res_shape, false);
            res_batch = res_batch.bitwise_right_shift(&codec.arange_bits);
            let ones = Tensor::ones_like(&res_batch).to_device(device);
            res_batch = res_batch.bitwise_and_tensor(&ones);

            let res_flat = res_batch.flatten(0, -1);

            let res_packed = encode::packbits(&res_flat);

            let shape = [res_batch.size()[0], embedding_dim / 8 * (nbits as i64)];
            chk_res_list.push(res_packed.reshape(&shape));
        }

        let chk_codes = Tensor::cat(&chk_codes_list, 0);
        let chk_residuals = Tensor::cat(&chk_res_list, 0);
        let chunk_num_embeddings = chk_codes.size()[0] as usize;

        let chk_codes_fpath = Path::new(&index_path).join(&format!("{}.codes.npy", chk_idx));
        chk_codes
            .to_device(Device::Cpu)
            .write_npy(&chk_codes_fpath)?;

        let chk_res_fpath = Path::new(&index_path).join(&format!("{}.residuals.npy", chk_idx));
        chk_residuals
            .to_device(Device::Cpu)
            .write_npy(&chk_res_fpath)?;

        let chk_doclens_npy_path = Path::new(&index_path).join(format!("doclens.{}.npy", chk_idx));
        Tensor::from_slice(&chk_doclens).write_npy(chk_doclens_npy_path)?;

        let chk_doclens_path = Path::new(&index_path).join(format!("doclens.{}.json", chk_idx));
        let dl_file = File::create(chk_doclens_path)?;
        let buf_writer = BufWriter::new(dl_file);
        serde_json::to_writer(buf_writer, &chk_doclens)?;

        let chk_meta = json!({
            "passage_offset": chk_offset,
            "num_passages": chk_doclens.len(),
            "num_embeddings": chunk_num_embeddings,
            "embedding_offset": current_emb_offset,
        });
        let chk_meta_fpath = Path::new(&index_path).join(format!("{}.metadata.json", chk_idx));
        let meta_f_w = File::create(chk_meta_fpath)?;
        let buf_writer_meta = BufWriter::new(meta_f_w);
        serde_json::to_writer(buf_writer_meta, &chk_meta)?;

        chunk_stats.push(encode::ChunkStats {
            embedding_offset: current_emb_offset,
            num_embeddings: chunk_num_embeddings,
        });
        current_emb_offset += chunk_num_embeddings;
    }
    Ok(chunk_stats)
}
