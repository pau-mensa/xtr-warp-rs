use anyhow::{anyhow, bail, Result};
use chrono::Utc;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use tch::{Device, Kind, Tensor};

use super::{compact, source::EmbeddingSource};
use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::{IndexConfig, IndexMetadata, IndexPlan};
use crate::index::encode::{encode_chunks, compress_into_codes, EncodeResult, CHUNK_SIZE, CODE_BATCH_SIZE};

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

    let encode_result = encode_chunks(
        &index_plan,
        embeddings_source,
        &centroids,
        &codec,
        &config.index_path,
        config.device,
        config.embedding_dim,
        None, // auto-assign passage IDs 0..N
        0,    // start chunk index
    )?;

    finalize_ivf_and_compact(config, &index_plan, &encode_result, &centroids)?;

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
        (n_docs as f64 / (CHUNK_SIZE as f64).min(1.0 + n_docs as f64)).ceil() as usize;

    let mut rng = if let Some(seed_value) = seed {
        Box::new(StdRng::seed_from_u64(seed_value)) as Box<dyn RngCore>
    } else {
        Box::new(rand::rng()) as Box<dyn RngCore>
    };
    let (total_doc_len, sample_pids, sampled_embeddings) = if source.get_doc(0).is_some() {
        let mut total: i64 = 0;
        for idx in 0..n_docs {
            let doc = source
                .get_doc(idx)
                .ok_or_else(|| anyhow!("Missing embedding at index {}", idx))?;
            total += doc.size()[0];
        }
        let (pids, embeddings) =
            sample_embeddings_in_memory(source, n_docs, config.embedding_dim, &mut *rng, config.device)?;
        (total, pids, embeddings)
    } else {
        let (pids, embeddings, total) = sample_embeddings_streaming(
            source,
            n_docs,
            config.embedding_dim,
            &mut *rng,
            config.device,
        )?;
        (total, pids, embeddings)
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

    Ok((index_plan, sample_pids, sampled_embeddings))
}

fn sample_embeddings_in_memory(
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
    Ok((sample_pids, sampled_embeddings))
}

fn sample_embeddings_streaming(
    source: &mut dyn EmbeddingSource,
    n_docs: usize,
    embedding_dim: u32,
    rng: &mut dyn RngCore,
    device: Device,
) -> Result<(Vec<i64>, Tensor, i64)> {
    let sample_k_float = 16.0 * (120.0 * n_docs as f64).sqrt();
    let k = (1.0 + sample_k_float).min(n_docs as f64) as usize;

    let mut sample_tensors: Vec<Tensor> = Vec::with_capacity(k);
    let mut sample_pids: Vec<i64> = Vec::with_capacity(k);
    let mut total_doc_len: i64 = 0;
    let mut seen: i64 = 0;
    let mut doc_offset: i64 = 0;

    let chunk_iter = source.chunk_iter(CHUNK_SIZE)?;
    for chunk in chunk_iter {
        let chunk = chunk?;
        total_doc_len += chunk.doclens.iter().sum::<i64>();
        for doc in &chunk.embeddings {
            if (seen as usize) < k {
                sample_tensors.push(doc.copy());
                sample_pids.push(doc_offset);
            } else {
                let j = (rng.next_u64() % (seen as u64 + 1)) as usize;
                if j < k {
                    sample_tensors[j] = doc.copy();
                    sample_pids[j] = doc_offset;
                }
            }
            seen += 1;
            doc_offset += 1;
        }
    }

    if k == 0 {
        let empty = Tensor::zeros(&[0, embedding_dim as i64], (Kind::Half, device));
        return Ok((Vec::new(), empty, total_doc_len));
    }

    let sample_refs: Vec<&Tensor> = sample_tensors.iter().collect();
    let sampled_embeddings = Tensor::cat(&sample_refs, 0)
        .to_kind(Kind::Half)
        .to_device(device);
    Ok((sample_pids, sampled_embeddings, total_doc_len))
}

fn finalize_ivf_and_compact(
    config: &IndexConfig,
    plan: &IndexPlan,
    encode_result: &EncodeResult,
    centroids: &Tensor,
) -> Result<()> {
    let final_avg_doclen = if plan.n_docs > 0 {
        encode_result.total_embeddings as f64 / plan.n_docs as f64
    } else {
        0.0
    };

    let meta = IndexMetadata {
        num_chunks: plan.num_chunks,
        nbits: plan.nbits,
        num_partitions: plan.est_total_embs,
        num_embeddings: encode_result.total_embeddings,
        avg_doclen: final_avg_doclen,
        num_passages: plan.n_docs,
        next_passage_id: plan.n_docs as i64,
        num_centroids: centroids.size()[0] as usize,
        dim: config.embedding_dim as usize,
        created_at: Utc::now().to_rfc3339(),
    };
    meta.save(&config.index_path)?;

    compact::compact_index_counting_sort(
        &config.index_path,
        plan.num_chunks,
        centroids.size()[0] as usize,
        config.embedding_dim as usize,
        plan.nbits as usize,
        config.device,
        &encode_result.global_centroid_counts,
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

    let heldout_codes = compress_into_codes(&heldout_samples, &initial_codec.centroids);

    let mut recon_embs_vec = Vec::new();
    for code_batch_idxs in heldout_codes.split(CODE_BATCH_SIZE, 0) {
        recon_embs_vec.push(initial_codec.centroids.index_select(0, &code_batch_idxs));
    }
    let heldout_recon_embs = Tensor::cat(&recon_embs_vec, 0);

    let heldout_res_raw = (&heldout_samples - &heldout_recon_embs).to_kind(Kind::Float);

    // Compute cluster_threshold: 75th percentile of residual L2 norms.
    // Used later by centroid expansion to detect outlier embeddings.
    {
        let residual_norms = heldout_res_raw.norm_scalaropt_dim(2, &[1], false);
        let n = residual_norms.size()[0];
        let k = ((0.75 * n as f64).ceil() as i64).max(1).min(n);
        let (threshold, _) = residual_norms.flatten(0, -1).kthvalue(k, 0, false);
        threshold
            .to_device(Device::Cpu)
            .write_npy(Path::new(index_path).join("cluster_threshold.npy"))?;
    }

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
