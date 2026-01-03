use anyhow::{Context, Result};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use regex::Regex;
use serde_json;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use tch::{Device, IndexOp, Kind, Tensor};

use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::IndexConfig;

/// Creates a new WARP index from a collection of document embeddings.
/// Result containing the index metadata on success
pub fn create_index(
    config: &IndexConfig,
    document_embeddings: Vec<Tensor>,
    centroids: Tensor,
    seed: Option<u64>,
) -> Result<()> {
    // Create the index directory if it doesn't exist
    std::fs::create_dir_all(&config.index_path)?;

    let n_docs = document_embeddings.len();
    let num_chunks = (n_docs as f64 / 25_000f64.min(1.0 + n_docs as f64)).ceil() as usize;

    let avg_doc_len = document_embeddings
        .iter()
        .map(|emb| emb.size()[0] as f64)
        .sum::<f64>()
        / n_docs as f64;

    let mut est_total_embs_f64 = (n_docs as f64) * avg_doc_len;
    est_total_embs_f64 = (16.0 * est_total_embs_f64.sqrt()).log2().floor();
    let est_total_embs = 2f64.powf(est_total_embs_f64) as i64;

    // Save plan.json
    let plan_fpath = config.index_path.join("plan.json");
    let plan_data = json!({ "nbits": config.nbits, "num_chunks": num_chunks });
    let mut plan_file = File::create(plan_fpath)?;
    writeln!(plan_file, "{}", serde_json::to_string_pretty(&plan_data)?)?;

    // Sample embeddings for training
    let sample_k_float = 16.0 * (120.0 * n_docs as f64).sqrt();
    let k = (1.0 + sample_k_float).min(n_docs as f64) as usize;
    let mut rng = if let Some(seed_value) = seed {
        Box::new(StdRng::seed_from_u64(seed_value)) as Box<dyn RngCore>
    } else {
        Box::new(rand::rng()) as Box<dyn RngCore>
    };

    let mut passage_indices: Vec<u32> = (0..n_docs as u32).collect();
    passage_indices.shuffle(&mut *rng);
    let sample_pids: Vec<u32> = passage_indices.into_iter().take(k).collect();

    let mut sample_tensors_vec: Vec<&Tensor> = Vec::with_capacity(k);

    for &pid in &sample_pids {
        sample_tensors_vec.push(&document_embeddings[pid as usize]);
    }

    let sampled_embeddings = Tensor::cat(&sample_tensors_vec, 0)
        .to_kind(Kind::Half)
        .to_device(config.device);

    let path_str = config
        .index_path
        .as_path()
        .to_str()
        .expect("index_path is not valid UTF-8");

    // Train residual codec using sampled embeddings
    let codec = train_residual_codec(
        &sampled_embeddings,
        &centroids,
        config.nbits,
        config.embedding_dim,
        config.device,
        &path_str,
    )?;

    // Encode all documents into codes and residuals
    let chunk_stats = compress_into_residuals(
        n_docs,
        num_chunks,
        config.nbits,
        config.embedding_dim as i64,
        &document_embeddings,
        config.device,
        &path_str,
        &codec,
    )?;

    let chk_emb_offsets: Vec<usize> = chunk_stats.iter().map(|s| s.embedding_offset).collect();
    let total_num_embs = chunk_stats
        .last()
        .map(|s| s.embedding_offset + s.num_embeddings)
        .unwrap_or(0);
    // Build inverted file (IVF) structure
    build_ivf(
        &chk_emb_offsets,
        est_total_embs,
        total_num_embs,
        num_chunks,
        &path_str,
        config.device,
    )?;

    // Finalize and create metadata
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

    // Compact the index (reorganize data by centroid)
    compact_index(
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
    nbits: i64,
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
    for code_batch_idxs in heldout_codes.split((1 << 20) as i64, 0) {
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

/// Bit weights for packing 8 bits into a byte
const BIT_WEIGHTS: [i64; 8] = [128, 64, 32, 16, 8, 4, 2, 1];

pub fn packbits(res: &Tensor) -> Tensor {
    let bits_mat = res.reshape(&[-1, 8]);
    let weights = Tensor::from_slice(&BIT_WEIGHTS)
        .to_device(res.device())
        .to_kind(Kind::Float);
    let packed = bits_mat
        .to_kind(Kind::Float)
        //.to_device(res.device())
        .matmul(&weights)
        .to_kind(Kind::Uint8);
    packed
}

pub struct ChunkStats {
    pub embedding_offset: usize,
    pub num_embeddings: usize,
}

pub fn compress_into_residuals(
    n_docs: usize,
    n_chunks: usize,
    nbits: i64,
    embedding_dim: i64,
    document_embeddings: &Vec<Tensor>,
    device: Device,
    index_path: &str,
    codec: &ResidualCodec,
) -> Result<Vec<ChunkStats>> {
    const CHUNK_SIZE: usize = 25_000;
    let proc_chunk_sz = CHUNK_SIZE.min(1 + n_docs);
    let mut chunk_stats = Vec::with_capacity(n_chunks);
    let mut current_emb_offset: usize = 0;

    for chk_idx in 0..n_chunks {
        let chk_offset = chk_idx * proc_chunk_sz;
        let chk_end_offset = (chk_offset + proc_chunk_sz).min(n_docs);

        let chk_embs_vec: Vec<Tensor> = document_embeddings[chk_offset..chk_end_offset]
            .iter()
            .map(|t| t.to_kind(Kind::Half))
            .collect();
        let chk_doclens: Vec<i64> = chk_embs_vec.iter().map(|e| e.size()[0]).collect();
        let chk_embs_tensor = Tensor::cat(&chk_embs_vec, 0)
            .to_kind(Kind::Half)
            .to_device(device);

        let mut chk_codes_list: Vec<Tensor> = Vec::new();
        let mut chk_res_list: Vec<Tensor> = Vec::new();

        for emb_batch in chk_embs_tensor.split(1 << 18, 0) {
            let code_batch = compress_into_codes(&emb_batch, &codec.centroids);
            chk_codes_list.push(code_batch.shallow_clone());

            let mut recon_centroids_batches: Vec<Tensor> = Vec::new();
            for sub_code_batch in code_batch.split(1 << 20, 0) {
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

            let res_packed = packbits(&res_flat);

            let shape = [res_batch.size()[0], embedding_dim / 8 * nbits];
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

        let chk_doclens_fpath = Path::new(&index_path).join(format!("doclens.{}.json", chk_idx));
        let dl_file = File::create(chk_doclens_fpath)?;
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

        chunk_stats.push(ChunkStats {
            embedding_offset: current_emb_offset,
            num_embeddings: chunk_num_embeddings,
        });
        current_emb_offset += chunk_num_embeddings;
    }
    Ok(chunk_stats)
}

pub fn optimize_ivf(
    ivf: &Tensor,
    ivf_lens: &Tensor,
    idx_path: &str,
    device: Device,
) -> Result<(Tensor, Tensor)> {
    let mut doclen_files: BTreeMap<i64, String> = BTreeMap::new();
    let doclen_re =
        Regex::new(r"doclens\.(\d+)\.json").context("Failed to compile regex for doclens files")?;

    for dir_entry_res in
        fs::read_dir(idx_path).with_context(|| format!("Failed to read directory: {}", idx_path))?
    {
        let dir_entry =
            dir_entry_res.with_context(|| format!("Failed to read entry in {}", idx_path))?;
        let fname = dir_entry.file_name();
        if let Some(fname_str) = fname.to_str() {
            if let Some(caps) = doclen_re.captures(fname_str) {
                if let Some(id_cap) = caps.get(1) {
                    let id = id_cap
                        .as_str()
                        .parse::<i64>()
                        .with_context(|| format!("Failed to parse chunk ID from {}", fname_str))?;
                    doclen_files.insert(id, dir_entry.path().to_str().unwrap().to_string());
                }
            }
        }
    }

    let mut all_doclens: Vec<i64> = Vec::new();
    for (_id, fpath) in doclen_files {
        let file = fs::File::open(&fpath)
            .with_context(|| format!("Failed to open doclens file: {}", fpath))?;
        let reader = BufReader::new(file);
        let chunk_doclens: Vec<i64> = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse JSON from {}", fpath))?;
        all_doclens.extend(chunk_doclens);
    }

    let total_embs: i64 = all_doclens.iter().sum();

    let mut emb_to_pid_vec: Vec<i64> = Vec::with_capacity(total_embs as usize);
    let mut pid_counter: i64 = 0;
    for &doc_len in &all_doclens {
        for _ in 0..doc_len {
            emb_to_pid_vec.push(pid_counter);
        }
        pid_counter += 1;
    }

    let emb_to_pid = Tensor::from_slice(&emb_to_pid_vec)
        .to_device(device)
        .to_kind(Kind::Int64);

    let pids_in_ivf = emb_to_pid.index_select(0, ivf);
    let mut unique_pids_list: Vec<Tensor> = Vec::new();
    let mut new_ivf_lens_vec: Vec<i64> = Vec::new();
    let ivf_lens_vec: Vec<i64> = Vec::<i64>::try_from(ivf_lens)?;
    let mut ivf_offset: i64 = 0;

    for &len in &ivf_lens_vec {
        let pids_seg = pids_in_ivf.narrow(0, ivf_offset, len);
        let (unique_pids, _, _) = pids_seg.unique_dim(0, true, false, false);
        unique_pids_list.push(unique_pids.copy());
        new_ivf_lens_vec.push(unique_pids.size1().unwrap_or(0));
        ivf_offset += len;
    }

    let pids_in_ivf = Tensor::cat(&unique_pids_list, 0);
    let new_ivf_lens = Tensor::from_slice(&new_ivf_lens_vec)
        .to_device(device)
        .to_kind(Kind::Int64);

    Ok((pids_in_ivf, new_ivf_lens))
}

/// Builds the inverted file (IVF) structure from codes.
/// # Returns
/// Tuple of (ivf_data, ivf_lengths)
fn build_ivf(
    chk_emb_offsets: &Vec<usize>,
    est_total_embs: i64,
    total_num_embs: usize,
    n_chunks: usize,
    index_path: &str,
    device: Device,
) -> Result<()> {
    let all_codes = Tensor::zeros(&[total_num_embs as i64], (Kind::Int64, device));

    for chk_idx in 0..n_chunks {
        let chk_offset_global = chk_emb_offsets[chk_idx];
        let codes_fpath_for_global = Path::new(index_path).join(format!("{}.codes.npy", chk_idx));
        let codes_from_file = Tensor::read_npy(&codes_fpath_for_global)?.to_device(device);
        let codes_in_chk_count = codes_from_file.size()[0];
        all_codes
            .narrow(0, chk_offset_global as i64, codes_in_chk_count)
            .copy_(&codes_from_file);
    }
    let (sorted_codes, sorted_indices) = all_codes.sort(0, false);
    let code_counts = sorted_codes.bincount::<Tensor>(None, est_total_embs);

    let (opt_ivf, opt_ivf_lens) = optimize_ivf(&sorted_indices, &code_counts, index_path, device)
        .context("Failed to optimize IVF")?;

    let opt_ivf_fpath = Path::new(index_path).join("ivf.npy");
    opt_ivf
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .write_npy(&opt_ivf_fpath)?;
    let opt_ivf_lens_fpath = Path::new(index_path).join("ivf_lengths.npy");
    opt_ivf_lens
        .to_device(Device::Cpu)
        .write_npy(&opt_ivf_lens_fpath)?;
    Ok(())
}

/// Compacts the index by reorganizing embeddings to be contiguous per centroid
/// Creates sizes_compacted.npy and codes_compacted.npy files
pub fn compact_index(
    index_path: &Path,
    num_chunks: usize,
    num_centroids: usize,
    embedding_dim: usize,
    nbits: usize,
    device: Device,
) -> Result<()> {
    // First pass: count embeddings per centroid
    let mut centroid_sizes = vec![0i64; num_centroids];
    let mut total_embeddings = 0i64;

    for chunk_idx in 0..num_chunks {
        let codes_path = index_path.join(format!("{}.codes.npy", chunk_idx));
        let codes = Tensor::read_npy(&codes_path)?;

        // Count embeddings per centroid
        for i in 0..codes.size()[0] {
            let centroid_id = codes.i(i).int64_value(&[]) as usize;
            centroid_sizes[centroid_id] += 1;
            total_embeddings += 1;
        }
    }

    // Create sizes_compacted tensor
    let sizes_compacted = Tensor::from_slice(&centroid_sizes).to_device(device);
    let sizes_path = index_path.join("sizes.compacted.npy");
    sizes_compacted
        .to_device(Device::Cpu)
        .write_npy(&sizes_path)?;

    // Calculate offsets for each centroid
    let mut offsets = vec![0i64; num_centroids + 1];
    for i in 0..num_centroids {
        offsets[i + 1] = offsets[i] + centroid_sizes[i];
    }

    // Create storage for compacted data
    let residual_dim = (embedding_dim * nbits) / 8;
    let compacted_residuals = Tensor::zeros(
        &[total_embeddings, residual_dim as i64],
        (Kind::Uint8, device),
    );
    let compacted_codes = Tensor::zeros(&[total_embeddings], (Kind::Int, device));

    // Track current write position for each centroid
    let mut centroid_positions = offsets[0..num_centroids].to_vec();

    // Second pass: reorganize data by centroid
    let mut passage_offset = 0i32;

    for chunk_idx in 0..num_chunks {
        // Load chunk data
        let codes_path = index_path.join(format!("{}.codes.npy", chunk_idx));
        let residuals_path = index_path.join(format!("{}.residuals.npy", chunk_idx));
        let doclens_path = index_path.join(format!("doclens.{}.json", chunk_idx));

        let codes = Tensor::read_npy(&codes_path)?;
        let residuals = Tensor::read_npy(&residuals_path)?;

        // Load document lengths
        let doclens_file = File::open(&doclens_path)?;
        let doclens: Vec<i64> = serde_json::from_reader(BufReader::new(doclens_file))?;

        // Create passage IDs for this chunk
        let mut passage_ids = Vec::new();
        for (doc_idx, &doc_len) in doclens.iter().enumerate() {
            for _ in 0..doc_len {
                passage_ids.push(passage_offset + doc_idx as i32);
            }
        }
        let passage_ids_tensor = Tensor::from_slice(&passage_ids).to_device(device);

        // Place each embedding in its centroid's section
        for emb_idx in 0..codes.size()[0] {
            let centroid_id = codes.i(emb_idx).int64_value(&[]) as usize;
            let write_pos = centroid_positions[centroid_id];

            // Write residual
            let residual = residuals.i(emb_idx);
            compacted_residuals.i(write_pos).copy_(&residual);

            // Write passage ID
            let pid = passage_ids_tensor.i(emb_idx);
            compacted_codes.i(write_pos).copy_(&pid);

            centroid_positions[centroid_id] += 1;
        }

        passage_offset += doclens.len() as i32;
    }

    // Save compacted data
    let residuals_compacted_path = index_path.join("residuals.compacted.npy");
    compacted_residuals
        .to_device(Device::Cpu)
        .write_npy(&residuals_compacted_path)?;

    let codes_compacted_path = index_path.join("codes.compacted.npy");
    compacted_codes
        .to_device(Device::Cpu)
        .write_npy(&codes_compacted_path)?;

    // Also save the offsets for quick access during search
    let offsets_tensor = Tensor::from_slice(&offsets).to_device(device);
    let offsets_path = index_path.join("offsets.compacted.npy");
    offsets_tensor
        .to_device(Device::Cpu)
        .write_npy(&offsets_path)?;

    Ok(())
}
