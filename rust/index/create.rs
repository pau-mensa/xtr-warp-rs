use anyhow::{bail, Result};
use chrono::Utc;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};
use serde::Deserialize;
use serde_json::json;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;
use tch::{Device, Kind, Tensor};

use super::{compact, source::EmbeddingSource};
use crate::index::encode::{compress_into_codes, encode_chunks, EncodeResult, CODE_BATCH_SIZE};
use crate::utils::residual_codec::ResidualCodec;
use crate::utils::types::{IndexConfig, IndexMetadata, IndexPlan};

/// Creates a new WARP index from a collection of document embeddings.
/// Result containing the index metadata on success
pub fn create_index(
    config: &IndexConfig,
    embeddings_source: &mut dyn EmbeddingSource,
    centroids: Tensor,
    seed: Option<u64>,
    resume: bool,
    show_progress: bool,
) -> Result<()> {
    // Create the index directory if it doesn't exist
    std::fs::create_dir_all(&config.index_path)?;

    let path_str = config
        .index_path
        .as_path()
        .to_str()
        .expect("index_path is not valid UTF-8");
    let resumed = if resume {
        load_resume_plan_and_codec(config, embeddings_source, &centroids)?
    } else {
        None
    };
    let (index_plan, codec) = if let Some(resumed) = resumed {
        if show_progress {
            eprintln!(
                "Loaded residual codec and index plan from checkpoint; skipping planning sample"
            );
        }
        resumed
    } else {
        let (index_plan, sample_pids, sampled_embeddings) =
            plan_and_sample(config, embeddings_source, seed)?;
        if show_progress {
            eprintln!(
                "Index plan: {} documents, {:.0} token embeddings estimated, {} chunks; residual-codec sample: {} tokens",
                index_plan.n_docs,
                index_plan.avg_doc_len * index_plan.n_docs as f64,
                index_plan.num_chunks,
                sampled_embeddings.size()[0]
            );
        }

        let plan_fpath = config.index_path.join("plan.json");
        let plan_data = json!({
            "nbits": index_plan.nbits,
            "num_chunks": index_plan.num_chunks,
            "n_docs": index_plan.n_docs,
            "avg_doc_len": index_plan.avg_doc_len,
            "est_total_embs": index_plan.est_total_embs,
            "embedding_dim": config.embedding_dim,
            "indexing_chunk_size": config.indexing_chunk_size,
            "compression_batch_size": config.compression_batch_size,
            "codec_sample_max_tokens": config.codec_sample_max_tokens,
        });
        let mut plan_file = File::create(plan_fpath)?;
        writeln!(plan_file, "{}", serde_json::to_string_pretty(&plan_data)?)?;

        let pids_fpath = Path::new(&path_str).join("pids.npy");
        Tensor::from_slice(&sample_pids).write_npy(&pids_fpath)?;

        if show_progress {
            eprintln!("Training residual codec");
        }
        let codec = train_residual_codec(
            &sampled_embeddings,
            &centroids,
            config.nbits,
            config.embedding_dim,
            config.device,
            &path_str,
        )?;
        (index_plan, codec)
    };

    if show_progress {
        eprintln!(
            "Encoding {} chunks ({} documents/chunk, {} tokens/compression batch)",
            index_plan.num_chunks, config.indexing_chunk_size, config.compression_batch_size
        );
    }
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
        config.indexing_chunk_size,
        config.compression_batch_size,
        resume,
        show_progress,
    )?;

    if show_progress {
        eprintln!("Encoding complete; compacting index with bounded disk partitions");
    }
    finalize_and_compact(
        config,
        &index_plan,
        &encode_result,
        &centroids,
        show_progress,
    )?;

    if show_progress {
        eprintln!("Index build complete");
    }

    Ok(())
}

#[derive(Deserialize)]
struct StoredPlan {
    nbits: u8,
    num_chunks: usize,
    n_docs: usize,
    avg_doc_len: f64,
    est_total_embs: i64,
    embedding_dim: u32,
    indexing_chunk_size: usize,
    compression_batch_size: i64,
    codec_sample_max_tokens: usize,
}

fn load_resume_plan_and_codec(
    config: &IndexConfig,
    source: &dyn EmbeddingSource,
    centroids: &Tensor,
) -> Result<Option<(IndexPlan, ResidualCodec)>> {
    let required = [
        config.index_path.join("plan.json"),
        config.index_path.join("pids.npy"),
        config.index_path.join("avg_residual.npy"),
        config.index_path.join("bucket_cutoffs.npy"),
        config.index_path.join("bucket_weights.npy"),
    ];
    if required.iter().any(|path| !path.is_file()) {
        return Ok(None);
    }

    let stored: StoredPlan = serde_json::from_reader(BufReader::new(File::open(&required[0])?))?;
    anyhow::ensure!(
        stored.n_docs == source.num_docs(),
        "resume document count changed"
    );
    anyhow::ensure!(stored.nbits == config.nbits, "resume nbits changed");
    anyhow::ensure!(
        stored.embedding_dim == config.embedding_dim,
        "resume embedding dimension changed"
    );
    anyhow::ensure!(
        stored.indexing_chunk_size == config.indexing_chunk_size,
        "resume indexing_chunk_size changed"
    );
    anyhow::ensure!(
        stored.compression_batch_size == config.compression_batch_size,
        "resume compression_batch_size changed"
    );
    anyhow::ensure!(
        stored.codec_sample_max_tokens == config.codec_sample_max_tokens,
        "resume codec_sample_max_tokens changed"
    );

    let avg_residual = Tensor::read_npy(&required[2])?.to_device(config.device);
    let bucket_cutoffs = Tensor::read_npy(&required[3])?.to_device(config.device);
    let bucket_weights = Tensor::read_npy(&required[4])?.to_device(config.device);
    let codec = ResidualCodec::load(
        config.nbits,
        centroids.to_kind(Kind::Half),
        avg_residual,
        Some(bucket_cutoffs),
        Some(bucket_weights),
        config.device,
    )?;
    Ok(Some((
        IndexPlan {
            n_docs: stored.n_docs,
            num_chunks: stored.num_chunks,
            avg_doc_len: stored.avg_doc_len,
            est_total_embs: stored.est_total_embs,
            nbits: stored.nbits,
        },
        codec,
    )))
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
    let num_chunks = (n_docs as f64 / (config.indexing_chunk_size as f64).min(1.0 + n_docs as f64))
        .ceil() as usize;

    let mut rng = if let Some(seed_value) = seed {
        Box::new(StdRng::seed_from_u64(seed_value)) as Box<dyn RngCore>
    } else {
        Box::new(rand::rng()) as Box<dyn RngCore>
    };
    let (sample_pids, sampled_embeddings, total_doc_len) = sample_embeddings_bounded(
        source,
        &mut *rng,
        config.device,
        config.indexing_chunk_size,
        config.codec_sample_max_tokens,
    )?;

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

fn sample_embeddings_bounded(
    source: &mut dyn EmbeddingSource,
    rng: &mut dyn RngCore,
    device: Device,
    chunk_size: usize,
    max_tokens: usize,
) -> Result<(Vec<i64>, Tensor, i64)> {
    anyhow::ensure!(max_tokens > 0, "codec_sample_max_tokens must be positive");
    // Keep documents with the smallest random priorities. Removing the largest
    // priorities whenever the token budget is exceeded produces a deterministic,
    // corpus-wide sample without ever retaining the heuristic's tens of thousands
    // of long documents at once.
    let mut priorities: BinaryHeap<(u64, i64)> = BinaryHeap::new();
    let mut samples: HashMap<i64, Tensor> = HashMap::new();
    let mut sampled_tokens = 0usize;
    let mut total_doc_len: i64 = 0;
    let mut doc_offset: i64 = 0;

    let chunk_iter = source.chunk_iter(chunk_size)?;
    for chunk in chunk_iter {
        let chunk = chunk?;
        total_doc_len += chunk.doclens.iter().sum::<i64>();
        for doc in &chunk.embeddings {
            let doc_tokens = doc.size()[0] as usize;
            if doc_tokens <= max_tokens {
                let priority = rng.next_u64();
                // Once the budget is full, a document that would be the first
                // eviction candidate is dropped without copying it. The result
                // is identical to inserting and immediately evicting it.
                let evicted_immediately = sampled_tokens + doc_tokens > max_tokens
                    && priorities
                        .peek()
                        .is_some_and(|top| (priority, doc_offset) >= *top);
                if !evicted_immediately {
                    priorities.push((priority, doc_offset));
                    samples.insert(doc_offset, doc.copy());
                    sampled_tokens += doc_tokens;
                    while sampled_tokens > max_tokens {
                        let (_, removed_pid) = priorities.pop().expect("sample heap is not empty");
                        if let Some(removed) = samples.remove(&removed_pid) {
                            sampled_tokens -= removed.size()[0] as usize;
                        }
                    }
                }
            }
            doc_offset += 1;
        }
    }

    anyhow::ensure!(
        !samples.is_empty(),
        "No document fits within codec_sample_max_tokens={}",
        max_tokens
    );
    let mut sample_entries: Vec<(i64, Tensor)> = samples.into_iter().collect();
    sample_entries.sort_unstable_by_key(|(pid, _)| *pid);
    let sample_pids: Vec<i64> = sample_entries.iter().map(|(pid, _)| *pid).collect();
    let sample_refs: Vec<&Tensor> = sample_entries.iter().map(|(_, tensor)| tensor).collect();
    let sampled_embeddings = Tensor::cat(&sample_refs, 0)
        .to_kind(Kind::Half)
        .to_device(device);
    Ok((sample_pids, sampled_embeddings, total_doc_len))
}

fn finalize_and_compact(
    config: &IndexConfig,
    plan: &IndexPlan,
    encode_result: &EncodeResult,
    centroids: &Tensor,
    show_progress: bool,
) -> Result<()> {
    let final_avg_doclen = if plan.n_docs > 0 {
        encode_result.total_embeddings as f64 / plan.n_docs as f64
    } else {
        0.0
    };

    let meta = IndexMetadata {
        num_chunks: plan.num_chunks,
        nbits: plan.nbits,
        // Search hyperparameter tuning reads `num_partitions`, so it must
        // describe the codebook actually written rather than the automatic
        // corpus-size heuristic (which may differ when K is overridden).
        num_partitions: centroids.size()[0],
        num_embeddings: encode_result.total_embeddings,
        avg_doclen: final_avg_doclen,
        num_passages: plan.n_docs,
        next_passage_id: plan.n_docs as i64,
        num_centroids: centroids.size()[0] as usize,
        dim: config.embedding_dim as usize,
        created_at: Utc::now().to_rfc3339(),
    };
    meta.save(&config.index_path)?;

    compact::compact_index(
        &config.index_path,
        plan.num_chunks,
        centroids.size()[0] as usize,
        config.embedding_dim as usize,
        plan.nbits as usize,
        config.device,
        &std::collections::HashSet::new(),
        show_progress,
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
