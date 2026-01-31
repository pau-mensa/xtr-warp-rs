use anyhow::{bail, Result};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tch::{Device, IndexOp, Kind, Tensor};

fn build_ivf_from_compacted(
    codes_compacted: &Tensor,
    sizes_vec: &[i64],
    index_path: &Path,
    device: Device,
) -> Result<()> {
    let mut ivf_list: Vec<Tensor> = Vec::with_capacity(sizes_vec.len());
    let mut ivf_lens_vec: Vec<i64> = Vec::with_capacity(sizes_vec.len());

    let mut offset: i64 = 0;
    for &size in sizes_vec {
        if size == 0 {
            ivf_lens_vec.push(0);
            continue;
        }
        let segment = codes_compacted.narrow(0, offset, size);
        let (unique_pids, _, _) = segment.unique_dim(0, true, false, false);
        ivf_lens_vec.push(unique_pids.size()[0]);
        ivf_list.push(unique_pids);
        offset += size;
    }

    let ivf = if ivf_list.is_empty() {
        Tensor::zeros(&[0], (Kind::Int64, device))
    } else {
        Tensor::cat(&ivf_list, 0)
    };
    let ivf_lens = Tensor::from_slice(&ivf_lens_vec)
        .to_device(device)
        .to_kind(Kind::Int64);

    let ivf_fpath = index_path.join("ivf.npy");
    ivf.to_device(Device::Cpu).write_npy(&ivf_fpath)?;

    let ivf_lens_fpath = index_path.join("ivf_lengths.npy");
    ivf_lens.to_device(Device::Cpu).write_npy(&ivf_lens_fpath)?;

    Ok(())
}

pub fn compact_index_counting_sort(
    index_path: &Path,
    num_chunks: usize,
    num_centroids: usize,
    embedding_dim: usize,
    nbits: usize,
    device: Device,
    global_centroid_counts: &Tensor,
) -> Result<()> {
    let sizes_compacted = global_centroid_counts
        .to_device(device)
        .to_kind(Kind::Int64);
    let sizes_vec: Vec<i64> = sizes_compacted.to_device(Device::Cpu).try_into()?;

    let mut offsets_vec = vec![0i64; num_centroids + 1];
    for i in 0..num_centroids {
        offsets_vec[i + 1] = offsets_vec[i] + sizes_vec[i];
    }

    let total_embeddings = offsets_vec[num_centroids];
    let sizes_path = index_path.join("sizes.compacted.npy");
    sizes_compacted
        .to_device(Device::Cpu)
        .write_npy(&sizes_path)?;

    let residual_dim = (embedding_dim * nbits) / 8;
    let compacted_residuals = Tensor::zeros(
        &[total_embeddings, residual_dim as i64],
        (Kind::Uint8, device),
    );
    let compacted_codes = Tensor::zeros(&[total_embeddings], (Kind::Int64, device));

    let mut write_offsets = offsets_vec[..num_centroids].to_vec();
    let mut passage_offset: i64 = 0;

    for chunk_idx in 0..num_chunks {
        let codes_path = index_path.join(format!("{}.codes.npy", chunk_idx));
        let residuals_path = index_path.join(format!("{}.residuals.npy", chunk_idx));
        let doclens_path = index_path.join(format!("doclens.{}.npy", chunk_idx));

        let codes = Tensor::read_npy(&codes_path)?.to_device(device);
        let residuals = Tensor::read_npy(&residuals_path)?.to_device(device);
        let doclens = Tensor::read_npy(&doclens_path)?
            .to_device(device)
            .to_kind(Kind::Int64);

        let num_passages = doclens.size()[0];
        let pids_base = Tensor::arange_start(
            passage_offset,
            passage_offset + num_passages,
            (Kind::Int64, device),
        );
        let chunk_total_embeddings = codes.size()[0];
        let pids = Tensor::repeat_interleave_self_tensor(
            &pids_base,
            &doclens,
            0,
            Some(chunk_total_embeddings),
        );

        let doclens_sum = doclens.sum(Kind::Int64).int64_value(&[]);
        if doclens_sum != chunk_total_embeddings {
            bail!(
                "doclens sum ({}) does not match embeddings count ({}) for chunk {}",
                doclens_sum,
                chunk_total_embeddings,
                chunk_idx
            );
        }

        let sort_result = codes.sort(0, false);
        let sorted_codes = sort_result.0;
        let sorted_idx = sort_result.1;
        let sorted_residuals = residuals.index_select(0, &sorted_idx);
        let sorted_pids = pids.index_select(0, &sorted_idx);

        let chunk_counts = sorted_codes.bincount::<Tensor>(None, num_centroids as i64);
        let chunk_counts_vec: Vec<i64> = chunk_counts.to_device(Device::Cpu).try_into()?;

        let mut local_offset: i64 = 0;
        for (centroid_id, &count) in chunk_counts_vec.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let write_pos = write_offsets[centroid_id];
            compacted_residuals
                .narrow(0, write_pos, count)
                .copy_(&sorted_residuals.narrow(0, local_offset, count));
            compacted_codes
                .narrow(0, write_pos, count)
                .copy_(&sorted_pids.narrow(0, local_offset, count));
            write_offsets[centroid_id] += count;
            local_offset += count;
        }

        passage_offset += num_passages;
    }

    let residuals_compacted_path = index_path.join("residuals.compacted.npy");
    compacted_residuals
        .to_device(Device::Cpu)
        .write_npy(&residuals_compacted_path)?;

    let codes_compacted_path = index_path.join("codes.compacted.npy");
    compacted_codes
        .to_device(Device::Cpu)
        .write_npy(&codes_compacted_path)?;

    let offsets_tensor = Tensor::from_slice(&offsets_vec).to_device(device);
    let offsets_path = index_path.join("offsets.compacted.npy");
    offsets_tensor
        .to_device(Device::Cpu)
        .write_npy(&offsets_path)?;

    build_ivf_from_compacted(&compacted_codes, &sizes_vec, index_path, device)?;

    Ok(())
}

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
