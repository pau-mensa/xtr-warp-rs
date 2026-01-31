use anyhow::Result;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tch::{Device, IndexOp, Kind, Tensor};

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
