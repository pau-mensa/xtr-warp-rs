use anyhow::{Context, Result};
use regex::Regex;
use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use tch::{Device, Kind, Tensor};

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
        let file = File::open(&fpath)
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
pub fn build_ivf(
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
