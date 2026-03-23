use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;
use tch::{Device, Kind, Tensor};

/// Add passage IDs to the tombstone file
pub fn delete_from_index(passage_ids: &[i64], index_path: &Path) -> Result<()> {
    let mut tombstones = load_tombstones(index_path)?;
    for &pid in passage_ids {
        tombstones.insert(pid);
    }
    let mut sorted: Vec<i64> = tombstones.into_iter().collect();
    sorted.sort_unstable();
    Tensor::from_slice(&sorted).write_npy(index_path.join("deleted_pids.npy"))?;
    Ok(())
}

/// Load tombstone set from `deleted_pids.npy`, or return empty set if absent.
pub fn load_tombstones(index_path: &Path) -> Result<HashSet<i64>> {
    let path = index_path.join("deleted_pids.npy");
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let pids: Vec<i64> = Tensor::read_npy(&path)?
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .try_into()?;
    Ok(pids.into_iter().collect())
}

/// Save a tombstone set to disk, or remove the file if empty.
pub fn save_tombstones(tombstones: &HashSet<i64>, index_path: &Path) -> Result<()> {
    if tombstones.is_empty() {
        clear_tombstones(index_path)
    } else {
        let mut sorted: Vec<i64> = tombstones.iter().copied().collect();
        sorted.sort_unstable();
        Tensor::from_slice(&sorted).write_npy(index_path.join("deleted_pids.npy"))?;
        Ok(())
    }
}

/// Remove the tombstone file if it exists.
pub fn clear_tombstones(index_path: &Path) -> Result<()> {
    let path = index_path.join("deleted_pids.npy");
    if path.exists() {
        std::fs::remove_file(&path)?;
    }
    Ok(())
}
