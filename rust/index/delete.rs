use anyhow::Result;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Add passage IDs to the tombstone file
pub fn delete_from_index(passage_ids: &[i64], index_path: &Path) -> Result<()> {
    let mut tombstones = load_tombstones(index_path)?;
    for &pid in passage_ids {
        tombstones.insert(pid);
    }
    let mut sorted: Vec<i64> = tombstones.into_iter().collect();
    sorted.sort_unstable();
    write_i64_npy(&sorted, &index_path.join("deleted_pids.npy"))?;
    Ok(())
}

/// Load tombstone set from `deleted_pids.npy`, or return empty set if absent.
pub fn load_tombstones(index_path: &Path) -> Result<HashSet<i64>> {
    let path = index_path.join("deleted_pids.npy");
    if !path.exists() {
        return Ok(HashSet::new());
    }
    let pids = read_i64_npy(&path)?;
    Ok(pids.into_iter().collect())
}

/// Save a tombstone set to disk, or remove the file if empty.
pub fn save_tombstones(tombstones: &HashSet<i64>, index_path: &Path) -> Result<()> {
    if tombstones.is_empty() {
        clear_tombstones(index_path)
    } else {
        let mut sorted: Vec<i64> = tombstones.iter().copied().collect();
        sorted.sort_unstable();
        write_i64_npy(&sorted, &index_path.join("deleted_pids.npy"))
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

/// Write a 1-D i64 array as a NumPy `.npy` file
fn write_i64_npy(data: &[i64], path: &Path) -> Result<()> {
    let header = format!(
        "{{'descr': '<i8', 'fortran_order': False, 'shape': ({},), }}",
        data.len()
    );
    let prefix_len = 10usize; // magic(6) + version(2) + header_len(2)
    let total = prefix_len + header.len() + 1; // +1 for newline
    let padding = ((total + 63) / 64) * 64 - total;
    let padded_header = format!("{}{}\n", header, " ".repeat(padding));

    let mut file = File::create(path)?;
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?;
    file.write_all(&(padded_header.len() as u16).to_le_bytes())?;
    file.write_all(padded_header.as_bytes())?;

    for &val in data {
        file.write_all(&val.to_le_bytes())?;
    }
    Ok(())
}

/// Read a 1-D i64 `.npy` file into a `Vec<i64>`
fn read_i64_npy(path: &Path) -> Result<Vec<i64>> {
    let data = std::fs::read(path)?;
    anyhow::ensure!(
        data.len() >= 10 && &data[..6] == b"\x93NUMPY",
        "Not a valid NPY file: {:?}",
        path
    );

    let major = data[6];
    let header_len: usize;
    let header_start: usize;

    if major == 1 {
        header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
        header_start = 10;
    } else if major == 2 {
        anyhow::ensure!(data.len() >= 12, "NPY v2 header too short");
        header_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        header_start = 12;
    } else {
        anyhow::bail!("Unsupported NPY version {}", major);
    }

    let data_offset = header_start + header_len;
    let bytes = &data[data_offset..];
    let num_elements = bytes.len() / 8;
    let mut result = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let start = i * 8;
        let val = i64::from_le_bytes([
            bytes[start],
            bytes[start + 1],
            bytes[start + 2],
            bytes[start + 3],
            bytes[start + 4],
            bytes[start + 5],
            bytes[start + 6],
            bytes[start + 7],
        ]);
        result.push(val);
    }
    Ok(result)
}
