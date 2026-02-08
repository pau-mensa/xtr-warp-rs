use anyhow::{bail, Context, Result};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};

/// Represents the source of embeddings
pub trait EmbeddingSource {
    fn num_docs(&self) -> usize;
    fn chunk_iter(
        &mut self,
        chunk_size: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<DocChunk>> + '_>>;
    fn get_doc(&self, _idx: usize) -> Option<&Tensor> {
        None
    }
}

pub struct DocChunk {
    pub embeddings: Vec<Tensor>, // per-doc tensors in the chunk
    pub doclens: Vec<i64>,
}

pub struct InMemoryEmbeddingSource {
    embeddings: Vec<Tensor>,
}

impl InMemoryEmbeddingSource {
    pub fn new(embeddings: Vec<Tensor>) -> Self {
        Self { embeddings }
    }

    #[allow(dead_code)]
    pub fn embeddings(&self) -> &[Tensor] {
        &self.embeddings
    }
}

impl EmbeddingSource for InMemoryEmbeddingSource {
    fn num_docs(&self) -> usize {
        self.embeddings.len()
    }

    fn chunk_iter(
        &mut self,
        chunk_size: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<DocChunk>> + '_>> {
        let embeddings = &self.embeddings;
        let total = embeddings.len();
        let iter = (0..total).step_by(chunk_size).map(move |offset| {
            let end = (offset + chunk_size).min(total);
            let chunk_embeddings: Vec<Tensor> = embeddings[offset..end]
                .iter()
                .map(|t| t.shallow_clone())
                .collect();
            let doclens: Vec<i64> = chunk_embeddings.iter().map(|e| e.size()[0]).collect();
            Ok(DocChunk {
                embeddings: chunk_embeddings,
                doclens,
            })
        });
        Ok(Box::new(iter))
    }

    fn get_doc(&self, idx: usize) -> Option<&Tensor> {
        self.embeddings.get(idx)
    }
}

impl From<Vec<Tensor>> for InMemoryEmbeddingSource {
    fn from(embeddings: Vec<Tensor>) -> Self {
        Self::new(embeddings)
    }
}

pub struct DiskEmbeddingSource {
    files: Vec<PathBuf>,
    num_docs: usize,
}

impl DiskEmbeddingSource {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let files = list_embedding_files(path)?;
        let mut num_docs = 0usize;
        for file in &files {
            num_docs += count_docs_for_file(file)?;
        }
        Ok(Self { files, num_docs })
    }
}

impl EmbeddingSource for DiskEmbeddingSource {
    fn num_docs(&self) -> usize {
        self.num_docs
    }

    fn chunk_iter(
        &mut self,
        chunk_size: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<DocChunk>> + '_>> {
        let files = self.files.clone();
        Ok(Box::new(DiskChunkIter::new(files, chunk_size)))
    }
}

struct DiskChunkIter {
    files: Vec<PathBuf>,
    file_idx: usize,
    chunk_size: usize,
    current_docs: VecDeque<Tensor>,
    current_doclens: VecDeque<i64>,
    failed: bool,
}

impl DiskChunkIter {
    fn new(files: Vec<PathBuf>, chunk_size: usize) -> Self {
        Self {
            files,
            file_idx: 0,
            chunk_size,
            current_docs: VecDeque::new(),
            current_doclens: VecDeque::new(),
            failed: false,
        }
    }
}

impl Iterator for DiskChunkIter {
    type Item = Result<DocChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }

        let mut chunk_embeddings = Vec::with_capacity(self.chunk_size);
        let mut chunk_doclens = Vec::with_capacity(self.chunk_size);

        while chunk_embeddings.len() < self.chunk_size {
            if self.current_docs.is_empty() {
                if self.file_idx >= self.files.len() {
                    break;
                }
                let file = self.files[self.file_idx].clone();
                self.file_idx += 1;
                match load_doc_batch(&file) {
                    Ok((docs, doclens)) => {
                        self.current_docs = VecDeque::from(docs);
                        self.current_doclens = VecDeque::from(doclens);
                    },
                    Err(err) => {
                        self.failed = true;
                        return Some(Err(err));
                    },
                }
            }

            if self.current_docs.is_empty() {
                continue;
            }

            let remaining = self.chunk_size - chunk_embeddings.len();
            let take = remaining.min(self.current_docs.len());
            for _ in 0..take {
                if let (Some(doc), Some(doclen)) = (
                    self.current_docs.pop_front(),
                    self.current_doclens.pop_front(),
                ) {
                    chunk_embeddings.push(doc);
                    chunk_doclens.push(doclen);
                } else {
                    break;
                }
            }
        }

        if chunk_embeddings.is_empty() {
            None
        } else {
            Some(Ok(DocChunk {
                embeddings: chunk_embeddings,
                doclens: chunk_doclens,
            }))
        }
    }
}

fn list_embedding_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        if is_embeddings_file(path) {
            return Ok(vec![path.to_path_buf()]);
        }
        bail!("embeddings path is not a .npy file: {}", path.display());
    }

    if !path.is_dir() {
        bail!(
            "embeddings path is not a file or directory: {}",
            path.display()
        );
    }

    let mut files: Vec<PathBuf> = fs::read_dir(path)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|entry| is_embeddings_file(entry))
        .filter(|entry| !is_doclens_file(entry))
        .collect();

    files.sort();

    if files.is_empty() {
        bail!("no embedding .npy files found in {}", path.display());
    }

    Ok(files)
}

fn is_doclens_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.ends_with(".doclens.npy"))
        .unwrap_or(false)
}

fn is_embeddings_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("npy"))
        .unwrap_or(false)
}

fn doclens_sidecar_path(emb_path: &Path) -> Option<PathBuf> {
    let stem = emb_path.file_stem()?.to_str()?;
    let parent = emb_path.parent().unwrap_or_else(|| Path::new(""));
    let npy_path = parent.join(format!("{stem}.doclens.npy"));
    if npy_path.exists() {
        return Some(npy_path);
    }
    None
}

fn count_docs_for_file(path: &Path) -> Result<usize> {
    if let Some(doclens_path) = doclens_sidecar_path(path) {
        let doclens = load_doclens(&doclens_path)?;
        return Ok(doclens.len());
    } else {
        bail!(
            "Embeddings require a doclens sidecar: {} (expected {}.doclens.npy)",
            path.display(),
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("batch")
        )
    }
}

fn load_doclens(path: &Path) -> Result<Vec<i64>> {
    let tensor = Tensor::read_npy(path)
        .with_context(|| format!("failed to read doclens from {}", path.display()))?;
    let tensor = tensor.to_device(Device::Cpu).to_kind(Kind::Int64);
    let doclens: Vec<i64> = tensor.try_into()?;
    Ok(doclens)
}

fn load_doc_batch(path: &Path) -> Result<(Vec<Tensor>, Vec<i64>)> {
    let embeddings = load_embeddings_tensor(path)?;

    let doclens_path = doclens_sidecar_path(path);
    let doclens = doclens_path.as_ref().map(|p| load_doclens(p)).transpose()?;

    match embeddings.dim() {
        2 => load_batch(embeddings, doclens, path),
        other => bail!(
            "unsupported embeddings tensor rank {} in {}",
            other,
            path.display()
        ),
    }
}

fn load_embeddings_tensor(path: &Path) -> Result<Tensor> {
    return Tensor::read_npy(path)
        .with_context(|| format!("failed to read embeddings from {}", path.display()));
}

fn load_batch(
    embeddings: Tensor,
    doclens: Option<Vec<i64>>,
    path: &Path,
) -> Result<(Vec<Tensor>, Vec<i64>)> {
    let doclens = match doclens {
        Some(doclens) => doclens,
        None => {
            bail!(
                "2D embeddings require a doclens sidecar: {} (expected {}.doclens.npy)",
                path.display(),
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("batch")
            );
        },
    };

    let total_embeddings = embeddings.size()[0];
    let mut offset = 0i64;
    let mut docs = Vec::with_capacity(doclens.len());
    for &doc_len in &doclens {
        let doc = embeddings.narrow(0, offset, doc_len);
        docs.push(doc);
        offset += doc_len;
    }
    if offset != total_embeddings {
        bail!(
            "doclens sum {} does not match embeddings count {} for {}",
            offset,
            total_embeddings,
            path.display()
        );
    }

    Ok((docs, doclens))
}
