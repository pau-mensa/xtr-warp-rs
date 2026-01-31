use anyhow::Result;
use tch::Tensor;

/// Represents the source of embeddings
pub trait EmbeddingSource {
    fn num_docs(&self) -> usize;
    fn chunk_iter(&mut self, chunk_size: usize) -> Result<Box<dyn Iterator<Item = DocChunk> + '_>>;
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

    fn chunk_iter(&mut self, chunk_size: usize) -> Result<Box<dyn Iterator<Item = DocChunk> + '_>> {
        let embeddings = &self.embeddings;
        let total = embeddings.len();
        let iter = (0..total).step_by(chunk_size).map(move |offset| {
            let end = (offset + chunk_size).min(total);
            let chunk_embeddings: Vec<Tensor> = embeddings[offset..end]
                .iter()
                .map(|t| t.shallow_clone())
                .collect();
            let doclens: Vec<i64> = chunk_embeddings.iter().map(|e| e.size()[0]).collect();
            DocChunk {
                embeddings: chunk_embeddings,
                doclens,
            }
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
