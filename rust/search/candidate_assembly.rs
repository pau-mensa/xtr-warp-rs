use rayon::{prelude::*, ThreadPool};

use crate::search::sharded_scorer::{QueryState, ShardOut};

/// Per-query CPU-resident Pass A1a assembly output.
pub(super) struct QueryFlats {
    capacities: Vec<i64>,
    sizes: Vec<i32>,
    passage_ids: Vec<i64>,
    scores: Vec<f32>,
}

/// Batch-level flattened candidate arrays plus per-query slice offsets.
pub(super) struct BatchFlats {
    pub(super) active: Vec<usize>,
    pub(super) capacities: Vec<i64>,
    pub(super) sizes: Vec<i32>,
    pub(super) passage_ids: Vec<i64>,
    pub(super) scores: Vec<f32>,
    pub(super) cell_offsets: Vec<usize>,
    pub(super) cand_offsets: Vec<usize>,
}

/// Pass A1a: per-query CPU assembly from per-shard outputs. Consumes
/// `shard_outs` — every inner `Option<ShardOut>` is moved out.
/// `early_resolved[b] = true` skips queries already resolved upstream
/// (e.g. empty-subset short-circuit).
pub(super) fn assemble_query_flats(
    thread_pool: &ThreadPool,
    num_shards: usize,
    early_resolved: &[bool],
    per_query: &[Option<QueryState>],
    shard_outs: Vec<Vec<Option<ShardOut>>>,
) -> Vec<Option<QueryFlats>> {
    thread_pool.install(|| {
        shard_outs
            .into_par_iter()
            .enumerate()
            .map(|(b, mut shard_outs_b)| -> Option<QueryFlats> {
                if early_resolved[b] {
                    return None;
                }
                let qs = per_query[b].as_ref().unwrap();
                let (nc, shard_cells) = qs.assembly_view();

                let mut capacities = vec![0i64; nc];
                let mut sizes = vec![0i32; nc];
                let mut per_cell_pids: Vec<Vec<i64>> = vec![Vec::new(); nc];
                let mut per_cell_scores: Vec<Vec<f32>> = vec![Vec::new(); nc];

                for si in 0..num_shards {
                    let cells = &shard_cells[si];
                    if cells.is_empty() {
                        continue;
                    }
                    let out = match shard_outs_b[si].take() {
                        Some(out) => out,
                        None => continue,
                    };

                    for (local, &global_cell) in cells.iter().enumerate() {
                        capacities[global_cell] = out.capacities[local];
                        sizes[global_cell] = out.sizes[local];
                        let start = out.offsets[local] as usize;
                        let end = out.offsets[local + 1] as usize;
                        per_cell_pids[global_cell] = out.passage_ids[start..end].to_vec();
                        per_cell_scores[global_cell] = out.scores[start..end].to_vec();
                    }
                }

                let mut passage_ids: Vec<i64> = Vec::new();
                let mut scores: Vec<f32> = Vec::new();
                for i in 0..nc {
                    passage_ids.extend(&per_cell_pids[i]);
                    scores.extend(&per_cell_scores[i]);
                }

                Some(QueryFlats {
                    capacities,
                    sizes,
                    passage_ids,
                    scores,
                })
            })
            .collect()
    })
}

/// Pass A1b: serial concat into master arrays, preserving batch order.
pub(super) fn concat_query_flats(
    query_flats: Vec<Option<QueryFlats>>,
    batch_len: usize,
) -> BatchFlats {
    let mut active: Vec<usize> = Vec::with_capacity(batch_len);
    let mut capacities: Vec<i64> = Vec::new();
    let mut sizes: Vec<i32> = Vec::new();
    let mut passage_ids: Vec<i64> = Vec::new();
    let mut scores: Vec<f32> = Vec::new();
    let mut cell_offsets: Vec<usize> = vec![0];
    let mut cand_offsets: Vec<usize> = vec![0];

    for (b, result) in query_flats.into_iter().enumerate() {
        if let Some(flats) = result {
            capacities.extend_from_slice(&flats.capacities);
            sizes.extend_from_slice(&flats.sizes);
            passage_ids.extend(flats.passage_ids);
            scores.extend(flats.scores);
            cell_offsets.push(capacities.len());
            cand_offsets.push(passage_ids.len());
            active.push(b);
        }
    }

    BatchFlats {
        active,
        capacities,
        sizes,
        passage_ids,
        scores,
        cell_offsets,
        cand_offsets,
    }
}
