#include <torch/torch.h>

#include <cstring>
#include <limits>
#include <utility>
#include <vector>

extern thread_local char *torch_last_err;

#define PROTECT(x)             \
  try {                        \
    x                          \
  } catch (const std::exception &e) { \
    torch_last_err = strdup(e.what()); \
  }

namespace {

// Flat reduction: max per (pid, token), then sum with MSE fill.
std::pair<torch::Tensor, torch::Tensor> gpu_merge_impl(
    const torch::Tensor &candidate_sizes,
    const torch::Tensor &candidate_pids,
    const torch::Tensor &candidate_scores,
    const torch::Tensor &mse_estimates,
    int64_t nprobe,
    int64_t k,
    int64_t max_candidates) {
  torch::NoGradGuard no_grad;

  if (candidate_pids.numel() == 0) {
    auto device = candidate_pids.device();
    auto empty_pid = torch::empty({0},
                                  torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto empty_scores = torch::empty({0},
                                     torch::TensorOptions().dtype(torch::kFloat).device(device));
    return {empty_pid, empty_scores};
  }

  const auto device = candidate_pids.device();
  const auto sizes = candidate_sizes.to(device, torch::kInt64, /*non_blocking=*/false, /*copy=*/false);
  const auto pids = candidate_pids.to(device, torch::kInt64, /*non_blocking=*/false, /*copy=*/false);
  const auto scores = candidate_scores.to(device, torch::kFloat, /*non_blocking=*/false, /*copy=*/false);
  auto mse = mse_estimates.to(device, torch::kFloat, /*non_blocking=*/false, /*copy=*/false);

  // Token index per cell, repeated per candidate
  const auto num_cells = sizes.size(0);
  auto token_indices = torch::arange(num_cells, torch::TensorOptions().device(device).dtype(torch::kInt64));
  token_indices = torch::div(token_indices, nprobe, /*rounding_mode=*/"trunc");
  auto candidate_tokens = torch::repeat_interleave(token_indices, sizes, /*dim=*/0);
  const auto num_tokens = (num_cells + nprobe - 1) / nprobe;

  // Flatten token+pid into a combined id to compactly deduplicate
  auto combined_ids = pids * num_tokens + candidate_tokens;
  auto sort_result = combined_ids.sort(0);
  auto sorted_ids = std::get<0>(sort_result);
  auto sort_idx = std::get<1>(sort_result);
  auto sorted_scores = scores.index({sort_idx});

  // Unique ids and inverse for max reduction
  auto unique_result = torch::unique_consecutive(sorted_ids, /*return_inverse=*/true, /*return_counts=*/false);
  auto unique_ids = std::get<0>(unique_result);
  auto inverse = std::get<1>(unique_result);
  auto max_init = torch::full(unique_ids.sizes(),
                              -std::numeric_limits<float>::infinity(),
                              sorted_scores.options());
  auto max_per_id = torch::index_reduce(max_init, 0, inverse, sorted_scores, "amax", /*include_self=*/true);

  // Split combined id back into pid and token
  auto pid = torch::div(unique_ids, num_tokens, /*rounding_mode=*/"trunc");
  auto token = unique_ids - pid * num_tokens;
  token = token.to(torch::kInt64);

  // Prepare MSE vector (pad to num_tokens if needed)
  if (mse.size(0) < num_tokens) {
    auto pad = torch::zeros({num_tokens - mse.size(0)}, mse.options());
    mse = torch::cat({mse, pad}, 0);
  }
  mse = mse.narrow(0, 0, num_tokens);

  auto sum_mse = mse.sum();
  auto mse_for_tokens = mse.index({token});
  auto delta = max_per_id - mse_for_tokens;

  // Reduce by pid: since keys were sorted, pid is non-decreasing
  auto pid_result = torch::unique_consecutive(pid,
                                              /*return_inverse=*/true,
                                              /*return_counts=*/true);
  auto unique_pids = std::get<0>(pid_result);
  auto pid_result_inv = std::get<1>(pid_result);
  auto pid_counts = std::get<2>(pid_result).to(torch::kInt64);

  // Deterministic per-PID sum using cumulative sums to avoid nondeterministic atomics.
  // Exclusive prefix to get exact per-pid sums.
  auto delta_cumsum = delta.cumsum(0);
  auto prefix = torch::zeros({delta_cumsum.size(0) + 1}, delta_cumsum.options());
  prefix.index_put_({torch::arange(delta_cumsum.size(0), delta_cumsum.options().dtype(torch::kInt64)) + 1},
                    delta_cumsum);

  auto end_indices = pid_counts.cumsum(0) - 1;
  auto start_indices = end_indices - pid_counts + 1;
  auto sums_at_end = prefix.index({end_indices + 1});
  auto sums_before = prefix.index({start_indices});
  auto deltas_per_pid = sums_at_end - sums_before;
  auto totals = deltas_per_pid + sum_mse;

  if (totals.numel() == 0) {
    auto empty_pid = torch::empty({0},
                                  torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto empty_scores_out = torch::empty({0},
                                         torch::TensorOptions().dtype(torch::kFloat).device(device));
    return {empty_pid, empty_scores_out};
  }

  const auto num_candidates = totals.size(0);
  const auto budget = std::max<int64_t>(k, std::min<int64_t>(max_candidates, num_candidates));
  const auto take_k = std::min<int64_t>(budget, num_candidates);

  auto topk = torch::topk(totals, take_k, 0, /*largest=*/true, /*sorted=*/true);
  auto top_scores = std::get<0>(topk);
  auto top_indices = std::get<1>(topk);
  auto top_pids = unique_pids.index({top_indices});

  return {top_pids, top_scores};
}

}  // namespace

extern "C" void xtr_warp_gpu_merge(
    const torch::Tensor *candidate_sizes,
    const torch::Tensor *candidate_pids,
    const torch::Tensor *candidate_scores,
    const torch::Tensor *mse_estimates,
    int64_t nprobe,
    int64_t k,
    int64_t max_candidates,
    torch::Tensor **out_pids,
    torch::Tensor **out_scores) {
  PROTECT({
    auto result = gpu_merge_impl(*candidate_sizes,
                                 *candidate_pids,
                                 *candidate_scores,
                                 *mse_estimates,
                                 nprobe,
                                 k,
                                 max_candidates);
    *out_pids = new torch::Tensor(std::move(result.first));
    *out_scores = new torch::Tensor(std::move(result.second));
  });
}
