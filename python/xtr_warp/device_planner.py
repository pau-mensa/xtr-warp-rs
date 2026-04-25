"""VRAM-aware shard ratio planning for sharded index loading.

:func:`compute_device_ratios` fills accelerator VRAM first and gives any
remainder to ``cpu``. :func:`warn_on_vram_overflow` is a best-effort sanity
check called from :meth:`xtr_warp.search.XTRWarp._load_sharded` before the
Rust loader runs.
"""
from __future__ import annotations

import logging

import torch

from . import xtr_warp_rs

logger = logging.getLogger(__name__)

# 50 MB reserve for matmul scratch + allocator fragmentation.
_ACCEL_RESERVE_BYTES = 50 * 1024 * 1024


def compute_device_ratios(devices: list[str], index: str) -> dict[str, float]:
    """Suggest a ``device → ratio`` map for sharded loading.

    Strategy: assign each accelerator a ratio capped by free VRAM (less the
    centroid + bucket-weight overhead and a small allocator reserve). If any
    work is left over, give it to ``cpu``.
    """
    try:
        mem_est = xtr_warp_rs.estimate_index_memory(index)
    except Exception as e:
        logger.warning(
            "Could not estimate index memory: %s — using equal ratios", e
        )
        n = len(devices)
        return {d: 1.0 / n for d in devices}

    shardable = mem_est.get("pids", 0) + mem_est.get("residuals", 0)
    if shardable == 0:
        n = len(devices)
        return {d: 1.0 / n for d in devices}

    accel_overhead = (
        mem_est.get("centroids", 0)
        + mem_est.get("bucket_weights", 0)
        + _ACCEL_RESERVE_BYTES
    )

    accelerators = [d for d in devices if d != "cpu"]
    has_cpu = "cpu" in devices

    ratios: dict[str, float] = {}
    remaining = 1.0

    for i, dev in enumerate(accelerators):
        if remaining <= 0:
            break
        try:
            dev_idx = int(dev.split(":")[-1]) if ":" in dev else 0
            free, _ = torch.cuda.mem_get_info(dev_idx)
        except Exception:
            continue

        usable = max(0, free - (accel_overhead if i == 0 else 0))
        ratio = min(usable / shardable, remaining)
        ratios[dev] = ratio
        remaining -= ratio

    if remaining > 0 and has_cpu:
        ratios["cpu"] = remaining
    elif remaining > 0 and not has_cpu:
        # Distribute remainder proportionally across existing devices.
        if ratios:
            assigned = sum(ratios.values())
            if assigned > 0:
                for dev in ratios:
                    ratios[dev] /= assigned
        else:
            # Fallback: equal split.
            for dev in devices:
                ratios[dev] = 1.0 / len(devices)

    return ratios


def warn_on_vram_overflow(ratios: dict[str, float], index: str) -> None:
    """Best-effort log warning if a CUDA shard's expected size exceeds free VRAM."""
    try:
        mem_est = xtr_warp_rs.estimate_index_memory(index)
        shardable = mem_est.get("pids", 0) + mem_est.get("residuals", 0)
        for dev, ratio in ratios.items():
            if dev.startswith("cuda") and torch.cuda.is_available():
                dev_idx = int(dev.split(":")[-1]) if ":" in dev else 0
                free, _ = torch.cuda.mem_get_info(dev_idx)
                needed = ratio * shardable
                overhead = mem_est.get("centroids", 0) + mem_est.get(
                    "bucket_weights", 0
                )
                if needed + overhead > free:
                    logger.warning(
                        "Device %s: estimated shard size %.0f MB + overhead %.0f MB "
                        "exceeds free VRAM %.0f MB — may OOM",
                        dev,
                        needed / 1e6,
                        overhead / 1e6,
                        free / 1e6,
                    )
    except Exception:
        pass  # best-effort check
