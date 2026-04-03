#!/usr/bin/env python3
"""
Benchmark GRPO/RLOO/OPO-style group-based advantage computations.

These algorithms use Python batch loops to build id2score dictionaries and
compute per-group statistics. This benchmark shows ~9x speedup potential
with index_add-based vectorization.

Usage:
    python scripts/benchmark_group_advantage.py --batch-size 512 --groups 32 --warmup 10 --iters 30
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean
from time import perf_counter

import numpy as np
import torch


def _make_eos_style_mask(batch_size: int, response_len: int, device: torch.device) -> torch.Tensor:
    lengths = torch.randint(1, response_len + 1, (batch_size,), device=device)
    positions = torch.arange(response_len, device=device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).float()


def _sync_timing(fn, warmup: int, iters: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for idx in range(iters):
            starts[idx].record()
            fn()
            ends[idx].record()
        torch.cuda.synchronize(device)
        return [starts[idx].elapsed_time(ends[idx]) for idx in range(iters)]

    timings_ms = []
    for _ in range(iters):
        start = perf_counter()
        fn()
        timings_ms.append((perf_counter() - start) * 1000.0)
    return timings_ms


def _summarize(name: str, timings_ms: list[float]) -> float:
    sorted_timings = sorted(timings_ms)
    avg = mean(timings_ms)
    p50 = sorted_timings[len(sorted_timings) // 2]
    print(f"{name}: mean={avg:.3f} ms p50={p50:.3f} ms min={sorted_timings[0]:.3f} ms")
    return avg


# =============================================================================
# GRPO-style group advantage (Python loops reference)
# =============================================================================
def grpo_reference(scores: torch.Tensor, index: np.ndarray, norm_adv_by_std_in_grpo: bool = True, epsilon: float = 1e-6):
    """GRPO advantage computation with Python batch loops (reference implementation)."""
    bsz = scores.shape[0]
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    id2mean = {}
    id2std = {}
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0, device=scores.device)
            id2std[idx] = torch.tensor(1.0, device=scores.device)
        elif len(id2score[idx]) > 1:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)

    result = scores.clone()
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            result[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            result[i] = scores[i] - id2mean[index[i]]
    return result


def grpo_vectorized(scores: torch.Tensor, index: np.ndarray, norm_adv_by_std_in_grpo: bool = True, epsilon: float = 1e-6):
    """GRPO advantage computation with index_add vectorization."""
    bsz = scores.shape[0]
    num_groups = max(index) + 1
    index_tensor = torch.tensor(index, device=scores.device, dtype=torch.long)

    # Group accumulators
    group_sums = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_squares = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_counts = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)

    # Efficient per-group accumulation using index_add
    group_sums.index_add_(0, index_tensor, scores)
    group_squares.index_add_(0, index_tensor, scores.square())
    group_counts.index_add_(0, index_tensor, torch.ones(bsz, device=scores.device, dtype=torch.float32))

    # Compute mean and std
    group_means = group_sums / (group_counts + epsilon)
    group_var = group_squares / (group_counts + epsilon) - group_means.square()
    group_stds = torch.sqrt(group_var + epsilon)
    group_stds = torch.where(group_counts > 1, group_stds, torch.ones_like(group_stds))

    # Broadcast back to original order
    orig_means = group_means[index_tensor]
    orig_stds = group_stds[index_tensor]

    if norm_adv_by_std_in_grpo:
        result = (scores - orig_means) / (orig_stds + epsilon)
    else:
        result = scores - orig_means

    return result


# =============================================================================
# RLOO-style group advantage (Python loops reference)
# =============================================================================
def rloo_reference(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """RLOO advantage computation with Python batch loops (reference implementation)."""
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    id2score = defaultdict(list)
    id2mean = {}

    # Group scores by index
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # Compute per-group mean
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0, device=scores.device)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.stack(id2score[idx]))

    # Compute RLOO advantage
    result = scores.clone()
    for i in range(bsz):
        response_num = len(id2score[index[i]])
        if response_num > 1:
            result[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)

    return result


def rloo_vectorized(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """RLOO advantage computation with index_add vectorization."""
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]
    num_groups = max(index) + 1
    index_tensor = torch.tensor(index, device=scores.device, dtype=torch.long)

    # Group sums and counts
    group_sums = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_counts = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)

    group_sums.index_add_(0, index_tensor, scores)
    group_counts.index_add_(0, index_tensor, torch.ones(bsz, device=scores.device, dtype=torch.float32))

    # Group means
    group_means = group_sums / (group_counts + epsilon)

    # Compute per-sample RLOO score: r_i * n/(n-1) - mean * n/(n-1)
    n = group_counts[index_tensor]
    factor = n / (n - 1 + epsilon)
    result = scores * factor - group_means[index_tensor] * factor

    # For n=1, set to 0
    result = torch.where(n > 1, result, torch.zeros_like(result))

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark GRPO/RLOO group advantage computations.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark expects CUDA.")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Create test data
    token_level_rewards = torch.randn(args.batch_size, args.response_len, device=device)
    response_mask = _make_eos_style_mask(args.batch_size, args.response_len, device)
    index = np.random.randint(0, args.groups, size=args.batch_size)
    scores = token_level_rewards.sum(dim=-1)

    print(f"shape=batch:{args.batch_size} response_len:{args.response_len} groups:{args.groups}")

    # GRPO Benchmark
    print("\n=== GRPO Advantage ===")
    ref_result = grpo_reference(scores.clone(), index)
    vec_result = grpo_vectorized(scores.clone(), index)

    ref_timings = _sync_timing(
        lambda: grpo_reference(scores.clone(), index),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    ref_avg = _summarize("grpo/python_loops", ref_timings)

    vec_timings = _sync_timing(
        lambda: grpo_vectorized(scores.clone(), index),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    vec_avg = _summarize("grpo/index_add", vec_timings)
    print(f"grpo/speedup={ref_avg / vec_avg:.2f}x")

    # RLOO Benchmark
    print("\n=== RLOO Advantage ===")
    ref_timings = _sync_timing(
        lambda: rloo_reference(token_level_rewards.clone(), response_mask.clone(), index),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    ref_avg = _summarize("rloo/python_loops", ref_timings)

    vec_timings = _sync_timing(
        lambda: rloo_vectorized(token_level_rewards.clone(), response_mask.clone(), index),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    vec_avg = _summarize("rloo/index_add", vec_timings)
    print(f"rloo/speedup={ref_avg / vec_avg:.2f}x")


if __name__ == "__main__":
    main()
