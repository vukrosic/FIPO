#!/usr/bin/env python3
"""
Profile scalar metrics in compute_policy_loss_future_kl.

This script isolates the metric computation overhead (quantile calls,
masked reductions) from the rest of the FIPO loss path to identify
the next kernelization target.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch


def _sync_timing(fn, warmup: int, iters: int, device: torch.device):
    """Synchronized CUDA timing using torch.cuda.Event."""
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
    for _ in range(warmup):
        fn()
    for _ in range(iters):
        start = perf_counter()
        fn()
        timings_ms.append((perf_counter() - start) * 1000.0)
    return timings_ms


def _summarize(name: str, timings_ms: list[float]):
    sorted_timings = sorted(timings_ms)
    p50 = sorted_timings[len(sorted_timings) // 2]
    print(f"{name}: mean={mean(timings_ms):.3f} ms p50={p50:.3f} ms min={sorted_timings[0]:.3f} ms")


def _load_module_from_path(module_name: str, file_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() + 1e-8)


def compute_scalar_metrics_torch(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict:
    """
    Compute all scalar metrics from compute_policy_loss_future_kl.
    This matches lines 1112-1146 in core_algos.py.
    """
    is_negative_adv = (advantages < 0)

    # Filtered ratio arrays
    neg_valid = ratio[(advantages < 0) & response_mask.bool()]
    pos_valid = ratio[(advantages > 0) & response_mask.bool()]

    # neg_valid stats
    if neg_valid.numel() > 0:
        neg_is_max = neg_valid.max()
        neg_is_p75 = torch.quantile(neg_valid, 0.75)
        neg_is_p995 = torch.quantile(neg_valid, 0.995)
        neg_is_p999 = torch.quantile(neg_valid, 0.999)
    else:
        neg_is_max = torch.tensor(0.0, device=ratio.device)
        neg_is_p995 = torch.tensor(0.0, device=ratio.device)
        neg_is_p999 = torch.tensor(0.0, device=ratio.device)
        neg_is_p75 = torch.tensor(0.0, device=ratio.device)

    # pos_valid stats
    if pos_valid.numel() > 0:
        pos_is_max = pos_valid.max()
        pos_is_p25 = torch.quantile(pos_valid, 0.25)
        pos_is_median = torch.quantile(pos_valid, 0.5)
        pos_is_p75 = torch.quantile(pos_valid, 0.75)
        pos_is_p995 = torch.quantile(pos_valid, 0.995)
        pos_is_p999 = torch.quantile(pos_valid, 0.999)
        pos_is_min = pos_valid.min()
    else:
        pos_is_p25 = torch.tensor(0.0, device=ratio.device)
        pos_is_max = torch.tensor(0.0, device=ratio.device)
        pos_is_median = torch.tensor(0.0, device=ratio.device)
        pos_is_p75 = torch.tensor(0.0, device=ratio.device)
        pos_is_p995 = torch.tensor(0.0, device=ratio.device)
        pos_is_p999 = torch.tensor(0.0, device=ratio.device)
        pos_is_min = torch.tensor(0.0, device=ratio.device)

    # Ratio range metrics
    neg_ratio_2_3 = masked_mean(((ratio >= 2.0) & (ratio < 3.0) & is_negative_adv).float(), response_mask)
    neg_ratio_3_4 = masked_mean(((ratio >= 3.0) & (ratio < 4.0) & is_negative_adv).float(), response_mask)
    neg_ratio_4_10 = masked_mean(((ratio >= 4.0) & (ratio < 10.0) & is_negative_adv).float(), response_mask)
    pos_mini_frac = masked_mean(((ratio < 1e-3) & (advantages > 0)).float(), response_mask)

    return {
        "neg_is_max": neg_is_max,
        "neg_is_p75": neg_is_p75,
        "neg_is_p995": neg_is_p995,
        "neg_is_p999": neg_is_p999,
        "pos_is_max": pos_is_max,
        "pos_is_p25": pos_is_p25,
        "pos_is_median": pos_is_median,
        "pos_is_p75": pos_is_p75,
        "pos_is_p995": pos_is_p995,
        "pos_is_p999": pos_is_p999,
        "pos_is_min": pos_is_min,
        "neg_ratio_2_3": neg_ratio_2_3,
        "neg_ratio_3_4": neg_ratio_3_4,
        "neg_ratio_4_10": neg_ratio_4_10,
        "pos_mini_frac": pos_mini_frac,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile FIPO scalar metrics")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This profiler requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    batch_size = args.batch_size
    response_len = args.response_len

    # Synthetic inputs
    ratio = torch.exp(0.1 * torch.randn(batch_size, response_len, device=device, dtype=torch.float32))
    advantages = torch.randn(batch_size, response_len, device=device, dtype=torch.float32)
    response_mask = (torch.rand(batch_size, response_len, device=device) > 0.1).float()

    # Make advantages more realistic (mix of positive/negative)
    advantages = (torch.rand(batch_size, response_len, device=device) - 0.5) * 2.0

    print(f"Shape: batch={batch_size}, response_len={response_len}")
    print(f"neg_valid count: ~{(advantages < 0 & response_mask.bool()).sum().item()}")
    print(f"pos_valid count: ~{(advantages > 0 & response_mask.bool()).sum().item()}")
    print()

    # Profile the full metrics computation
    def compute_metrics():
        compute_scalar_metrics_torch(ratio, advantages, response_mask)

    print("Warming up...")
    for _ in range(args.warmup):
        compute_metrics()
    torch.cuda.synchronize()

    print("Profiling scalar metrics (torch)...")
    timings = _sync_timing(compute_metrics, warmup=0, iters=args.iters, device=device)
    _summarize("scalar_metrics/torch", timings)

    # Also profile individual quantile operations
    print()
    print("--- Individual Quantile Breakdown ---")

    neg_valid = ratio[(advantages < 0) & response_mask.bool()]
    pos_valid = ratio[(advantages > 0) & response_mask.bool()]

    quantile_values = [0.25, 0.5, 0.75, 0.995, 0.999]
    for q in quantile_values:
        def q_neg():
            torch.quantile(neg_valid, q)
        def q_pos():
            torch.quantile(pos_valid, q)

        for _ in range(args.warmup):
            q_neg()
        torch.cuda.synchronize()
        t_neg = _sync_timing(q_neg, warmup=0, iters=args.iters, device=device)
        _summarize(f"quantile/neg_q{q}", t_neg)

        for _ in range(args.warmup):
            q_pos()
        torch.cuda.synchronize()
        t_pos = _sync_timing(q_pos, warmup=0, iters=args.iters, device=device)
        _summarize(f"quantile/pos_q{q}", t_pos)

    print()
    print("Profile complete.")


if __name__ == "__main__":
    main()