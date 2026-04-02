#!/usr/bin/env python3
"""
Benchmark: compare original scalar metrics pattern vs fused compute_ratio_metrics.

Original pattern (in core_algos.py):
    neg_valid = ratio[(advantages < 0) & response_mask.bool()]
    neg_is_max = neg_valid.max()
    neg_is_p75 = torch.quantile(neg_valid, 0.75)
    ... (5 quantiles + 1 max per array)

Optimized pattern:
    - Uses compute_masked_quantiles which does single sort for all percentiles
    - Avoids materializing intermediate neg_valid/pos_valid tensors
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch


def _sync_timing(fn, warmup: int, iters: int, device: torch.device):
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


def compute_scalar_metrics_original(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_c: float,
) -> dict:
    """Original pattern from core_algos.py lines 1115-1146."""
    is_negative_adv = (advantages < 0)

    # Filtered ratio arrays (expensive - creates non-contiguous tensors)
    neg_valid = ratio[(advantages < 0) & response_mask.bool()]
    pos_valid = ratio[(advantages > 0) & response_mask.bool()]

    # neg_valid stats (5 quantile calls on same array)
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

    # pos_valid stats (5 quantile calls on same array)
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
    neg_ratio_2_3 = (((ratio >= 2.0) & (ratio < 3.0) & is_negative_adv).float() * response_mask).sum() / (response_mask.sum() + 1e-8)
    neg_ratio_3_4 = (((ratio >= 3.0) & (ratio < 4.0) & is_negative_adv).float() * response_mask).sum() / (response_mask.sum() + 1e-8)
    neg_ratio_4_10 = (((ratio >= 4.0) & (ratio < clip_ratio_c) & is_negative_adv).float() * response_mask).sum() / (response_mask.sum() + 1e-8)
    pos_mini_frac = (((ratio < 1e-3) & (advantages > 0)).float() * response_mask).sum() / (response_mask.sum() + 1e-8)

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
    parser = argparse.ArgumentParser(description="Benchmark ratio metrics computation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--clip-ratio-c", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load future_kl module
    repo_root = Path(__file__).resolve().parents[1]
    future_kl_mod = _load_module_from_path(
        "future_kl", repo_root / "verl" / "utils" / "kernel" / "future_kl.py"
    )
    compute_ratio_metrics = future_kl_mod.compute_ratio_metrics
    HAVE_TRITON = future_kl_mod.HAVE_TRITON

    batch_size = args.batch_size
    response_len = args.response_len

    # Synthetic inputs
    ratio = torch.exp(0.1 * torch.randn(batch_size, response_len, device=device, dtype=torch.float32))
    advantages = torch.randn(batch_size, response_len, device=device, dtype=torch.float32)
    response_mask = (torch.rand(batch_size, response_len, device=device) > 0.1).float()
    # Make advantages more realistic (mix of positive/negative)
    advantages = (torch.rand(batch_size, response_len, device=device) - 0.5) * 2.0

    print(f"Shape: batch={batch_size}, response_len={response_len}")
    print(f"Triton available: {HAVE_TRITON}")
    print()

    # Benchmark original pattern
    def original_path():
        compute_scalar_metrics_original(ratio, advantages, response_mask, clip_ratio_c=args.clip_ratio_c)

    print("Warming up original...")
    for _ in range(args.warmup):
        original_path()
    torch.cuda.synchronize()

    print("Benchmarking original pattern...")
    t_original = _sync_timing(original_path, warmup=0, iters=args.iters, device=device)
    _summarize("original/torch", t_original)

    # Benchmark fused pattern
    def fused_path():
        compute_ratio_metrics(ratio, advantages, response_mask, clip_ratio_c=args.clip_ratio_c)

    print()
    print("Warming up fused...")
    for _ in range(args.warmup):
        fused_path()
    torch.cuda.synchronize()

    print("Benchmarking fused pattern...")
    t_fused = _sync_timing(fused_path, warmup=0, iters=args.iters, device=device)
    _summarize("fused/torch", t_fused)

    print()
    speedup = mean(t_original) / mean(t_fused)
    print(f"Speedup: {speedup:.2f}x")

    # Verify correctness
    print()
    print("Verifying correctness...")
    orig_result = compute_scalar_metrics_original(ratio, advantages, response_mask, clip_ratio_c=args.clip_ratio_c)
    fused_result = compute_ratio_metrics(ratio, advantages, response_mask, clip_ratio_c=args.clip_ratio_c)

    # Compare key values
    all_ok = True
    for key in orig_result:
        orig_val = orig_result[key].item()
        fused_val = fused_result[key].item()
        if abs(orig_val) > 1e-6 or abs(fused_val) > 1e-6:
            rel_diff = abs(orig_val - fused_val) / (abs(orig_val) + 1e-8)
            if rel_diff > 1e-4:
                print(f"  WARN {key}: orig={orig_val:.6f}, fused={fused_val:.6f}, rel_diff={rel_diff:.6f}")
                all_ok = False
    if all_ok:
        print("  All values match (or are near-zero)")
    else:
        print("  Some values differ significantly")


if __name__ == "__main__":
    main()
