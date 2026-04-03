#!/usr/bin/env python3
"""
Benchmark for Fused KL Loss Kernel.

Compares the fused compute_kl_loss kernel (Triton) against the original
two-step approach: kl_penalty + agg_loss.
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

    # CPU fallback
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
    print(f"{name}: mean={mean(timings_ms):.4f} ms p50={p50:.4f} ms min={sorted_timings[0]:.4f} ms")


def _load_module_from_path(module_name: str, file_path: Path):
    """Load a module directly from a file path, bypassing package __init__."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def original_kl_loss(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    response_mask: torch.Tensor,
):
    """Original two-step KL loss: kl_penalty + agg_loss."""
    # kl_penalty (low_var_kl mode)
    kl = ref_logprob - logprob
    kl = torch.clamp(kl, min=-20.0, max=20.0)
    ratio = torch.exp(kl)
    kld = ratio - kl - 1.0
    kld = torch.clamp(kld, min=-10.0, max=10.0)

    # agg_loss (token-mean mode)
    masked_kld = kld * response_mask
    kl_loss = masked_kld.sum() / (response_mask.sum() + 1e-8)
    return kl_loss


def main():
    parser = argparse.ArgumentParser(description="KL Loss Kernel Benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load kernel module
    repo_root = Path(__file__).resolve().parents[1]
    future_kl_mod = _load_module_from_path(
        "future_kl", repo_root / "verl" / "utils" / "kernel" / "future_kl.py"
    )
    compute_kl_loss = future_kl_mod.compute_kl_loss
    HAVE_TRITON = future_kl_mod.HAVE_TRITON

    batch_size = args.batch_size
    response_len = args.response_len

    # Create inputs
    logprob = torch.randn(batch_size, response_len, device=device, dtype=torch.float32)
    ref_logprob = torch.randn(batch_size, response_len, device=device, dtype=torch.float32)
    response_mask = (torch.rand(batch_size, response_len, device=device) > 0.1).float()

    print(f"Shape: batch={batch_size}, response_len={response_len}, dtype=float32")
    print(f"Triton available: {HAVE_TRITON}")
    print()

    # Warmup
    print("Warming up...")
    for _ in range(args.warmup):
        original_kl_loss(logprob, ref_logprob, response_mask)
        compute_kl_loss(logprob, ref_logprob, response_mask, impl="torch")
        if HAVE_TRITON:
            compute_kl_loss(logprob, ref_logprob, response_mask, impl="triton")
    torch.cuda.synchronize()

    # Benchmark original (two-step)
    print("\nBenchmarking original two-step path (CPU/CUDA)...")
    def original_path():
        original_kl_loss(logprob, ref_logprob, response_mask)

    timings_original = _sync_timing(original_path, warmup=0, iters=args.iters, device=device)
    _summarize("kl_loss/original", timings_original)

    # Benchmark fused torch
    print("\nBenchmarking fused torch path...")
    def fused_torch_path():
        compute_kl_loss(logprob, ref_logprob, response_mask, impl="torch")

    timings_fused_torch = _sync_timing(fused_torch_path, warmup=0, iters=args.iters, device=device)
    _summarize("kl_loss/fused_torch", timings_fused_torch)

    # Benchmark fused triton
    if HAVE_TRITON:
        print("\nBenchmarking fused triton path...")
        def fused_triton_path():
            compute_kl_loss(logprob, ref_logprob, response_mask, impl="triton")

        timings_fused_triton = _sync_timing(fused_triton_path, warmup=0, iters=args.iters, device=device)
        _summarize("kl_loss/fused_triton", timings_fused_triton)

        print()
        print("Speedup (original -> triton): {:.2f}x".format(
            mean(timings_original) / mean(timings_fused_triton)))
        print("Speedup (torch -> triton): {:.2f}x".format(
            mean(timings_fused_torch) / mean(timings_fused_triton)))

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
