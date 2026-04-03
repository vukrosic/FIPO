#!/usr/bin/env python3
"""Benchmark entropy_from_logits dispatcher against the legacy CUDA path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import verl.utils.torch_functional as verl_F


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
    for _ in range(iters):
        start = perf_counter()
        fn()
        timings_ms.append((perf_counter() - start) * 1000.0)
    return timings_ms


def _summarize(name: str, timings_ms: list[float]):
    sorted_timings = sorted(timings_ms)
    p50 = sorted_timings[len(sorted_timings) // 2]
    print(f"{name}: mean={mean(timings_ms):.3f} ms p50={p50:.3f} ms min={sorted_timings[0]:.3f} ms")


def _entropy_reference(logits: torch.Tensor) -> torch.Tensor:
    pd = torch.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Benchmark entropy_from_logits dispatcher")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    padded_logits = torch.randn(args.batch_size, args.seq_len, args.vocab_size, device=device, dtype=dtype)
    rmpad_logits = torch.randn(args.batch_size * args.seq_len, args.vocab_size, device=device, dtype=dtype)

    print(
        f"Shape: batch={args.batch_size}, seq_len={args.seq_len}, vocab={args.vocab_size}, dtype={dtype}"
    )
    print()

    def padded_reference():
        _entropy_reference(padded_logits)

    def padded_dispatch():
        verl_F.entropy_from_logits(padded_logits)

    def rmpad_reference():
        _entropy_reference(rmpad_logits)

    def rmpad_dispatch():
        verl_F.entropy_from_logits(rmpad_logits)

    t_padded_ref = _sync_timing(padded_reference, warmup=args.warmup, iters=args.iters, device=device)
    t_padded_dispatch = _sync_timing(padded_dispatch, warmup=args.warmup, iters=args.iters, device=device)
    t_rmpad_ref = _sync_timing(rmpad_reference, warmup=args.warmup, iters=args.iters, device=device)
    t_rmpad_dispatch = _sync_timing(rmpad_dispatch, warmup=args.warmup, iters=args.iters, device=device)

    _summarize("padded/reference", t_padded_ref)
    _summarize("padded/dispatch", t_padded_dispatch)
    print(f"padded/speedup: {mean(t_padded_ref) / mean(t_padded_dispatch):.2f}x")
    print()
    _summarize("rmpad/reference", t_rmpad_ref)
    _summarize("rmpad/dispatch", t_rmpad_dispatch)
    print(f"rmpad/speedup: {mean(t_rmpad_ref) / mean(t_rmpad_dispatch):.2f}x")


if __name__ == "__main__":
    main()
