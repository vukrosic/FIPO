#!/usr/bin/env python3
"""Benchmark the torch_functional logprobs fallback against the Triton gathered-logprob kernel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import verl.utils.torch_functional as verl_F
from verl.utils.kernel.logprob import compute_token_logprob


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


def _dispatch_kernel(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 2:
        output = compute_token_logprob(logits.unsqueeze(0), labels.unsqueeze(0), impl="triton").squeeze(0)
    else:
        output = compute_token_logprob(logits, labels, impl="triton")
    if logits.dtype in {torch.float16, torch.bfloat16}:
        output = output.to(logits.dtype)
    return output


def main():
    parser = argparse.ArgumentParser(description="Benchmark logprobs dispatcher fallback")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark expects CUDA.")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logits = torch.randn(args.batch_size, args.seq_len, args.vocab_size, device=device, dtype=dtype)
    labels = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    with mock.patch.object(verl_F, "FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False), mock.patch.object(
        verl_F, "NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False
    ):
        baseline = verl_F.logprobs_from_logits_v2(logits, labels)
        fused = verl_F.logprobs_from_logits(logits, labels)
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(fused.float(), baseline.float(), rtol=tol, atol=tol)

    print(f"padded_shape=batch:{args.batch_size} seq_len:{args.seq_len} vocab:{args.vocab_size} dtype={dtype}")
    baseline_ms = _summarize(
        "padded/v2",
        _sync_timing(lambda: verl_F.logprobs_from_logits_v2(logits, labels), args.warmup, args.iters, device),
    )
    fused_ms = _summarize(
        "padded/dispatch",
        _sync_timing(lambda: _dispatch_kernel(logits, labels), args.warmup, args.iters, device),
    )
    print(f"padded/speedup={baseline_ms / fused_ms:.2f}x")
    print()

    total_tokens = args.batch_size * args.seq_len
    logits_flat = logits.reshape(total_tokens, args.vocab_size)
    labels_flat = labels.reshape(total_tokens)
    baseline = verl_F.logprobs_from_logits_v2(logits_flat, labels_flat)
    fused = _dispatch_kernel(logits_flat, labels_flat)
    torch.testing.assert_close(fused.float(), baseline.float(), rtol=tol, atol=tol)

    print(f"rmpad_shape=tokens:{total_tokens} vocab:{args.vocab_size} dtype={dtype}")
    baseline_ms = _summarize(
        "rmpad/v2",
        _sync_timing(lambda: verl_F.logprobs_from_logits_v2(logits_flat, labels_flat), args.warmup, args.iters, device),
    )
    fused_ms = _summarize(
        "rmpad/dispatch",
        _sync_timing(lambda: _dispatch_kernel(logits_flat, labels_flat), args.warmup, args.iters, device),
    )
    print(f"rmpad/speedup={baseline_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    main()
