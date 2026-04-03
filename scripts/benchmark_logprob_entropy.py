#!/usr/bin/env python3
"""Benchmark fused logprob + entropy helper on actor-like shapes."""

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

from verl.utils.kernel.entropy_from_logits import compute_entropy_from_logits
from verl.utils.kernel.logprob import compute_token_logprob
from verl.utils.kernel.logprob_entropy import compute_logprob_and_entropy


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


def _separate_padded(logits: torch.Tensor, token_ids: torch.Tensor):
    log_probs = compute_token_logprob(logits, token_ids, impl="triton")
    entropy = compute_entropy_from_logits(logits, impl="triton")
    return log_probs, entropy


def _separate_flat(logits: torch.Tensor, token_ids: torch.Tensor):
    logits_btv = logits.unsqueeze(0)
    token_ids_bt = token_ids.unsqueeze(0)
    log_probs = compute_token_logprob(logits_btv, token_ids_bt, impl="triton").squeeze(0)
    entropy = compute_entropy_from_logits(logits_btv, impl="triton").squeeze(0)
    return log_probs, entropy


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused logprob+entropy helper")
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
    token_ids = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    separate_lp, separate_ent = _separate_padded(logits, token_ids)
    fused_lp, fused_ent = compute_logprob_and_entropy(logits, token_ids, impl="triton")
    tol = 5e-3 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(fused_lp, separate_lp, rtol=tol, atol=tol)
    torch.testing.assert_close(fused_ent, separate_ent, rtol=tol, atol=tol)

    print(f"padded_shape=batch:{args.batch_size} seq_len:{args.seq_len} vocab:{args.vocab_size} dtype={dtype}")
    separate_ms = _summarize(
        "padded/separate",
        _sync_timing(lambda: _separate_padded(logits, token_ids), args.warmup, args.iters, device),
    )
    fused_ms = _summarize(
        "padded/fused",
        _sync_timing(lambda: compute_logprob_and_entropy(logits, token_ids, impl="triton"), args.warmup, args.iters, device),
    )
    print(f"padded/speedup={separate_ms / fused_ms:.2f}x")
    print()

    flat_tokens = args.batch_size * args.seq_len
    logits_flat = logits.reshape(flat_tokens, args.vocab_size)
    token_ids_flat = token_ids.reshape(flat_tokens)
    separate_lp, separate_ent = _separate_flat(logits_flat, token_ids_flat)
    fused_lp, fused_ent = compute_logprob_and_entropy(logits_flat, token_ids_flat, impl="triton")
    torch.testing.assert_close(fused_lp, separate_lp, rtol=tol, atol=tol)
    torch.testing.assert_close(fused_ent, separate_ent, rtol=tol, atol=tol)

    print(f"rmpad_shape=tokens:{flat_tokens} vocab:{args.vocab_size} dtype={dtype}")
    separate_ms = _summarize(
        "rmpad/separate",
        _sync_timing(lambda: _separate_flat(logits_flat, token_ids_flat), args.warmup, args.iters, device),
    )
    fused_ms = _summarize(
        "rmpad/fused",
        _sync_timing(lambda: compute_logprob_and_entropy(logits_flat, token_ids_flat, impl="triton"), args.warmup, args.iters, device),
    )
    print(f"rmpad/speedup={separate_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    main()
