#!/usr/bin/env python3
"""Benchmark for the fused gathered log-probability kernel (FIPO-026).

Compares:
  - naive:   full log_softmax(logits, dim=-1).gather(...)  [materialises (B,T,V)]
  - torch:   compute_token_logprob_torch  (same math, explicit reference)
  - triton:  compute_token_logprob_triton (fused single-pass kernel)

Usage:
  python scripts/benchmark_logprob_kernel.py
  python scripts/benchmark_logprob_kernel.py --batch-size 32 --seq-len 2048 \
      --vocab-size 32768 --dtype bfloat16 --warmup 5 --iters 20
"""

import argparse
import time

import torch
import torch.nn.functional as F

from verl.utils.kernel.logprob import (
    HAVE_TRITON,
    _TRITON_MAX_VOCAB,
    compute_token_logprob_torch,
    compute_token_logprob_triton,
)


def _naive(logits, token_ids):
    lp = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def run(batch_size: int, seq_len: int, vocab_size: int, dtype: torch.dtype,
        warmup: int, iters: int) -> None:
    if not torch.cuda.is_available():
        print("CUDA not available – skipping GPU benchmark.")
        return

    device = "cuda"
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=dtype)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    numel = batch_size * seq_len
    logits_gb = logits.numel() * logits.element_size() / 1e9
    print(f"\nLogits shape : ({batch_size}, {seq_len}, {vocab_size})  [{logits_gb:.2f} GB]")
    print(f"dtype        : {dtype}")
    print(f"warmup / iters : {warmup} / {iters}")
    print("-" * 60)

    naive_ms = None
    try:
        naive_ms = _bench(lambda: _naive(logits, token_ids), warmup, iters)
        print(f"naive (torch log_softmax+gather) : {naive_ms:.3f} ms")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"naive (torch log_softmax+gather) : OOM  (materialising (B,T,V) exhausted GPU memory)")

    torch_ms = None
    try:
        torch_ms = _bench(lambda: compute_token_logprob_torch(logits, token_ids), warmup, iters)
        print(f"torch (reference impl)           : {torch_ms:.3f} ms")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"torch (reference impl)           : OOM")

    if HAVE_TRITON:
        triton_ms = _bench(lambda: compute_token_logprob_triton(logits, token_ids), warmup, iters)
        ref_ms = naive_ms or torch_ms
        speedup_str = f"   ({ref_ms / triton_ms:.2f}x vs {'naive' if naive_ms else 'torch'})" if ref_ms else ""
        print(f"triton (fused kernel)            : {triton_ms:.3f} ms{speedup_str}")
    else:
        print(f"triton                           : SKIPPED (Triton not installed)")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark gathered logprob kernel")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    # Sweep over several vocab sizes to show memory-bandwidth cliff
    vocab_sweep = [args.vocab_size]
    if args.vocab_size == 32768:
        vocab_sweep = [1024, 4096, 8192, 16384, 32768]

    for V in vocab_sweep:
        run(args.batch_size, args.seq_len, V, dtype, args.warmup, args.iters)


if __name__ == "__main__":
    main()
