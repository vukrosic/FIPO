#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch


def _load_future_kl_module():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "kernel" / "future_kl.py"
    spec = importlib.util.spec_from_file_location("future_kl_benchmark_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _summarize(name: str, timings_ms: list[float]):
    sorted_timings = sorted(timings_ms)
    p50 = sorted_timings[len(sorted_timings) // 2]
    print(f"{name}: mean={mean(timings_ms):.3f} ms p50={p50:.3f} ms min={sorted_timings[0]:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Future-KL torch vs Triton implementations.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--decay-rate", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark expects CUDA.")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    future_kl_mod = _load_future_kl_module()
    compute_future_kl = future_kl_mod.compute_future_kl
    compute_future_kl_chunked_reference = future_kl_mod.compute_future_kl_chunked_reference
    compute_influence_weights = future_kl_mod.compute_influence_weights
    compute_influence_weights_torch = future_kl_mod.compute_influence_weights_torch
    compute_masked_mean = future_kl_mod.compute_masked_mean
    compute_masked_mean_torch = future_kl_mod.compute_masked_mean_torch
    have_triton = future_kl_mod.HAVE_TRITON

    gamma = 2 ** (-1.0 / args.decay_rate)
    kl_response_input = torch.randn(args.batch_size, args.response_len, device=device, dtype=dtype)
    print(f"input_dtype={dtype} compute_dtype=torch.float32")

    def torch_path():
        return compute_future_kl(kl_response_input.float(), gamma, impl="torch", chunk_size=args.chunk_size)

    def triton_path():
        return compute_future_kl(kl_response_input.float(), gamma, impl="triton", chunk_size=args.chunk_size)

    if have_triton:
        torch_future = compute_future_kl_chunked_reference(kl_response_input.float(), gamma, chunk_size=args.chunk_size)
        triton_future = triton_path()
        torch.testing.assert_close(triton_future, torch_future, rtol=2e-3, atol=2e-3)

    helper_torch = _sync_timing(
        torch_path,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    _summarize("helper/torch", helper_torch)

    if have_triton:
        helper_triton = _sync_timing(
            triton_path,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        _summarize("helper/triton", helper_triton)
        print(f"helper/speedup={mean(helper_torch) / mean(helper_triton):.2f}x")

    future_kl = torch.randn(args.batch_size, args.response_len, device=device, dtype=torch.float32)
    advantages = torch.randn(args.batch_size, args.response_len, device=device, dtype=torch.float32)
    ratio = torch.exp(0.1 * torch.randn(args.batch_size, args.response_len, device=device, dtype=torch.float32))

    if have_triton:
        torch_raw, torch_influence, _, _ = compute_influence_weights_torch(
            future_kl=future_kl,
            advantages=advantages,
            ratio=ratio,
            clip_ratio=0.2,
            clip_high_only=True,
            safe_threshold=10.0,
        )
        triton_raw, triton_influence, _, _ = compute_influence_weights(
            future_kl=future_kl,
            advantages=advantages,
            ratio=ratio,
            clip_ratio=0.2,
            clip_high_only=True,
            safe_threshold=10.0,
            impl="triton",
        )
        torch.testing.assert_close(triton_raw, torch_raw, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(triton_influence, torch_influence, rtol=1e-5, atol=1e-5)

    influence_torch = _sync_timing(
        lambda: compute_influence_weights(
            future_kl=future_kl,
            advantages=advantages,
            ratio=ratio,
            clip_ratio=0.2,
            clip_high_only=True,
            safe_threshold=10.0,
            impl="torch",
        ),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    _summarize("influence/torch", influence_torch)

    if have_triton:
        influence_triton = _sync_timing(
            lambda: compute_influence_weights(
                future_kl=future_kl,
                advantages=advantages,
                ratio=ratio,
                clip_ratio=0.2,
                clip_high_only=True,
                safe_threshold=10.0,
                impl="triton",
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        _summarize("influence/triton", influence_triton)
        print(f"influence/speedup={mean(influence_torch) / mean(influence_triton):.2f}x")

    values = torch.randn(args.batch_size, args.response_len, device=device, dtype=torch.float32)
    mask = (torch.rand(args.batch_size, args.response_len, device=device) > 0.1).float()

    if have_triton:
        torch_mean = compute_masked_mean_torch(values=values, mask=mask)
        triton_mean = compute_masked_mean(values=values, mask=mask, impl="triton")
        torch.testing.assert_close(triton_mean, torch_mean, rtol=1e-5, atol=1e-5)

    masked_mean_torch = _sync_timing(
        lambda: compute_masked_mean(values=values, mask=mask, impl="torch"),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    _summarize("masked_mean/torch", masked_mean_torch)

    if have_triton:
        masked_mean_triton = _sync_timing(
            lambda: compute_masked_mean(values=values, mask=mask, impl="triton"),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        _summarize("masked_mean/triton", masked_mean_triton)
        print(f"masked_mean/speedup={mean(masked_mean_torch) / mean(masked_mean_triton):.2f}x")


if __name__ == "__main__":
    main()
