#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch


def _make_eos_style_mask(batch_size: int, response_len: int, device: torch.device) -> torch.Tensor:
    lengths = torch.randint(1, response_len + 1, (batch_size,), device=device)
    positions = torch.arange(response_len, device=device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).float()


def _load_advantage_module():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "kernel" / "advantage_kernels.py"
    spec = importlib.util.spec_from_file_location("advantage_kernel_benchmark_module", module_path)
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


def _summarize(name: str, timings_ms: list[float]) -> float:
    sorted_timings = sorted(timings_ms)
    avg = mean(timings_ms)
    p50 = sorted_timings[len(sorted_timings) // 2]
    print(f"{name}: mean={avg:.3f} ms p50={p50:.3f} ms min={sorted_timings[0]:.3f} ms")
    return avg


def main():
    parser = argparse.ArgumentParser(description="Benchmark discounted-return and GAE Triton kernels.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark expects CUDA.")

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    advantage_mod = _load_advantage_module()
    compute_discounted_returns = advantage_mod.compute_discounted_returns
    compute_gae_advantages_returns = advantage_mod.compute_gae_advantages_returns

    rewards = torch.randn(args.batch_size, args.response_len, device=device, dtype=dtype)
    values = torch.randn(args.batch_size, args.response_len, device=device, dtype=dtype)
    response_mask = _make_eos_style_mask(args.batch_size, args.response_len, device)

    torch_returns = compute_discounted_returns(rewards, response_mask, args.gamma, impl="torch")
    triton_returns = compute_discounted_returns(rewards, response_mask, args.gamma, impl="triton")
    torch.testing.assert_close(triton_returns, torch_returns, rtol=2e-4, atol=2e-4)

    print(
        f"shape=batch:{args.batch_size} response_len:{args.response_len} dtype={dtype} gamma={args.gamma} lam={args.lam}"
    )
    torch_timings = _sync_timing(
        lambda: compute_discounted_returns(rewards, response_mask, args.gamma, impl="torch"),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    torch_avg = _summarize("discounted_returns/torch", torch_timings)

    triton_timings = _sync_timing(
        lambda: compute_discounted_returns(rewards, response_mask, args.gamma, impl="triton"),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    triton_avg = _summarize("discounted_returns/triton", triton_timings)
    print(f"discounted_returns/speedup={torch_avg / triton_avg:.2f}x")

    remax_rewards = rewards * response_mask
    torch_remax = _sync_timing(
        lambda: compute_discounted_returns(remax_rewards, response_mask, 1.0, impl="torch"),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    remax_torch_avg = _summarize("remax_returns/torch", torch_remax)

    triton_remax = _sync_timing(
        lambda: compute_discounted_returns(remax_rewards, response_mask, 1.0, impl="triton"),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    remax_triton_avg = _summarize("remax_returns/triton", triton_remax)
    print(f"remax_returns/speedup={remax_torch_avg / remax_triton_avg:.2f}x")

    torch_gae_advantages, torch_gae_returns = compute_gae_advantages_returns(
        rewards,
        values,
        response_mask,
        args.gamma,
        args.lam,
        impl="torch",
    )
    triton_gae_advantages, triton_gae_returns = compute_gae_advantages_returns(
        rewards,
        values,
        response_mask,
        args.gamma,
        args.lam,
        impl="triton",
    )
    torch.testing.assert_close(triton_gae_advantages, torch_gae_advantages, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(triton_gae_returns, torch_gae_returns, rtol=2e-4, atol=2e-4)

    torch_gae = _sync_timing(
        lambda: compute_gae_advantages_returns(rewards, values, response_mask, args.gamma, args.lam, impl="torch"),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    torch_gae_avg = _summarize("gae/torch", torch_gae)

    triton_gae = _sync_timing(
        lambda: compute_gae_advantages_returns(rewards, values, response_mask, args.gamma, args.lam, impl="triton"),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    triton_gae_avg = _summarize("gae/triton", triton_gae)
    print(f"gae/speedup={torch_gae_avg / triton_gae_avg:.2f}x")


if __name__ == "__main__":
    main()
