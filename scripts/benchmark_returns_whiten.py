#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch


def _load_core_algos():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    module_path = repo_root / "verl" / "trainer" / "ppo" / "core_algos.py"
    spec = importlib.util.spec_from_file_location("returns_whiten_benchmark_core_algos", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def _run_impl(core_algos, rewards: torch.Tensor, response_mask: torch.Tensor, gamma: float, impl: str):
    config = types.SimpleNamespace(gamma=gamma, reinforce_plus_plus_impl=impl)
    return core_algos.compute_reinforce_plus_plus_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        config=config,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark the REINFORCE++ returns+whiten fast path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--gamma", type=float, default=0.99)
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

    core_algos = _load_core_algos()
    rewards = torch.randn(args.batch_size, args.response_len, device=device, dtype=dtype)
    response_mask = _make_eos_style_mask(args.batch_size, args.response_len, device)

    torch_adv, torch_returns = _run_impl(core_algos, rewards, response_mask, args.gamma, "torch")
    auto_adv, auto_returns = _run_impl(core_algos, rewards, response_mask, args.gamma, "auto")

    if dtype == torch.float32:
        triton_adv, triton_returns = _run_impl(core_algos, rewards, response_mask, args.gamma, "triton")
        torch.testing.assert_close(triton_returns, torch_returns, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(triton_adv, torch_adv, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(auto_returns, triton_returns, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(auto_adv, triton_adv, rtol=2e-4, atol=2e-4)
    else:
        torch.testing.assert_close(auto_returns.float(), torch_returns.float(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(auto_adv.float(), torch_adv.float(), rtol=1e-5, atol=1e-5)

    print(
        f"shape=batch:{args.batch_size} response_len:{args.response_len} dtype={dtype} gamma={args.gamma}"
    )

    torch_timings = _sync_timing(
        lambda: _run_impl(core_algos, rewards, response_mask, args.gamma, "torch"),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    torch_avg = _summarize("reinforce_plus_plus/torch", torch_timings)

    auto_timings = _sync_timing(
        lambda: _run_impl(core_algos, rewards, response_mask, args.gamma, "auto"),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    auto_avg = _summarize("reinforce_plus_plus/auto", auto_timings)
    print(f"reinforce_plus_plus/auto_speedup={torch_avg / auto_avg:.2f}x")

    if dtype == torch.float32:
        triton_timings = _sync_timing(
            lambda: _run_impl(core_algos, rewards, response_mask, args.gamma, "triton"),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        triton_avg = _summarize("reinforce_plus_plus/triton", triton_timings)
        print(f"reinforce_plus_plus/triton_speedup={torch_avg / triton_avg:.2f}x")


if __name__ == "__main__":
    main()
