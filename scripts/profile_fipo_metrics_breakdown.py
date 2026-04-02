#!/usr/bin/env python3
"""
Detailed breakdown of scalar metrics cost in compute_policy_loss_future_kl.
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
    print(f"{name}: mean={mean(timings_ms):.3f} ms p50={p50:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Breakdown FIPO scalar metrics cost")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    batch_size = args.batch_size
    response_len = args.response_len

    ratio = torch.exp(0.1 * torch.randn(batch_size, response_len, device=device, dtype=torch.float32))
    advantages = torch.randn(batch_size, response_len, device=device, dtype=torch.float32)
    response_mask = (torch.rand(batch_size, response_len, device=device) > 0.1).float()
    advantages = (torch.rand(batch_size, response_len, device=device) - 0.5) * 2.0

    print(f"Shape: batch={batch_size}, response_len={response_len}")
    print()

    # 1. Filtering cost
    def filter_neg():
        return ratio[(advantages < 0) & response_mask.bool()]

    def filter_pos():
        return ratio[(advantages > 0) & response_mask.bool()]

    for _ in range(args.warmup):
        neg_valid = filter_neg()
        pos_valid = filter_pos()
    torch.cuda.synchronize()

    t_neg = _sync_timing(filter_neg, warmup=0, iters=args.iters, device=device)
    _summarize("filter/neg_valid", t_neg)

    t_pos = _sync_timing(filter_pos, warmup=0, iters=args.iters, device=device)
    _summarize("filter/pos_valid", t_pos)

    # 2. Max/min cost
    neg_valid = filter_neg()
    pos_valid = filter_pos()

    def neg_max():
        return neg_valid.max()

    def pos_max():
        return pos_valid.max()

    def neg_min():
        return neg_valid.min()

    def pos_min():
        return pos_valid.min()

    for _ in range(args.warmup):
        neg_max()
        pos_max()
    torch.cuda.synchronize()

    t = _sync_timing(neg_max, warmup=0, iters=args.iters, device=device)
    _summarize("reduce/neg_max", t)

    t = _sync_timing(pos_max, warmup=0, iters=args.iters, device=device)
    _summarize("reduce/pos_max", t)

    t = _sync_timing(neg_min, warmup=0, iters=args.iters, device=device)
    _summarize("reduce/neg_min", t)

    t = _sync_timing(pos_min, warmup=0, iters=args.iters, device=device)
    _summarize("reduce/pos_min", t)

    # 3. Quantile breakdown
    def q_neg(q):
        def fn():
            torch.quantile(neg_valid, q)
        return fn

    def q_pos(q):
        def fn():
            torch.quantile(pos_valid, q)
        return fn

    for q in [0.25, 0.5, 0.75, 0.995, 0.999]:
        for _ in range(args.warmup):
            q_neg(q)()
        torch.cuda.synchronize()
        t = _sync_timing(q_neg(q), warmup=0, iters=args.iters, device=device)
        _summarize(f"quantile/neg_{q}", t)

        for _ in range(args.warmup):
            q_pos(q)()
        torch.cuda.synchronize()
        t = _sync_timing(q_pos(q), warmup=0, iters=args.iters, device=device)
        _summarize(f"quantile/pos_{q}", t)

    # 4. Masked mean operations
    is_negative_adv = (advantages < 0)

    def neg_ratio_2_3():
        return (((ratio >= 2.0) & (ratio < 3.0) & is_negative_adv).float() * response_mask).sum() / (response_mask.sum() + 1e-8)

    def neg_ratio_3_4():
        return (((ratio >= 3.0) & (ratio < 4.0) & is_negative_adv).float() * response_mask).sum() / (response_mask.sum() + 1e-8)

    def neg_ratio_4_10():
        return (((ratio >= 4.0) & (ratio < 10.0) & is_negative_adv).float() * response_mask).sum() / (response_mask.sum() + 1e-8)

    def pos_mini_frac():
        return (((ratio < 1e-3) & (advantages > 0)).float() * response_mask).sum() / (response_mask.sum() + 1e-8)

    for fn in [neg_ratio_2_3, neg_ratio_3_4, neg_ratio_4_10, pos_mini_frac]:
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        t = _sync_timing(fn, warmup=0, iters=args.iters, device=device)
        _summarize(f"masked_mean/{fn.__name__}", t)

    print()
    print("Profile complete.")


if __name__ == "__main__":
    main()