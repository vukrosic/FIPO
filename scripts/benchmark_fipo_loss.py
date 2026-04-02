#!/usr/bin/env python3
"""
End-to-end FIPO loss benchmark.

Measures the full compute_policy_loss_future_kl path including:
- future_kl computation
- influence_weights computation
- weighted advantage computation
- PPO clipping
- loss aggregation

This is a standalone script that replicates the FIPO loss computation
without importing the full Ray stack.
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
    """Load a module directly from a file path, bypassing package __init__."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean, matching verl_F.masked_mean."""
    return (values * mask).sum() / (mask.sum() + 1e-8)


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """Aggregate loss matrix into scalar, matching core_algos.agg_loss."""
    if loss_agg_mode == "token-mean":
        return masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        return torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)
        return torch.mean(seq_losses)
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def compute_policy_loss_future_kl_torch(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config,
    loss_agg_mode: str = "token-mean",
    future_kl_mod=None,
):
    """
    Torch implementation of FIPO policy loss.
    Replicates compute_policy_loss_future_kl from core_algos.py.
    """
    # Keep the policy-loss math in fp32 even when upstream log-probs come from bf16 kernels.
    old_log_prob = old_log_prob.float()
    log_prob = log_prob.float()
    advantages = advantages.float()
    response_mask = response_mask.float()

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    batch_size, response_len = log_prob.shape
    device = log_prob.device

    chunk_size = config.policy_loss.chunk_size
    decay_rate = config.policy_loss.decay_rate
    gamma = 2 ** (-1.0 / decay_rate)
    future_kl_impl = config.policy_loss.future_kl_impl

    # Compute future kl using negative_approx_kl and response_mask
    filter_threshold = torch.log(torch.tensor(clip_ratio_c, device=device, dtype=log_prob.dtype))
    is_negative_adv = (advantages < 0)
    ignore_mask = negative_approx_kl > filter_threshold
    participation_mask = ~ignore_mask
    kl_response_premask = negative_approx_kl * response_mask.to(log_prob.dtype)
    kl_response = kl_response_premask * participation_mask.to(log_prob.dtype)

    # Compute future_kl
    _compute_future_kl = future_kl_mod.compute_future_kl
    if future_kl_impl == "torch" or (future_kl_impl == "auto" and not torch.cuda.is_available()):
        future_kl = _compute_future_kl(
            kl_response=kl_response,
            gamma=gamma,
            impl="torch",
            chunk_size=chunk_size,
        )
    else:
        future_kl = _compute_future_kl(
            kl_response=kl_response,
            gamma=gamma,
            impl="triton",
        )

    clip_ratio_val = config.policy_loss.future_kl_clip_ratio
    clip_high_only = config.policy_loss.future_kl_clip_high_only
    safe_threshold = config.policy_loss.safety_thresh

    _compute_influence_weights = future_kl_mod.compute_influence_weights
    raw_influence_weights, influence_weights, lower_bound, upper_bound = _compute_influence_weights(
        future_kl=future_kl,
        advantages=advantages,
        ratio=ratio,
        clip_ratio=clip_ratio_val,
        clip_high_only=clip_high_only,
        safe_threshold=safe_threshold,
        impl="torch",  # always use torch for this benchmark comparison
    )
    influence_weights = influence_weights.detach()
    raw_influence_weights = raw_influence_weights.detach()

    clip_frac_upper = masked_mean((influence_weights >= upper_bound - 1e-7).float(), response_mask)
    clip_frac_lower = masked_mean((influence_weights <= lower_bound + 1e-7).float(), response_mask)
    total_clip_frac = clip_frac_upper + clip_frac_lower

    influence_weights_mean_raw = masked_mean(raw_influence_weights, response_mask)
    valid_vals_raw = raw_influence_weights[response_mask.to(dtype=torch.bool, device=influence_weights.device)]
    raw_influence_weights_min = valid_vals_raw.min()
    raw_influence_weights_max = valid_vals_raw.max()

    influence_weights_mean = masked_mean(influence_weights, response_mask)
    valid_vals = influence_weights[response_mask.to(dtype=torch.bool, device=influence_weights.device)]
    influence_weights_min = valid_vals.min()
    influence_weights_max = valid_vals.max()

    weighted_advantages = advantages * influence_weights

    pg_losses1 = -weighted_advantages * ratio
    pg_losses2 = -weighted_advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -weighted_advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    lower_clip_mask = (
        (advantages < 0) &
        (clip_pg_losses1 > pg_losses3) &
        response_mask.bool()
    )
    low_clip_token_counts = lower_clip_mask.sum(dim=1)

    seq_has_low_clip = (low_clip_token_counts > 1)
    seq_valid_mask = (~seq_has_low_clip).unsqueeze(1)

    final_mask = response_mask.bool() & seq_valid_mask
    final_mask_f = final_mask.to(log_prob.dtype)

    pg_losses = torch.where(weighted_advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=final_mask_f, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def main():
    parser = argparse.ArgumentParser(description="End-to-end FIPO loss benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-impls", action="store_true", default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Load kernel modules directly
    repo_root = Path(__file__).resolve().parents[1]
    future_kl_mod = _load_module_from_path(
        "future_kl", repo_root / "verl" / "utils" / "kernel" / "future_kl.py"
    )
    compute_future_kl = future_kl_mod.compute_future_kl
    compute_influence_weights = future_kl_mod.compute_influence_weights
    compute_influence_weights_torch = future_kl_mod.compute_influence_weights_torch
    HAVE_TRITON = future_kl_mod.HAVE_TRITON

    dtype = getattr(torch, args.dtype)

    # Create synthetic data matching production shapes
    batch_size = args.batch_size
    response_len = args.response_len

    # Inputs for compute_policy_loss_future_kl
    old_log_prob = torch.randn(batch_size, response_len, device=device, dtype=dtype)
    log_prob = torch.randn(batch_size, response_len, device=device, dtype=dtype)
    advantages = torch.randn(batch_size, response_len, device=device, dtype=dtype)
    response_mask = (torch.rand(batch_size, response_len, device=device) > 0.1).float()

    print(f"Shape: batch={batch_size}, response_len={response_len}, dtype={dtype}")
    print(f"Triton available: {HAVE_TRITON}")
    print()

    # Create config objects
    class PolicyLossConfig:
        chunk_size = 128
        decay_rate = 32.0
        future_kl_impl = "auto"
        future_kl_clip_ratio = 0.0
        future_kl_clip_high_only = False
        safety_thresh = 4.0

    class MockConfig:
        clip_ratio = 0.2
        clip_ratio_low = None
        clip_ratio_high = None
        policy_loss = PolicyLossConfig()

        def get(self, key, default=None):
            return getattr(self, key, default)

    config = MockConfig()

    # Benchmark the full torch path
    def torch_full_path():
        compute_policy_loss_future_kl_torch(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=config,
            loss_agg_mode="token-mean",
            future_kl_mod=future_kl_mod,
        )

    # Warmup
    print("Warming up torch full path...")
    for _ in range(args.warmup):
        torch_full_path()
    torch.cuda.synchronize()

    print()
    print("Running full FIPO loss benchmark...")

    timings_full = _sync_timing(torch_full_path, warmup=0, iters=args.iters, device=device)
    _summarize("fipo_loss/full_torch", timings_full)

    # Also benchmark the individual hotspots
    print()
    print("--- Individual Hotspot Breakdown ---")

    # Create inputs for individual benchmarking
    kl_response = torch.randn(batch_size, response_len, device=device, dtype=torch.float32)
    future_kl = torch.randn(batch_size, response_len, device=device, dtype=torch.float32)
    ratio = torch.exp(0.1 * torch.randn(batch_size, response_len, device=device, dtype=torch.float32))

    gamma = 2 ** (-1.0 / config.policy_loss.decay_rate)

    # future_kl torch benchmark
    def future_kl_torch_path():
        compute_future_kl(kl_response, gamma, impl="torch", chunk_size=config.policy_loss.chunk_size)

    # future_kl triton benchmark
    def future_kl_triton_path():
        compute_future_kl(kl_response, gamma, impl="triton")

    # Warmup
    for _ in range(args.warmup):
        future_kl_torch_path()
    torch.cuda.synchronize()

    if HAVE_TRITON:
        for _ in range(args.warmup):
            future_kl_triton_path()
        torch.cuda.synchronize()

    timings_fkl_torch = _sync_timing(future_kl_torch_path, warmup=0, iters=args.iters, device=device)
    _summarize("future_kl/torch", timings_fkl_torch)

    if HAVE_TRITON:
        timings_fkl_triton = _sync_timing(future_kl_triton_path, warmup=0, iters=args.iters, device=device)
        _summarize("future_kl/triton", timings_fkl_triton)
        print(f"future_kl/speedup: {mean(timings_fkl_torch) / mean(timings_fkl_triton):.2f}x")

    # influence_weights torch benchmark
    def influence_torch_path():
        compute_influence_weights(
            future_kl=future_kl,
            advantages=advantages.float(),
            ratio=ratio,
            clip_ratio=0.2,
            clip_high_only=True,
            safe_threshold=4.0,
            impl="torch",
        )

    # influence_weights triton benchmark
    def influence_triton_path():
        compute_influence_weights(
            future_kl=future_kl,
            advantages=advantages.float(),
            ratio=ratio,
            clip_ratio=0.2,
            clip_high_only=True,
            safe_threshold=4.0,
            impl="triton",
        )

    # Warmup
    for _ in range(args.warmup):
        influence_torch_path()
    torch.cuda.synchronize()

    if HAVE_TRITON:
        for _ in range(args.warmup):
            influence_triton_path()
        torch.cuda.synchronize()

    timings_inf_torch = _sync_timing(influence_torch_path, warmup=0, iters=args.iters, device=device)
    _summarize("influence_weights/torch", timings_inf_torch)

    if HAVE_TRITON:
        timings_inf_triton = _sync_timing(influence_triton_path, warmup=0, iters=args.iters, device=device)
        _summarize("influence_weights/triton", timings_inf_triton)
        print(f"influence_weights/speedup: {mean(timings_inf_torch) / mean(timings_inf_triton):.2f}x")

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()