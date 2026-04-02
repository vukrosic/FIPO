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


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ensure_namespace(module_name: str, module_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    module = types.ModuleType(module_name)
    module.__path__ = [str(module_path)]
    sys.modules[module_name] = module
    return module


def _load_linear_ce_modules():
    verl_root = REPO_ROOT / "verl"
    utils_root = verl_root / "utils"
    kernel_root = utils_root / "kernel"

    _ensure_namespace("verl", verl_root)
    _ensure_namespace("verl.utils", utils_root)
    kernel_pkg = _ensure_namespace("verl.utils.kernel", kernel_root)

    device_mod = _load_module("verl.utils.device", utils_root / "device.py")
    kernels_mod = _load_module("verl.utils.kernel.kernels", kernel_root / "kernels.py")
    linear_ce_mod = _load_module("verl.utils.kernel.linear_cross_entropy", kernel_root / "linear_cross_entropy.py")

    sys.modules["verl.utils"].device = device_mod
    kernel_pkg.kernels = kernels_mod
    kernel_pkg.linear_cross_entropy = linear_ce_mod
    return kernels_mod, linear_ce_mod


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


def _parse_int_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _make_inputs(
    *,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    hidden = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype, generator=generator)
    weight = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype, generator=generator)
    labels = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.long, generator=generator)
    dlogprobs = torch.randn(num_tokens, device=device, dtype=torch.float32, generator=generator)
    dentropy = torch.randn(num_tokens, device=device, dtype=torch.float32, generator=generator)
    return hidden, weight, labels, dlogprobs, dentropy


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton linear cross entropy forward/backward paths.")
    parser.add_argument("--num-tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reduction", choices=["none", "sum", "mean"], default="none")
    parser.add_argument("--benchmark", choices=["forward", "backward", "full", "all"], default="all")
    parser.add_argument("--backward-method", choices=["split_n", "total_fuse", "total_separate"], default="split_n")
    parser.add_argument("--forward-vocab-per-split", type=int, default=None)
    parser.add_argument("--backward-vocab-per-split", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--matrix-num-tokens", default="2048,4096")
    parser.add_argument("--matrix-hidden-sizes", default="3584,4096")
    parser.add_argument("--matrix-vocab-sizes", default="32768,38016")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark expects CUDA.")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    kernels_mod, linear_ce_mod = _load_linear_ce_modules()
    linear_cross_entropy = linear_ce_mod.linear_cross_entropy

    backward_method_map = {
        "split_n": kernels_mod.BackwardEnum._Split_Dlogits_N,
        "total_fuse": kernels_mod.BackwardEnum._Total_Fuse_MN,
        "total_separate": kernels_mod.BackwardEnum._Total_Separate,
    }
    kernels_mod.set_backward_method(backward_method_map[args.backward_method])

    if args.forward_vocab_per_split is not None and hasattr(kernels_mod, "set_forward_vocab_per_split"):
        kernels_mod.set_forward_vocab_per_split(args.forward_vocab_per_split)
    if args.backward_vocab_per_split is not None and hasattr(kernels_mod, "set_backward_vocab_per_split"):
        kernels_mod.set_backward_vocab_per_split(args.backward_vocab_per_split)

    shape_list: list[tuple[int, int, int]]
    if args.matrix:
        shape_list = [
            (num_tokens, hidden_size, vocab_size)
            for num_tokens in _parse_int_list(args.matrix_num_tokens)
            for hidden_size in _parse_int_list(args.matrix_hidden_sizes)
            for vocab_size in _parse_int_list(args.matrix_vocab_sizes)
        ]
    else:
        shape_list = [(args.num_tokens, args.hidden_size, args.vocab_size)]

    for shape_idx, (num_tokens, hidden_size, vocab_size) in enumerate(shape_list):
        if hidden_size % 128 != 0:
            raise ValueError(f"hidden_size must be divisible by 128, got {hidden_size}")

        hidden, weight, labels, dlogprobs, dentropy = _make_inputs(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            dtype=dtype,
            device=device,
            seed=args.seed + shape_idx,
        )
        if args.reduction != "none":
            dlogprobs = dlogprobs.mean()
            dentropy = dentropy.mean()

        print(
            f"shape=num_tokens:{num_tokens} hidden_size:{hidden_size} vocab_size:{vocab_size} "
            f"dtype={dtype} backward={args.backward_method} "
            f"forward_split={getattr(kernels_mod._config, '_forward_vocab_per_split', 'na')} "
            f"backward_split={getattr(kernels_mod._config, '_backward_vocab_per_split', 'na')}"
        )

        forward_outputs = kernels_mod.efficient_entropy_forward(
            hidden,
            weight,
            labels,
            kernels_mod.get_entropy_reduction_enum_number(args.reduction),
            args.temperature,
            None,
        )

        logprobs, entropy, maximum, accumulate, entropy_b = forward_outputs
        if not torch.isfinite(logprobs).all() or not torch.isfinite(entropy).all():
            raise AssertionError("Initial forward pass produced non-finite outputs.")

        if args.benchmark in {"forward", "all"}:
            forward_timings = _sync_timing(
                lambda: kernels_mod.efficient_entropy_forward(
                    hidden,
                    weight,
                    labels,
                    kernels_mod.get_entropy_reduction_enum_number(args.reduction),
                    args.temperature,
                    None,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            _summarize("forward", forward_timings)

        if args.benchmark in {"backward", "all"}:
            backward_timings = _sync_timing(
                lambda: kernels_mod.efficient_entropy_backward(
                    dlogprobs,
                    dentropy,
                    hidden,
                    weight,
                    labels,
                    maximum,
                    accumulate,
                    entropy_b,
                    kernels_mod.get_entropy_reduction_enum_number(args.reduction),
                    False,
                    args.temperature,
                    None,
                ),
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            _summarize("backward", backward_timings)

        if args.benchmark in {"full", "all"}:
            hidden_full = hidden.detach().clone().requires_grad_(True)
            weight_full = weight.detach().clone().requires_grad_(True)

            def full_step():
                hidden_full.grad = None
                weight_full.grad = None
                logprobs_full, entropy_full = linear_cross_entropy(
                    hidden_full,
                    weight_full,
                    labels,
                    args.temperature,
                    args.reduction,
                    None,
                )
                if args.reduction == "none":
                    torch.autograd.backward((logprobs_full, entropy_full), (dlogprobs, dentropy))
                else:
                    torch.autograd.backward((logprobs_full, entropy_full), (dlogprobs, dentropy))

            full_timings = _sync_timing(
                full_step,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            _summarize("full", full_timings)

        print()


if __name__ == "__main__":
    main()
