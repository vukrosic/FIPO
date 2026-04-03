"""Benchmark GMPO (Geo-Mean) policy loss: torch vs triton."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verl.utils.kernel.gmpo_loss import (
    compute_gmpo_loss_torch,
    compute_gmpo_loss_triton,
    HAVE_TRITON,
)


def _bench(fn, *args, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    B, T = args.batch_size, args.response_len
    device = "cuda"

    torch.manual_seed(0)
    lp  = torch.randn(B, T, device=device) * 0.3
    olp = torch.randn(B, T, device=device) * 0.3
    adv = torch.randn(B, T, device=device)
    m   = (torch.rand(B, T, device=device) > 0.1).float()

    print(f"Shape: ({B}, {T})")

    torch_ms = _bench(compute_gmpo_loss_torch, lp, olp, adv, m,
                       warmup=args.warmup, iters=args.iters)
    print(f"  torch:  {torch_ms:.3f} ms")

    if HAVE_TRITON:
        triton_ms = _bench(compute_gmpo_loss_triton, lp, olp, adv, m,
                           warmup=args.warmup, iters=args.iters)
        print(f"  triton: {triton_ms:.3f} ms")
        print(f"  speedup: {torch_ms / triton_ms:.2f}x")

        # Verify correctness
        pg_t, cf_t, kl_t, cfl_t = compute_gmpo_loss_triton(lp, olp, adv, m)
        pg_r, cf_r, kl_r, cfl_r = compute_gmpo_loss_torch(lp, olp, adv, m)
        print(f"  loss match: {torch.allclose(pg_t, pg_r, rtol=1e-3, atol=1e-3)}")
        print(f"  kl match:   {torch.allclose(kl_t, kl_r, rtol=1e-3, atol=1e-3)}")


if __name__ == "__main__":
    main()
