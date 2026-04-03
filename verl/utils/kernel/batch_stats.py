"""Fused batch statistics kernel (FIPO-035).

compute_data_metrics in metric_utils.py calls torch.mean(), torch.max(),
torch.min() separately for 6+ categories — each launches a separate CUDA
kernel.  This fuses mean/max/min into a single pass.

For 1-D tensors (the common case in metrics), a single Triton kernel computes
all three statistics in one scan.

Public API
----------
compute_batch_stats(x, impl="auto") -> (mean, max_val, min_val)
compute_masked_batch_stats(x, mask, impl="auto") -> (mean, max_val, min_val)
"""

from __future__ import annotations

from typing import Literal

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


BatchStatsImpl = Literal["auto", "torch", "triton"]


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_batch_stats_torch(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mean, max, min in separate passes (torch baseline)."""
    return torch.mean(x), torch.max(x), torch.min(x)


def compute_masked_batch_stats_torch(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute masked mean, max, min (torch baseline)."""
    valid = x[mask.bool()]
    if valid.numel() == 0:
        z = torch.tensor(0.0, device=x.device)
        return z, z, z
    return torch.mean(valid), torch.max(valid), torch.min(valid)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=2),
        ],
        key=["N"],
    )
    @triton.jit
    def _batch_stats_kernel(
        x_ptr,
        sum_ptr,
        max_ptr,
        min_ptr,
        count_ptr,
        N: tl.int32,
        BLOCK: tl.constexpr,
    ):
        """Single program computing sum, max, min over flat tensor."""
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        valid = off < N

        x = tl.load(x_ptr + off, mask=valid, other=0.0).to(tl.float32)

        # For max/min, use -inf/inf for invalid positions
        x_for_max = tl.where(valid, x, float('-inf'))
        x_for_min = tl.where(valid, x, float('inf'))

        local_sum = tl.sum(x * valid.to(tl.float32), axis=0)
        local_max = tl.max(x_for_max, axis=0)
        local_min = tl.min(x_for_min, axis=0)
        local_count = tl.sum(valid.to(tl.float32), axis=0)

        tl.atomic_add(sum_ptr, local_sum)
        tl.atomic_max(max_ptr, local_max)
        tl.atomic_min(min_ptr, local_min)
        tl.atomic_add(count_ptr, local_count)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 256},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 512},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
        ],
        key=["N"],
    )
    @triton.jit
    def _masked_batch_stats_kernel(
        x_ptr,
        mask_ptr,
        sum_ptr,
        max_ptr,
        min_ptr,
        count_ptr,
        N: tl.int32,
        BLOCK: tl.constexpr,
    ):
        """Single program computing masked sum, max, min."""
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        valid = off < N

        x = tl.load(x_ptr + off, mask=valid, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr + off, mask=valid, other=0.0).to(tl.float32)
        active = (m > 0.0) & valid

        x_masked = x * active.to(tl.float32)
        x_for_max = tl.where(active, x, float('-inf'))
        x_for_min = tl.where(active, x, float('inf'))

        local_sum = tl.sum(x_masked, axis=0)
        local_max = tl.max(x_for_max, axis=0)
        local_min = tl.min(x_for_min, axis=0)
        local_count = tl.sum(active.to(tl.float32), axis=0)

        tl.atomic_add(sum_ptr, local_sum)
        tl.atomic_max(max_ptr, local_max)
        tl.atomic_min(min_ptr, local_min)
        tl.atomic_add(count_ptr, local_count)


def compute_batch_stats_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_flat = x.float().contiguous().reshape(-1)
    N = x_flat.numel()

    sum_out = torch.zeros((), device=x.device, dtype=torch.float32)
    max_out = torch.full((), float('-inf'), device=x.device, dtype=torch.float32)
    min_out = torch.full((), float('inf'), device=x.device, dtype=torch.float32)
    count_out = torch.zeros((), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _batch_stats_kernel[grid](x_flat, sum_out, max_out, min_out, count_out, N)

    mean_out = sum_out / (count_out + 1e-8)
    return mean_out, max_out, min_out


def compute_masked_batch_stats_triton(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_flat = x.float().contiguous().reshape(-1)
    m_flat = mask.float().contiguous().reshape(-1)
    N = x_flat.numel()

    sum_out = torch.zeros((), device=x.device, dtype=torch.float32)
    max_out = torch.full((), float('-inf'), device=x.device, dtype=torch.float32)
    min_out = torch.full((), float('inf'), device=x.device, dtype=torch.float32)
    count_out = torch.zeros((), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _masked_batch_stats_kernel[grid](x_flat, m_flat, sum_out, max_out, min_out, count_out, N)

    count = count_out.item()
    if count == 0:
        z = torch.tensor(0.0, device=x.device)
        return z, z, z

    mean_out = sum_out / count_out
    return mean_out, max_out, min_out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_batch_stats(
    x: torch.Tensor,
    impl: BatchStatsImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mean, max, min in a single pass.

    Returns: (mean, max_val, min_val) as scalar float32 tensors.
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and x.is_cuda else "torch"
    if resolved == "triton":
        return compute_batch_stats_triton(x)
    return compute_batch_stats_torch(x)


def compute_masked_batch_stats(
    x: torch.Tensor,
    mask: torch.Tensor,
    impl: BatchStatsImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute masked mean, max, min in a single pass.

    Returns: (mean, max_val, min_val) as scalar float32 tensors.
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and x.is_cuda else "torch"
    if resolved == "triton":
        return compute_masked_batch_stats_triton(x, mask)
    return compute_masked_batch_stats_torch(x, mask)
