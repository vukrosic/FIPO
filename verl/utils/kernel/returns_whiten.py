"""Fused discounted-returns + advantage whitening (FIPO-040).

REINFORCE++ computes advantages as:

  1. returns = discounted_returns(rewards, mask, gamma)   # reverse scan (B,T)
  2. advantages = masked_whiten(returns, mask)            # normalize (B,T)

This requires 3 memory passes over (B,T):
  Pass 1 – reverse scan   : write returns
  Pass 2 – accumulate stats: read returns, atomic-add to (sum, sum_sq, count)
  Pass 3 – apply whitening : read returns, write advantages

The fused kernel eliminates Pass 2 by accumulating stats DURING the reverse
scan in Pass 1, reducing total memory traffic from 3 reads + 2 writes to
2 reads + 2 writes.

Pass 1 (reverse scan + stats):
  For each batch row (BLOCK_ROWS rows per program):
    reverse-scan to build returns[b,t];
    atomically accumulate (sum, sum_sq, count) from each stored value.

Pass 2 (whitening):
  Read returns, compute (val - mean) * rsqrt(var + eps) * mask.

Public API
----------
compute_returns_and_whiten(
    rewards, mask, gamma, shift_mean=True, impl="auto", return_returns=False
) -> advantages: (B, T) float32
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


ReturnsWhitenImpl = Literal["auto", "torch", "triton"]


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_returns_and_whiten_torch(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    shift_mean: bool = True,
    return_returns: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference: discounted returns then masked whiten.

    Args:
        rewards: (B, T) float32
        mask:    (B, T) float32  (1 for valid tokens, 0 for padding)
        gamma:   discount factor
        shift_mean: if True subtract mean (standard whitening)

    Returns:
        advantages: (B, T) float32
        returns: (B, T) float32 when return_returns=True
    """
    m = mask.float()
    r = rewards.float()

    # Discounted returns via reverse scan
    returns = torch.zeros_like(r)
    carry = torch.zeros(r.shape[0], device=r.device, dtype=r.dtype)
    for step in range(r.shape[1]):
        col = r.shape[1] - step - 1
        carry = r[:, col] + gamma * carry
        returns[:, col] = carry
        carry = carry * m[:, col]

    # Masked whiten
    mask_sum = m.sum()
    if mask_sum <= 1:
        advantages = returns * m
        if return_returns:
            return advantages, returns
        return advantages

    mean = (returns * m).sum() / (mask_sum + 1e-8)
    centered = returns - mean
    var = (centered * centered * m).sum() / (mask_sum + 1e-8)
    var = var * mask_sum / (mask_sum - 1)  # Bessel's correction
    whitened = (returns - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    advantages = whitened * m
    if return_returns:
        return advantages, returns
    return advantages


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=1),
        ],
        key=["T"],
    )
    @triton.jit
    def _returns_scan_and_accum(
        rewards_ptr,       # (B, T)
        mask_ptr,          # (B, T)
        returns_ptr,       # (B, T) output buffer
        sum_ptr,           # scalar accumulator
        sum_sq_ptr,        # scalar accumulator
        count_ptr,         # scalar accumulator
        B: tl.int32,
        T: tl.int32,
        stride_br: tl.int64,   # row stride in rewards
        stride_bm: tl.int64,   # row stride in mask
        stride_bo: tl.int64,   # row stride in returns
        gamma: tl.float32,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Reverse-scan returns; accumulate whitening stats with ONE atomic_add per block.

        Accumulates (sum, sum_sq, count) locally in registers during the scan
        and flushes with a single atomic_add after the loop — avoids O(T) atomics.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_valid = rows < B

        carry = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
        # Local accumulators — flushed once after the loop
        local_sum    = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
        local_sum_sq = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
        local_count  = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

        for step in range(T):
            col = T - step - 1

            r = tl.load(
                rewards_ptr + rows * stride_br + col,
                mask=row_valid, other=0.0,
            ).to(tl.float32)
            m = tl.load(
                mask_ptr + rows * stride_bm + col,
                mask=row_valid, other=0.0,
            ).to(tl.float32)

            carry = r + gamma * carry

            tl.store(
                returns_ptr + rows * stride_bo + col,
                carry,
                mask=row_valid,
            )

            # Local accumulation (no atomics — just register adds)
            masked_val = carry * m
            local_sum    += masked_val
            local_sum_sq += masked_val * masked_val
            local_count  += m

            carry = carry * m

        # Single atomic_add per block after the loop (BLOCK_ROWS → 1 scalar)
        tl.atomic_add(sum_ptr,    tl.sum(local_sum,    axis=0))
        tl.atomic_add(sum_sq_ptr, tl.sum(local_sum_sq, axis=0))
        tl.atomic_add(count_ptr,  tl.sum(local_count,  axis=0))

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
    def _whiten_apply(
        returns_ptr,
        mask_ptr,
        output_ptr,
        N: tl.int32,
        mean: tl.float32,
        rsqrt_var: tl.float32,
        shift_mean: tl.int32,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        valid = off < N

        v = tl.load(returns_ptr + off, mask=valid, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr   + off, mask=valid, other=0.0).to(tl.float32)

        whitened = (v - mean) * rsqrt_var
        if shift_mean == 0:
            whitened = whitened + mean

        tl.store(output_ptr + off, whitened * m, mask=valid)


def compute_returns_and_whiten_triton(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    shift_mean: bool = True,
    return_returns: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused reverse-scan + whitening (FIPO-040).

    Args:
        rewards: (B, T) float32 CUDA tensor
        mask:    (B, T) float32 CUDA tensor
        gamma:   discount factor

    Returns:
        advantages: (B, T) float32
        returns: (B, T) float32 when return_returns=True
    """
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available.")
    if not rewards.is_cuda or not mask.is_cuda:
        raise RuntimeError("Triton returns+whiten requires CUDA tensors.")

    rewards_f = rewards.float().contiguous()
    mask_f = mask.float().contiguous()
    B, T = rewards_f.shape

    returns = torch.empty_like(rewards_f)
    sum_out    = torch.zeros((), device=rewards.device, dtype=torch.float32)
    sum_sq_out = torch.zeros((), device=rewards.device, dtype=torch.float32)
    count_out  = torch.zeros((), device=rewards.device, dtype=torch.float32)

    grid_scan = lambda meta: (triton.cdiv(B, meta["BLOCK_ROWS"]),)
    _returns_scan_and_accum[grid_scan](
        rewards_f, mask_f, returns,
        sum_out, sum_sq_out, count_out,
        B, T,
        rewards_f.stride(0), mask_f.stride(0), returns.stride(0),
        float(gamma),
    )

    count = count_out.item()
    if count <= 1:
        advantages = returns * mask_f
        if return_returns:
            return advantages, returns
        return advantages

    s  = sum_out.item()
    sq = sum_sq_out.item()
    mean   = s  / (count + 1e-8)
    # Bessel's correction: var_biased * n/(n-1)
    var_biased = sq / (count + 1e-8) - mean * mean
    var        = var_biased * count / (count - 1)
    rsqrt_var  = 1.0 / (var + 1e-8) ** 0.5

    N = B * T
    ret_flat  = returns.reshape(-1)
    mask_flat = mask_f.reshape(-1)
    output    = torch.empty(N, device=rewards.device, dtype=torch.float32)

    grid_apply = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _whiten_apply[grid_apply](
        ret_flat, mask_flat, output, N,
        float(mean), float(rsqrt_var), int(shift_mean),
    )

    advantages = output.reshape(B, T)
    if return_returns:
        return advantages, returns
    return advantages


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_returns_and_whiten(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    shift_mean: bool = True,
    impl: ReturnsWhitenImpl = "auto",
    return_returns: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Discounted returns followed by advantage whitening (fused).

    Computes REINFORCE++ advantages in 2 GPU passes instead of 3:
      Pass 1 — reverse scan writes returns + accumulates whitening statistics
      Pass 2 — applies (val - mean) * rsqrt(var) * mask

    Args:
        rewards:    (B, T) float32 or bfloat16
        mask:       (B, T) float32  (1=valid token, 0=padding)
        gamma:      discount factor (e.g. 0.99)
        shift_mean: if True zero-mean the advantages (standard whitening)
        impl:       "auto" | "torch" | "triton"

    Returns:
        advantages: (B, T) float32
        returns: (B, T) float32 when return_returns=True
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and rewards.is_cuda else "torch"

    if resolved == "triton":
        return compute_returns_and_whiten_triton(
            rewards,
            mask,
            gamma,
            shift_mean,
            return_returns=return_returns,
        )
    return compute_returns_and_whiten_torch(
        rewards,
        mask,
        gamma,
        shift_mean,
        return_returns=return_returns,
    )
