"""Fused advantage normalization kernel (FIPO-038).

In REINFORCE++ and other algorithms, after computing advantages the pipeline
does:
  advantages = masked_whiten(returns, mask)  # zero-mean, unit-variance
  advantages = advantages * mask             # zero out padding

This requires 3 passes: (1) mean, (2) variance, (3) apply whitening.
The fused Triton kernel does it in 2 passes: (1) accumulate sum/sum_sq/count,
(2) apply whitening + mask in one scan.

For the case where advantages are already computed and just need normalization
(e.g., GRPO with norm_adv_by_std), this avoids the intermediate tensor.

Public API
----------
compute_fused_advantage_norm(values, mask, shift_mean=True, impl="auto")
    -> normalized: (B, T) float32, already multiplied by mask
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


AdvNormImpl = Literal["auto", "torch", "triton"]


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_fused_advantage_norm_torch(
    values: torch.Tensor,
    mask: torch.Tensor,
    shift_mean: bool = True,
) -> torch.Tensor:
    """Reference: whiten + mask multiply."""
    m = mask.float()
    mask_sum = m.sum()
    if mask_sum <= 1:
        return values * m

    mean = (values * m).sum() / (mask_sum + 1e-8)
    centered = values - mean
    var = (centered * centered * m).sum() / (mask_sum + 1e-8)
    # Bessel's correction
    var = var * mask_sum / (mask_sum - 1)

    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened * m


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
        ],
        key=["N"],
    )
    @triton.jit
    def _adv_norm_accum(
        values_ptr,
        mask_ptr,
        sum_ptr,
        sum_sq_ptr,
        count_ptr,
        N: tl.int32,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        valid = off < N

        v = tl.load(values_ptr + off, mask=valid, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr + off, mask=valid, other=0.0).to(tl.float32)

        weighted = v * m
        tl.atomic_add(sum_ptr, tl.sum(weighted, axis=0))
        tl.atomic_add(sum_sq_ptr, tl.sum(v * v * m, axis=0))
        tl.atomic_add(count_ptr, tl.sum(m, axis=0))

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
    def _adv_norm_apply(
        values_ptr,
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

        v = tl.load(values_ptr + off, mask=valid, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr + off, mask=valid, other=0.0).to(tl.float32)

        whitened = (v - mean) * rsqrt_var
        if shift_mean == 0:
            whitened = whitened + mean
        result = whitened * m

        tl.store(output_ptr + off, result, mask=valid)


def compute_fused_advantage_norm_triton(
    values: torch.Tensor,
    mask: torch.Tensor,
    shift_mean: bool = True,
) -> torch.Tensor:
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available.")

    v_flat = values.float().contiguous().reshape(-1)
    m_flat = mask.float().contiguous().reshape(-1)
    N = v_flat.numel()

    sum_out = torch.zeros((), device=values.device, dtype=torch.float32)
    sum_sq_out = torch.zeros((), device=values.device, dtype=torch.float32)
    count_out = torch.zeros((), device=values.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _adv_norm_accum[grid](v_flat, m_flat, sum_out, sum_sq_out, count_out, N)

    count = count_out.item()
    if count <= 1:
        return values * mask.float()

    s = sum_out.item()
    sq = sum_sq_out.item()
    mean = s / (count + 1e-8)
    var = (sq / (count + 1e-8)) - mean * mean
    # Bessel's correction
    var = var * count / (count - 1)
    rsqrt_var = 1.0 / (var + 1e-8) ** 0.5

    output = torch.empty_like(v_flat)
    _adv_norm_apply[grid](v_flat, m_flat, output, N, mean, rsqrt_var, int(shift_mean))

    return output.reshape(values.shape)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_fused_advantage_norm(
    values: torch.Tensor,
    mask: torch.Tensor,
    shift_mean: bool = True,
    impl: AdvNormImpl = "auto",
) -> torch.Tensor:
    """Fused advantage normalization (whiten + mask).

    Returns: (B, T) float32, whitened and masked.
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and values.is_cuda else "torch"

    if resolved == "triton":
        return compute_fused_advantage_norm_triton(values, mask, shift_mean)
    return compute_fused_advantage_norm_torch(values, mask, shift_mean)
