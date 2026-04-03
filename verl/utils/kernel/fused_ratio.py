"""Fused ratio + KL computation kernel (FIPO-037).

Every PPO-family policy loss starts with the same three operations:
  neg_kl = log_prob - old_log_prob
  ratio  = exp(neg_kl)
  ppo_kl = masked_mean(-neg_kl, mask)

This materializes both neg_kl (B,T) and ratio (B,T).  The fused kernel
computes ratio in-place and ppo_kl as a by-product of a single scan, avoiding
the separate neg_kl allocation and the masked_mean kernel launch.

Also provides a variant that clamps neg_kl before exponentiation (used by
vanilla PPO: clamp(neg_kl, -20, 20)).

Public API
----------
compute_fused_ratio(log_prob, old_log_prob, response_mask, clamp_min=-20.0,
                    clamp_max=20.0, impl="auto")
    -> (ratio: (B,T), ppo_kl: scalar)
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


FusedRatioImpl = Literal["auto", "torch", "triton"]


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_fused_ratio_torch(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    clamp_min: float = -20.0,
    clamp_max: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: compute ratio and ppo_kl."""
    neg_kl = log_prob - old_log_prob
    neg_kl_clamped = torch.clamp(neg_kl, min=clamp_min, max=clamp_max)
    ratio = torch.exp(neg_kl_clamped)
    mask = response_mask.float()
    ppo_kl = (-neg_kl * mask).sum() / (mask.sum() + 1e-8)
    return ratio, ppo_kl


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
    def _fused_ratio_kernel(
        lp_ptr,
        olp_ptr,
        mask_ptr,
        ratio_ptr,           # output (N,)
        kl_sum_ptr,          # output scalar (atomic)
        mask_sum_ptr,        # output scalar (atomic)
        N: tl.int32,
        clamp_min: tl.float32,
        clamp_max: tl.float32,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        valid = off < N

        lp  = tl.load(lp_ptr  + off, mask=valid, other=0.0).to(tl.float32)
        olp = tl.load(olp_ptr + off, mask=valid, other=0.0).to(tl.float32)
        msk = tl.load(mask_ptr + off, mask=valid, other=0.0).to(tl.float32)

        neg_kl = lp - olp
        neg_kl_clamped = tl.minimum(tl.maximum(neg_kl, clamp_min), clamp_max)
        ratio = tl.exp(neg_kl_clamped)

        tl.store(ratio_ptr + off, ratio, mask=valid)

        # Accumulate KL and mask sums
        local_kl = tl.sum((-neg_kl) * msk * valid.to(tl.float32), axis=0)
        local_mask = tl.sum(msk * valid.to(tl.float32), axis=0)
        tl.atomic_add(kl_sum_ptr, local_kl)
        tl.atomic_add(mask_sum_ptr, local_mask)


def compute_fused_ratio_triton(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    clamp_min: float = -20.0,
    clamp_max: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available.")

    lp  = log_prob.float().contiguous().reshape(-1)
    olp = old_log_prob.float().contiguous().reshape(-1)
    msk = response_mask.float().contiguous().reshape(-1)
    N = lp.numel()

    ratio = torch.empty_like(lp)
    kl_sum = torch.zeros((), device=lp.device, dtype=torch.float32)
    mask_sum = torch.zeros((), device=lp.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _fused_ratio_kernel[grid](
        lp, olp, msk, ratio, kl_sum, mask_sum,
        N, float(clamp_min), float(clamp_max),
    )

    ppo_kl = kl_sum / (mask_sum + 1e-8)
    return ratio.reshape(log_prob.shape), ppo_kl


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_fused_ratio(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    clamp_min: float = -20.0,
    clamp_max: float = 20.0,
    impl: FusedRatioImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute importance ratio and token-mean KL in one pass.

    Returns:
        ratio:  (B, T) float32 — exp(clamp(log_prob - old_log_prob))
        ppo_kl: scalar float32 — masked_mean(-(log_prob - old_log_prob))
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and log_prob.is_cuda else "torch"

    if resolved == "triton":
        return compute_fused_ratio_triton(log_prob, old_log_prob, response_mask, clamp_min, clamp_max)
    return compute_fused_ratio_torch(log_prob, old_log_prob, response_mask, clamp_min, clamp_max)
