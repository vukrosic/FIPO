"""Fused GPG policy loss kernel (FIPO-036).

GPG (Group Policy Gradient) is the simplest policy loss variant:
  pg_losses = -log_prob * advantages
  pg_loss = agg_loss(pg_losses, mask, mode)

This fuses the multiplication and aggregation into a single Triton kernel,
avoiding the intermediate (B, T) pg_losses tensor allocation.

Public API
----------
compute_gpg_loss(log_prob, advantages, response_mask, loss_agg_mode="token-mean",
                 impl="auto") -> (pg_loss, 0, 0, 0)
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


GpgImpl = Literal["auto", "torch", "triton"]


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_gpg_loss_torch(
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation matching core_algos.compute_policy_loss_gpg."""
    mask = response_mask.float()
    pg_losses = -log_prob * advantages

    if loss_agg_mode == "token-mean":
        pg_loss = (pg_losses * mask).sum() / (mask.sum() + 1e-8)
    elif loss_agg_mode == "seq-mean-token-sum":
        pg_loss = (pg_losses * mask).sum(dim=-1).mean()
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_sum = (pg_losses * mask).sum(dim=-1)
        seq_count = mask.sum(dim=-1)
        pg_loss = (seq_sum / seq_count).mean()
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        pg_loss = (pg_losses * mask).sum() / pg_losses.shape[-1]
    else:
        raise ValueError(f"Unknown mode: {loss_agg_mode}")

    z = torch.tensor(0.0, device=pg_loss.device)
    return pg_loss, z, z, z


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_T": 256},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK_T": 512},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK_T": 1024}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_T": 2048}, num_warps=8, num_stages=2),
        ],
        key=["T"],
    )
    @triton.jit
    def _gpg_loss_kernel(
        lp_ptr,
        adv_ptr,
        mask_ptr,
        # per-sequence outputs (B,)
        seq_sum_ptr,
        seq_count_ptr,
        # scalars
        B: tl.int32,
        T: tl.int32,
        stride_b: tl.int64,
        BLOCK_T: tl.constexpr,
    ):
        """One program per batch element: fused -lp*adv*mask sum."""
        bid = tl.program_id(0)
        if bid >= B:
            return

        row = bid * stride_b
        num_chunks = tl.cdiv(T, BLOCK_T)

        acc_sum = 0.0
        acc_count = 0.0

        for chunk in range(num_chunks):
            t_off = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
            valid = t_off < T
            lp  = tl.load(lp_ptr   + row + t_off, mask=valid, other=0.0).to(tl.float32)
            adv = tl.load(adv_ptr  + row + t_off, mask=valid, other=0.0).to(tl.float32)
            msk = tl.load(mask_ptr + row + t_off, mask=valid, other=0.0).to(tl.float32)

            pg = -lp * adv
            acc_sum += tl.sum(pg * msk, axis=0)
            acc_count += tl.sum(msk, axis=0)

        tl.store(seq_sum_ptr + bid, acc_sum)
        tl.store(seq_count_ptr + bid, acc_count)


def compute_gpg_loss_triton(
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available.")

    B, T = log_prob.shape
    lp  = log_prob.float().contiguous()
    adv = advantages.float().contiguous()
    msk = response_mask.float().contiguous()
    assert lp.stride(0) == adv.stride(0) == msk.stride(0)

    seq_sums   = torch.empty(B, device=log_prob.device, dtype=torch.float32)
    seq_counts = torch.empty(B, device=log_prob.device, dtype=torch.float32)

    grid = lambda meta: (B,)
    _gpg_loss_kernel[grid](
        lp, adv, msk,
        seq_sums, seq_counts,
        B, T, lp.stride(0),
    )

    if loss_agg_mode == "token-mean":
        pg_loss = seq_sums.sum() / (seq_counts.sum() + 1e-8)
    elif loss_agg_mode == "seq-mean-token-sum":
        pg_loss = seq_sums.mean()
    elif loss_agg_mode == "seq-mean-token-mean":
        pg_loss = (seq_sums / seq_counts).mean()
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        pg_loss = seq_sums.sum() / T
    else:
        raise ValueError(f"Unknown mode: {loss_agg_mode}")

    z = torch.tensor(0.0, device=pg_loss.device)
    return pg_loss, z, z, z


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_gpg_loss(
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    impl: GpgImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused GPG policy loss.

    Returns:
        pg_loss:           scalar float32
        pg_clipfrac:       scalar float32 (always 0)
        ppo_kl:            scalar float32 (always 0)
        pg_clipfrac_lower: scalar float32 (always 0)
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and log_prob.is_cuda else "torch"

    if resolved == "triton":
        return compute_gpg_loss_triton(log_prob, advantages, response_mask, loss_agg_mode)
    return compute_gpg_loss_torch(log_prob, advantages, response_mask, loss_agg_mode)
