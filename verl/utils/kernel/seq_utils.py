"""Fused sequence-level reduction kernels (FIPO-028).

Operations like summing / averaging token-level quantities along the sequence
axis (with a response mask) appear throughout PPO/GRPO/GSPO training.  Each
naive call launches 2-3 separate CUDA kernels; this module fuses them into one.

Provided functions
------------------
compute_seq_logprob(log_probs, response_mask, impl="auto")
    -> (seq_logprob_sum, seq_lengths)
    Computes per-sequence log-probability sums and sequence lengths in one pass.
    Useful for GSPO, Geo-Mean PPO, and other sequence-level objective variants.

compute_seq_mean(values, response_mask, impl="auto")
    -> (seq_means,)  shape (B,)
    Per-sequence masked mean: sum(values * mask, dim=-1) / sum(mask, dim=-1).

Both return float32 regardless of input dtype.
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


SeqUtilsImpl = Literal["auto", "torch", "triton"]


# ---------------------------------------------------------------------------
# Torch references
# ---------------------------------------------------------------------------

def compute_seq_logprob_torch(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence sum of log-probs and sequence lengths.

    Args:
        log_probs:      (B, T) float32 or bfloat16
        response_mask:  (B, T) float32  0/1 mask

    Returns:
        seq_lp_sum: (B,) float32 – sum(log_probs * mask, dim=-1)
        seq_len:    (B,) float32 – sum(mask, dim=-1).clamp(min=1)
    """
    lp = log_probs.float()
    mask = response_mask.float()
    seq_lp_sum = (lp * mask).sum(dim=-1)
    seq_len    = mask.sum(dim=-1).clamp(min=1.0)
    return seq_lp_sum, seq_len


def compute_seq_mean_torch(
    values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sequence masked mean.

    Args:
        values:         (B, T) float32 or bfloat16
        response_mask:  (B, T) float32

    Returns:
        seq_means: (B,) float32
    """
    v    = values.float()
    mask = response_mask.float()
    return (v * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Triton kernels
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
    def _seq_logprob_kernel(
        lp_ptr,           # (B, T)
        mask_ptr,         # (B, T)
        lp_sum_ptr,       # (B,) output
        seq_len_ptr,      # (B,) output
        B: tl.int32,
        T: tl.int32,
        stride_b_lp: tl.int64,
        stride_b_mask: tl.int64,
        BLOCK_T: tl.constexpr,
    ):
        """One program per batch element; streams over T in BLOCK_T chunks."""
        bid = tl.program_id(0)
        if bid >= B:
            return

        lp_base   = lp_ptr   + bid * stride_b_lp
        mask_base = mask_ptr + bid * stride_b_mask

        acc_lp  = 0.0
        acc_len = 0.0

        num_chunks = tl.cdiv(T, BLOCK_T)
        for chunk in range(num_chunks):
            t_offsets = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
            valid = t_offsets < T

            lp   = tl.load(lp_base   + t_offsets, mask=valid, other=0.0).to(tl.float32)
            mask = tl.load(mask_base + t_offsets, mask=valid, other=0.0).to(tl.float32)

            acc_lp  = acc_lp  + tl.sum(lp * mask, axis=0)
            acc_len = acc_len + tl.sum(mask,       axis=0)

        tl.store(lp_sum_ptr  + bid, acc_lp)
        tl.store(seq_len_ptr + bid, tl.maximum(acc_len, 1.0))


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
    def _seq_mean_kernel(
        values_ptr,       # (B, T)
        mask_ptr,         # (B, T)
        out_ptr,          # (B,)
        B: tl.int32,
        T: tl.int32,
        stride_b_v: tl.int64,
        stride_b_m: tl.int64,
        BLOCK_T: tl.constexpr,
    ):
        bid = tl.program_id(0)
        if bid >= B:
            return

        v_base    = values_ptr + bid * stride_b_v
        mask_base = mask_ptr   + bid * stride_b_m

        acc_v   = 0.0
        acc_len = 0.0

        num_chunks = tl.cdiv(T, BLOCK_T)
        for chunk in range(num_chunks):
            t_offsets = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
            valid = t_offsets < T

            v    = tl.load(v_base    + t_offsets, mask=valid, other=0.0).to(tl.float32)
            mask = tl.load(mask_base + t_offsets, mask=valid, other=0.0).to(tl.float32)

            acc_v   = acc_v   + tl.sum(v * mask, axis=0)
            acc_len = acc_len + tl.sum(mask,     axis=0)

        tl.store(out_ptr + bid, acc_v / tl.maximum(acc_len, 1.0))


def compute_seq_logprob_triton(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available.")
    if not log_probs.is_cuda or not response_mask.is_cuda:
        raise RuntimeError("Triton seq_logprob requires CUDA tensors.")

    B, T = log_probs.shape
    lp_sum  = torch.empty(B, device=log_probs.device, dtype=torch.float32)
    seq_len = torch.empty(B, device=log_probs.device, dtype=torch.float32)

    grid = lambda meta: (B,)
    _seq_logprob_kernel[grid](
        log_probs.contiguous(),
        response_mask.contiguous(),
        lp_sum,
        seq_len,
        B, T,
        log_probs.stride(0),
        response_mask.stride(0),
    )
    return lp_sum, seq_len


def compute_seq_mean_triton(
    values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available.")
    if not values.is_cuda or not response_mask.is_cuda:
        raise RuntimeError("Triton seq_mean requires CUDA tensors.")

    B, T = values.shape
    out = torch.empty(B, device=values.device, dtype=torch.float32)

    grid = lambda meta: (B,)
    _seq_mean_kernel[grid](
        values.contiguous(),
        response_mask.contiguous(),
        out,
        B, T,
        values.stride(0),
        response_mask.stride(0),
    )
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_seq_logprob(
    log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    impl: SeqUtilsImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence sum of log-probs and lengths.

    Returns:
        seq_lp_sum: (B,) float32
        seq_len:    (B,) float32  (clamped to ≥ 1)
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and log_probs.is_cuda else "torch"

    if resolved == "triton":
        return compute_seq_logprob_triton(log_probs, response_mask)
    return compute_seq_logprob_torch(log_probs, response_mask)


def compute_seq_mean(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    impl: SeqUtilsImpl = "auto",
) -> torch.Tensor:
    """Per-sequence masked mean.

    Returns:
        seq_means: (B,) float32
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and values.is_cuda else "torch"

    if resolved == "triton":
        return compute_seq_mean_triton(values, response_mask)
    return compute_seq_mean_torch(values, response_mask)
