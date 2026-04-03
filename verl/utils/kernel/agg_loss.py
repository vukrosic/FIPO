"""Fused agg_loss kernel (FIPO-034).

The agg_loss function is called by every policy loss variant to reduce a
(B, T) loss matrix to a scalar.  It supports four aggregation modes:

  token-mean:            sum(loss * mask) / sum(mask)
  seq-mean-token-sum:    mean_b( sum_t(loss * mask) )
  seq-mean-token-mean:   mean_b( sum_t(loss * mask) / sum_t(mask) )
  seq-mean-token-sum-norm: sum_b( sum_t(loss * mask) ) / T

Each mode normally requires 2-3 kernel launches (mul, sum, div).  The Triton
kernel does one pass per sequence (B programs), accumulating the masked sum
and mask count, then writes per-sequence results.  The final scalar reduction
(mean over B) is done torch-side since B is small.

Public API
----------
compute_agg_loss(loss_mat, loss_mask, mode, impl="auto") -> scalar Tensor
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


AggLossImpl = Literal["auto", "torch", "triton"]

# Mode codes for Triton kernel
_MODE_TOKEN_MEAN = 0
_MODE_SEQ_MEAN_TOKEN_SUM = 1
_MODE_SEQ_MEAN_TOKEN_MEAN = 2
_MODE_SEQ_MEAN_TOKEN_SUM_NORM = 3

_MODE_MAP = {
    "token-mean": _MODE_TOKEN_MEAN,
    "seq-mean-token-sum": _MODE_SEQ_MEAN_TOKEN_SUM,
    "seq-mean-token-mean": _MODE_SEQ_MEAN_TOKEN_MEAN,
    "seq-mean-token-sum-norm": _MODE_SEQ_MEAN_TOKEN_SUM_NORM,
}


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_agg_loss_torch(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Reference implementation matching core_algos.agg_loss."""
    mask = loss_mask.float()
    if mode == "token-mean":
        return (loss_mat * mask).sum() / (mask.sum() + 1e-8)
    elif mode == "seq-mean-token-sum":
        return (loss_mat * mask).sum(dim=-1).mean()
    elif mode == "seq-mean-token-mean":
        seq_sum = (loss_mat * mask).sum(dim=-1)
        seq_count = mask.sum(dim=-1)
        return (seq_sum / seq_count).mean()
    elif mode == "seq-mean-token-sum-norm":
        return (loss_mat * mask).sum() / loss_mat.shape[-1]
    else:
        raise ValueError(f"Unknown mode: {mode}")


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
    def _agg_loss_kernel(
        loss_ptr,
        mask_ptr,
        # per-sequence outputs (B,)
        seq_result_ptr,
        seq_count_ptr,
        # scalars
        B: tl.int32,
        T: tl.int32,
        stride_b: tl.int64,
        stride_mask_b: tl.int64,
        BLOCK_T: tl.constexpr,
    ):
        """One program per batch element."""
        bid = tl.program_id(0)
        if bid >= B:
            return

        row_loss = bid * stride_b
        row_mask = bid * stride_mask_b
        num_chunks = tl.cdiv(T, BLOCK_T)

        acc_sum = 0.0
        acc_count = 0.0

        for chunk in range(num_chunks):
            t_off = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
            valid = t_off < T
            loss_val = tl.load(loss_ptr + row_loss + t_off, mask=valid, other=0.0).to(tl.float32)
            mask_val = tl.load(mask_ptr + row_mask + t_off, mask=valid, other=0.0).to(tl.float32)
            acc_sum += tl.sum(loss_val * mask_val, axis=0)
            acc_count += tl.sum(mask_val, axis=0)

        tl.store(seq_result_ptr + bid, acc_sum)
        tl.store(seq_count_ptr + bid, acc_count)


def compute_agg_loss_triton(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available.")

    B, T = loss_mat.shape
    loss = loss_mat.float().contiguous()
    mask = loss_mask.float().contiguous()

    seq_results = torch.empty(B, device=loss.device, dtype=torch.float32)
    seq_counts = torch.empty(B, device=loss.device, dtype=torch.float32)

    grid = lambda meta: (B,)
    _agg_loss_kernel[grid](
        loss, mask,
        seq_results, seq_counts,
        B, T,
        loss.stride(0),
        mask.stride(0),
    )

    if mode == "token-mean":
        total_sum = seq_results.sum()
        total_count = seq_counts.sum()
        return total_sum / (total_count + 1e-8)
    elif mode == "seq-mean-token-sum":
        return seq_results.mean()
    elif mode == "seq-mean-token-mean":
        return (seq_results / seq_counts).mean()
    elif mode == "seq-mean-token-sum-norm":
        return seq_results.sum() / T
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    mode: str,
    impl: AggLossImpl = "auto",
) -> torch.Tensor:
    """Fused loss aggregation for all 4 modes.

    Returns:
        scalar float32 tensor
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and loss_mat.is_cuda else "torch"

    if resolved == "triton":
        return compute_agg_loss_triton(loss_mat, loss_mask, mode)
    return compute_agg_loss_torch(loss_mat, loss_mask, mode)
