"""Fused per-token entropy kernel (FIPO-039).

Computes per-token entropy H(p) = logsumexp(logits) - sum(softmax(logits)*logits)
WITHOUT materialising the full (B, T, V) softmax or log-softmax tensor.

For each (b, t) position the kernel streams over the vocab dimension in
BLOCK_V-wide chunks, maintaining an online logsumexp accumulator and a parallel
weighted-sum accumulator.  The algorithm is identical to the online logsumexp
trick used in FIPO-026 (logprob.py), extended with a second running scalar:

    running_max     — current maximum logit seen so far
    running_sum_exp — sum of exp(logit - running_max) over all vocab so far
    running_sx      — sum of exp(logit - running_max) * logit over all vocab

At the end of the V scan:

    logsumexp = running_max + log(running_sum_exp)
    entropy   = logsumexp - running_sx / running_sum_exp

The kernel supports float32 and bfloat16 inputs (upcast to float32 internally)
and arbitrary vocab sizes (no upper-bound restriction).

Public API
----------
compute_entropy_from_logits(logits, impl="auto") -> (B, T) float32
  logits : (B, T, V) float32 or bfloat16
  returns: (B, T)    float32  per-token Shannon entropy in nats
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


EntropyImpl = Literal["auto", "torch", "triton"]

_TRITON_MAX_VOCAB = 262144  # practical upper-bound for the streaming kernel


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_entropy_from_logits_torch(logits: torch.Tensor) -> torch.Tensor:
    """Per-token entropy via softmax (reference implementation).

    Args:
        logits: (B, T, V) float32 or bfloat16

    Returns:
        entropy: (B, T) float32
    """
    logits_f = logits.float()
    pd = torch.softmax(logits_f, dim=-1)              # (B, T, V)
    lse = torch.logsumexp(logits_f, dim=-1)           # (B, T)
    entropy = lse - (pd * logits_f).sum(dim=-1)       # (B, T)
    return entropy


# ---------------------------------------------------------------------------
# Triton streaming kernel
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_V": 512},  num_warps=4,  num_stages=1),
            triton.Config({"BLOCK_V": 1024}, num_warps=4,  num_stages=1),
            triton.Config({"BLOCK_V": 2048}, num_warps=8,  num_stages=1),
            triton.Config({"BLOCK_V": 4096}, num_warps=8,  num_stages=1),
            triton.Config({"BLOCK_V": 8192}, num_warps=16, num_stages=1),
        ],
        key=["V"],
    )
    @triton.jit
    def _entropy_streaming_kernel(
        logits_ptr,   # (total_tokens, V) row-major
        output_ptr,   # (total_tokens,)  float32 output
        total_tokens: tl.int32,
        V: tl.int32,
        stride_row: tl.int64,   # stride between (b,t) rows in logits
        stride_out: tl.int64,   # stride between (b,t) in output
        BLOCK_V: tl.constexpr,
    ):
        """One program per (b, t) position.  Streams over V in BLOCK_V chunks.

        Accumulators (all float32):
            running_max     — online maximum of logits seen
            running_sum_exp — sum of exp(logit - running_max)
            running_sx      — sum of exp(logit - running_max) * logit
        """
        pid = tl.program_id(axis=0)
        if pid >= total_tokens:
            return

        row_ptr = logits_ptr + pid * stride_row

        running_max = -1e9
        running_sum_exp = 0.0
        running_sx = 0.0  # sum of exp(logit - running_max) * logit

        num_chunks = tl.cdiv(V, BLOCK_V)
        for chunk_id in range(num_chunks):
            v_offsets = chunk_id * BLOCK_V + tl.arange(0, BLOCK_V)
            valid = v_offsets < V

            x = tl.load(row_ptr + v_offsets, mask=valid, other=-1e9).to(tl.float32)

            # Chunk max over valid positions only
            chunk_max = tl.max(tl.where(valid, x, -1e9), axis=0)
            new_max = tl.maximum(running_max, chunk_max)

            # Rescale old accumulators to new max
            scale = tl.exp(running_max - new_max)
            running_sum_exp = running_sum_exp * scale
            running_sx = running_sx * scale

            # Accumulate new chunk
            exp_shifted = tl.where(valid, tl.exp(x - new_max), 0.0)
            running_sum_exp += tl.sum(exp_shifted, axis=0)
            running_sx += tl.sum(tl.where(valid, exp_shifted * x, 0.0), axis=0)

            running_max = new_max

        # Compute entropy: H = logsumexp - sum(p * logits)
        # where sum(p * logits) = running_sx / running_sum_exp
        logsumexp = running_max + tl.log(running_sum_exp)
        entropy = logsumexp - running_sx / running_sum_exp

        tl.store(output_ptr + pid * stride_out, entropy)


def compute_entropy_from_logits_triton(logits: torch.Tensor) -> torch.Tensor:
    """Streaming Triton entropy kernel — no (B,T,V) materialisation.

    Args:
        logits: (B, T, V) float32 or bfloat16, CUDA tensor

    Returns:
        entropy: (B, T) float32 CUDA tensor
    """
    if not HAVE_TRITON:
        raise RuntimeError("Triton is not installed.")
    if not logits.is_cuda:
        raise RuntimeError("Triton entropy requires CUDA tensors.")

    B, T, V = logits.shape

    if V > _TRITON_MAX_VOCAB:
        return compute_entropy_from_logits_torch(logits)

    # Flatten to (B*T, V)
    logits_2d = logits.reshape(B * T, V).contiguous()
    output_1d = torch.empty(B * T, device=logits.device, dtype=torch.float32)

    total_tokens = B * T
    grid = lambda meta: (total_tokens,)

    _entropy_streaming_kernel[grid](
        logits_2d,
        output_1d,
        total_tokens,
        V,
        logits_2d.stride(0),
        output_1d.stride(0),
    )

    return output_1d.reshape(B, T)


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def compute_entropy_from_logits(
    logits: torch.Tensor,
    impl: EntropyImpl = "auto",
) -> torch.Tensor:
    """Per-token Shannon entropy from logits (in nats).

    Avoids materialising the (B, T, V) softmax tensor by using an online
    single-pass streaming accumulation over the vocab dimension.

    Args:
        logits: (B, T, V) float32 or bfloat16
        impl:   "auto" | "torch" | "triton"

    Returns:
        entropy: (B, T) float32
    """
    resolved = impl
    if resolved == "auto":
        resolved = (
            "triton"
            if HAVE_TRITON and logits.is_cuda
            else "torch"
        )

    if resolved == "triton":
        return compute_entropy_from_logits_triton(logits)
    if resolved == "torch":
        return compute_entropy_from_logits_torch(logits)

    raise ValueError(f"Unsupported entropy implementation: {impl!r}")
