"""Fused per-token log-prob + entropy from logits (FIPO-041).

Computing both log-probs and entropy from the same logits tensor is common
in PPO/RLHF actor updates:

    log_prob = log_softmax(logits)[token_id]  # needed for policy loss
    entropy  = -sum(softmax(logits) * log_softmax(logits))  # needed for entropy bonus

The naive approach reads logits (B, T, V) TWICE (once per call) and
materialises the full (B, T, V) softmax/log_softmax each time.

This kernel reads logits ONCE and returns both outputs in a single streaming
pass over the vocabulary dimension using the online logsumexp trick:

  For each (b, t) position, stream over vocab chunks [v, v+BLOCK_V):
    - Maintain running_max, running_sum_exp (for logsumexp)
    - Maintain running_sx (sum of exp(logit-max)*logit, for entropy)
    - Record selected_logit[token_id] for log_prob

  After the scan:
    logsumexp  = running_max + log(running_sum_exp)
    log_prob   = selected_logit - logsumexp
    entropy    = logsumexp - running_sx / running_sum_exp

This fuses three reads through (B,T,V) into ONE — a 3x memory bandwidth
reduction for logits, which dominates the compute at typical LLM vocab sizes.

Supports float32 and bfloat16 inputs (upcast to float32 internally).
Supports logits with shape (..., V) and token_ids with matching leading shape.

Public API
----------
compute_logprob_and_entropy(logits, token_ids, impl="auto")
    -> (log_probs: (...) float32, entropy: (...) float32)
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


LogprobEntropyImpl = Literal["auto", "torch", "triton"]

_TRITON_MAX_VOCAB = 262144


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_logprob_and_entropy_torch(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: log_softmax gather + entropy.  Reads logits twice.

    Args:
        logits:    (..., V) float32 or bfloat16
        token_ids: (...)    int64

    Returns:
        log_probs: (...) float32
        entropy:   (...) float32
    """
    if logits.ndim < 2:
        raise ValueError(f"logits must have at least 2 dims, got shape={tuple(logits.shape)}")
    if token_ids.shape != logits.shape[:-1]:
        raise ValueError(
            f"token_ids shape must match logits leading dims, got logits={tuple(logits.shape)} token_ids={tuple(token_ids.shape)}"
        )

    leading_shape = token_ids.shape
    logits_f = logits.float()
    pd = torch.softmax(logits_f, dim=-1)
    lse = torch.logsumexp(logits_f, dim=-1)
    log_probs = logits_f.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1) - lse
    entropy = lse - (pd * logits_f).sum(dim=-1)
    log_probs = log_probs.reshape(leading_shape)
    entropy = entropy.reshape(leading_shape)
    return log_probs, entropy


# ---------------------------------------------------------------------------
# Triton kernel
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
    def _logprob_entropy_kernel(
        logits_ptr,     # (total_tokens, V) row-major
        token_ids_ptr,  # (total_tokens,)   int32
        lp_out_ptr,     # (total_tokens,)   float32
        ent_out_ptr,    # (total_tokens,)   float32
        total_tokens: tl.int32,
        V: tl.int32,
        stride_row: tl.int64,  # stride between (b,t) rows in logits
        BLOCK_V: tl.constexpr,
    ):
        """One program per (b, t).  Single pass over V for both outputs.

        Accumulators (all float32):
            running_max     — online maximum
            running_sum_exp — sum of exp(logit - running_max)
            running_sx      — sum of exp(logit - running_max) * logit  (for entropy)
            selected_logit  — logits value at token_id                 (for logprob)
        """
        pid = tl.program_id(axis=0)
        if pid >= total_tokens:
            return

        row_ptr  = logits_ptr + pid * stride_row
        token_id = tl.load(token_ids_ptr + pid).to(tl.int32)

        running_max     = -1e9
        running_sum_exp = 0.0
        running_sx      = 0.0   # sum(exp(x - max) * x) for entropy
        selected_logit  = 0.0   # logit at token_id

        num_chunks = tl.cdiv(V, BLOCK_V)
        for chunk_id in range(num_chunks):
            v_offsets = chunk_id * BLOCK_V + tl.arange(0, BLOCK_V)
            valid     = v_offsets < V

            x = tl.load(row_ptr + v_offsets, mask=valid, other=-1e9).to(tl.float32)

            # Online logsumexp update
            chunk_max = tl.max(tl.where(valid, x, -1e9), axis=0)
            new_max   = tl.maximum(running_max, chunk_max)
            scale     = tl.exp(running_max - new_max)

            exp_shifted  = tl.where(valid, tl.exp(x - new_max), 0.0)
            running_sum_exp = running_sum_exp * scale + tl.sum(exp_shifted, axis=0)
            running_sx      = running_sx * scale + tl.sum(
                tl.where(valid, exp_shifted * x, 0.0), axis=0
            )
            running_max = new_max

            # Capture selected logit (exactly one chunk owns token_id)
            is_selected    = (v_offsets == token_id) & valid
            selected_logit = selected_logit + tl.sum(
                tl.where(is_selected, x, 0.0), axis=0
            )

        logsumexp      = running_max + tl.log(running_sum_exp)
        log_prob       = selected_logit - logsumexp
        entropy        = logsumexp - running_sx / running_sum_exp

        tl.store(lp_out_ptr  + pid, log_prob)
        tl.store(ent_out_ptr + pid, entropy)


def compute_logprob_and_entropy_triton(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass fused logprob + entropy.  Reads logits once.

    Args:
        logits:    (..., V) float32 or bfloat16, CUDA tensor
        token_ids: (...)    int64 CUDA tensor

    Returns:
        log_probs: (...) float32
        entropy:   (...) float32
    """
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available.")
    if not logits.is_cuda or not token_ids.is_cuda:
        raise RuntimeError("Triton logprob+entropy requires CUDA tensors.")
    if logits.ndim < 2:
        raise ValueError(f"logits must have at least 2 dims, got shape={tuple(logits.shape)}")
    if token_ids.shape != logits.shape[:-1]:
        raise ValueError(
            f"token_ids shape must match logits leading dims, got logits={tuple(logits.shape)} token_ids={tuple(token_ids.shape)}"
        )

    leading_shape = token_ids.shape
    V = logits.shape[-1]

    if V > _TRITON_MAX_VOCAB:
        return compute_logprob_and_entropy_torch(logits, token_ids)

    total_tokens = token_ids.numel()
    logits_2d = logits.reshape(total_tokens, V).contiguous()
    token_ids_1d = token_ids.reshape(total_tokens).to(torch.int32).contiguous()

    lp_out = torch.empty(total_tokens, device=logits.device, dtype=torch.float32)
    ent_out = torch.empty(total_tokens, device=logits.device, dtype=torch.float32)

    grid = lambda meta: (total_tokens,)

    _logprob_entropy_kernel[grid](
        logits_2d,
        token_ids_1d,
        lp_out,
        ent_out,
        total_tokens,
        V,
        logits_2d.stride(0),
    )

    return lp_out.reshape(leading_shape), ent_out.reshape(leading_shape)


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def compute_logprob_and_entropy(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    impl: LogprobEntropyImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused per-token log-probability and entropy from logits.

    Reads logits ONCE instead of twice — eliminates one full (B,T,V) pass
    compared to calling compute_token_logprob + compute_entropy_from_logits
    separately.  Also avoids materialising (B,T,V) softmax/log_softmax.

    Args:
        logits:    (..., V) float32 or bfloat16
        token_ids: (...)    int64 — target token at each position
        impl:      "auto" | "torch" | "triton"

    Returns:
        log_probs: (...) float32
        entropy:   (...) float32
    """
    resolved = impl
    if resolved == "auto":
        resolved = (
            "triton"
            if HAVE_TRITON and logits.is_cuda and token_ids.is_cuda
            else "torch"
        )

    if resolved == "triton":
        return compute_logprob_and_entropy_triton(logits, token_ids)
    if resolved == "torch":
        return compute_logprob_and_entropy_torch(logits, token_ids)

    raise ValueError(f"Unsupported logprob_entropy implementation: {impl!r}")
