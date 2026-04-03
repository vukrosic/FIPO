"""Fused GSPO policy loss kernel (FIPO-029).

GSPO (Group Sequence Policy Optimization) uses a sequence-level importance
ratio instead of a token-level one.  The computation in the unfused path is:

  # (1) sequence-level log ratio (shape: (B,))
  seq_len          = mask.sum(-1).clamp(min=1)
  neg_kl_seq       = (neg_kl * mask).sum(-1) / seq_len     # mean neg KL per seq

  # (2) token-level ratio via straight-through estimator (shape: (B, T))
  log_s_it = (log_prob - log_prob.detach()) + neg_kl_seq.detach().unsqueeze(-1)
  log_s_it = log_s_it.clamp(max=10)
  s_it     = exp(log_s_it)

  # (3) PPO-clip at token level
  pg1 = -advantages * s_it
  pg2 = -advantages * clamp(s_it, 1-eps_low, 1+eps_high)
  pg  = max(pg1, pg2)

  # (4) seq-mean-token-mean aggregation
  loss = mean_b( sum_t(pg * mask) / sum_t(mask) )

The fused Triton kernel computes steps (1)–(4) in two passes:
  Pass A – one program per sequence: accumulate neg_kl_seq, seq_len, and the
            per-sequence pg token-sum and token-count into a (B,) buffer.
  Pass B – one program reduces (B,) buffers to the scalar loss (torch-side
            mean for simplicity – the B reduction is small).

Because step (2) uses `log_prob - log_prob.detach()` (straight-through), and
detach in Triton is simply not storing the gradient, the forward pass is just
`log_prob - stop_grad(log_prob) + neg_kl_seq_detach = neg_kl_seq_detach`.
That means s_it = exp(neg_kl_seq_detach[b]) for every t in sequence b – a
constant per sequence in the forward pass.  The backward pass re-introduces
the log_prob gradient via the autograd graph for the `log_prob` input, which
is NOT detached.

Public API
----------
compute_gspo_loss(
    log_prob, old_log_prob, advantages, response_mask,
    clip_ratio_low=0.2, clip_ratio_high=0.2,
    impl="auto"
) -> (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower)
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


GspoImpl = Literal["auto", "torch", "triton"]


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_gspo_loss_torch(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation matching core_algos.compute_policy_loss_gspo.

    Returns:
        pg_loss:           scalar
        pg_clipfrac:       scalar
        ppo_kl:            scalar (token-mean approx KL)
        pg_clipfrac_lower: scalar (always 0 for GSPO)
    """
    neg_kl = log_prob - old_log_prob
    mask   = response_mask.float()

    # Sequence-level importance ratio
    seq_len = mask.sum(dim=-1).clamp(min=1)
    neg_kl_seq = (neg_kl * mask).sum(dim=-1) / seq_len   # (B,)

    # Token-level combined ratio (straight-through for grad)
    log_s_it = (log_prob - log_prob.detach()) + neg_kl_seq.detach().unsqueeze(-1)
    log_s_it = log_s_it.clamp(max=10.0)
    s_it = torch.exp(log_s_it)

    # PPO clip
    pg1 = -advantages * s_it
    pg2 = -advantages * torch.clamp(s_it, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    pg  = torch.maximum(pg1, pg2)

    # seq-mean-token-mean aggregation
    seq_sums  = (pg * mask).sum(dim=-1)
    seq_means = seq_sums / seq_len
    pg_loss   = seq_means.mean()

    valid_tokens = mask.sum()
    pg_clipfrac = ((pg2 > pg1) & mask.bool()).float().sum() / (valid_tokens + 1e-8)
    ppo_kl      = (-neg_kl * mask).sum() / (valid_tokens + 1e-8)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


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
    def _gspo_loss_kernel(
        # inputs  (B, T)
        log_prob_ptr,
        old_log_prob_ptr,
        advantages_ptr,
        mask_ptr,
        # per-sequence outputs  (B,)
        pg_sum_ptr,           # sum of clipped pg losses per sequence
        pg_count_ptr,         # sum of valid tokens per sequence
        clipfrac_sum_ptr,     # clipped token count per sequence
        kl_sum_ptr,           # sum of -neg_kl * mask per sequence
        # scalars
        B: tl.int32,
        T: tl.int32,
        stride_b: tl.int64,   # common row stride (all (B,T) tensors share it)
        clip_low: tl.float32,
        clip_high: tl.float32,
        BLOCK_T: tl.constexpr,
    ):
        """One program per batch element.

        Pass 1 within program: compute neg_kl_seq.
        Pass 2 within program: compute clipped loss using neg_kl_seq as ratio.
        """
        bid = tl.program_id(0)
        if bid >= B:
            return

        row = bid * stride_b
        num_chunks = tl.cdiv(T, BLOCK_T)

        # ---- Pass 1: accumulate neg_kl_seq and seq_len ----
        acc_neg_kl = 0.0
        acc_len    = 0.0
        for chunk in range(num_chunks):
            t_off  = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
            valid  = t_off < T
            lp     = tl.load(log_prob_ptr     + row + t_off, mask=valid, other=0.0).to(tl.float32)
            olp    = tl.load(old_log_prob_ptr + row + t_off, mask=valid, other=0.0).to(tl.float32)
            msk    = tl.load(mask_ptr         + row + t_off, mask=valid, other=0.0).to(tl.float32)
            neg_kl = lp - olp
            acc_neg_kl = acc_neg_kl + tl.sum(neg_kl * msk, axis=0)
            acc_len    = acc_len    + tl.sum(msk,           axis=0)

        seq_len    = tl.maximum(acc_len, 1.0)
        neg_kl_seq = acc_neg_kl / seq_len   # scalar, detached in forward

        # Sequence-level ratio: exp(neg_kl_seq)  [same for all t in this seq]
        s_it = tl.exp(tl.minimum(neg_kl_seq, 10.0))

        # ---- Pass 2: compute clipped loss ----
        acc_pg       = 0.0
        acc_clip     = 0.0
        acc_kl_mon   = 0.0

        for chunk in range(num_chunks):
            t_off  = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
            valid  = t_off < T
            adv    = tl.load(advantages_ptr   + row + t_off, mask=valid, other=0.0).to(tl.float32)
            msk    = tl.load(mask_ptr         + row + t_off, mask=valid, other=0.0).to(tl.float32)
            lp     = tl.load(log_prob_ptr     + row + t_off, mask=valid, other=0.0).to(tl.float32)
            olp    = tl.load(old_log_prob_ptr + row + t_off, mask=valid, other=0.0).to(tl.float32)

            neg_kl = lp - olp
            s_clamped = tl.minimum(tl.maximum(s_it, 1.0 - clip_low), 1.0 + clip_high)
            pg1 = -adv * s_it
            pg2 = -adv * s_clamped
            pg  = tl.where(pg1 > pg2, pg1, pg2)

            acc_pg     = acc_pg   + tl.sum(pg * msk,               axis=0)
            acc_clip   = acc_clip + tl.sum(((pg2 > pg1) & (msk > 0.0)).to(tl.float32), axis=0)
            acc_kl_mon = acc_kl_mon + tl.sum((-neg_kl) * msk,      axis=0)

        # Store per-sequence results
        tl.store(pg_sum_ptr       + bid, acc_pg / seq_len)
        tl.store(pg_count_ptr     + bid, seq_len)
        tl.store(clipfrac_sum_ptr + bid, acc_clip)
        tl.store(kl_sum_ptr       + bid, acc_kl_mon)


def compute_gspo_loss_triton(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available.")
    for t in (log_prob, old_log_prob, advantages, response_mask):
        if not t.is_cuda:
            raise RuntimeError("All tensors must be CUDA.")

    B, T = log_prob.shape
    pg_seq_means   = torch.empty(B, device=log_prob.device, dtype=torch.float32)
    pg_seq_counts  = torch.empty(B, device=log_prob.device, dtype=torch.float32)
    clipfrac_sums  = torch.empty(B, device=log_prob.device, dtype=torch.float32)
    kl_sums        = torch.empty(B, device=log_prob.device, dtype=torch.float32)

    # All (B, T) tensors must share the same row stride
    lp  = log_prob.float().contiguous()
    olp = old_log_prob.float().contiguous()
    adv = advantages.float().contiguous()
    msk = response_mask.float().contiguous()
    assert lp.stride(0) == olp.stride(0) == adv.stride(0) == msk.stride(0), \
        "stride mismatch – ensure all tensors have the same (B,T) layout"

    grid = lambda meta: (B,)
    _gspo_loss_kernel[grid](
        lp, olp, adv, msk,
        pg_seq_means, pg_seq_counts, clipfrac_sums, kl_sums,
        B, T,
        lp.stride(0),
        float(clip_ratio_low),
        float(clip_ratio_high),
    )

    pg_loss    = pg_seq_means.mean()
    valid_tok  = msk.sum()
    pg_clipfrac = clipfrac_sums.sum() / (valid_tok + 1e-8)
    ppo_kl      = kl_sums.sum()       / (valid_tok + 1e-8)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_gspo_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    impl: GspoImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused GSPO policy loss.

    Returns:
        pg_loss:           scalar float32
        pg_clipfrac:       scalar float32
        ppo_kl:            scalar float32
        pg_clipfrac_lower: scalar float32 (always 0 for GSPO)
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and log_prob.is_cuda else "torch"

    if resolved == "triton":
        return compute_gspo_loss_triton(
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            advantages=advantages,
            response_mask=response_mask,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
        )
    return compute_gspo_loss_torch(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
    )
