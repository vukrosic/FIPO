"""Fused GMPO (Geometric-Mean Policy Optimization) loss kernel (FIPO-030).

GMPO computes a *sequence-level* geometric-mean importance ratio by
exponentiating the mean of clipped per-token log-ratios:

  neg_kl        = log_prob - old_log_prob                          # (B, T)
  sgn           = sign(advantages)
  neg_kl_clamp  = clamp(neg_kl, -clip_low, clip_high)
  neg_kl_min    = sgn * min(sgn * neg_kl, sgn * neg_kl_clamp)     # wider clip
  mask_sum      = mask.sum(-1)                                     # (B,)
  ratio         = exp( (neg_kl_min * mask).sum(-1) / mask_sum )    # (B,)
  advantage_seq = (adv * mask).sum(-1) / mask_sum                  # (B,)
  pg_loss       = mean( -advantage_seq * ratio )                   # scalar

The fused Triton kernel does this in a single pass per sequence:
  - One program per batch element
  - Accumulates neg_kl_min_sum, adv_sum, mask_sum, clip counts in one scan
  - Computes ratio = exp(neg_kl_min_sum / mask_sum) then loss = -adv_mean * ratio

Public API
----------
compute_gmpo_loss(
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


GmpoImpl = Literal["auto", "torch", "triton"]


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_gmpo_loss_torch(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation matching core_algos.compute_policy_loss_geo_mean.

    Returns:
        pg_loss:           scalar
        pg_clipfrac:       scalar (fraction of tokens clipped when adv > 0)
        ppo_kl:            scalar (token-mean approx KL)
        pg_clipfrac_lower: scalar (fraction of tokens clipped when adv < 0)
    """
    neg_kl = log_prob - old_log_prob
    mask = response_mask.float()
    valid_tokens = mask.sum()

    # Token-mean KL (monitoring)
    ppo_kl = (-neg_kl * mask).sum() / (valid_tokens + 1e-8)

    # Token-level wider clip
    sgn = torch.sign(advantages)
    neg_kl_clamp = torch.clamp(neg_kl, -clip_ratio_low, clip_ratio_high)
    neg_kl_min = sgn * torch.min(sgn * neg_kl, sgn * neg_kl_clamp)

    # Sequence-level geometric mean ratio
    mask_sum = mask.sum(dim=-1)  # (B,)
    ratio = torch.exp((neg_kl_min * mask).sum(dim=-1) / (mask_sum + 1e-8))
    advantage_seq = (advantages * mask).sum(dim=-1) / (mask_sum + 1e-8)

    pg_losses = -advantage_seq * ratio
    pg_loss = pg_losses.mean()

    # Clip fraction metrics
    clipped = torch.ne(neg_kl, neg_kl_clamp)
    pg_clipfrac = ((clipped & (advantages > 0)).float() * mask).sum() / (valid_tokens + 1e-8)
    pg_clipfrac_lower = ((clipped & (advantages < 0)).float() * mask).sum() / (valid_tokens + 1e-8)

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
    def _gmpo_loss_kernel(
        # inputs  (B, T)
        log_prob_ptr,
        old_log_prob_ptr,
        advantages_ptr,
        mask_ptr,
        # per-sequence outputs  (B,)
        pg_loss_ptr,          # -adv_mean * ratio per sequence
        mask_sum_ptr,         # valid token count per sequence
        clipfrac_upper_ptr,   # clipped & adv>0 count
        clipfrac_lower_ptr,   # clipped & adv<0 count
        kl_sum_ptr,           # sum of -neg_kl * mask per sequence
        # scalars
        B: tl.int32,
        T: tl.int32,
        stride_b: tl.int64,
        clip_low: tl.float32,
        clip_high: tl.float32,
        BLOCK_T: tl.constexpr,
    ):
        """One program per batch element — single pass."""
        bid = tl.program_id(0)
        if bid >= B:
            return

        row = bid * stride_b
        num_chunks = tl.cdiv(T, BLOCK_T)

        acc_neg_kl_min = 0.0   # sum of neg_kl_min * mask
        acc_adv        = 0.0   # sum of adv * mask
        acc_mask       = 0.0   # sum of mask
        acc_kl_mon     = 0.0   # sum of -neg_kl * mask (monitoring)
        acc_clip_upper = 0.0   # clipped & adv > 0
        acc_clip_lower = 0.0   # clipped & adv < 0

        for chunk in range(num_chunks):
            t_off = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
            valid = t_off < T

            lp  = tl.load(log_prob_ptr     + row + t_off, mask=valid, other=0.0).to(tl.float32)
            olp = tl.load(old_log_prob_ptr + row + t_off, mask=valid, other=0.0).to(tl.float32)
            adv = tl.load(advantages_ptr   + row + t_off, mask=valid, other=0.0).to(tl.float32)
            msk = tl.load(mask_ptr         + row + t_off, mask=valid, other=0.0).to(tl.float32)

            neg_kl = lp - olp
            neg_kl_clamp = tl.minimum(tl.maximum(neg_kl, -clip_low), clip_high)

            # wider clip: sgn * min(sgn * neg_kl, sgn * neg_kl_clamp)
            # Equivalent: where(adv > 0, min(neg_kl, neg_kl_clamp),
            #              where(adv < 0, max(neg_kl, neg_kl_clamp), neg_kl_clamp))
            # But sign(0) = 0, so for adv==0 tokens: sgn*min(0,0) = 0
            sgn = tl.where(adv > 0, 1.0, tl.where(adv < 0, -1.0, 0.0))
            neg_kl_min = sgn * tl.minimum(sgn * neg_kl, sgn * neg_kl_clamp)

            clipped = (neg_kl != neg_kl_clamp)

            acc_neg_kl_min += tl.sum(neg_kl_min * msk, axis=0)
            acc_adv        += tl.sum(adv * msk, axis=0)
            acc_mask       += tl.sum(msk, axis=0)
            acc_kl_mon     += tl.sum((-neg_kl) * msk, axis=0)
            acc_clip_upper += tl.sum((clipped & (adv > 0)).to(tl.float32) * msk, axis=0)
            acc_clip_lower += tl.sum((clipped & (adv < 0)).to(tl.float32) * msk, axis=0)

        mask_sum = tl.maximum(acc_mask, 1.0)
        ratio = tl.exp(acc_neg_kl_min / mask_sum)
        adv_mean = acc_adv / mask_sum
        pg_loss_val = -adv_mean * ratio

        tl.store(pg_loss_ptr        + bid, pg_loss_val)
        tl.store(mask_sum_ptr       + bid, acc_mask)
        tl.store(clipfrac_upper_ptr + bid, acc_clip_upper)
        tl.store(clipfrac_lower_ptr + bid, acc_clip_lower)
        tl.store(kl_sum_ptr         + bid, acc_kl_mon)


def compute_gmpo_loss_triton(
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
    pg_losses      = torch.empty(B, device=log_prob.device, dtype=torch.float32)
    mask_sums      = torch.empty(B, device=log_prob.device, dtype=torch.float32)
    clipfrac_upper = torch.empty(B, device=log_prob.device, dtype=torch.float32)
    clipfrac_lower = torch.empty(B, device=log_prob.device, dtype=torch.float32)
    kl_sums        = torch.empty(B, device=log_prob.device, dtype=torch.float32)

    lp  = log_prob.float().contiguous()
    olp = old_log_prob.float().contiguous()
    adv = advantages.float().contiguous()
    msk = response_mask.float().contiguous()
    assert lp.stride(0) == olp.stride(0) == adv.stride(0) == msk.stride(0)

    grid = lambda meta: (B,)
    _gmpo_loss_kernel[grid](
        lp, olp, adv, msk,
        pg_losses, mask_sums, clipfrac_upper, clipfrac_lower, kl_sums,
        B, T,
        lp.stride(0),
        float(clip_ratio_low),
        float(clip_ratio_high),
    )

    pg_loss = pg_losses.mean()
    valid_tok = msk.sum()
    pg_cf = clipfrac_upper.sum() / (valid_tok + 1e-8)
    ppo_kl = kl_sums.sum() / (valid_tok + 1e-8)
    pg_cfl = clipfrac_lower.sum() / (valid_tok + 1e-8)

    return pg_loss, pg_cf, ppo_kl, pg_cfl


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_gmpo_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    impl: GmpoImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused GMPO policy loss.

    Returns:
        pg_loss:           scalar float32
        pg_clipfrac:       scalar float32
        ppo_kl:            scalar float32
        pg_clipfrac_lower: scalar float32
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if HAVE_TRITON and log_prob.is_cuda else "torch"

    if resolved == "triton":
        return compute_gmpo_loss_triton(
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            advantages=advantages,
            response_mask=response_mask,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
        )
    return compute_gmpo_loss_torch(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
    )
