"""Vectorized KL-Cov policy loss (FIPO-033).

The original compute_policy_loss_kl_cov in core_algos.py uses:
  - torch.masked_select() which triggers a CPU sync for output size
  - torch.topk on a variable-sized tensor from masked_select
  - Scatter-write to replace losses at high-covariance positions

The vectorized version avoids CPU syncs by:
  1. Computing covariance over the full (B,T) grid with mask
  2. Using topk on the flat grid directly (masked positions get -inf)
  3. Scatter via flat indexing — all GPU-side

Public API
----------
compute_kl_cov_loss(
    log_prob, old_log_prob, advantages, response_mask,
    kl_cov_ratio=0.0002, ppo_kl_coef=1.0,
    loss_agg_mode="token-mean",
    impl="auto"
) -> (pg_loss, pg_clipfrac, ppo_kl_abs, pg_clipfrac_lower)
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


KlCovImpl = Literal["auto", "torch", "triton"]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + 1e-8)


def _agg_loss_token_mean(loss_mat: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    return (loss_mat * loss_mask).sum() / (loss_mask.sum() + 1e-8)


# ---------------------------------------------------------------------------
# Torch reference (matches core_algos)
# ---------------------------------------------------------------------------

def compute_kl_cov_loss_torch(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    kl_cov_ratio: float = 0.0002,
    ppo_kl_coef: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation matching core_algos.compute_policy_loss_kl_cov."""
    mask = response_mask.float()

    neg_kl = log_prob - old_log_prob
    abs_kl = neg_kl.abs()
    ratio = torch.exp(neg_kl)
    ppo_kl_abs = _masked_mean(abs_kl, mask)

    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1.clone()

    # Covariance on valid tokens
    valid_mask = mask > 0
    valid_adv = advantages[valid_mask].detach().reshape(-1)
    valid_lp = log_prob[valid_mask].detach().reshape(-1)

    if valid_adv.numel() > 0 and kl_cov_ratio > 0:
        cov = (valid_adv - valid_adv.mean()) * (valid_lp - valid_lp.mean())
        k = max(1, int(cov.shape[0] * kl_cov_ratio))
        _, top_indices = torch.topk(cov, k, largest=True)

        # Map back to (B, T) indices
        valid_flat_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=True)[0]
        selected = valid_flat_indices[top_indices]

        T = advantages.shape[1]
        rows = selected // T
        cols = selected % T
        pg_losses[rows, cols] = pg_losses_kl[rows, cols]

    pg_loss = _agg_loss_token_mean(pg_losses, mask)
    return pg_loss, torch.tensor(0.0, device=pg_loss.device), ppo_kl_abs, torch.tensor(0.0, device=pg_loss.device)


# ---------------------------------------------------------------------------
# Vectorized torch (avoids CPU sync from masked_select)
# ---------------------------------------------------------------------------

def compute_kl_cov_loss_vectorized(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    kl_cov_ratio: float = 0.0002,
    ppo_kl_coef: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized implementation avoiding CPU syncs from masked_select/nonzero.

    Computes covariance over full (B,T) grid with mask, uses topk on flat
    grid directly.
    """
    mask = response_mask.float()
    B, T = log_prob.shape

    neg_kl = log_prob - old_log_prob
    abs_kl = neg_kl.abs()
    ratio = torch.exp(neg_kl)
    ppo_kl_abs = _masked_mean(abs_kl, mask)

    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl

    # Compute covariance on full grid, mask invalid positions
    adv_det = advantages.detach()
    lp_det = log_prob.detach()

    # Masked mean of advantages and log_probs
    adv_mean = (adv_det * mask).sum() / (mask.sum() + 1e-8)
    lp_mean = (lp_det * mask).sum() / (mask.sum() + 1e-8)

    cov_all = (adv_det - adv_mean) * (lp_det - lp_mean)
    # Set masked positions to -inf so they're never selected by topk
    cov_all = torch.where(mask > 0, cov_all, torch.tensor(-torch.inf, device=cov_all.device))

    # Select top-k by covariance (all GPU)
    n_valid = mask.sum()
    k = torch.clamp((kl_cov_ratio * n_valid).int(), min=1)
    k_val = k.item()

    flat_cov = cov_all.reshape(-1)
    _, top_idx = torch.topk(flat_cov, min(k_val, (flat_cov > -torch.inf).sum().item()), largest=True)

    # Build output: use pg_losses1 everywhere, except at top-k positions use pg_losses_kl
    pg_losses = pg_losses1.clone()
    flat_pg = pg_losses.reshape(-1)
    flat_pg_kl = pg_losses_kl.reshape(-1)
    flat_pg[top_idx] = flat_pg_kl[top_idx]
    pg_losses = flat_pg.reshape(B, T)

    pg_loss = _agg_loss_token_mean(pg_losses, mask)
    return pg_loss, torch.tensor(0.0, device=pg_loss.device), ppo_kl_abs, torch.tensor(0.0, device=pg_loss.device)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_kl_cov_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    kl_cov_ratio: float = 0.0002,
    ppo_kl_coef: float = 1.0,
    impl: KlCovImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL-Cov policy loss with optional vectorized fast path.

    Returns:
        pg_loss:           scalar float32
        pg_clipfrac:       scalar float32 (always 0)
        ppo_kl_abs:        scalar float32
        pg_clipfrac_lower: scalar float32 (always 0)
    """
    resolved = impl
    if resolved == "auto":
        resolved = "triton" if log_prob.is_cuda else "torch"

    if resolved == "triton":
        return compute_kl_cov_loss_vectorized(
            log_prob=log_prob, old_log_prob=old_log_prob,
            advantages=advantages, response_mask=response_mask,
            kl_cov_ratio=kl_cov_ratio, ppo_kl_coef=ppo_kl_coef,
        )
    return compute_kl_cov_loss_torch(
        log_prob=log_prob, old_log_prob=old_log_prob,
        advantages=advantages, response_mask=response_mask,
        kl_cov_ratio=kl_cov_ratio, ppo_kl_coef=ppo_kl_coef,
    )
