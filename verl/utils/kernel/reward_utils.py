"""Fused reward computation kernel (FIPO-027).

Computes token-level rewards with KL penalty in a single kernel pass,
supporting all four KL penalty variants:

  k1 / kl  : kl = logp - ref_logp            (linear, biased)
  k2 / mse : kl = 0.5 * (logp - ref_logp)^2  (quadratic)
  k3 / lvk : kl = exp(ref-logp) - (ref-logp) - 1  (low-variance, unbiased)
  abs      : kl = |logp - ref_logp|

Fused operation:
  kl    = kl_fn(log_prob, ref_log_prob)
  kl    = kl * response_mask
  reward = token_scores - kl * kl_ratio

Also returns the masked-mean KL for monitoring.

Public API
----------
compute_rewards_fused(
    token_scores, log_prob, ref_log_prob, response_mask,
    kl_ratio=1.0, kl_type="k1", impl="auto"
) -> (rewards: Tensor, kl_mean: Tensor)

Both tensors are float32.  token_scores / log_prob / ref_log_prob may be
float32 or bfloat16; they are upcast to float32 inside the kernel.
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


RewardImpl = Literal["auto", "torch", "triton"]

# KL type int codes passed to the Triton kernel
_KL_K1 = 0   # logp - ref_logp
_KL_K2 = 1   # 0.5 * (logp - ref_logp)^2
_KL_K3 = 2   # low_var_kl: exp(ref-logp)-(ref-logp)-1
_KL_ABS = 3  # |logp - ref_logp|

_KL_TYPE_MAP = {
    "k1": _KL_K1, "kl": _KL_K1,
    "k2": _KL_K2, "mse": _KL_K2,
    "k3": _KL_K3, "low_var_kl": _KL_K3, "lvk": _KL_K3,
    "abs": _KL_ABS,
}


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def compute_rewards_fused_torch(
    token_scores: torch.Tensor,
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    kl_ratio: float = 1.0,
    kl_type: str = "k1",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation supporting all KL variants.

    Returns:
        rewards:  (B, T) float32 – token_scores - kl * kl_ratio (masked)
        kl_mean:  scalar float32 – masked mean KL over valid tokens
    """
    s = token_scores.float()
    lp = log_prob.float()
    rlp = ref_log_prob.float()
    mask = response_mask.float()

    kt = kl_type.lower()
    if kt in ("k1", "kl"):
        kl = lp - rlp
    elif kt in ("k2", "mse"):
        kl = 0.5 * (lp - rlp).square()
    elif kt in ("k3", "low_var_kl", "lvk"):
        diff = torch.clamp(rlp - lp, min=-20.0, max=20.0)
        kl = torch.exp(diff) - diff - 1.0
        kl = torch.clamp(kl, min=-10.0, max=10.0)
    elif kt == "abs":
        kl = (lp - rlp).abs()
    else:
        raise ValueError(f"Unknown kl_type: {kl_type!r}")

    kl_masked = kl * mask
    rewards = (s - kl_masked * kl_ratio) * mask
    kl_mean = kl_masked.sum() / (mask.sum() + 1e-8)
    return rewards, kl_mean


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        ],
        key=["numel"],
    )
    @triton.jit
    def _fused_rewards_kernel(
        scores_ptr,
        log_prob_ptr,
        ref_log_prob_ptr,
        mask_ptr,
        rewards_ptr,
        kl_sum_ptr,
        mask_sum_ptr,
        numel,
        kl_ratio: tl.float32,
        kl_type: tl.int32,   # 0=k1, 1=k2, 2=k3, 3=abs
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < numel

        scores   = tl.load(scores_ptr     + offsets, mask=valid, other=0.0).to(tl.float32)
        lp       = tl.load(log_prob_ptr   + offsets, mask=valid, other=0.0).to(tl.float32)
        rlp      = tl.load(ref_log_prob_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        mask     = tl.load(mask_ptr        + offsets, mask=valid, other=0.0).to(tl.float32)

        # KL computation (compile-time branch on kl_type constexpr not possible,
        # so use runtime tl.where chains)
        diff = lp - rlp                     # k1 base

        # k2: 0.5 * diff^2
        kl_k2 = 0.5 * diff * diff

        # k3: exp(rlp - lp) - (rlp - lp) - 1, clamped
        neg_diff = tl.minimum(tl.maximum(rlp - lp, -20.0), 20.0)
        kl_k3 = tl.exp(neg_diff) - neg_diff - 1.0
        kl_k3 = tl.minimum(tl.maximum(kl_k3, -10.0), 10.0)

        # abs: |diff|
        kl_abs = tl.abs(diff)

        # Select based on kl_type
        kl = tl.where(kl_type == 0, diff,
             tl.where(kl_type == 1, kl_k2,
             tl.where(kl_type == 2, kl_k3,
                      kl_abs)))

        kl_masked = kl * mask
        rewards   = (scores - kl_masked * kl_ratio) * mask

        tl.store(rewards_ptr + offsets, rewards, mask=valid)
        tl.atomic_add(kl_sum_ptr,  tl.sum(kl_masked, axis=0))
        tl.atomic_add(mask_sum_ptr, tl.sum(mask,     axis=0))


def compute_rewards_fused_triton(
    token_scores: torch.Tensor,
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    kl_ratio: float = 1.0,
    kl_type: str = "k1",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAVE_TRITON:
        raise RuntimeError("Triton not available")
    for t in (token_scores, log_prob, ref_log_prob, response_mask):
        if not t.is_cuda:
            raise RuntimeError("All tensors must be on CUDA")

    kl_int = _KL_TYPE_MAP.get(kl_type.lower())
    if kl_int is None:
        raise ValueError(f"Unknown kl_type: {kl_type!r}. Choose from {list(_KL_TYPE_MAP)}")

    numel = token_scores.numel()
    rewards   = torch.empty_like(token_scores, dtype=torch.float32)
    kl_sum    = torch.zeros((), device=token_scores.device, dtype=torch.float32)
    mask_sum  = torch.zeros((), device=token_scores.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _fused_rewards_kernel[grid](
        token_scores.contiguous(),
        log_prob.contiguous(),
        ref_log_prob.contiguous(),
        response_mask.contiguous(),
        rewards,
        kl_sum,
        mask_sum,
        numel,
        float(kl_ratio),
        int(kl_int),
    )

    kl_mean = kl_sum / (mask_sum + 1e-8)
    return rewards, kl_mean


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def compute_rewards_fused(
    token_scores: torch.Tensor,
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    kl_ratio: float = 1.0,
    kl_type: str = "k1",
    impl: RewardImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused reward computation with configurable KL penalty.

    Computes:
        kl     = kl_fn(log_prob, ref_log_prob)          # one of k1/k2/k3/abs
        reward = (token_scores - kl * kl_ratio) * mask

    Args:
        token_scores:  (B, T)  float32 or bfloat16 – raw reward signal
        log_prob:      (B, T)  float32 or bfloat16 – current policy log probs
        ref_log_prob:  (B, T)  float32 or bfloat16 – reference policy log probs
        response_mask: (B, T)  float32 – 1 for valid tokens, 0 for padding
        kl_ratio:      float   – KL penalty coefficient
        kl_type:       str     – "k1"/"kl", "k2"/"mse", "k3"/"low_var_kl"/"lvk", "abs"
        impl:          str     – "auto" | "torch" | "triton"

    Returns:
        rewards:  (B, T) float32 – per-token rewards (zero outside mask)
        kl_mean:  scalar float32 – masked-mean KL for monitoring
    """
    resolved = impl
    if resolved == "auto":
        resolved = (
            "triton"
            if HAVE_TRITON and token_scores.is_cuda
            else "torch"
        )

    if resolved == "triton":
        return compute_rewards_fused_triton(
            token_scores=token_scores,
            log_prob=log_prob,
            ref_log_prob=ref_log_prob,
            response_mask=response_mask,
            kl_ratio=kl_ratio,
            kl_type=kl_type,
        )
    return compute_rewards_fused_torch(
        token_scores=token_scores,
        log_prob=log_prob,
        ref_log_prob=ref_log_prob,
        response_mask=response_mask,
        kl_ratio=kl_ratio,
        kl_type=kl_type,
    )
