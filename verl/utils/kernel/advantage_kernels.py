from __future__ import annotations

from typing import Literal

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


AdvantageKernelImpl = Literal["auto", "torch", "triton"]


def compute_discounted_returns_torch(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    returns = torch.zeros_like(token_level_rewards)
    carry = torch.zeros(token_level_rewards.shape[0], device=token_level_rewards.device, dtype=token_level_rewards.dtype)
    gamma_t = torch.tensor(gamma, device=token_level_rewards.device, dtype=token_level_rewards.dtype)

    for step in range(token_level_rewards.shape[1]):
        col = token_level_rewards.shape[1] - step - 1
        carry = token_level_rewards[:, col] + gamma_t * carry
        returns[:, col] = carry
        carry = carry * response_mask[:, col]

    return returns


if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=1),
        ],
        key=["response_len"],
    )
    @triton.jit
    def _discounted_returns_reverse_scan_kernel(
        rewards_ptr,
        mask_ptr,
        output_ptr,
        batch_size,
        response_len,
        rewards_stride_batch: tl.int64,
        rewards_stride_seq: tl.int64,
        mask_stride_batch: tl.int64,
        mask_stride_seq: tl.int64,
        output_stride_batch: tl.int64,
        output_stride_seq: tl.int64,
        gamma: tl.float32,
        BLOCK_ROWS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < batch_size
        carry = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

        for step in range(0, response_len):
            col = response_len - step - 1
            rewards = tl.load(
                rewards_ptr + rows * rewards_stride_batch + col * rewards_stride_seq,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            response = tl.load(
                mask_ptr + rows * mask_stride_batch + col * mask_stride_seq,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            carry = rewards + gamma * carry
            tl.store(
                output_ptr + rows * output_stride_batch + col * output_stride_seq,
                carry,
                mask=row_mask,
            )
            carry = carry * response


def compute_discounted_returns_triton(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    if not HAVE_TRITON:
        raise RuntimeError("Triton is not installed.")
    if not token_level_rewards.is_cuda or not response_mask.is_cuda:
        raise RuntimeError("Triton discounted returns require CUDA tensors.")
    if token_level_rewards.dtype != torch.float32 or response_mask.dtype != torch.float32:
        raise RuntimeError("Triton discounted returns currently support float32 inputs only.")

    output = torch.empty_like(token_level_rewards)
    grid = lambda meta: (triton.cdiv(token_level_rewards.shape[0], meta["BLOCK_ROWS"]),)
    _discounted_returns_reverse_scan_kernel[grid](
        token_level_rewards,
        response_mask,
        output,
        token_level_rewards.shape[0],
        token_level_rewards.shape[1],
        token_level_rewards.stride(0),
        token_level_rewards.stride(1),
        response_mask.stride(0),
        response_mask.stride(1),
        output.stride(0),
        output.stride(1),
        float(gamma),
    )
    return output


def compute_discounted_returns(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    impl: AdvantageKernelImpl = "auto",
) -> torch.Tensor:
    resolved_impl = impl
    if resolved_impl == "auto":
        resolved_impl = (
            "triton"
            if HAVE_TRITON
            and token_level_rewards.is_cuda
            and response_mask.is_cuda
            and token_level_rewards.dtype == torch.float32
            and response_mask.dtype == torch.float32
            else "torch"
        )

    if resolved_impl == "triton":
        return compute_discounted_returns_triton(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            gamma=gamma,
        )
    if resolved_impl == "torch":
        return compute_discounted_returns_torch(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            gamma=gamma,
        )

    raise ValueError(f"Unsupported discounted returns implementation: {impl}")


def compute_gae_advantages_returns_torch(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(token_level_rewards)
    next_values = torch.zeros(token_level_rewards.shape[0], device=token_level_rewards.device, dtype=token_level_rewards.dtype)
    lastgaelam = torch.zeros(token_level_rewards.shape[0], device=token_level_rewards.device, dtype=token_level_rewards.dtype)
    gamma_t = torch.tensor(gamma, device=token_level_rewards.device, dtype=token_level_rewards.dtype)
    lam_t = torch.tensor(lam, device=token_level_rewards.device, dtype=token_level_rewards.dtype)

    for step in range(token_level_rewards.shape[1]):
        col = token_level_rewards.shape[1] - step - 1
        delta = token_level_rewards[:, col] + gamma_t * next_values - values[:, col]
        lastgaelam_candidate = delta + gamma_t * lam_t * lastgaelam
        next_values = values[:, col] * response_mask[:, col] + (1 - response_mask[:, col]) * next_values
        lastgaelam = lastgaelam_candidate * response_mask[:, col] + (1 - response_mask[:, col]) * lastgaelam
        advantages[:, col] = lastgaelam

    returns = advantages + values
    return advantages, returns


if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=1, num_stages=1),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=2, num_stages=1),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=1),
        ],
        key=["response_len"],
    )
    @triton.jit
    def _gae_reverse_scan_kernel(
        rewards_ptr,
        values_ptr,
        mask_ptr,
        advantages_ptr,
        returns_ptr,
        batch_size,
        response_len,
        rewards_stride_batch: tl.int64,
        rewards_stride_seq: tl.int64,
        values_stride_batch: tl.int64,
        values_stride_seq: tl.int64,
        mask_stride_batch: tl.int64,
        mask_stride_seq: tl.int64,
        advantages_stride_batch: tl.int64,
        advantages_stride_seq: tl.int64,
        returns_stride_batch: tl.int64,
        returns_stride_seq: tl.int64,
        gamma: tl.float32,
        lam: tl.float32,
        BLOCK_ROWS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < batch_size
        next_values = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
        lastgaelam = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)

        for step in range(0, response_len):
            col = response_len - step - 1
            rewards = tl.load(
                rewards_ptr + rows * rewards_stride_batch + col * rewards_stride_seq,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            values = tl.load(
                values_ptr + rows * values_stride_batch + col * values_stride_seq,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            response = tl.load(
                mask_ptr + rows * mask_stride_batch + col * mask_stride_seq,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)

            delta = rewards + gamma * next_values - values
            lastgaelam_candidate = delta + gamma * lam * lastgaelam
            next_values = values * response + (1.0 - response) * next_values
            lastgaelam = lastgaelam_candidate * response + (1.0 - response) * lastgaelam

            tl.store(
                advantages_ptr + rows * advantages_stride_batch + col * advantages_stride_seq,
                lastgaelam,
                mask=row_mask,
            )
            tl.store(
                returns_ptr + rows * returns_stride_batch + col * returns_stride_seq,
                lastgaelam + values,
                mask=row_mask,
            )


def compute_gae_advantages_returns_triton(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not HAVE_TRITON:
        raise RuntimeError("Triton is not installed.")
    if not token_level_rewards.is_cuda or not values.is_cuda or not response_mask.is_cuda:
        raise RuntimeError("Triton GAE requires CUDA tensors.")
    if token_level_rewards.dtype != torch.float32 or values.dtype != torch.float32 or response_mask.dtype != torch.float32:
        raise RuntimeError("Triton GAE currently supports float32 inputs only.")

    advantages = torch.empty_like(token_level_rewards)
    returns = torch.empty_like(token_level_rewards)
    grid = lambda meta: (triton.cdiv(token_level_rewards.shape[0], meta["BLOCK_ROWS"]),)
    _gae_reverse_scan_kernel[grid](
        token_level_rewards,
        values,
        response_mask,
        advantages,
        returns,
        token_level_rewards.shape[0],
        token_level_rewards.shape[1],
        token_level_rewards.stride(0),
        token_level_rewards.stride(1),
        values.stride(0),
        values.stride(1),
        response_mask.stride(0),
        response_mask.stride(1),
        advantages.stride(0),
        advantages.stride(1),
        returns.stride(0),
        returns.stride(1),
        float(gamma),
        float(lam),
    )
    return advantages, returns


def compute_gae_advantages_returns(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
    impl: AdvantageKernelImpl = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_impl = impl
    if resolved_impl == "auto":
        resolved_impl = (
            "triton"
            if HAVE_TRITON
            and token_level_rewards.is_cuda
            and values.is_cuda
            and response_mask.is_cuda
            and token_level_rewards.dtype == torch.float32
            and values.dtype == torch.float32
            and response_mask.dtype == torch.float32
            else "torch"
        )

    if resolved_impl == "triton":
        return compute_gae_advantages_returns_triton(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
    if resolved_impl == "torch":
        return compute_gae_advantages_returns_torch(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )

    raise ValueError(f"Unsupported GAE implementation: {impl}")


# =============================================================================
# GRPO-style group advantage (vectorized with index_add)
# =============================================================================


def compute_grpo_outcome_advantage_torch(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO advantage computation using index_add for per-group accumulation.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) - group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: whether to scale advantage by std

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) - same as advantages for GRPO
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bsz = scores.shape[0]
    num_groups = int(index.max()) + 1
    index_tensor = index.to(torch.long)

    # Group accumulators
    group_sums = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_squares = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_counts = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)

    # Efficient per-group accumulation using index_add
    group_sums.index_add_(0, index_tensor, scores)
    group_squares.index_add_(0, index_tensor, scores.square())
    group_counts.index_add_(0, index_tensor, torch.ones(bsz, device=scores.device, dtype=torch.float32))

    # Compute mean and std
    group_means = group_sums / (group_counts + epsilon)
    # Use sample variance (n-1 denominator) to match torch.std behavior
    group_var = (group_squares / (group_counts + epsilon) - group_means.square()) * (group_counts / (group_counts - 1 + epsilon))
    group_stds = torch.sqrt(group_var + epsilon)
    # For groups of size 1, set std=1 and mean=0 to match reference behavior
    group_stds = torch.where(group_counts > 1, group_stds, torch.ones_like(group_stds))
    group_means = torch.where(group_counts > 1, group_means, torch.zeros_like(group_means))

    # Broadcast back to original order
    orig_means = group_means[index_tensor]
    orig_stds = group_stds[index_tensor]

    if norm_adv_by_std_in_grpo:
        advantages = (scores - orig_means) / (orig_stds + epsilon)
    else:
        advantages = scores - orig_means

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


# =============================================================================
# RLOO-style group advantage (vectorized with index_add)
# =============================================================================


def compute_rloo_outcome_advantage_torch(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized RLOO advantage computation using index_add for per-group accumulation.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) - group ID per sample
        epsilon: float for numerical stability

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) - same as advantages for RLOO
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bsz = scores.shape[0]
    num_groups = int(index.max()) + 1
    index_tensor = index.to(torch.long)

    # Group sums and counts
    group_sums = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_counts = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)

    group_sums.index_add_(0, index_tensor, scores)
    group_counts.index_add_(0, index_tensor, torch.ones(bsz, device=scores.device, dtype=torch.float32))

    # Group means
    group_means = group_sums / (group_counts + epsilon)

    # Compute per-sample RLOO score: r_i * n/(n-1) - mean * n/(n-1)
    n = group_counts[index_tensor]
    factor = n / (n - 1 + epsilon)
    advantages = scores * factor - group_means[index_tensor] * factor

    # For n=1, keep the original score (the RLOO formula is not defined for n=1)
    advantages = torch.where(n > 1, advantages, scores)

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


# =============================================================================
# OPO-style group advantage (vectorized with index_add)
# =============================================================================


def compute_opo_outcome_advantage_torch(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized OPO advantage computation using index_add for per-group accumulation.

    OPO computes a length-normalized baseline per group: sum(len_i * score_i) / sum(len_i)

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) - group ID per sample
        epsilon: float for numerical stability

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) - same as advantages for OPO
    """
    response_length = response_mask.sum(dim=-1)  # (bs,)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bsz = scores.shape[0]
    num_groups = int(index.max()) + 1
    index_tensor = index.to(torch.long)

    # Group weighted sums, length sums, and counts
    # baseline = sum(len_i * score_i) / sum(len_i)
    group_weighted_sums = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_length_sums = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)
    group_counts = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)

    weighted_scores = response_length * scores  # len_i * score_i
    group_weighted_sums.index_add_(0, index_tensor, weighted_scores)
    group_length_sums.index_add_(0, index_tensor, response_length.float())
    group_counts.index_add_(0, index_tensor, torch.ones(bsz, device=scores.device, dtype=torch.float32))

    # Compute baseline per group
    group_baselines = group_weighted_sums / (group_length_sums + epsilon)
    # For groups of size 1, set baseline=0 to match reference behavior
    group_baselines = torch.where(group_counts > 1, group_baselines, torch.zeros_like(group_baselines))

    # Broadcast back and subtract baseline
    advantages = scores - group_baselines[index_tensor]
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


# =============================================================================
# GPG-style group advantage (vectorized with index_add)
# =============================================================================


def compute_gpg_outcome_advantage_torch(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GPG advantage computation using index_add for per-group accumulation.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) - group ID per sample
        epsilon: float for numerical stability
        f_norm: float normalization factor

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) - same as advantages for GPG
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bsz = scores.shape[0]
    num_groups = int(index.max()) + 1
    index_tensor = index.to(torch.long)

    # Group accumulators
    group_sums = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_counts = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)

    group_sums.index_add_(0, index_tensor, scores)
    group_counts.index_add_(0, index_tensor, torch.ones(bsz, device=scores.device, dtype=torch.float32))

    # Compute alpha: bsz / non_zero_count
    m = torch.count_nonzero(scores)
    alpha = bsz / m.clamp(min=1)

    # Compute mean
    group_means = group_sums / (group_counts + epsilon)
    # For groups of size 1, set mean=0 to match reference behavior
    group_means = torch.where(group_counts > 1, group_means, torch.zeros_like(group_means))

    # Broadcast back and apply GPG formula
    orig_means = group_means[index_tensor]
    advantages = alpha * (scores - orig_means) / f_norm
    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


# =============================================================================
# GRPO_PASSK-style group advantage (vectorized with topk + scatter)
# =============================================================================


def compute_grpo_passk_outcome_advantage_torch(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO Pass@k advantage computation using topk and scatter.

    The advantage is computed as r_max - r_second_max for each group,
    assigned only to the sample with r_max.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) - group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: whether to scale advantage by std

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) - same as advantages for GRPO
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bsz = scores.shape[0]
    num_groups = int(index.max()) + 1
    index_tensor = index.to(torch.long)

    # Compute per-group statistics for std normalization
    group_counts = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)
    group_sums = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_squares = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)

    group_counts.index_add_(0, index_tensor, torch.ones(bsz, device=scores.device, dtype=torch.float32))
    group_sums.index_add_(0, index_tensor, scores)
    group_squares.index_add_(0, index_tensor, scores.square())

    # Compute mean and std per group
    group_means = group_sums / (group_counts + epsilon)
    group_var = (group_squares / (group_counts + epsilon) - group_means.square()) * (
        group_counts / (group_counts - 1 + epsilon)
    )
    group_stds = torch.sqrt(group_var + epsilon)
    # For groups of size 1, set std=1 to match reference behavior
    group_stds = torch.where(group_counts > 1, group_stds, torch.ones_like(group_stds))

    # Find top-2 per group using topk on reshaped view
    # Assumes equal group sizes (standard GRPO assumption)
    k = bsz // num_groups
    assert bsz % num_groups == 0, f"GRPO_PASSK requires equal group sizes, got bsz={bsz}, num_groups={num_groups}"

    # Reshape scores to (num_groups, k) for per-group topk
    scores_view = scores.view(num_groups, k)
    sorted_values, sorted_indices = torch.sort(scores_view, dim=1, descending=True)

    # max is at sorted_indices[:, 0], second max at sorted_indices[:, 1]
    max_values = sorted_values[:, 0]
    second_max_values = sorted_values[:, 1]

    # Map local indices to global flattened indices
    # group_offsets[g] = g * k (starting index of group g in flattened array)
    group_offsets = torch.arange(num_groups, device=scores.device, dtype=torch.long) * k
    max_global_idx = group_offsets + sorted_indices[:, 0]
    second_max_global_idx = group_offsets + sorted_indices[:, 1]

    # Compute advantage: r_max - r_second_max
    adv_per_group = max_values - second_max_values
    if norm_adv_by_std_in_grpo:
        adv_per_group = adv_per_group / (group_stds + epsilon)

    # Scatter advantages to original positions (only the max sample gets non-zero)
    advantages = torch.zeros(bsz, device=scores.device, dtype=scores.dtype)
    advantages[max_global_idx] = adv_per_group

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


# =============================================================================
# REINFORCE++ baseline-style group advantage (vectorized with index_add)
# =============================================================================


def compute_reinforce_plus_plus_baseline_outcome_advantage_torch(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized REINFORCE++ baseline advantage computation using index_add for per-group accumulation.

    The algorithm:
    1. Group scores by index
    2. For groups of size 1: mean = 0
    3. For groups of size > 1: mean = average of scores in group
    4. Subtract mean from each score
    5. Tile to full sequence length and apply response mask
    6. Apply masked whitening
    7. Apply response mask again

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) - group ID per sample
        epsilon: float for numerical stability

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length) - same as advantages for REINFORCE++ baseline
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bsz = scores.shape[0]
    num_groups = int(index.max()) + 1
    index_tensor = index.to(torch.long)

    # Group accumulators
    group_sums = torch.zeros(num_groups, device=scores.device, dtype=scores.dtype)
    group_counts = torch.zeros(num_groups, device=scores.device, dtype=torch.float32)

    # Efficient per-group accumulation using index_add
    group_sums.index_add_(0, index_tensor, scores)
    group_counts.index_add_(0, index_tensor, torch.ones(bsz, device=scores.device, dtype=torch.float32))

    # Compute group means
    # For groups of size 1, set mean=0 to match reference behavior
    group_means = group_sums / (group_counts + epsilon)
    group_means = torch.where(group_counts > 1, group_means, torch.zeros_like(group_means))

    # Broadcast back and subtract mean
    advantages = scores - group_means[index_tensor]

    # Tile to full sequence length and apply response mask
    response_length = response_mask.shape[-1]
    advantages = advantages.unsqueeze(-1).tile([1, response_length]) * response_mask

    # Apply masked whitening (same as verl_F.masked_whiten)
    # Compute masked mean: sum(values * mask) / sum(mask)
    mask_sum = response_mask.sum()
    masked_mean = (advantages * response_mask).sum() / (mask_sum + epsilon)
    # Compute masked variance with Bessel's correction
    centered = advantages - masked_mean
    masked_var = (centered * centered * response_mask).sum() / (mask_sum - 1 + epsilon)
    # Whiten: (values - mean) / sqrt(var)
    advantages = centered * torch.rsqrt(masked_var + epsilon)

    # Apply response mask again
    advantages = advantages * response_mask

    return advantages, advantages
