#!/usr/bin/env python3
"""
Test parity between vectorized group advantage kernels and reference Python-loop implementations.
"""

from __future__ import annotations

import importlib.util
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _load_advantage_module():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "kernel" / "advantage_kernels.py"
    spec = importlib.util.spec_from_file_location("advantage_kernel_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ADVANTAGE_KERNELS = _load_advantage_module()


def _make_eos_style_mask(batch_size: int, response_len: int, device: str | torch.device = "cpu") -> torch.Tensor:
    lengths = torch.randint(1, response_len + 1, (batch_size,), device=device)
    positions = torch.arange(response_len, device=device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).float()


# =============================================================================
# GRPO Reference Implementation (Python loops)
# =============================================================================
def grpo_reference(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    norm_adv_by_std_in_grpo: bool = True,
    epsilon: float = 1e-6,
):
    """GRPO advantage computation with Python batch loops (reference implementation)."""
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    id2mean = {}
    id2std = {}
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0, device=scores.device)
            id2std[idx] = torch.tensor(1.0, device=scores.device)
        elif len(id2score[idx]) > 1:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
            id2std[idx] = torch.std(scores_tensor)

    result = scores.clone()
    for i in range(bsz):
        if norm_adv_by_std_in_grpo:
            result[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        else:
            result[i] = scores[i] - id2mean[index[i]]
    result = result.unsqueeze(-1) * response_mask
    return result


# =============================================================================
# RLOO Reference Implementation (Python loops)
# =============================================================================
def rloo_reference(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    """RLOO advantage computation with Python batch loops (reference implementation)."""
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    id2score = defaultdict(list)
    id2mean = {}

    # Group scores by index
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    # Compute per-group mean
    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0, device=scores.device)
        elif len(id2score[idx]) > 1:
            id2mean[idx] = torch.mean(torch.stack(id2score[idx]))

    # Compute RLOO advantage
    result = scores.clone()
    for i in range(bsz):
        response_num = len(id2score[index[i]])
        if response_num > 1:
            result[i] = (
                scores[i] * response_num / (response_num - 1)
                - id2mean[index[i]] * response_num / (response_num - 1)
            )

    result = result.unsqueeze(-1) * response_mask
    return result


# =============================================================================
# OPO Reference Implementation (Python loops)
# =============================================================================
def opo_reference(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    """OPO advantage computation with Python batch loops (reference implementation)."""
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2len[index[i]].append(response_length[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2bsl[idx] = torch.tensor(0.0)
        elif len(id2score[idx]) > 1:
            score_tensor = torch.stack(id2score[idx])
            len_tensor = torch.stack(id2len[idx])
            id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    result = scores.clone()
    for i in range(bsz):
        result[i] = scores[i] - id2bsl[index[i]]
    result = result.unsqueeze(-1) * response_mask
    return result


# =============================================================================
# GPG Reference Implementation (Python loops)
# =============================================================================
def gpg_reference(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
):
    """GPG advantage computation with Python batch loops (reference implementation)."""
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    bsz = scores.shape[0]
    m = torch.count_nonzero(scores)
    alpha = bsz / m.clamp(min=1)

    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        if len(id2score[idx]) == 1:
            id2mean[idx] = torch.tensor(0.0, device=scores.device)
        elif len(id2score[idx]) > 1:
            scores_tensor = torch.stack(id2score[idx])
            id2mean[idx] = torch.mean(scores_tensor)
        else:
            raise ValueError(f"no score in prompt index: {idx}")

    result = scores.clone()
    for i in range(bsz):
        result[i] = alpha * (scores[i] - id2mean[index[i]]) / f_norm
    result = result.unsqueeze(-1) * response_mask
    return result


# =============================================================================
# GRPO_PASSK Reference Implementation (Python loops)
# =============================================================================
def grpo_passk_reference(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """GRPO_PASSK advantage computation with Python batch loops (reference implementation).

    Only the best response per group gets a non-zero advantage: r_max - r_second_max.
    """
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    id2score = defaultdict(list)
    id2indices = defaultdict(list)

    for i in range(bsz):
        id2score[index[i]].append(scores[i])
        id2indices[index[i]].append(i)

    advantages = torch.zeros_like(scores)
    for idx in id2score:
        rewards = torch.stack(id2score[idx])  # (k,)
        if rewards.numel() < 2:
            raise ValueError(
                f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
            )
        topk, topk_idx = torch.topk(rewards, 2)
        r_max, r_second_max = topk[0], topk[1]
        i_max = id2indices[idx][topk_idx[0].item()]
        advantage = r_max - r_second_max
        if norm_adv_by_std_in_grpo:
            std = torch.std(rewards)
            advantage = advantage / (std + epsilon)
        advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages


# =============================================================================
# REINFORCE++ baseline Reference Implementation (Python loops)
# =============================================================================
def _masked_whiten_reference(values, mask, epsilon=1e-8):
    """Inlined masked whitening for reference implementation."""
    mask_sum = mask.sum()
    mean = (values * mask).sum() / (mask_sum + epsilon)
    centered = values - mean
    var = (centered * centered * mask).sum() / (mask_sum - 1 + epsilon)
    return centered * torch.rsqrt(var + epsilon)


def reinforce_plus_plus_baseline_reference(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    """REINFORCE++ baseline advantage computation with Python batch loops (reference implementation)."""
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]
    response_length = response_mask.shape[-1]

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = _masked_whiten_reference(scores, response_mask) * response_mask

    return scores


class GroupAdvantageKernelCPUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_grpo_vectorized_matches_reference(self):
        """Test that GRPO vectorized kernel matches reference Python-loop implementation."""
        batch_size = 128
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference implementation
        ref_result = grpo_reference(
            token_level_rewards.clone(), response_mask.clone(), index, norm_adv_by_std_in_grpo=True
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=True,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_grpo_vectorized_no_std_norm_matches_reference(self):
        """Test GRPO without std normalization matches reference."""
        batch_size = 128
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference implementation
        ref_result = grpo_reference(
            token_level_rewards.clone(), response_mask.clone(), index, norm_adv_by_std_in_grpo=False
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=False,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_rloo_vectorized_matches_reference(self):
        """Test that RLOO vectorized kernel matches reference Python-loop implementation."""
        batch_size = 128
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference implementation
        ref_result = rloo_reference(
            token_level_rewards.clone(), response_mask.clone(), index
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_rloo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_opo_vectorized_matches_reference(self):
        """Test that OPO vectorized kernel matches reference Python-loop implementation."""
        batch_size = 128
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference implementation
        ref_result = opo_reference(
            token_level_rewards.clone(), response_mask.clone(), index
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_opo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_gpg_vectorized_matches_reference(self):
        """Test that GPG vectorized kernel matches reference Python-loop implementation."""
        batch_size = 128
        response_len = 512
        num_groups = 32
        f_norm = 1.5

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference implementation
        ref_result = gpg_reference(
            token_level_rewards.clone(), response_mask.clone(), index, f_norm=f_norm
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_gpg_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            f_norm=f_norm,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_grpo_with_single_sample_per_group(self):
        """Test GRPO when some groups have only one sample (std=1, mean=0)."""
        batch_size = 64
        response_len = 256
        num_groups = 64  # Each group has exactly 1 sample

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.arange(batch_size)  # Each sample in its own group

        # Reference: with only 1 sample, std=1, mean=0, so advantage = score
        ref_result = grpo_reference(
            token_level_rewards.clone(), response_mask.clone(), index, norm_adv_by_std_in_grpo=True
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=True,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_rloo_with_single_sample_per_group(self):
        """Test RLOO when some groups have only one sample (should be 0)."""
        batch_size = 64
        response_len = 256
        num_groups = 64  # Each group has exactly 1 sample

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.arange(batch_size)  # Each sample in its own group

        # Reference: with only 1 sample, RLOO advantage = 0
        ref_result = rloo_reference(
            token_level_rewards.clone(), response_mask.clone(), index
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_rloo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_grpo_response_mask_application(self):
        """Test that response_mask is properly applied to advantages."""
        batch_size = 32
        response_len = 128
        num_groups = 8

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.random.randint(0, num_groups, size=batch_size)

        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=True,
        )

        # After masking, positions beyond response length should be 0
        lengths = response_mask.sum(dim=-1).long()
        for i in range(batch_size):
            length = lengths[i].item()
            if length < response_len:
                self.assertTrue(
                    torch.all(adv_result[i, length:] == 0),
                    f"Sample {i}: positions after length {length} should be 0",
                )

    def test_grpo_large_batch(self):
        """Test GRPO with large batch size."""
        batch_size = 1024
        response_len = 1024
        num_groups = 128

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference implementation
        ref_result = grpo_reference(
            token_level_rewards.clone(), response_mask.clone(), index, norm_adv_by_std_in_grpo=True
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=True,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_grpo_passk_vectorized_matches_reference(self):
        """Test that GRPO_PASSK vectorized kernel matches reference Python-loop implementation."""
        batch_size = 128
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        # Use round-robin assignment for equal group sizes (required for GRPO_PASSK vectorization)
        index = np.repeat(np.arange(num_groups), batch_size // num_groups)

        # Reference implementation
        ref_result = grpo_passk_reference(
            token_level_rewards.clone(), response_mask.clone(), index, norm_adv_by_std_in_grpo=True
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_passk_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=True,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_grpo_passk_vectorized_no_std_norm_matches_reference(self):
        """Test GRPO_PASSK without std normalization matches reference."""
        batch_size = 128
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        # Use round-robin assignment for equal group sizes (required for GRPO_PASSK vectorization)
        index = np.repeat(np.arange(num_groups), batch_size // num_groups)

        # Reference implementation
        ref_result = grpo_passk_reference(
            token_level_rewards.clone(), response_mask.clone(), index, norm_adv_by_std_in_grpo=False
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_passk_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=False,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_grpo_passk_only_one_sample_gets_advantage(self):
        """Test that only the sample with max reward gets non-zero advantage."""
        batch_size = 64
        response_len = 256
        num_groups = 16

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        # Use round-robin assignment for equal group sizes (required for GRPO_PASSK vectorization)
        index = np.repeat(np.arange(num_groups), batch_size // num_groups)

        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_passk_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=True,
        )

        # Count non-zero advantages per group
        for g in range(num_groups):
            group_mask = index == g
            group_adv = adv_result[group_mask].sum(dim=-1)  # Sum over sequence length
            # Only one sample should have non-zero advantage
            non_zero_count = (group_adv.abs() > 1e-6).sum().item()
            self.assertEqual(
                non_zero_count, 1,
                f"Group {g}: expected exactly 1 non-zero advantage, got {non_zero_count}"
            )

    def test_grpo_passk_large_batch(self):
        """Test GRPO_PASSK with large batch size."""
        batch_size = 1024
        response_len = 1024
        num_groups = 128

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        # Use round-robin assignment for equal group sizes (required for GRPO_PASSK vectorization)
        index = np.repeat(np.arange(num_groups), batch_size // num_groups)

        # Reference implementation
        ref_result = grpo_passk_reference(
            token_level_rewards.clone(), response_mask.clone(), index, norm_adv_by_std_in_grpo=True
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_passk_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=True,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_reinforce_plus_plus_baseline_vectorized_matches_reference(self):
        """Test that REINFORCE++ baseline vectorized kernel matches reference Python-loop implementation."""
        batch_size = 128
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference implementation
        ref_result = reinforce_plus_plus_baseline_reference(
            token_level_rewards.clone(), response_mask.clone(), index
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_reinforce_plus_plus_baseline_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_reinforce_plus_plus_baseline_with_single_sample_per_group(self):
        """Test REINFORCE++ baseline when some groups have only one sample (mean=0)."""
        batch_size = 64
        response_len = 256
        num_groups = 64  # Each group has exactly 1 sample

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.arange(batch_size)  # Each sample in its own group

        # Reference implementation
        ref_result = reinforce_plus_plus_baseline_reference(
            token_level_rewards.clone(), response_mask.clone(), index
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_reinforce_plus_plus_baseline_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)

    def test_reinforce_plus_plus_baseline_large_batch(self):
        """Test REINFORCE++ baseline with large batch size."""
        batch_size = 1024
        response_len = 1024
        num_groups = 128

        token_level_rewards = torch.randn(batch_size, response_len)
        response_mask = _make_eos_style_mask(batch_size, response_len)
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference implementation
        ref_result = reinforce_plus_plus_baseline_reference(
            token_level_rewards.clone(), response_mask.clone(), index
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long)
        adv_result, _ = ADVANTAGE_KERNELS.compute_reinforce_plus_plus_baseline_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
        )

        torch.testing.assert_close(adv_result, ref_result, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU tests")
class GroupAdvantageKernelCUDATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)

    def test_grpo_cuda_matches_reference(self):
        """Test GRPO CUDA kernel matches reference."""
        batch_size = 256
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len, device="cuda")
        response_mask = _make_eos_style_mask(batch_size, response_len, device="cuda")
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference (CPU)
        ref_result = grpo_reference(
            token_level_rewards.cpu().clone(),
            response_mask.cpu().clone(),
            index,
            norm_adv_by_std_in_grpo=True,
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long, device="cuda")
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=True,
        )

        torch.testing.assert_close(adv_result.cpu(), ref_result, rtol=1e-4, atol=1e-4)

    def test_rloo_cuda_matches_reference(self):
        """Test RLOO CUDA kernel matches reference."""
        batch_size = 256
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len, device="cuda")
        response_mask = _make_eos_style_mask(batch_size, response_len, device="cuda")
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference (CPU)
        ref_result = rloo_reference(
            token_level_rewards.cpu().clone(), response_mask.cpu().clone(), index
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long, device="cuda")
        adv_result, _ = ADVANTAGE_KERNELS.compute_rloo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
        )

        torch.testing.assert_close(adv_result.cpu(), ref_result, rtol=1e-4, atol=1e-4)

    def test_opo_cuda_matches_reference(self):
        """Test OPO CUDA kernel matches reference."""
        batch_size = 256
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len, device="cuda")
        response_mask = _make_eos_style_mask(batch_size, response_len, device="cuda")
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference (CPU)
        ref_result = opo_reference(
            token_level_rewards.cpu().clone(), response_mask.cpu().clone(), index
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long, device="cuda")
        adv_result, _ = ADVANTAGE_KERNELS.compute_opo_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
        )

        torch.testing.assert_close(adv_result.cpu(), ref_result, rtol=1e-4, atol=1e-4)

    def test_gpg_cuda_matches_reference(self):
        """Test GPG CUDA kernel matches reference."""
        batch_size = 256
        response_len = 512
        num_groups = 32
        f_norm = 1.5

        token_level_rewards = torch.randn(batch_size, response_len, device="cuda")
        response_mask = _make_eos_style_mask(batch_size, response_len, device="cuda")
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference (CPU)
        ref_result = gpg_reference(
            token_level_rewards.cpu().clone(),
            response_mask.cpu().clone(),
            index,
            f_norm=f_norm,
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long, device="cuda")
        adv_result, _ = ADVANTAGE_KERNELS.compute_gpg_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            f_norm=f_norm,
        )

        torch.testing.assert_close(adv_result.cpu(), ref_result, rtol=1e-4, atol=1e-4)

    def test_grpo_passk_cuda_matches_reference(self):
        """Test GRPO_PASSK CUDA kernel matches reference."""
        batch_size = 256
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len, device="cuda")
        response_mask = _make_eos_style_mask(batch_size, response_len, device="cuda")
        # Use round-robin assignment for equal group sizes (required for GRPO_PASSK vectorization)
        index = np.repeat(np.arange(num_groups), batch_size // num_groups)

        # Reference (CPU)
        ref_result = grpo_passk_reference(
            token_level_rewards.cpu().clone(),
            response_mask.cpu().clone(),
            index,
            norm_adv_by_std_in_grpo=True,
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long, device="cuda")
        adv_result, _ = ADVANTAGE_KERNELS.compute_grpo_passk_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
            norm_adv_by_std_in_grpo=True,
        )

        torch.testing.assert_close(adv_result.cpu(), ref_result, rtol=1e-4, atol=1e-4)

    def test_reinforce_plus_plus_baseline_cuda_matches_reference(self):
        """Test REINFORCE++ baseline CUDA kernel matches reference."""
        batch_size = 256
        response_len = 512
        num_groups = 32

        token_level_rewards = torch.randn(batch_size, response_len, device="cuda")
        response_mask = _make_eos_style_mask(batch_size, response_len, device="cuda")
        index = np.random.randint(0, num_groups, size=batch_size)

        # Reference (CPU)
        ref_result = reinforce_plus_plus_baseline_reference(
            token_level_rewards.cpu().clone(),
            response_mask.cpu().clone(),
            index,
        )

        # Vectorized kernel
        index_tensor = torch.tensor(index, dtype=torch.long, device="cuda")
        adv_result, _ = ADVANTAGE_KERNELS.compute_reinforce_plus_plus_baseline_outcome_advantage_torch(
            token_level_rewards,
            response_mask,
            index_tensor,
        )

        torch.testing.assert_close(adv_result.cpu(), ref_result, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
