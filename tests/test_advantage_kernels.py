from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

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


def _manual_gae_advantages_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    next_values = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    lastgaelam = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)

    for step in range(rewards.shape[1]):
        col = rewards.shape[1] - step - 1
        delta = rewards[:, col] + gamma * next_values - values[:, col]
        lastgaelam_candidate = delta + gamma * lam * lastgaelam
        next_values = values[:, col] * response_mask[:, col] + (1 - response_mask[:, col]) * next_values
        lastgaelam = lastgaelam_candidate * response_mask[:, col] + (1 - response_mask[:, col]) * lastgaelam
        advantages[:, col] = lastgaelam

    returns = advantages + values
    return advantages, returns


class AdvantageKernelCPUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_discounted_returns_torch_matches_manual_loop(self):
        rewards = torch.randn(5, 17, dtype=torch.float32)
        response_mask = _make_eos_style_mask(5, 17)
        gamma = 0.97

        expected = torch.zeros_like(rewards)
        carry = torch.zeros(rewards.shape[0], dtype=rewards.dtype)
        for step in range(rewards.shape[1]):
            col = rewards.shape[1] - step - 1
            carry = rewards[:, col] + gamma * carry
            expected[:, col] = carry
            carry = carry * response_mask[:, col]

        actual = ADVANTAGE_KERNELS.compute_discounted_returns_torch(rewards, response_mask, gamma)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_remax_style_returns_match_reverse_cumsum(self):
        rewards = torch.randn(4, 19, dtype=torch.float32)
        response_mask = _make_eos_style_mask(4, 19)

        expected = (rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        actual = ADVANTAGE_KERNELS.compute_discounted_returns_torch(rewards * response_mask, response_mask, 1.0)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_gae_torch_matches_manual_loop(self):
        rewards = torch.randn(5, 23, dtype=torch.float32)
        values = torch.randn(5, 23, dtype=torch.float32)
        response_mask = _make_eos_style_mask(5, 23)
        gamma = 0.99
        lam = 0.95

        expected_advantages, expected_returns = _manual_gae_advantages_returns(rewards, values, response_mask, gamma, lam)
        actual_advantages, actual_returns = ADVANTAGE_KERNELS.compute_gae_advantages_returns_torch(
            rewards,
            values,
            response_mask,
            gamma,
            lam,
        )
        torch.testing.assert_close(actual_advantages, expected_advantages, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual_returns, expected_returns, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton advantage kernel tests")
@unittest.skipUnless(ADVANTAGE_KERNELS.HAVE_TRITON, "Triton is required for Triton advantage kernel tests")
class AdvantageKernelCUDATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_triton_discounted_returns_matches_torch(self):
        rewards = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        response_mask = _make_eos_style_mask(16, 257, device="cuda")
        gamma = 0.99

        expected = ADVANTAGE_KERNELS.compute_discounted_returns(rewards, response_mask, gamma, impl="torch")
        actual = ADVANTAGE_KERNELS.compute_discounted_returns(rewards, response_mask, gamma, impl="triton")
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    def test_triton_remax_returns_matches_torch(self):
        rewards = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        response_mask = _make_eos_style_mask(16, 257, device="cuda")

        expected = ADVANTAGE_KERNELS.compute_discounted_returns(rewards * response_mask, response_mask, 1.0, impl="torch")
        actual = ADVANTAGE_KERNELS.compute_discounted_returns(rewards * response_mask, response_mask, 1.0, impl="triton")
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    def test_triton_gae_matches_torch(self):
        rewards = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        values = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        response_mask = _make_eos_style_mask(16, 257, device="cuda")
        gamma = 0.99
        lam = 0.95

        expected_advantages, expected_returns = ADVANTAGE_KERNELS.compute_gae_advantages_returns(
            rewards,
            values,
            response_mask,
            gamma,
            lam,
            impl="torch",
        )
        actual_advantages, actual_returns = ADVANTAGE_KERNELS.compute_gae_advantages_returns(
            rewards,
            values,
            response_mask,
            gamma,
            lam,
            impl="triton",
        )
        torch.testing.assert_close(actual_advantages, expected_advantages, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(actual_returns, expected_returns, rtol=2e-4, atol=2e-4)

    def test_bfloat16_inputs_can_be_upcast_before_triton_gae(self):
        rewards = torch.randn(16, 257, device="cuda", dtype=torch.bfloat16)
        values = torch.randn(16, 257, device="cuda", dtype=torch.bfloat16)
        response_mask = _make_eos_style_mask(16, 257, device="cuda")
        gamma = 0.99
        lam = 0.95

        expected_advantages, expected_returns = ADVANTAGE_KERNELS.compute_gae_advantages_returns(
            rewards.float(),
            values.float(),
            response_mask.float(),
            gamma,
            lam,
            impl="torch",
        )
        actual_advantages, actual_returns = ADVANTAGE_KERNELS.compute_gae_advantages_returns(
            rewards.float(),
            values.float(),
            response_mask.float(),
            gamma,
            lam,
            impl="triton",
        )
        torch.testing.assert_close(actual_advantages, expected_advantages, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(actual_returns, expected_returns, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
