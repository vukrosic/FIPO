"""Parity tests for the REINFORCE++ returns-plus-whitening kernel."""

from __future__ import annotations

import importlib.util
import types
import unittest
from pathlib import Path

import torch


def _load(rel: str):
    path = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RETURNS_WHITEN = _load("verl/utils/kernel/returns_whiten.py")
CORE_ALGOS = _load("verl/trainer/ppo/core_algos.py")


def _make_eos_style_mask(batch_size: int, response_len: int, device: str | torch.device = "cpu") -> torch.Tensor:
    lengths = torch.randint(1, response_len + 1, (batch_size,), device=device)
    positions = torch.arange(response_len, device=device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).float()


def _manual_discounted_returns(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    rewards_f = rewards.float()
    mask_f = mask.float()
    returns = torch.zeros_like(rewards_f)
    carry = torch.zeros(rewards.shape[0], device=rewards.device, dtype=torch.float32)

    for step in range(rewards.shape[1]):
        col = rewards.shape[1] - step - 1
        carry = rewards_f[:, col] + gamma * carry
        returns[:, col] = carry
        carry = carry * mask_f[:, col]

    return returns


def _manual_returns_and_whiten(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    shift_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = _manual_discounted_returns(rewards, mask, gamma)
    mask_f = mask.float()
    mask_sum = mask_f.sum()

    if mask_sum <= 1:
        return returns * mask_f, returns

    mean = (returns * mask_f).sum() / (mask_sum + 1e-8)
    centered = returns - mean
    var = (centered * centered * mask_f).sum() / (mask_sum + 1e-8)
    var = var * mask_sum / (mask_sum - 1)
    advantages = (returns - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        advantages = advantages + mean
    return advantages * mask_f, returns


class ReturnsWhitenCPUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_torch_matches_manual_reference(self):
        rewards = torch.randn(4, 64)
        mask = _make_eos_style_mask(4, 64)

        ref_adv, ref_returns = _manual_returns_and_whiten(rewards, mask, gamma=0.99)
        out_adv, out_returns = RETURNS_WHITEN.compute_returns_and_whiten(
            rewards,
            mask,
            gamma=0.99,
            impl="torch",
            return_returns=True,
        )

        torch.testing.assert_close(out_returns, ref_returns, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_adv, ref_adv, rtol=1e-6, atol=1e-6)

    def test_shift_mean_false_matches_reference(self):
        rewards = torch.randn(3, 33)
        mask = _make_eos_style_mask(3, 33)

        ref_adv, ref_returns = _manual_returns_and_whiten(rewards, mask, gamma=0.95, shift_mean=False)
        out_adv, out_returns = RETURNS_WHITEN.compute_returns_and_whiten(
            rewards,
            mask,
            gamma=0.95,
            shift_mean=False,
            impl="torch",
            return_returns=True,
        )

        torch.testing.assert_close(out_returns, ref_returns, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_adv, ref_adv, rtol=1e-6, atol=1e-6)

    def test_auto_falls_back_to_torch_on_cpu(self):
        rewards = torch.randn(2, 17)
        mask = _make_eos_style_mask(2, 17)

        ref_adv, ref_returns = RETURNS_WHITEN.compute_returns_and_whiten(
            rewards,
            mask,
            gamma=0.99,
            impl="torch",
            return_returns=True,
        )
        out_adv, out_returns = RETURNS_WHITEN.compute_returns_and_whiten(
            rewards,
            mask,
            gamma=0.99,
            impl="auto",
            return_returns=True,
        )

        torch.testing.assert_close(out_returns, ref_returns, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(out_adv, ref_adv, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(RETURNS_WHITEN.HAVE_TRITON, "Triton required")
class ReturnsWhitenCUDATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_triton_matches_torch_float32(self):
        rewards = torch.randn(16, 2048, device="cuda", dtype=torch.float32)
        mask = _make_eos_style_mask(16, 2048, device="cuda")

        ref_adv, ref_returns = RETURNS_WHITEN.compute_returns_and_whiten(
            rewards,
            mask,
            gamma=0.99,
            impl="torch",
            return_returns=True,
        )
        out_adv, out_returns = RETURNS_WHITEN.compute_returns_and_whiten(
            rewards,
            mask,
            gamma=0.99,
            impl="triton",
            return_returns=True,
        )

        torch.testing.assert_close(out_returns, ref_returns, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(out_adv, ref_adv, rtol=2e-4, atol=2e-4)

    def test_triton_matches_torch_bfloat16_reference(self):
        rewards = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
        mask = _make_eos_style_mask(8, 1024, device="cuda")

        ref_adv, ref_returns = RETURNS_WHITEN.compute_returns_and_whiten(
            rewards.float(),
            mask,
            gamma=0.99,
            impl="torch",
            return_returns=True,
        )
        out_adv, out_returns = RETURNS_WHITEN.compute_returns_and_whiten(
            rewards,
            mask,
            gamma=0.99,
            impl="triton",
            return_returns=True,
        )

        torch.testing.assert_close(out_returns, ref_returns, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(out_adv, ref_adv, rtol=5e-3, atol=5e-3)

    def test_triton_handles_all_masked(self):
        rewards = torch.randn(4, 64, device="cuda")
        mask = torch.zeros(4, 64, device="cuda")

        out_adv, out_returns = RETURNS_WHITEN.compute_returns_and_whiten(
            rewards,
            mask,
            gamma=0.99,
            impl="triton",
            return_returns=True,
        )

        self.assertTrue((out_adv.abs() < 1e-7).all())
        self.assertTrue(torch.isfinite(out_returns).all())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(RETURNS_WHITEN.HAVE_TRITON, "Triton required")
class ReinforcePlusPlusIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)

    def _run_core(self, rewards: torch.Tensor, mask: torch.Tensor, impl: str):
        config = types.SimpleNamespace(gamma=0.99, reinforce_plus_plus_impl=impl)
        return CORE_ALGOS.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=rewards,
            response_mask=mask,
            config=config,
        )

    def test_core_triton_matches_torch_on_float32(self):
        rewards = torch.randn(16, 1024, device="cuda", dtype=torch.float32)
        mask = _make_eos_style_mask(16, 1024, device="cuda")

        ref_adv, ref_returns = self._run_core(rewards, mask, "torch")
        out_adv, out_returns = self._run_core(rewards, mask, "triton")

        torch.testing.assert_close(out_returns, ref_returns, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(out_adv, ref_adv, rtol=2e-4, atol=2e-4)

    def test_core_auto_uses_triton_on_float32(self):
        rewards = torch.randn(8, 512, device="cuda", dtype=torch.float32)
        mask = _make_eos_style_mask(8, 512, device="cuda")

        auto_adv, auto_returns = self._run_core(rewards, mask, "auto")
        triton_adv, triton_returns = self._run_core(rewards, mask, "triton")

        torch.testing.assert_close(auto_returns, triton_returns, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(auto_adv, triton_adv, rtol=2e-4, atol=2e-4)

    def test_core_auto_falls_back_to_torch_on_bfloat16(self):
        rewards = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
        mask = _make_eos_style_mask(8, 512, device="cuda")

        auto_adv, auto_returns = self._run_core(rewards, mask, "auto")
        torch_adv, torch_returns = self._run_core(rewards, mask, "torch")

        torch.testing.assert_close(auto_returns.float(), torch_returns.float(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(auto_adv.float(), torch_adv.float(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
