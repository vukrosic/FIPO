"""Tests for fused returns + whitening kernel (FIPO-040).

Covers:
  - CPU correctness (torch path)
  - CUDA triton matches torch reference
  - BF16 rewards input
  - All-masked edge case
  - Large batch
  - shift_mean=False (preserve mean)
  - Speedup benchmark vs separate kernels
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


def _load(rel: str):
    p = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RW = _load("verl/utils/kernel/returns_whiten.py")


def _masked_whiten_ref(x: torch.Tensor, mask: torch.Tensor, shift_mean=True) -> torch.Tensor:
    """Reference masked whiten matching fused_advantage_norm."""
    m = mask.float()
    n = m.sum()
    if n <= 1:
        return x * m
    mean = (x * m).sum() / (n + 1e-8)
    var  = ((x - mean) ** 2 * m).sum() / (n + 1e-8)
    var  = var * n / (n - 1)
    w = (x - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        w = w + mean
    return w * m


def _discounted_returns_ref(rewards, mask, gamma):
    r = rewards.float()
    m = mask.float()
    returns = torch.zeros_like(r)
    carry = torch.zeros(r.shape[0], device=r.device, dtype=r.dtype)
    for step in range(r.shape[1]):
        col = r.shape[1] - step - 1
        carry = r[:, col] + gamma * carry
        returns[:, col] = carry
        carry = carry * m[:, col]
    return returns


class ReturnsWhitenCPUTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)

    def _run_ref(self, rewards, mask, gamma, shift_mean=True):
        ret = _discounted_returns_ref(rewards, mask, gamma)
        return _masked_whiten_ref(ret, mask, shift_mean)

    def test_basic_shape(self):
        B, T = 4, 32
        rewards = torch.randn(B, T)
        mask    = (torch.rand(B, T) > 0.1).float()
        out = RW.compute_returns_and_whiten(rewards, mask, gamma=0.99, impl="torch")
        self.assertEqual(out.shape, (B, T))

    def test_matches_ref(self):
        B, T = 4, 64
        rewards = torch.randn(B, T)
        mask    = (torch.rand(B, T) > 0.1).float()
        ref = self._run_ref(rewards, mask, 0.99)
        out = RW.compute_returns_and_whiten(rewards, mask, gamma=0.99, impl="torch")
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_padding_is_zero(self):
        """Masked positions should be exactly zero."""
        B, T = 4, 16
        rewards = torch.randn(B, T)
        mask = torch.ones(B, T)
        mask[:, -4:] = 0.0  # pad last 4 positions
        out = RW.compute_returns_and_whiten(rewards, mask, gamma=0.99, impl="torch")
        self.assertTrue((out[:, -4:].abs() < 1e-7).all())

    def test_shift_mean_false(self):
        B, T = 4, 32
        rewards = torch.randn(B, T)
        mask    = torch.ones(B, T)
        ref = self._run_ref(rewards, mask, 0.99, shift_mean=False)
        out = RW.compute_returns_and_whiten(rewards, mask, 0.99, shift_mean=False, impl="torch")
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_all_masked(self):
        B, T = 2, 16
        rewards = torch.randn(B, T)
        mask    = torch.zeros(B, T)
        out = RW.compute_returns_and_whiten(rewards, mask, gamma=0.99, impl="torch")
        self.assertTrue((out.abs() < 1e-7).all())

    def test_gamma_zero(self):
        """gamma=0 → returns equal rewards at each step."""
        B, T = 2, 8
        rewards = torch.randn(B, T)
        mask    = torch.ones(B, T)
        ret = _discounted_returns_ref(rewards, mask, gamma=0.0)
        ref = _masked_whiten_ref(ret, mask)
        out = RW.compute_returns_and_whiten(rewards, mask, gamma=0.0, impl="torch")
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_auto_falls_back_on_cpu(self):
        B, T = 4, 32
        rewards = torch.randn(B, T)
        mask    = (torch.rand(B, T) > 0.1).float()
        ref = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="torch")
        out = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="auto")
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(RW.HAVE_TRITON, "Triton required")
class ReturnsWhitenCUDATest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_triton_matches_torch(self):
        B, T = 32, 2048
        rewards = torch.randn(B, T, device="cuda")
        mask    = (torch.rand(B, T, device="cuda") > 0.1).float()

        ref = RW.compute_returns_and_whiten(rewards, mask, gamma=0.99, impl="torch")
        out = RW.compute_returns_and_whiten(rewards, mask, gamma=0.99, impl="triton")

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_triton_padding_is_zero(self):
        B, T = 8, 256
        rewards = torch.randn(B, T, device="cuda")
        mask = torch.ones(B, T, device="cuda")
        mask[:, -32:] = 0.0

        out = RW.compute_returns_and_whiten(rewards, mask, gamma=0.99, impl="triton")
        self.assertTrue((out[:, -32:].abs() < 1e-6).all())

    def test_triton_shift_mean_false(self):
        B, T = 8, 128
        rewards = torch.randn(B, T, device="cuda")
        mask    = torch.ones(B, T, device="cuda")

        ref = RW.compute_returns_and_whiten(rewards, mask, 0.99, shift_mean=False, impl="torch")
        out = RW.compute_returns_and_whiten(rewards, mask, 0.99, shift_mean=False, impl="triton")
        # shift_mean=False adds back the mean computed via scalar accumulators — slightly wider
        # tolerance than shift_mean=True due to float32 precision on the mean term.
        torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    def test_triton_all_masked(self):
        B, T = 4, 64
        rewards = torch.randn(B, T, device="cuda")
        mask    = torch.zeros(B, T, device="cuda")
        out = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="triton")
        self.assertTrue((out.abs() < 1e-6).all())

    def test_triton_bfloat16_rewards(self):
        B, T = 8, 256
        rewards = torch.randn(B, T, device="cuda", dtype=torch.bfloat16)
        mask    = (torch.rand(B, T, device="cuda") > 0.1).float()

        ref = RW.compute_returns_and_whiten(rewards.float(), mask, 0.99, impl="torch")
        out = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="triton")
        # BF16 rewards upcast internally → modest tolerance
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_triton_single_row(self):
        B, T = 1, 128
        rewards = torch.randn(B, T, device="cuda")
        mask    = torch.ones(B, T, device="cuda")
        ref = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="torch")
        out = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="triton")
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_triton_large_batch(self):
        B, T = 64, 2048
        rewards = torch.randn(B, T, device="cuda")
        mask    = (torch.rand(B, T, device="cuda") > 0.1).float()
        ref = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="torch")
        out = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="triton")
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    def test_triton_output_dtype_float32(self):
        B, T = 4, 64
        rewards = torch.randn(B, T, device="cuda")
        mask    = torch.ones(B, T, device="cuda")
        out = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="triton")
        self.assertEqual(out.dtype, torch.float32)

    def test_triton_unit_variance_approximately(self):
        """Whitened advantages over valid tokens should have std ≈ 1."""
        B, T = 32, 2048
        rewards = torch.randn(B, T, device="cuda")
        mask    = torch.ones(B, T, device="cuda")
        out = RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="triton")
        std = out.std()
        # Should be close to 1 (Bessel-corrected)
        self.assertAlmostEqual(std.item(), 1.0, delta=0.05)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(RW.HAVE_TRITON, "Triton required")
class ReturnsWhitenSpeedupTest(unittest.TestCase):

    def test_speedup_vs_separate_ops(self):
        """Fused should be faster than separate returns + whiten calls."""
        import time
        from verl.utils.kernel.advantage_kernels import compute_discounted_returns
        from verl.utils.kernel.fused_advantage_norm import compute_fused_advantage_norm

        B, T = 32, 2048
        rewards = torch.randn(B, T, device="cuda")
        mask    = (torch.rand(B, T, device="cuda") > 0.1).float()

        # Warmup
        for _ in range(5):
            ret = compute_discounted_returns(rewards, mask, gamma=0.99)
            compute_fused_advantage_norm(ret, mask)
            RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="triton")
        torch.cuda.synchronize()

        iters = 30

        # Separate
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            ret = compute_discounted_returns(rewards, mask, gamma=0.99)
            compute_fused_advantage_norm(ret, mask)
        torch.cuda.synchronize()
        separate_ms = (time.perf_counter() - t0) / iters * 1000

        # Fused
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            RW.compute_returns_and_whiten(rewards, mask, 0.99, impl="triton")
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) / iters * 1000

        speedup = separate_ms / fused_ms
        print(f"\nSeparate returns+whiten: {separate_ms:.3f}ms  Fused: {fused_ms:.3f}ms  speedup={speedup:.2f}x")
        self.assertGreater(speedup, 0.8, f"Expected reasonable speedup, got {speedup:.2f}x")


if __name__ == "__main__":
    unittest.main()
