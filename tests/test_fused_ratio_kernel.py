"""Tests for fused ratio + KL kernel (FIPO-037)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


def _load(name, rel):
    path = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load("fused_ratio", "verl/utils/kernel/fused_ratio.py")


class TestFusedRatioTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _tensors(self, B=4, T=16, device="cpu"):
        lp  = torch.randn(B, T, device=device) * 0.3
        olp = torch.randn(B, T, device=device) * 0.3
        m   = (torch.rand(B, T, device=device) > 0.2).float()
        return lp, olp, m

    def test_shapes(self):
        lp, olp, m = self._tensors()
        ratio, kl = MOD.compute_fused_ratio_torch(lp, olp, m)
        self.assertEqual(ratio.shape, lp.shape)
        self.assertEqual(kl.shape, ())

    def test_ratio_positive(self):
        lp, olp, m = self._tensors()
        ratio, _ = MOD.compute_fused_ratio_torch(lp, olp, m)
        self.assertTrue((ratio > 0).all())

    def test_equal_lp_ratio_one(self):
        B, T = 4, 16
        lp = olp = torch.zeros(B, T)
        m = torch.ones(B, T)
        ratio, kl = MOD.compute_fused_ratio_torch(lp, olp, m)
        torch.testing.assert_close(ratio, torch.ones(B, T), atol=1e-5, rtol=0)
        self.assertAlmostEqual(kl.item(), 0.0, delta=1e-6)

    def test_kl_nonneg(self):
        """KL = masked_mean(old_lp - lp) which can be negative, but for identical policies = 0."""
        lp, olp, m = self._tensors()
        _, kl = MOD.compute_fused_ratio_torch(lp, olp, m)
        self.assertFalse(kl.isnan())

    def test_clamping(self):
        """Large neg_kl values should be clamped."""
        B, T = 2, 4
        lp = torch.full((B, T), 30.0)
        olp = torch.zeros(B, T)
        m = torch.ones(B, T)
        ratio, _ = MOD.compute_fused_ratio_torch(lp, olp, m, clamp_max=20.0)
        expected = torch.full((B, T), torch.exp(torch.tensor(20.0)).item())
        torch.testing.assert_close(ratio, expected, atol=1e-3, rtol=1e-3)

    def test_dispatch_cpu(self):
        lp, olp, m = self._tensors()
        r1 = MOD.compute_fused_ratio_torch(lp, olp, m)
        r2 = MOD.compute_fused_ratio(lp, olp, m, impl="torch")
        torch.testing.assert_close(r1[0], r2[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(r1[1], r2[1], atol=1e-5, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestFusedRatioTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _tensors(self, B=8, T=64, device="cuda"):
        lp  = torch.randn(B, T, device=device) * 0.3
        olp = torch.randn(B, T, device=device) * 0.3
        m   = (torch.rand(B, T, device=device) > 0.2).float()
        return lp, olp, m

    def test_ratio_matches_torch(self):
        lp, olp, m = self._tensors()
        r_t, kl_t = MOD.compute_fused_ratio_triton(lp, olp, m)
        r_r, kl_r = MOD.compute_fused_ratio_torch(lp, olp, m)
        torch.testing.assert_close(r_t, r_r, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(kl_t, kl_r, rtol=1e-3, atol=1e-3)

    def test_large_batch(self):
        lp, olp, m = self._tensors(B=32, T=2048)
        r_t, kl_t = MOD.compute_fused_ratio_triton(lp, olp, m)
        r_r, kl_r = MOD.compute_fused_ratio_torch(lp, olp, m)
        torch.testing.assert_close(r_t, r_r, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(kl_t, kl_r, rtol=1e-2, atol=1e-2)

    def test_auto_dispatch_cuda(self):
        lp, olp, m = self._tensors()
        r_a, kl_a = MOD.compute_fused_ratio(lp, olp, m, impl="auto")
        r_r, kl_r = MOD.compute_fused_ratio_torch(lp, olp, m)
        torch.testing.assert_close(r_a, r_r, rtol=1e-4, atol=1e-4)

    def test_bf16_input(self):
        lp, olp, m = self._tensors()
        r_t, _ = MOD.compute_fused_ratio_triton(lp.bfloat16(), olp.bfloat16(), m)
        r_r, _ = MOD.compute_fused_ratio_torch(lp, olp, m)
        torch.testing.assert_close(r_t, r_r, rtol=5e-2, atol=5e-2)

    def test_output_device(self):
        lp, olp, m = self._tensors()
        ratio, kl = MOD.compute_fused_ratio_triton(lp, olp, m)
        self.assertEqual(ratio.device.type, "cuda")
        self.assertEqual(kl.device.type, "cuda")

    def test_clamping_triton(self):
        B, T = 2, 4
        lp = torch.full((B, T), 30.0, device="cuda")
        olp = torch.zeros(B, T, device="cuda")
        m = torch.ones(B, T, device="cuda")
        ratio, _ = MOD.compute_fused_ratio_triton(lp, olp, m, clamp_max=20.0)
        expected_val = torch.exp(torch.tensor(20.0)).item()
        self.assertAlmostEqual(ratio[0, 0].item(), expected_val, delta=256.0)  # exp(20) ~ 4.8e8, float32 eps


if __name__ == "__main__":
    unittest.main(verbosity=2)
