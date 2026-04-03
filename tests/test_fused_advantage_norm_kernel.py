"""Tests for fused advantage normalization kernel (FIPO-038)."""

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


MOD = _load("fused_advantage_norm", "verl/utils/kernel/fused_advantage_norm.py")


class TestAdvNormTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _tensors(self, B=4, T=16, device="cpu"):
        v = torch.randn(B, T, device=device)
        m = (torch.rand(B, T, device=device) > 0.2).float()
        return v, m

    def test_output_shape(self):
        v, m = self._tensors()
        out = MOD.compute_fused_advantage_norm_torch(v, m)
        self.assertEqual(out.shape, v.shape)

    def test_zero_mean(self):
        """Whitened values should have ~zero masked mean."""
        v, m = self._tensors(B=16, T=64)
        out = MOD.compute_fused_advantage_norm_torch(v, m)
        masked_mean = (out * m).sum() / (m.sum() + 1e-8)
        self.assertAlmostEqual(masked_mean.item(), 0.0, delta=0.01)

    def test_unit_variance(self):
        """Whitened values should have ~unit masked variance."""
        v, m = self._tensors(B=16, T=64)
        out = MOD.compute_fused_advantage_norm_torch(v, m)
        masked_mean = (out * m).sum() / (m.sum() + 1e-8)
        masked_var = ((out - masked_mean) ** 2 * m).sum() / (m.sum() + 1e-8)
        # With Bessel's correction the variance should be ~1
        self.assertAlmostEqual(masked_var.item(), 1.0, delta=0.15)

    def test_masked_positions_zero(self):
        v, m = self._tensors()
        out = MOD.compute_fused_advantage_norm_torch(v, m)
        zero_mask = m == 0
        self.assertTrue((out[zero_mask] == 0).all())

    def test_all_masked_passthrough(self):
        B, T = 4, 16
        v = torch.randn(B, T)
        m = torch.zeros(B, T)
        out = MOD.compute_fused_advantage_norm_torch(v, m)
        torch.testing.assert_close(out, torch.zeros(B, T), atol=1e-6, rtol=0)

    def test_dispatch_cpu(self):
        v, m = self._tensors()
        r1 = MOD.compute_fused_advantage_norm_torch(v, m)
        r2 = MOD.compute_fused_advantage_norm(v, m, impl="torch")
        torch.testing.assert_close(r1, r2, atol=1e-5, rtol=0)

    def test_no_shift_mean(self):
        v, m = self._tensors(B=16, T=64)
        out = MOD.compute_fused_advantage_norm_torch(v, m, shift_mean=False)
        masked_mean_orig = (v * m).sum() / (m.sum() + 1e-8)
        masked_mean_out = (out * m).sum() / (m.sum() + 1e-8)
        # Mean should be preserved
        self.assertAlmostEqual(masked_mean_out.item(), masked_mean_orig.item(), delta=0.05)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestAdvNormTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _tensors(self, B=8, T=64, device="cuda"):
        v = torch.randn(B, T, device=device)
        m = (torch.rand(B, T, device=device) > 0.2).float()
        return v, m

    def test_matches_torch(self):
        v, m = self._tensors()
        out_t = MOD.compute_fused_advantage_norm_triton(v, m)
        out_r = MOD.compute_fused_advantage_norm_torch(v, m)
        torch.testing.assert_close(out_t, out_r, rtol=1e-3, atol=1e-3)

    def test_large_batch(self):
        v, m = self._tensors(B=32, T=2048)
        out_t = MOD.compute_fused_advantage_norm_triton(v, m)
        out_r = MOD.compute_fused_advantage_norm_torch(v, m)
        torch.testing.assert_close(out_t, out_r, rtol=1e-2, atol=1e-2)

    def test_masked_positions_zero(self):
        v, m = self._tensors()
        out = MOD.compute_fused_advantage_norm_triton(v, m)
        zero_mask = m == 0
        # Triton result at masked positions should be 0
        self.assertTrue((out[zero_mask].abs() < 1e-5).all())

    def test_auto_dispatch_cuda(self):
        v, m = self._tensors()
        out_a = MOD.compute_fused_advantage_norm(v, m, impl="auto")
        out_r = MOD.compute_fused_advantage_norm_torch(v, m)
        torch.testing.assert_close(out_a, out_r, rtol=1e-3, atol=1e-3)

    def test_bf16_input(self):
        v, m = self._tensors()
        out_t = MOD.compute_fused_advantage_norm_triton(v.bfloat16(), m)
        out_r = MOD.compute_fused_advantage_norm_torch(v, m)
        torch.testing.assert_close(out_t, out_r, rtol=5e-2, atol=5e-2)

    def test_no_shift_mean_matches(self):
        v, m = self._tensors()
        out_t = MOD.compute_fused_advantage_norm_triton(v, m, shift_mean=False)
        out_r = MOD.compute_fused_advantage_norm_torch(v, m, shift_mean=False)
        torch.testing.assert_close(out_t, out_r, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
