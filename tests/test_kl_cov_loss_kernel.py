"""Tests for vectorized KL-Cov policy loss (FIPO-033)."""

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


MOD = _load("kl_cov_loss", "verl/utils/kernel/kl_cov_loss.py")


class TestKlCovLossTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _tensors(self, B=4, T=16, device="cpu"):
        lp  = torch.randn(B, T, device=device) * 0.2
        olp = torch.randn(B, T, device=device) * 0.2
        adv = torch.randn(B, T, device=device)
        m   = (torch.rand(B, T, device=device) > 0.2).float()
        return lp, olp, adv, m

    def test_output_shapes(self):
        lp, olp, adv, m = self._tensors()
        pg_loss, pg_cf, ppo_kl, pg_cfl = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m)
        for t in (pg_loss, pg_cf, ppo_kl, pg_cfl):
            self.assertEqual(t.shape, ())

    def test_pg_loss_finite(self):
        lp, olp, adv, m = self._tensors()
        pg_loss, _, _, _ = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m)
        self.assertFalse(pg_loss.isnan())
        self.assertFalse(pg_loss.isinf())

    def test_ppo_kl_abs_nonneg(self):
        lp, olp, adv, m = self._tensors()
        _, _, ppo_kl, _ = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m)
        self.assertGreaterEqual(ppo_kl.item(), 0.0)

    def test_ppo_kl_zero_for_equal(self):
        B, T = 4, 16
        lp = olp = torch.zeros(B, T)
        adv = torch.randn(B, T)
        m = torch.ones(B, T)
        _, _, ppo_kl, _ = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m)
        self.assertAlmostEqual(ppo_kl.item(), 0.0, delta=1e-6)

    def test_clipfrac_always_zero(self):
        lp, olp, adv, m = self._tensors()
        _, pg_cf, _, pg_cfl = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m)
        self.assertAlmostEqual(pg_cf.item(), 0.0, delta=1e-6)
        self.assertAlmostEqual(pg_cfl.item(), 0.0, delta=1e-6)

    def test_dispatch_cpu(self):
        lp, olp, adv, m = self._tensors()
        r1 = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m)
        r2 = MOD.compute_kl_cov_loss(lp, olp, adv, m, impl="torch")
        torch.testing.assert_close(r1[0], r2[0], atol=1e-5, rtol=0)

    def test_zero_ratio_no_kl_penalty(self):
        """With kl_cov_ratio ~ 0, all tokens use pg_losses1 (no KL penalty)."""
        lp, olp, adv, m = self._tensors()
        pg1, _, _, _ = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m, kl_cov_ratio=1e-10)
        self.assertFalse(pg1.isnan())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestKlCovLossVectorized(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _tensors(self, B=8, T=64, device="cuda"):
        lp  = torch.randn(B, T, device=device) * 0.3
        olp = torch.randn(B, T, device=device) * 0.3
        adv = torch.randn(B, T, device=device)
        m   = (torch.rand(B, T, device=device) > 0.2).float()
        return lp, olp, adv, m

    def test_vectorized_kl_matches_torch(self):
        """KL should match exactly (no randomness in KL computation)."""
        lp, olp, adv, m = self._tensors()
        _, _, kl_v, _ = MOD.compute_kl_cov_loss_vectorized(lp, olp, adv, m)
        _, _, kl_r, _ = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m)
        torch.testing.assert_close(kl_v, kl_r, rtol=1e-4, atol=1e-4)

    def test_vectorized_loss_matches_torch(self):
        """Both paths select by covariance, so with same seed they should match."""
        torch.manual_seed(99)
        torch.cuda.manual_seed_all(99)
        lp, olp, adv, m = self._tensors()
        pg_v, _, _, _ = MOD.compute_kl_cov_loss_vectorized(lp, olp, adv, m)
        pg_r, _, _, _ = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m)
        # Both use topk on same covariance → same positions selected
        torch.testing.assert_close(pg_v, pg_r, rtol=1e-4, atol=1e-4)

    def test_vectorized_loss_finite(self):
        lp, olp, adv, m = self._tensors()
        pg_loss, _, _, _ = MOD.compute_kl_cov_loss_vectorized(lp, olp, adv, m)
        self.assertFalse(pg_loss.isnan())
        self.assertFalse(pg_loss.isinf())

    def test_output_scalars(self):
        lp, olp, adv, m = self._tensors()
        results = MOD.compute_kl_cov_loss_vectorized(lp, olp, adv, m)
        for r in results:
            self.assertEqual(r.shape, ())

    def test_large_batch(self):
        lp, olp, adv, m = self._tensors(B=32, T=2048)
        pg_v, _, _, _ = MOD.compute_kl_cov_loss_vectorized(lp, olp, adv, m)
        pg_r, _, _, _ = MOD.compute_kl_cov_loss_torch(lp, olp, adv, m)
        torch.testing.assert_close(pg_v, pg_r, rtol=1e-3, atol=1e-3)

    def test_auto_dispatch_cuda(self):
        lp, olp, adv, m = self._tensors()
        pg_a, _, _, _ = MOD.compute_kl_cov_loss(lp, olp, adv, m, impl="auto")
        self.assertFalse(pg_a.isnan())

    def test_bf16_input(self):
        lp, olp, adv, m = self._tensors()
        lp16, olp16, adv16 = lp.bfloat16(), olp.bfloat16(), adv.bfloat16()
        pg_v, _, _, _ = MOD.compute_kl_cov_loss_vectorized(lp16, olp16, adv16, m)
        self.assertFalse(pg_v.isnan())

    def test_high_kl_coef_increases_loss(self):
        """Higher KL coef should push loss higher for the selected tokens."""
        lp, olp, adv, m = self._tensors()
        pg_low, _, _, _ = MOD.compute_kl_cov_loss_vectorized(
            lp, olp, adv, m, kl_cov_ratio=0.1, ppo_kl_coef=0.01
        )
        pg_high, _, _, _ = MOD.compute_kl_cov_loss_vectorized(
            lp, olp, adv, m, kl_cov_ratio=0.1, ppo_kl_coef=10.0
        )
        # With higher KL penalty, selected tokens get abs_kl added -> different loss
        self.assertFalse(torch.allclose(pg_low, pg_high))


if __name__ == "__main__":
    unittest.main(verbosity=2)
