"""Tests for fused GSPO policy loss kernel (FIPO-029)."""

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


MOD = _load("gspo_loss", "verl/utils/kernel/gspo_loss.py")


def _ref(lp, olp, adv, m, low=0.2, high=0.2):
    return MOD.compute_gspo_loss_torch(lp, olp, adv, m, low, high)


class TestGspoLossTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _tensors(self, B=4, T=16, device="cpu"):
        lp  = torch.randn(B, T, device=device) * 0.2
        olp = torch.randn(B, T, device=device) * 0.2
        adv = torch.randn(B, T, device=device)
        m   = (torch.rand(B, T, device=device) > 0.2).float()
        return lp, olp, adv, m

    def test_output_shapes(self):
        lp, olp, adv, m = self._tensors()
        pg_loss, pg_cf, ppo_kl, pg_cfl = _ref(lp, olp, adv, m)
        for t in (pg_loss, pg_cf, ppo_kl, pg_cfl):
            self.assertEqual(t.shape, ())

    def test_pg_loss_finite(self):
        lp, olp, adv, m = self._tensors()
        pg_loss, _, _, _ = _ref(lp, olp, adv, m)
        self.assertFalse(pg_loss.isnan())
        self.assertFalse(pg_loss.isinf())

    def test_ppo_kl_non_negative_for_small_lp_diff(self):
        """For near-identical policies KL ~ 0."""
        B, T = 4, 16
        lp  = torch.randn(B, T) * 0.001
        olp = torch.randn(B, T) * 0.001
        adv = torch.randn(B, T)
        m   = torch.ones(B, T)
        _, _, ppo_kl, _ = _ref(lp, olp, adv, m)
        self.assertAlmostEqual(ppo_kl.item(), 0.0, delta=0.1)

    def test_equal_log_probs_clip_frac_zero(self):
        """When lp == olp the ratio is 1.0 → never clipped."""
        B, T = 4, 16
        lp = olp = torch.zeros(B, T)
        adv = torch.randn(B, T)
        m   = torch.ones(B, T)
        _, pg_cf, _, _ = _ref(lp, olp, adv, m)
        torch.testing.assert_close(pg_cf, torch.tensor(0.0), atol=1e-5, rtol=0)

    def test_pgclipl_always_zero(self):
        lp, olp, adv, m = self._tensors()
        _, _, _, pgl = _ref(lp, olp, adv, m)
        torch.testing.assert_close(pgl, torch.tensor(0.0), atol=1e-6, rtol=0)

    def test_dispatch_cpu(self):
        lp, olp, adv, m = self._tensors()
        r1 = _ref(lp, olp, adv, m)
        r2 = MOD.compute_gspo_loss(lp, olp, adv, m, impl="torch")
        torch.testing.assert_close(r1[0], r2[0], atol=1e-5, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestGspoLossTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)

    def _tensors(self, B=8, T=64, device="cuda"):
        lp  = torch.randn(B, T, device=device) * 0.3
        olp = torch.randn(B, T, device=device) * 0.3
        adv = torch.randn(B, T, device=device)
        m   = (torch.rand(B, T, device=device) > 0.2).float()
        return lp, olp, adv, m

    def test_triton_loss_close_to_torch(self):
        lp, olp, adv, m = self._tensors()
        pg_t, _, _, _ = MOD.compute_gspo_loss_triton(lp, olp, adv, m)
        pg_r, _, _, _ = MOD.compute_gspo_loss_torch( lp, olp, adv, m)
        torch.testing.assert_close(pg_t, pg_r, rtol=1e-3, atol=1e-3)

    def test_triton_kl_close_to_torch(self):
        lp, olp, adv, m = self._tensors()
        _, _, kl_t, _ = MOD.compute_gspo_loss_triton(lp, olp, adv, m)
        _, _, kl_r, _ = MOD.compute_gspo_loss_torch( lp, olp, adv, m)
        torch.testing.assert_close(kl_t, kl_r, rtol=1e-3, atol=1e-3)

    def test_triton_clipfrac_close_to_torch(self):
        lp, olp, adv, m = self._tensors()
        _, cf_t, _, _ = MOD.compute_gspo_loss_triton(lp, olp, adv, m)
        _, cf_r, _, _ = MOD.compute_gspo_loss_torch( lp, olp, adv, m)
        torch.testing.assert_close(cf_t, cf_r, rtol=1e-3, atol=1e-3)

    def test_output_scalars(self):
        lp, olp, adv, m = self._tensors()
        results = MOD.compute_gspo_loss_triton(lp, olp, adv, m)
        for r in results:
            self.assertEqual(r.shape, ())

    def test_finite_outputs(self):
        lp, olp, adv, m = self._tensors()
        for r in MOD.compute_gspo_loss_triton(lp, olp, adv, m):
            self.assertFalse(r.isnan(), f"NaN in output")
            self.assertFalse(r.isinf(), f"Inf in output")

    def test_large_batch(self):
        lp, olp, adv, m = self._tensors(B=32, T=2048)
        pg_t, _, _, _ = MOD.compute_gspo_loss_triton(lp, olp, adv, m)
        pg_r, _, _, _ = MOD.compute_gspo_loss_torch( lp, olp, adv, m)
        torch.testing.assert_close(pg_t, pg_r, rtol=1e-2, atol=1e-2)

    def test_equal_logprobs_no_clip(self):
        B, T = 4, 32
        lp = olp = torch.zeros(B, T, device="cuda")
        adv = torch.randn(B, T, device="cuda")
        m   = torch.ones(B, T, device="cuda")
        _, cf_t, _, _ = MOD.compute_gspo_loss_triton(lp, olp, adv, m)
        torch.testing.assert_close(cf_t, torch.tensor(0.0, device="cuda"), atol=1e-5, rtol=0)

    def test_auto_dispatch_cuda(self):
        lp, olp, adv, m = self._tensors()
        pg_a, _, _, _ = MOD.compute_gspo_loss(lp, olp, adv, m, impl="auto")
        pg_r, _, _, _ = MOD.compute_gspo_loss_torch(lp, olp, adv, m)
        torch.testing.assert_close(pg_a, pg_r, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
