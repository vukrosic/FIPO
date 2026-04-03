"""Tests for fused GMPO policy loss kernel (FIPO-030)."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load(name, rel):
    path = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load("gmpo_loss", "verl/utils/kernel/gmpo_loss.py")
CORE = _load("core_algos", "verl/trainer/ppo/core_algos.py")


def _ref(lp, olp, adv, m, low=0.2, high=0.2):
    return MOD.compute_gmpo_loss_torch(lp, olp, adv, m, low, high)


def _cfg(impl: str):
    return SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        policy_loss=SimpleNamespace(gmpo_impl=impl),
    )


class TestGmpoLossTorch(unittest.TestCase):
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

    def test_equal_logprobs_ratio_one(self):
        """When lp == olp, neg_kl=0, ratio=exp(0)=1."""
        B, T = 4, 16
        lp = olp = torch.zeros(B, T)
        adv = torch.randn(B, T)
        m = torch.ones(B, T)
        pg_loss, pg_cf, ppo_kl, _ = _ref(lp, olp, adv, m)
        # ratio should be 1, so pg_loss = -mean(adv)
        expected = -(adv.sum(dim=-1) / T).mean()
        torch.testing.assert_close(pg_loss, expected, atol=1e-5, rtol=1e-5)

    def test_ppo_kl_nonneg_for_identical(self):
        B, T = 4, 16
        lp = olp = torch.zeros(B, T)
        adv = torch.randn(B, T)
        m = torch.ones(B, T)
        _, _, ppo_kl, _ = _ref(lp, olp, adv, m)
        self.assertAlmostEqual(ppo_kl.item(), 0.0, delta=1e-6)

    def test_clipfrac_zero_no_clip(self):
        """When neg_kl is within clip range, clipfrac should be 0."""
        B, T = 4, 16
        lp = olp = torch.zeros(B, T)
        adv = torch.randn(B, T)
        m = torch.ones(B, T)
        _, pg_cf, _, pg_cfl = _ref(lp, olp, adv, m)
        self.assertAlmostEqual(pg_cf.item(), 0.0, delta=1e-5)
        self.assertAlmostEqual(pg_cfl.item(), 0.0, delta=1e-5)

    def test_dispatch_cpu(self):
        lp, olp, adv, m = self._tensors()
        r1 = _ref(lp, olp, adv, m)
        r2 = MOD.compute_gmpo_loss(lp, olp, adv, m, impl="torch")
        torch.testing.assert_close(r1[0], r2[0], atol=1e-5, rtol=0)

    def test_core_geo_mean_torch_matches_reference(self):
        lp, olp, adv, m = self._tensors()
        ref = _ref(lp, olp, adv, m)
        out = CORE.compute_policy_loss_geo_mean(olp, lp, adv, m, config=_cfg("torch"))
        for actual, expected in zip(out, ref):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestGmpoLossTriton(unittest.TestCase):
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
        pg_t, _, _, _ = MOD.compute_gmpo_loss_triton(lp, olp, adv, m)
        pg_r, _, _, _ = MOD.compute_gmpo_loss_torch(lp, olp, adv, m)
        torch.testing.assert_close(pg_t, pg_r, rtol=1e-3, atol=1e-3)

    def test_triton_kl_close_to_torch(self):
        lp, olp, adv, m = self._tensors()
        _, _, kl_t, _ = MOD.compute_gmpo_loss_triton(lp, olp, adv, m)
        _, _, kl_r, _ = MOD.compute_gmpo_loss_torch(lp, olp, adv, m)
        torch.testing.assert_close(kl_t, kl_r, rtol=1e-3, atol=1e-3)

    def test_triton_clipfrac_close_to_torch(self):
        lp, olp, adv, m = self._tensors()
        _, cf_t, _, cfl_t = MOD.compute_gmpo_loss_triton(lp, olp, adv, m)
        _, cf_r, _, cfl_r = MOD.compute_gmpo_loss_torch(lp, olp, adv, m)
        torch.testing.assert_close(cf_t, cf_r, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(cfl_t, cfl_r, rtol=1e-3, atol=1e-3)

    def test_output_scalars(self):
        lp, olp, adv, m = self._tensors()
        results = MOD.compute_gmpo_loss_triton(lp, olp, adv, m)
        for r in results:
            self.assertEqual(r.shape, ())

    def test_finite_outputs(self):
        lp, olp, adv, m = self._tensors()
        for r in MOD.compute_gmpo_loss_triton(lp, olp, adv, m):
            self.assertFalse(r.isnan())
            self.assertFalse(r.isinf())

    def test_large_batch(self):
        lp, olp, adv, m = self._tensors(B=32, T=2048)
        pg_t, _, _, _ = MOD.compute_gmpo_loss_triton(lp, olp, adv, m)
        pg_r, _, _, _ = MOD.compute_gmpo_loss_torch(lp, olp, adv, m)
        torch.testing.assert_close(pg_t, pg_r, rtol=1e-2, atol=1e-2)

    def test_equal_logprobs_ratio_one(self):
        B, T = 4, 32
        lp = olp = torch.zeros(B, T, device="cuda")
        adv = torch.randn(B, T, device="cuda")
        m = torch.ones(B, T, device="cuda")
        pg_t, _, _, _ = MOD.compute_gmpo_loss_triton(lp, olp, adv, m)
        expected = -(adv.sum(dim=-1) / T).mean()
        torch.testing.assert_close(pg_t, expected, atol=1e-4, rtol=1e-4)

    def test_auto_dispatch_cuda(self):
        lp, olp, adv, m = self._tensors()
        pg_a, _, _, _ = MOD.compute_gmpo_loss(lp, olp, adv, m, impl="auto")
        pg_r, _, _, _ = MOD.compute_gmpo_loss_torch(lp, olp, adv, m)
        torch.testing.assert_close(pg_a, pg_r, rtol=1e-3, atol=1e-3)

    def test_bf16_input(self):
        lp, olp, adv, m = self._tensors()
        lp16, olp16, adv16 = lp.bfloat16(), olp.bfloat16(), adv.bfloat16()
        pg_t, _, _, _ = MOD.compute_gmpo_loss_triton(lp16, olp16, adv16, m)
        pg_r, _, _, _ = MOD.compute_gmpo_loss_torch(lp, olp, adv, m)
        torch.testing.assert_close(pg_t, pg_r, rtol=5e-2, atol=5e-2)

    def test_all_masked(self):
        B, T = 4, 32
        lp = torch.randn(B, T, device="cuda") * 0.3
        olp = torch.randn(B, T, device="cuda") * 0.3
        adv = torch.randn(B, T, device="cuda")
        m = torch.zeros(B, T, device="cuda")
        pg_t, _, _, _ = MOD.compute_gmpo_loss_triton(lp, olp, adv, m)
        self.assertFalse(pg_t.isnan())

    def test_core_geo_mean_auto_matches_reference(self):
        lp, olp, adv, m = self._tensors(B=16, T=512)
        ref = MOD.compute_gmpo_loss_torch(lp, olp, adv, m)
        out = CORE.compute_policy_loss_geo_mean(olp, lp, adv, m, config=_cfg("auto"))
        for actual, expected in zip(out, ref):
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
