"""Tests for fused reward computation kernel (FIPO-027)."""

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


MOD = _load("reward_utils", "verl/utils/kernel/reward_utils.py")


def _ref(scores, lp, rlp, mask, kl_ratio, kl_type):
    return MOD.compute_rewards_fused_torch(scores, lp, rlp, mask, kl_ratio, kl_type)


class TestRewardsTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _tensors(self, B=4, T=16, device="cpu"):
        scores = torch.randn(B, T, device=device)
        lp     = torch.randn(B, T, device=device) * 0.5
        rlp    = torch.randn(B, T, device=device) * 0.5
        mask   = (torch.rand(B, T, device=device) > 0.2).float()
        return scores, lp, rlp, mask

    def test_k1_rewards_shape(self):
        s, lp, rlp, m = self._tensors()
        r, km = MOD.compute_rewards_fused_torch(s, lp, rlp, m, 0.1, "k1")
        self.assertEqual(r.shape, s.shape)
        self.assertEqual(km.shape, ())

    def test_k1_outside_mask_zero(self):
        s, lp, rlp, m = self._tensors()
        r, _ = MOD.compute_rewards_fused_torch(s, lp, rlp, m, 0.1, "k1")
        self.assertTrue((r[m == 0] == 0).all())

    def test_k1_formula(self):
        B, T = 2, 8
        s   = torch.ones(B, T)
        lp  = torch.full((B, T), 0.3)
        rlp = torch.full((B, T), 0.1)
        m   = torch.ones(B, T)
        r, km = MOD.compute_rewards_fused_torch(s, lp, rlp, m, 1.0, "k1")
        # kl = 0.3 - 0.1 = 0.2; reward = 1 - 0.2 = 0.8
        torch.testing.assert_close(r, torch.full((B, T), 0.8), atol=1e-5, rtol=0)
        torch.testing.assert_close(km, torch.tensor(0.2), atol=1e-5, rtol=0)

    def test_all_kl_types_smoke(self):
        s, lp, rlp, m = self._tensors()
        for kt in ("k1", "kl", "k2", "mse", "k3", "low_var_kl", "lvk", "abs"):
            r, km = MOD.compute_rewards_fused_torch(s, lp, rlp, m, 0.05, kt)
            self.assertEqual(r.shape, s.shape)
            self.assertFalse(r.isnan().any(), f"NaN in rewards for kl_type={kt}")

    def test_bfloat16_input(self):
        s, lp, rlp, m = self._tensors()
        r, km = MOD.compute_rewards_fused_torch(
            s.bfloat16(), lp.bfloat16(), rlp.bfloat16(), m, 0.1, "k1"
        )
        self.assertEqual(r.dtype, torch.float32)

    def test_output_float32(self):
        s, lp, rlp, m = self._tensors()
        r, km = MOD.compute_rewards_fused_torch(s, lp, rlp, m, 0.1, "k3")
        self.assertEqual(r.dtype, torch.float32)
        self.assertEqual(km.dtype, torch.float32)

    def test_kl_ratio_zero_returns_scores(self):
        s, lp, rlp, m = self._tensors()
        r, _ = MOD.compute_rewards_fused_torch(s, lp, rlp, m, 0.0, "k1")
        expected = s.float() * m
        torch.testing.assert_close(r, expected, atol=1e-5, rtol=0)

    def test_unknown_kl_type_raises(self):
        s, lp, rlp, m = self._tensors()
        with self.assertRaises(ValueError):
            MOD.compute_rewards_fused_torch(s, lp, rlp, m, 0.1, "unknown_type")

    def test_dispatch_cpu(self):
        s, lp, rlp, m = self._tensors()
        r1, km1 = MOD.compute_rewards_fused(s, lp, rlp, m, 0.1, "k1", impl="torch")
        r2, km2 = MOD.compute_rewards_fused(s, lp, rlp, m, 0.1, "k1", impl="auto")
        torch.testing.assert_close(r1, r2)
        torch.testing.assert_close(km1, km2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestRewardsTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _tensors(self, B=8, T=32, device="cuda"):
        s   = torch.randn(B, T, device=device)
        lp  = torch.randn(B, T, device=device) * 0.5
        rlp = torch.randn(B, T, device=device) * 0.5
        m   = (torch.rand(B, T, device=device) > 0.2).float()
        return s, lp, rlp, m

    def test_triton_matches_torch_k1(self):
        s, lp, rlp, m = self._tensors()
        r_t, km_t = MOD.compute_rewards_fused_triton(s, lp, rlp, m, 0.1, "k1")
        r_r, km_r = MOD.compute_rewards_fused_torch( s, lp, rlp, m, 0.1, "k1")
        torch.testing.assert_close(r_t, r_r, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(km_t, km_r, rtol=1e-4, atol=1e-4)

    def test_triton_matches_torch_k2(self):
        s, lp, rlp, m = self._tensors()
        r_t, km_t = MOD.compute_rewards_fused_triton(s, lp, rlp, m, 0.05, "k2")
        r_r, km_r = MOD.compute_rewards_fused_torch( s, lp, rlp, m, 0.05, "k2")
        torch.testing.assert_close(r_t, r_r, rtol=1e-4, atol=1e-4)

    def test_triton_matches_torch_k3(self):
        s, lp, rlp, m = self._tensors()
        r_t, km_t = MOD.compute_rewards_fused_triton(s, lp, rlp, m, 0.1, "k3")
        r_r, km_r = MOD.compute_rewards_fused_torch( s, lp, rlp, m, 0.1, "k3")
        torch.testing.assert_close(r_t, r_r, rtol=1e-4, atol=1e-4)

    def test_triton_matches_torch_abs(self):
        s, lp, rlp, m = self._tensors()
        r_t, _ = MOD.compute_rewards_fused_triton(s, lp, rlp, m, 0.1, "abs")
        r_r, _ = MOD.compute_rewards_fused_torch( s, lp, rlp, m, 0.1, "abs")
        torch.testing.assert_close(r_t, r_r, rtol=1e-4, atol=1e-4)

    def test_outside_mask_zero(self):
        s, lp, rlp, m = self._tensors()
        r, _ = MOD.compute_rewards_fused_triton(s, lp, rlp, m, 0.1, "k1")
        self.assertTrue((r[m == 0] == 0).all())

    def test_output_float32(self):
        s, lp, rlp, m = self._tensors()
        r, km = MOD.compute_rewards_fused_triton(s, lp, rlp, m, 0.1, "k1")
        self.assertEqual(r.dtype, torch.float32)
        self.assertEqual(km.dtype, torch.float32)

    def test_bfloat16_input(self):
        s, lp, rlp, m = self._tensors()
        r_t, _  = MOD.compute_rewards_fused_triton(
            s.bfloat16(), lp.bfloat16(), rlp.bfloat16(), m, 0.1, "k1"
        )
        r_r, _ = MOD.compute_rewards_fused_torch(
            s.bfloat16(), lp.bfloat16(), rlp.bfloat16(), m, 0.1, "k1"
        )
        torch.testing.assert_close(r_t, r_r, rtol=1e-2, atol=1e-2)

    def test_auto_dispatch_cuda(self):
        s, lp, rlp, m = self._tensors()
        r_auto, _ = MOD.compute_rewards_fused(s, lp, rlp, m, 0.1, "k1", impl="auto")
        r_ref,  _ = MOD.compute_rewards_fused_torch(s, lp, rlp, m, 0.1, "k1")
        torch.testing.assert_close(r_auto, r_ref, rtol=1e-4, atol=1e-4)

    def test_large_batch(self):
        s, lp, rlp, m = self._tensors(B=32, T=2048)
        r_t, _ = MOD.compute_rewards_fused_triton(s, lp, rlp, m, 0.1, "k1")
        r_r, _ = MOD.compute_rewards_fused_torch( s, lp, rlp, m, 0.1, "k1")
        torch.testing.assert_close(r_t, r_r, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
