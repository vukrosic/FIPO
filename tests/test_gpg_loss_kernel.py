"""Tests for fused GPG policy loss kernel (FIPO-036)."""

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


MOD = _load("fused_gpg_loss", "verl/utils/kernel/fused_gpg_loss.py")

MODES = ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]


class TestGpgLossTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _tensors(self, B=4, T=16, device="cpu"):
        lp  = torch.randn(B, T, device=device) * 0.5
        adv = torch.randn(B, T, device=device)
        m   = (torch.rand(B, T, device=device) > 0.2).float()
        return lp, adv, m

    def test_all_modes_finite(self):
        lp, adv, m = self._tensors()
        for mode in MODES:
            pg_loss, _, _, _ = MOD.compute_gpg_loss_torch(lp, adv, m, mode)
            self.assertFalse(pg_loss.isnan(), f"NaN for mode={mode}")
            self.assertFalse(pg_loss.isinf(), f"Inf for mode={mode}")

    def test_output_shapes(self):
        lp, adv, m = self._tensors()
        results = MOD.compute_gpg_loss_torch(lp, adv, m)
        for r in results:
            self.assertEqual(r.shape, ())

    def test_zeros_return_zero(self):
        B, T = 4, 16
        lp = torch.zeros(B, T)
        adv = torch.randn(B, T)
        m = torch.ones(B, T)
        pg_loss, _, _, _ = MOD.compute_gpg_loss_torch(lp, adv, m)
        self.assertAlmostEqual(pg_loss.item(), 0.0, delta=1e-6)

    def test_dispatch_cpu(self):
        lp, adv, m = self._tensors()
        for mode in MODES:
            r1 = MOD.compute_gpg_loss_torch(lp, adv, m, mode)
            r2 = MOD.compute_gpg_loss(lp, adv, m, mode, impl="torch")
            torch.testing.assert_close(r1[0], r2[0], atol=1e-5, rtol=0)

    def test_manual_token_mean(self):
        B, T = 2, 4
        lp = torch.ones(B, T)
        adv = torch.ones(B, T) * 2
        m = torch.ones(B, T)
        pg_loss, _, _, _ = MOD.compute_gpg_loss_torch(lp, adv, m, "token-mean")
        # -lp * adv = -2, mean = -2
        torch.testing.assert_close(pg_loss, torch.tensor(-2.0), atol=1e-5, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestGpgLossTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _tensors(self, B=8, T=64, device="cuda"):
        lp  = torch.randn(B, T, device=device) * 0.5
        adv = torch.randn(B, T, device=device)
        m   = (torch.rand(B, T, device=device) > 0.2).float()
        return lp, adv, m

    def test_all_modes_match_torch(self):
        lp, adv, m = self._tensors()
        for mode in MODES:
            pg_t, _, _, _ = MOD.compute_gpg_loss_triton(lp, adv, m, mode)
            pg_r, _, _, _ = MOD.compute_gpg_loss_torch(lp, adv, m, mode)
            torch.testing.assert_close(pg_t, pg_r, rtol=1e-3, atol=1e-3,
                                       msg=f"Mismatch for mode={mode}")

    def test_large_batch(self):
        lp, adv, m = self._tensors(B=32, T=2048)
        for mode in MODES:
            pg_t, _, _, _ = MOD.compute_gpg_loss_triton(lp, adv, m, mode)
            pg_r, _, _, _ = MOD.compute_gpg_loss_torch(lp, adv, m, mode)
            torch.testing.assert_close(pg_t, pg_r, rtol=1e-2, atol=1e-2,
                                       msg=f"Mismatch for mode={mode}")

    def test_output_scalars(self):
        lp, adv, m = self._tensors()
        results = MOD.compute_gpg_loss_triton(lp, adv, m)
        for r in results:
            self.assertEqual(r.shape, ())

    def test_auto_dispatch_cuda(self):
        lp, adv, m = self._tensors()
        for mode in MODES:
            pg_a, _, _, _ = MOD.compute_gpg_loss(lp, adv, m, mode, impl="auto")
            pg_r, _, _, _ = MOD.compute_gpg_loss_torch(lp, adv, m, mode)
            torch.testing.assert_close(pg_a, pg_r, rtol=1e-3, atol=1e-3)

    def test_bf16_input(self):
        lp, adv, m = self._tensors()
        lp16, adv16 = lp.bfloat16(), adv.bfloat16()
        for mode in MODES:
            pg_t, _, _, _ = MOD.compute_gpg_loss_triton(lp16, adv16, m, mode)
            pg_r, _, _, _ = MOD.compute_gpg_loss_torch(lp, adv, m, mode)
            torch.testing.assert_close(pg_t, pg_r, rtol=5e-2, atol=5e-2)

    def test_zeros_return_zero(self):
        B, T = 4, 32
        lp = torch.zeros(B, T, device="cuda")
        adv = torch.randn(B, T, device="cuda")
        m = torch.ones(B, T, device="cuda")
        pg_t, _, _, _ = MOD.compute_gpg_loss_triton(lp, adv, m)
        self.assertAlmostEqual(pg_t.item(), 0.0, delta=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
