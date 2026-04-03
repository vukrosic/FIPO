"""Tests for fused agg_loss kernel (FIPO-034)."""

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


MOD = _load("agg_loss", "verl/utils/kernel/agg_loss.py")

MODES = ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]


class TestAggLossTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _tensors(self, B=4, T=16, device="cpu"):
        loss = torch.randn(B, T, device=device)
        mask = (torch.rand(B, T, device=device) > 0.2).float()
        return loss, mask

    def test_all_modes_finite(self):
        loss, mask = self._tensors()
        for mode in MODES:
            result = MOD.compute_agg_loss_torch(loss, mask, mode)
            self.assertEqual(result.shape, (), f"mode={mode}")
            self.assertFalse(result.isnan(), f"NaN for mode={mode}")
            self.assertFalse(result.isinf(), f"Inf for mode={mode}")

    def test_token_mean_manual(self):
        B, T = 2, 4
        loss = torch.ones(B, T)
        mask = torch.ones(B, T)
        result = MOD.compute_agg_loss_torch(loss, mask, "token-mean")
        torch.testing.assert_close(result, torch.tensor(1.0), atol=1e-5, rtol=0)

    def test_seq_mean_token_sum_manual(self):
        B, T = 2, 4
        loss = torch.ones(B, T)
        mask = torch.ones(B, T)
        # sum per seq = 4, mean of [4, 4] = 4
        result = MOD.compute_agg_loss_torch(loss, mask, "seq-mean-token-sum")
        torch.testing.assert_close(result, torch.tensor(4.0), atol=1e-5, rtol=0)

    def test_dispatch_cpu(self):
        loss, mask = self._tensors()
        for mode in MODES:
            r1 = MOD.compute_agg_loss_torch(loss, mask, mode)
            r2 = MOD.compute_agg_loss(loss, mask, mode, impl="torch")
            torch.testing.assert_close(r1, r2, atol=1e-5, rtol=0)

    def test_invalid_mode_raises(self):
        loss, mask = self._tensors()
        with self.assertRaises(ValueError):
            MOD.compute_agg_loss_torch(loss, mask, "invalid")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestAggLossTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _tensors(self, B=8, T=64, device="cuda"):
        loss = torch.randn(B, T, device=device)
        mask = (torch.rand(B, T, device=device) > 0.2).float()
        return loss, mask

    def test_all_modes_match_torch(self):
        loss, mask = self._tensors()
        for mode in MODES:
            r_t = MOD.compute_agg_loss_triton(loss, mask, mode)
            r_r = MOD.compute_agg_loss_torch(loss, mask, mode)
            torch.testing.assert_close(r_t, r_r, rtol=1e-3, atol=1e-3,
                                       msg=f"Mismatch for mode={mode}")

    def test_large_batch(self):
        loss, mask = self._tensors(B=32, T=2048)
        for mode in MODES:
            r_t = MOD.compute_agg_loss_triton(loss, mask, mode)
            r_r = MOD.compute_agg_loss_torch(loss, mask, mode)
            torch.testing.assert_close(r_t, r_r, rtol=1e-2, atol=1e-2,
                                       msg=f"Mismatch for mode={mode}")

    def test_output_scalars(self):
        loss, mask = self._tensors()
        for mode in MODES:
            result = MOD.compute_agg_loss_triton(loss, mask, mode)
            self.assertEqual(result.shape, ())

    def test_auto_dispatch_cuda(self):
        loss, mask = self._tensors()
        for mode in MODES:
            r_a = MOD.compute_agg_loss(loss, mask, mode, impl="auto")
            r_r = MOD.compute_agg_loss_torch(loss, mask, mode)
            torch.testing.assert_close(r_a, r_r, rtol=1e-3, atol=1e-3)

    def test_bf16_input(self):
        loss, mask = self._tensors()
        loss16 = loss.bfloat16()
        for mode in MODES:
            r_t = MOD.compute_agg_loss_triton(loss16, mask, mode)
            r_r = MOD.compute_agg_loss_torch(loss, mask, mode)
            torch.testing.assert_close(r_t, r_r, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
