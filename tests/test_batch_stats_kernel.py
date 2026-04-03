"""Tests for fused batch statistics kernel (FIPO-035)."""

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


MOD = _load("batch_stats", "verl/utils/kernel/batch_stats.py")


class TestBatchStatsTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, mx, mn = MOD.compute_batch_stats_torch(x)
        self.assertAlmostEqual(mean.item(), 3.0, delta=1e-5)
        self.assertAlmostEqual(mx.item(), 5.0, delta=1e-5)
        self.assertAlmostEqual(mn.item(), 1.0, delta=1e-5)

    def test_negative_values(self):
        x = torch.tensor([-5.0, -1.0, 0.0, 3.0])
        mean, mx, mn = MOD.compute_batch_stats_torch(x)
        self.assertAlmostEqual(mean.item(), -0.75, delta=1e-5)
        self.assertAlmostEqual(mx.item(), 3.0, delta=1e-5)
        self.assertAlmostEqual(mn.item(), -5.0, delta=1e-5)

    def test_2d_tensor(self):
        x = torch.randn(10, 20)
        mean, mx, mn = MOD.compute_batch_stats_torch(x)
        torch.testing.assert_close(mean, x.mean(), atol=1e-5, rtol=0)
        torch.testing.assert_close(mx, x.max(), atol=1e-5, rtol=0)
        torch.testing.assert_close(mn, x.min(), atol=1e-5, rtol=0)


class TestMaskedBatchStatsTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        mean, mx, mn = MOD.compute_masked_batch_stats_torch(x, mask)
        # Valid: [1, 2, 4]
        self.assertAlmostEqual(mean.item(), 7.0/3, delta=1e-5)
        self.assertAlmostEqual(mx.item(), 4.0, delta=1e-5)
        self.assertAlmostEqual(mn.item(), 1.0, delta=1e-5)

    def test_all_masked(self):
        x = torch.tensor([1.0, 2.0])
        mask = torch.tensor([0.0, 0.0])
        mean, mx, mn = MOD.compute_masked_batch_stats_torch(x, mask)
        self.assertAlmostEqual(mean.item(), 0.0, delta=1e-5)

    def test_dispatch(self):
        x = torch.randn(100)
        mask = (torch.rand(100) > 0.3).float()
        r1 = MOD.compute_batch_stats(x, impl="torch")
        r2 = MOD.compute_batch_stats_torch(x)
        for a, b in zip(r1, r2):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestBatchStatsTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def test_matches_torch(self):
        x = torch.randn(10000, device="cuda")
        mean_t, mx_t, mn_t = MOD.compute_batch_stats_triton(x)
        mean_r, mx_r, mn_r = MOD.compute_batch_stats_torch(x)
        torch.testing.assert_close(mean_t, mean_r, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(mx_t, mx_r, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(mn_t, mn_r, rtol=1e-5, atol=1e-5)

    def test_2d_matches_torch(self):
        x = torch.randn(32, 2048, device="cuda")
        mean_t, mx_t, mn_t = MOD.compute_batch_stats_triton(x)
        mean_r, mx_r, mn_r = MOD.compute_batch_stats_torch(x)
        torch.testing.assert_close(mean_t, mean_r, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(mx_t, mx_r, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(mn_t, mn_r, rtol=1e-5, atol=1e-5)

    def test_masked_matches_torch(self):
        x = torch.randn(10000, device="cuda")
        mask = (torch.rand(10000, device="cuda") > 0.3).float()
        mean_t, mx_t, mn_t = MOD.compute_masked_batch_stats_triton(x, mask)
        mean_r, mx_r, mn_r = MOD.compute_masked_batch_stats_torch(x, mask)
        torch.testing.assert_close(mean_t, mean_r, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(mx_t, mx_r, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(mn_t, mn_r, rtol=1e-5, atol=1e-5)

    def test_auto_dispatch_cuda(self):
        x = torch.randn(1000, device="cuda")
        mean_a, mx_a, mn_a = MOD.compute_batch_stats(x, impl="auto")
        mean_r, mx_r, mn_r = MOD.compute_batch_stats_torch(x)
        torch.testing.assert_close(mean_a, mean_r, rtol=1e-3, atol=1e-3)

    def test_bf16_input(self):
        x = torch.randn(10000, device="cuda")
        x16 = x.bfloat16()
        mean_t, mx_t, mn_t = MOD.compute_batch_stats_triton(x16)
        mean_r, mx_r, mn_r = MOD.compute_batch_stats_torch(x)
        torch.testing.assert_close(mean_t, mean_r, rtol=5e-2, atol=5e-2)

    def test_output_device(self):
        x = torch.randn(100, device="cuda")
        mean, mx, mn = MOD.compute_batch_stats_triton(x)
        self.assertEqual(mean.device.type, "cuda")
        self.assertEqual(mx.device.type, "cuda")

    def test_single_element(self):
        x = torch.tensor([42.0], device="cuda")
        mean, mx, mn = MOD.compute_batch_stats_triton(x)
        self.assertAlmostEqual(mean.item(), 42.0, delta=1e-4)
        self.assertAlmostEqual(mx.item(), 42.0, delta=1e-4)
        self.assertAlmostEqual(mn.item(), 42.0, delta=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
