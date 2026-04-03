"""Tests for fused sequence-level reduction kernels (FIPO-028)."""

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


MOD = _load("seq_utils", "verl/utils/kernel/seq_utils.py")


class TestSeqLogprobTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shape(self):
        B, T = 4, 32
        lp = torch.randn(B, T)
        m  = (torch.rand(B, T) > 0.2).float()
        s, l = MOD.compute_seq_logprob_torch(lp, m)
        self.assertEqual(s.shape, (B,))
        self.assertEqual(l.shape, (B,))

    def test_matches_manual(self):
        B, T = 4, 16
        lp = torch.randn(B, T)
        m  = (torch.rand(B, T) > 0.2).float()
        s, l = MOD.compute_seq_logprob_torch(lp, m)
        expected_s = (lp * m).sum(-1)
        expected_l = m.sum(-1).clamp(min=1)
        torch.testing.assert_close(s, expected_s, atol=1e-5, rtol=0)
        torch.testing.assert_close(l, expected_l, atol=1e-5, rtol=0)

    def test_all_masked_returns_zero_sum(self):
        B, T = 2, 8
        lp = torch.randn(B, T)
        m  = torch.zeros(B, T)
        s, l = MOD.compute_seq_logprob_torch(lp, m)
        torch.testing.assert_close(s, torch.zeros(B), atol=1e-6, rtol=0)
        torch.testing.assert_close(l, torch.ones(B),  atol=1e-6, rtol=0)  # clamped

    def test_bfloat16_input(self):
        B, T = 2, 16
        lp = torch.randn(B, T).bfloat16()
        m  = (torch.rand(B, T) > 0.2).float()
        s, l = MOD.compute_seq_logprob_torch(lp, m)
        self.assertEqual(s.dtype, torch.float32)
        self.assertEqual(l.dtype, torch.float32)

    def test_float32_output(self):
        lp = torch.randn(2, 8)
        m  = torch.ones(2, 8)
        s, l = MOD.compute_seq_logprob_torch(lp, m)
        self.assertEqual(s.dtype, torch.float32)


class TestSeqMeanTorch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_shape(self):
        B, T = 4, 32
        v = torch.randn(B, T)
        m = (torch.rand(B, T) > 0.2).float()
        out = MOD.compute_seq_mean_torch(v, m)
        self.assertEqual(out.shape, (B,))

    def test_matches_manual(self):
        B, T = 4, 16
        v = torch.randn(B, T)
        m = (torch.rand(B, T) > 0.2).float()
        out = MOD.compute_seq_mean_torch(v, m)
        expected = (v * m).sum(-1) / m.sum(-1).clamp(min=1)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0)

    def test_all_ones_mask(self):
        B, T = 2, 8
        v = torch.randn(B, T)
        m = torch.ones(B, T)
        out = MOD.compute_seq_mean_torch(v, m)
        expected = v.mean(-1)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestSeqLogprobTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _make(self, B, T):
        lp = torch.randn(B, T, device="cuda")
        m  = (torch.rand(B, T, device="cuda") > 0.2).float()
        return lp, m

    def test_triton_matches_torch(self):
        lp, m = self._make(8, 64)
        s_t, l_t = MOD.compute_seq_logprob_triton(lp, m)
        s_r, l_r = MOD.compute_seq_logprob_torch(lp, m)
        torch.testing.assert_close(s_t, s_r, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(l_t, l_r, rtol=1e-4, atol=1e-4)

    def test_triton_large(self):
        lp, m = self._make(32, 2048)
        s_t, l_t = MOD.compute_seq_logprob_triton(lp, m)
        s_r, l_r = MOD.compute_seq_logprob_torch(lp, m)
        torch.testing.assert_close(s_t, s_r, rtol=1e-4, atol=1e-4)

    def test_dispatch_auto(self):
        lp, m = self._make(4, 64)
        s_a, l_a = MOD.compute_seq_logprob(lp, m, impl="auto")
        s_r, l_r = MOD.compute_seq_logprob_torch(lp, m)
        torch.testing.assert_close(s_a, s_r, rtol=1e-4, atol=1e-4)

    def test_seq_mean_triton(self):
        v = torch.randn(8, 64, device="cuda")
        m = (torch.rand(8, 64, device="cuda") > 0.2).float()
        t = MOD.compute_seq_mean_triton(v, m)
        r = MOD.compute_seq_mean_torch(v, m)
        torch.testing.assert_close(t, r, rtol=1e-4, atol=1e-4)

    def test_seq_mean_dispatch(self):
        v = torch.randn(4, 128, device="cuda")
        m = torch.ones(4, 128, device="cuda")
        t = MOD.compute_seq_mean(v, m, impl="auto")
        r = MOD.compute_seq_mean_torch(v, m)
        torch.testing.assert_close(t, r, rtol=1e-4, atol=1e-4)

    def test_output_float32(self):
        lp, m = self._make(4, 32)
        s, l = MOD.compute_seq_logprob_triton(lp, m)
        self.assertEqual(s.dtype, torch.float32)
        self.assertEqual(l.dtype, torch.float32)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestSeqMeanTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)

    def test_large_tensor(self):
        v = torch.randn(32, 2048, device="cuda")
        m = (torch.rand(32, 2048, device="cuda") > 0.15).float()
        t = MOD.compute_seq_mean_triton(v, m)
        r = MOD.compute_seq_mean_torch(v, m)
        torch.testing.assert_close(t, r, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
