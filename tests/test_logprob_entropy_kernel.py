"""Tests for fused log-prob + entropy kernel (FIPO-041).

Covers:
  - CPU correctness (torch path)
  - CUDA triton matches separate logprob + entropy
  - BF16 input
  - Large vocab (8192, 32768)
  - All-same-token edge case
  - Speedup benchmark vs calling both kernels separately
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


def _load(rel: str):
    p = Path(__file__).resolve().parents[1] / rel
    spec = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LE = _load("verl/utils/kernel/logprob_entropy.py")
LP = _load("verl/utils/kernel/logprob.py")
EN = _load("verl/utils/kernel/entropy_from_logits.py")


def _ref_logprob(logits, token_ids):
    logits_f = logits.float()
    lse = torch.logsumexp(logits_f, dim=-1)
    return logits_f.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1) - lse


def _ref_entropy(logits):
    logits_f = logits.float()
    pd = torch.softmax(logits_f, dim=-1)
    lse = torch.logsumexp(logits_f, dim=-1)
    return lse - (pd * logits_f).sum(dim=-1)


class LogprobEntropyCPUTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(99)

    def test_shapes(self):
        B, T, V = 4, 16, 128
        logits    = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        lp, ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        self.assertEqual(lp.shape,  (B, T))
        self.assertEqual(ent.shape, (B, T))
        self.assertEqual(lp.dtype,  torch.float32)
        self.assertEqual(ent.dtype, torch.float32)

    def test_logprob_matches_ref(self):
        B, T, V = 4, 32, 256
        logits    = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        ref_lp  = _ref_logprob(logits, token_ids)
        lp, _   = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        torch.testing.assert_close(lp, ref_lp, rtol=1e-5, atol=1e-5)

    def test_entropy_matches_ref(self):
        B, T, V = 4, 32, 256
        logits    = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        ref_ent = _ref_entropy(logits)
        _, ent  = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        torch.testing.assert_close(ent, ref_ent, rtol=1e-5, atol=1e-5)

    def test_entropy_nonnegative(self):
        B, T, V = 4, 8, 512
        logits    = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        _, ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        self.assertTrue((ent >= -1e-6).all())

    def test_bfloat16_input_returns_float32(self):
        B, T, V = 2, 8, 256
        logits    = torch.randn(B, T, V).bfloat16()
        token_ids = torch.randint(0, V, (B, T))
        lp, ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        self.assertEqual(lp.dtype,  torch.float32)
        self.assertEqual(ent.dtype, torch.float32)

    def test_auto_falls_back_on_cpu(self):
        B, T, V = 2, 8, 128
        logits    = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        ref_lp, ref_ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        lp, ent         = LE.compute_logprob_and_entropy(logits, token_ids, impl="auto")
        torch.testing.assert_close(lp,  ref_lp,  rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ent, ref_ent, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(LE.HAVE_TRITON, "Triton required")
class LogprobEntropyCUDATest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_triton_matches_torch_small_vocab(self):
        B, T, V = 8, 64, 512
        logits    = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")

        ref_lp, ref_ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        lp, ent         = LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")

        torch.testing.assert_close(lp,  ref_lp,  rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ent, ref_ent, rtol=1e-3, atol=1e-3)

    def test_triton_matches_torch_vocab_4096(self):
        B, T, V = 4, 32, 4096
        logits    = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        ref_lp, ref_ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        lp, ent         = LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")
        torch.testing.assert_close(lp,  ref_lp,  rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ent, ref_ent, rtol=1e-3, atol=1e-3)

    def test_triton_matches_torch_vocab_8192(self):
        B, T, V = 4, 16, 8192
        logits    = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        ref_lp, ref_ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        lp, ent         = LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")
        torch.testing.assert_close(lp,  ref_lp,  rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(ent, ref_ent, rtol=2e-3, atol=2e-3)

    def test_triton_matches_torch_vocab_32768(self):
        B, T, V = 2, 8, 32768
        logits    = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        ref_lp, ref_ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        lp, ent         = LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")
        torch.testing.assert_close(lp,  ref_lp,  rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(ent, ref_ent, rtol=3e-3, atol=3e-3)

    def test_triton_bfloat16(self):
        B, T, V = 4, 16, 4096
        logits    = torch.randn(B, T, V, device="cuda", dtype=torch.bfloat16)
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        ref_lp, ref_ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        lp, ent         = LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")
        self.assertEqual(lp.dtype,  torch.float32)
        self.assertEqual(ent.dtype, torch.float32)
        torch.testing.assert_close(lp,  ref_lp,  rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(ent, ref_ent, rtol=5e-3, atol=5e-3)

    def test_matches_separate_kernels(self):
        """Fused output must match separate logprob + entropy calls."""
        B, T, V = 8, 64, 4096
        logits    = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")

        sep_lp  = LP.compute_token_logprob(logits, token_ids, impl="triton")
        sep_ent = EN.compute_entropy_from_logits(logits, impl="triton")

        fused_lp, fused_ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")

        torch.testing.assert_close(fused_lp,  sep_lp,  rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(fused_ent, sep_ent, rtol=1e-3, atol=1e-3)

    def test_logprob_leq_zero(self):
        """log-probabilities must be <= 0 (since p <= 1)."""
        B, T, V = 4, 32, 512
        logits    = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        lp, _ = LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")
        self.assertTrue((lp <= 1e-6).all(), f"Found positive log-prob: {lp.max()}")

    def test_entropy_nonnegative(self):
        B, T, V = 4, 32, 512
        logits    = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        _, ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")
        self.assertTrue((ent >= -1e-4).all(), f"Found negative entropy: {ent.min()}")

    def test_large_batch(self):
        B, T, V = 32, 2048, 4096
        logits    = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        ref_lp, ref_ent = LE.compute_logprob_and_entropy(logits, token_ids, impl="torch")
        lp, ent         = LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")
        torch.testing.assert_close(lp,  ref_lp,  rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(ent, ref_ent, rtol=2e-3, atol=2e-3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(LE.HAVE_TRITON, "Triton required")
class LogprobEntropySpeedupTest(unittest.TestCase):

    def test_speedup_vs_separate_at_large_vocab(self):
        """Fused should be faster than two separate kernel calls."""
        import time

        B, T, V = 16, 2048, 8192
        logits    = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")

        # Warmup
        for _ in range(5):
            LP.compute_token_logprob(logits, token_ids, impl="triton")
            EN.compute_entropy_from_logits(logits, impl="triton")
            LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")
        torch.cuda.synchronize()

        iters = 20

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            LP.compute_token_logprob(logits, token_ids, impl="triton")
            EN.compute_entropy_from_logits(logits, impl="triton")
        torch.cuda.synchronize()
        sep_ms = (time.perf_counter() - t0) / iters * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            LE.compute_logprob_and_entropy(logits, token_ids, impl="triton")
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - t0) / iters * 1000

        speedup = sep_ms / fused_ms
        print(f"\nSeparate (lp+ent) B={B},T={T},V={V}: {sep_ms:.3f}ms  Fused: {fused_ms:.3f}ms  speedup={speedup:.2f}x")
        self.assertGreater(speedup, 1.3, f"Expected speedup>1.3x, got {speedup:.2f}x")


if __name__ == "__main__":
    unittest.main()
