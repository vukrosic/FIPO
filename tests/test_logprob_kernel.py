"""Tests for the fused gathered log-probability kernel (FIPO-026)."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F


def _ref_logprob(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Naive reference: full log_softmax + gather."""
    lp = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def _load_logprob_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "kernel" / "logprob.py"
    spec = importlib.util.spec_from_file_location("logprob", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MOD = _load_logprob_module()


class TestTokenLogprobTorch(unittest.TestCase):
    """CPU / torch-path tests."""

    def setUp(self):
        torch.manual_seed(0)

    def test_matches_reference_small(self):
        B, T, V = 4, 8, 64
        logits = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_torch(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_matches_reference_larger(self):
        B, T, V = 8, 32, 512
        logits = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_torch(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_bfloat16_input(self):
        """bfloat16 logits should produce float32 output matching the reference."""
        B, T, V = 4, 16, 128
        logits = torch.randn(B, T, V).bfloat16()
        token_ids = torch.randint(0, V, (B, T))
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_torch(logits, token_ids)
        # bfloat16 has low precision; use loose tolerance
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_single_vocab(self):
        """Edge case: vocab_size == 1 → log_softmax = 0 for every token."""
        B, T, V = 2, 4, 1
        logits = torch.randn(B, T, V)
        token_ids = torch.zeros(B, T, dtype=torch.long)
        actual = MOD.compute_token_logprob_torch(logits, token_ids)
        expected = torch.zeros(B, T)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    def test_output_dtype_is_float32(self):
        B, T, V = 2, 4, 32
        logits = torch.randn(B, T, V).bfloat16()
        token_ids = torch.randint(0, V, (B, T))
        out = MOD.compute_token_logprob_torch(logits, token_ids)
        self.assertEqual(out.dtype, torch.float32)

    def test_output_shape(self):
        B, T, V = 3, 10, 256
        logits = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        out = MOD.compute_token_logprob_torch(logits, token_ids)
        self.assertEqual(out.shape, (B, T))

    def test_log_probs_are_negative(self):
        """Log-probabilities must be ≤ 0."""
        B, T, V = 8, 16, 64
        logits = torch.randn(B, T, V) * 3
        token_ids = torch.randint(0, V, (B, T))
        out = MOD.compute_token_logprob_torch(logits, token_ids)
        self.assertTrue((out <= 1e-6).all(), "All log-probs should be ≤ 0")

    def test_numerical_stability_large_logits(self):
        """Kernel should be stable under large logit values."""
        B, T, V = 2, 4, 64
        logits = torch.randn(B, T, V) * 100.0
        token_ids = torch.randint(0, V, (B, T))
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_torch(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_all_same_logits(self):
        """Uniform logits → log_prob = -log(V) for every token."""
        B, T, V = 2, 4, 8
        logits = torch.ones(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        actual = MOD.compute_token_logprob_torch(logits, token_ids)
        expected = torch.full((B, T), -torch.log(torch.tensor(float(V))))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_dispatch_torch_cpu(self):
        """dispatch with impl='auto' on CPU goes through torch path."""
        B, T, V = 4, 8, 64
        logits = torch.randn(B, T, V)
        token_ids = torch.randint(0, V, (B, T))
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob(logits, token_ids, impl="auto")
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(MOD.HAVE_TRITON, "Triton required")
class TestTokenLogprobTriton(unittest.TestCase):
    """CUDA / Triton-path tests."""

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _make(self, B, T, V, dtype=torch.float32):
        logits = torch.randn(B, T, V, device="cuda", dtype=dtype)
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        return logits, token_ids

    def test_triton_matches_torch_small_vocab(self):
        logits, token_ids = self._make(4, 8, 256)
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_triton(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_triton_matches_torch_medium_vocab(self):
        logits, token_ids = self._make(4, 16, 4096)
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_triton(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_triton_matches_torch_large_vocab(self):
        logits, token_ids = self._make(2, 8, 32768)
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_triton(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_triton_bfloat16_input(self):
        logits, token_ids = self._make(4, 16, 512, dtype=torch.bfloat16)
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_triton(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_triton_output_is_float32(self):
        logits, token_ids = self._make(2, 4, 128, dtype=torch.bfloat16)
        out = MOD.compute_token_logprob_triton(logits, token_ids)
        self.assertEqual(out.dtype, torch.float32)

    def test_triton_output_shape(self):
        B, T, V = 3, 10, 1024
        logits, token_ids = self._make(B, T, V)
        out = MOD.compute_token_logprob_triton(logits, token_ids)
        self.assertEqual(out.shape, (B, T))

    def test_triton_log_probs_non_positive(self):
        logits, token_ids = self._make(8, 32, 512)
        out = MOD.compute_token_logprob_triton(logits, token_ids)
        self.assertTrue((out <= 1e-5).all(), "All log-probs must be ≤ 0")

    def test_triton_uniform_logits(self):
        B, T, V = 2, 4, 64
        logits = torch.ones(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        out = MOD.compute_token_logprob_triton(logits, token_ids)
        expected = torch.full((B, T), -torch.log(torch.tensor(float(V))))
        torch.testing.assert_close(out, expected.cuda(), atol=1e-4, rtol=1e-4)

    def test_triton_numerical_stability(self):
        """Large logit values should not cause NaN/Inf."""
        logits, token_ids = self._make(4, 8, 256)
        logits = logits * 100.0
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_triton(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_auto_dispatch_cuda_triton(self):
        """impl='auto' on CUDA with V <= _TRITON_MAX_VOCAB uses triton."""
        logits, token_ids = self._make(4, 16, 512)
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob(logits, token_ids, impl="auto")
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_auto_dispatch_large_vocab_fallback(self):
        """impl='auto' with V > _TRITON_MAX_VOCAB falls back to torch."""
        V = MOD._TRITON_MAX_VOCAB + 1
        B, T = 2, 4
        logits = torch.randn(B, T, V, device="cuda")
        token_ids = torch.randint(0, V, (B, T), device="cuda")
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob(logits, token_ids, impl="auto")
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_batch_size_1(self):
        logits, token_ids = self._make(1, 64, 512)
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_triton(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_seq_len_1(self):
        logits, token_ids = self._make(8, 1, 512)
        expected = _ref_logprob(logits, token_ids)
        actual = MOD.compute_token_logprob_triton(logits, token_ids)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
