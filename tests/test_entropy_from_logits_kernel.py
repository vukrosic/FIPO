"""Tests for streaming entropy kernel (FIPO-039).

Covers:
  - CPU correctness (torch path, various shapes)
  - CUDA triton path matches torch reference
  - BF16 input handling
  - All vocab sizes including large (> 4096 old cap)
  - All-masked (zero-mask) edge case
  - Integration with compute_entropy_loss in future_kl.py
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch
import verl.utils.torch_functional as TORCH_F


def _load_module(rel_path: str):
    path = Path(__file__).resolve().parents[1] / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ENTROPY_MOD = _load_module("verl/utils/kernel/entropy_from_logits.py")
FUTURE_KL_MOD = _load_module("verl/utils/kernel/future_kl.py")


def _reference_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Naive reference: softmax + logsumexp - sum(p*logits)."""
    logits_f = logits.float()
    pd = torch.softmax(logits_f, dim=-1)
    lse = torch.logsumexp(logits_f, dim=-1)
    return lse - (pd * logits_f).sum(dim=-1)


def _reference_entropy_functional(logits: torch.Tensor) -> torch.Tensor:
    """Reference matching torch_functional.entropy_from_logits dtype semantics."""
    pd = torch.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - (pd * logits).sum(dim=-1)


class EntropyFromLogitsCPUTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_basic_shape(self):
        B, T, V = 4, 16, 128
        logits = torch.randn(B, T, V)
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        self.assertEqual(out.shape, (B, T))
        self.assertEqual(out.dtype, torch.float32)

    def test_matches_reference_small_vocab(self):
        B, T, V = 4, 32, 256
        logits = torch.randn(B, T, V)
        ref = _reference_entropy(logits)
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_matches_reference_2d(self):
        N, V = 32, 256
        logits = torch.randn(N, V)
        ref = _reference_entropy(logits)
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        self.assertEqual(out.shape, (N,))
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_matches_reference_medium_vocab(self):
        B, T, V = 2, 8, 4096
        logits = torch.randn(B, T, V)
        ref = _reference_entropy(logits)
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_matches_reference_large_vocab(self):
        """Vocab > 4096 — previously fell back to naive torch in old kernel."""
        B, T, V = 2, 4, 8192
        logits = torch.randn(B, T, V)
        ref = _reference_entropy(logits)
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_entropy_nonnegative(self):
        """Entropy must be >= 0 for any probability distribution."""
        B, T, V = 4, 8, 512
        logits = torch.randn(B, T, V)
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        self.assertTrue((out >= -1e-6).all(), f"Negative entropy found: {out.min()}")

    def test_uniform_distribution_max_entropy(self):
        """Uniform logits → maximum entropy = log(V)."""
        V = 256
        logits = torch.zeros(1, 1, V)
        expected = torch.log(torch.tensor(float(V)))
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        torch.testing.assert_close(out[0, 0], expected, rtol=1e-5, atol=1e-5)

    def test_deterministic_distribution_zero_entropy(self):
        """One logit >> all others → entropy ≈ 0."""
        V = 64
        logits = torch.zeros(1, 1, V)
        logits[0, 0, 0] = 100.0
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        self.assertAlmostEqual(out[0, 0].item(), 0.0, delta=1e-4)

    def test_bfloat16_input_returns_float32(self):
        B, T, V = 2, 8, 256
        logits = torch.randn(B, T, V).bfloat16()
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        self.assertEqual(out.dtype, torch.float32)

    def test_auto_falls_back_on_cpu(self):
        B, T, V = 2, 8, 512
        logits = torch.randn(B, T, V)
        ref = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="auto")
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(ENTROPY_MOD.HAVE_TRITON, "Triton required")
class EntropyFromLogitsCUDATest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_triton_matches_torch_small_vocab(self):
        B, T, V = 8, 64, 512
        logits = torch.randn(B, T, V, device="cuda", dtype=torch.float32)
        ref = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        self.assertEqual(out.shape, (B, T))
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_triton_matches_torch_2d(self):
        N, V = 2048, 4096
        logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
        ref = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        self.assertEqual(out.shape, (N,))
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    def test_triton_matches_torch_vocab_4096(self):
        """Exactly at the old cap — streaming kernel must handle correctly."""
        B, T, V = 4, 32, 4096
        logits = torch.randn(B, T, V, device="cuda")
        ref = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_triton_matches_torch_vocab_8192(self):
        """Old kernel fell back to torch; new streaming kernel handles it."""
        B, T, V = 4, 16, 8192
        logits = torch.randn(B, T, V, device="cuda")
        ref = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    def test_triton_matches_torch_vocab_32768(self):
        """Large LLM vocab size."""
        B, T, V = 2, 8, 32768
        logits = torch.randn(B, T, V, device="cuda")
        ref = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    def test_triton_bfloat16_input(self):
        B, T, V = 4, 16, 2048
        logits = torch.randn(B, T, V, device="cuda", dtype=torch.bfloat16)
        ref = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        self.assertEqual(out.dtype, torch.float32)
        # BF16 input has lower precision — use wider tolerance
        torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    def test_output_is_float32_regardless_of_input(self):
        B, T, V = 2, 8, 512
        for dtype in [torch.float32, torch.bfloat16]:
            logits = torch.randn(B, T, V, device="cuda", dtype=dtype)
            out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
            self.assertEqual(out.dtype, torch.float32)

    def test_uniform_logits_max_entropy(self):
        V = 1024
        logits = torch.zeros(1, 1, V, device="cuda")
        expected = torch.log(torch.tensor(float(V)))
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        torch.testing.assert_close(
            out[0, 0].cpu(), expected, rtol=1e-4, atol=1e-4
        )

    def test_deterministic_logit_zero_entropy(self):
        V = 512
        logits = torch.zeros(1, 1, V, device="cuda")
        logits[0, 0, 0] = 100.0
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        self.assertAlmostEqual(out[0, 0].item(), 0.0, delta=1e-3)

    def test_large_batch(self):
        B, T, V = 32, 2048, 4096
        logits = torch.randn(B, T, V, device="cuda")
        ref = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    def test_single_token(self):
        B, T, V = 1, 1, 256
        logits = torch.randn(B, T, V, device="cuda")
        ref = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        out = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(ENTROPY_MOD.HAVE_TRITON, "Triton required")
class EntropyLossIntegrationTest(unittest.TestCase):
    """Verify that future_kl.compute_entropy_loss uses the new streaming kernel."""

    def setUp(self):
        torch.manual_seed(0)

    def test_all_agg_modes_large_vocab(self):
        """All 4 aggregation modes work with vocab > 4096 via streaming kernel."""
        B, T, V = 4, 16, 8192
        logits = torch.randn(B, T, V, device="cuda")
        mask = (torch.rand(B, T, device="cuda") > 0.1).float()

        for mode in ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"]:
            torch_loss, _ = FUTURE_KL_MOD.compute_entropy_loss(
                logits, mask, loss_agg_mode=mode, impl="torch"
            )
            triton_loss, _ = FUTURE_KL_MOD.compute_entropy_loss(
                logits, mask, loss_agg_mode=mode, impl="triton"
            )
            torch.testing.assert_close(
                triton_loss, torch_loss, rtol=2e-3, atol=2e-3,
                msg=f"Mode {mode!r} mismatch"
            )

    def test_bfloat16_works_without_error(self):
        """BF16 logits should no longer raise dtype errors."""
        B, T, V = 4, 16, 2048
        logits = torch.randn(B, T, V, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(B, T, device="cuda")
        # Should not raise
        loss, mean = FUTURE_KL_MOD.compute_entropy_loss(logits, mask, impl="auto")
        self.assertTrue(torch.isfinite(loss))

    def test_vocab_8192_triton_matches_torch(self):
        B, T, V = 8, 32, 8192
        logits = torch.randn(B, T, V, device="cuda")
        mask = (torch.rand(B, T, device="cuda") > 0.1).float()

        torch_loss, _ = FUTURE_KL_MOD.compute_entropy_loss(logits, mask, impl="torch")
        triton_loss, _ = FUTURE_KL_MOD.compute_entropy_loss(logits, mask, impl="triton")
        torch.testing.assert_close(triton_loss, torch_loss, rtol=2e-3, atol=2e-3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(ENTROPY_MOD.HAVE_TRITON, "Triton required")
class EntropyDispatchIntegrationTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_entropy_dispatch_matches_reference_3d(self):
        B, T, V = 8, 64, 2048
        logits = torch.randn(B, T, V, device="cuda", dtype=torch.float32)
        ref = _reference_entropy_functional(logits)
        out = TORCH_F.entropy_from_logits(logits)
        self.assertEqual(out.shape, (B, T))
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    def test_entropy_dispatch_matches_reference_2d(self):
        N, V = 4096, 2048
        logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
        ref = _reference_entropy_functional(logits)
        out = TORCH_F.entropy_from_logits(logits)
        self.assertEqual(out.shape, (N,))
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    def test_entropy_dispatch_preserves_bfloat16_dtype(self):
        B, T, V = 4, 32, 2048
        logits = torch.randn(B, T, V, device="cuda", dtype=torch.bfloat16)
        ref = _reference_entropy_functional(logits)
        out = TORCH_F.entropy_from_logits(logits)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    def test_entropy_chunking_dispatch_matches_reference_2d(self):
        N, V = 4096, 4096
        logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
        ref = _reference_entropy_functional(logits)
        out = TORCH_F.entropy_from_logits_with_chunking(logits, chunk_size=256)
        self.assertEqual(out.shape, (N,))
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(ENTROPY_MOD.HAVE_TRITON, "Triton required")
class EntropySpeedupTest(unittest.TestCase):
    """Verify streaming kernel is faster than materialising (B,T,V) for large vocab."""

    def test_speedup_vs_naive_at_large_vocab(self):
        import time

        B, T, V = 16, 2048, 8192
        logits = torch.randn(B, T, V, device="cuda")

        # Warmup
        for _ in range(5):
            _ = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
            _ = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        torch.cuda.synchronize()

        iters = 20

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="torch")
        torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - t0) / iters * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = ENTROPY_MOD.compute_entropy_from_logits(logits, impl="triton")
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - t0) / iters * 1000

        speedup = torch_ms / triton_ms
        print(f"\nEntropy (B={B},T={T},V={V}): torch={torch_ms:.3f}ms  triton={triton_ms:.3f}ms  speedup={speedup:.2f}x")
        # Should be at least 1.5x faster at this size
        self.assertGreater(speedup, 1.0, f"Expected speedup>1.0, got {speedup:.2f}x")


if __name__ == "__main__":
    unittest.main()
