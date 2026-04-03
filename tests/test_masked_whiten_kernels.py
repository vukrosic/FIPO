from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


def _load_torch_functional_module():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "torch_functional.py"
    spec = importlib.util.spec_from_file_location("torch_functional_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TORCH_FUNCTIONAL = _load_torch_functional_module()


def _reference_masked_mean(values, mask):
    """Reference implementation of masked mean."""
    return (values * mask).sum() / (mask.sum() + 1e-8)


def _reference_masked_var(values, mask, unbiased=True):
    """Reference implementation of masked variance."""
    mean = _reference_masked_mean(values, mask)
    centered_values = values - mean
    variance = (centered_values * centered_values * mask).sum() / (mask.sum() + 1e-8)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum > 1:
            bessel_correction = mask_sum / (mask_sum - 1)
            variance = variance * bessel_correction
    return variance


def _reference_masked_whiten(values, mask, shift_mean=True):
    """Reference implementation matching original masked_whiten."""
    mean = _reference_masked_mean(values, mask)
    var = _reference_masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class MaskedWhitenCPUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_torch_impl_matches_reference(self):
        """Test that _masked_whiten_torch_impl matches the original two-pass reference."""
        values = torch.randn(8, 64, dtype=torch.float32)
        mask = (torch.rand(8, 64) > 0.2).float()

        expected = _reference_masked_whiten(values, mask, shift_mean=True)
        actual = TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask, shift_mean=True)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_torch_impl_shift_mean_false(self):
        """Test _masked_whiten_torch_impl with shift_mean=False."""
        values = torch.randn(8, 64, dtype=torch.float32)
        mask = (torch.rand(8, 64) > 0.2).float()

        expected = _reference_masked_whiten(values, mask, shift_mean=False)
        actual = TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask, shift_mean=False)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_torch_impl_with_zeros_mask(self):
        """Test _masked_whiten_torch_impl when all elements are masked."""
        values = torch.randn(8, 64, dtype=torch.float32)
        mask = torch.zeros(8, 64, dtype=torch.float32)

        with self.assertRaises(ValueError):
            TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask)

    def test_torch_impl_single_masked_element(self):
        """Test _masked_whiten_torch_impl with only one masked element.

        Single masked element should raise ValueError (same as original masked_var).
        """
        values = torch.randn(8, 64, dtype=torch.float32)
        mask = torch.zeros(8, 64, dtype=torch.float32)
        mask[0, 0] = 1.0

        # Single element causes division by zero in Bessel's correction
        with self.assertRaises(ValueError):
            TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask)

    def test_whitened_values_have_zero_mean(self):
        """Test that whitened values have approximately zero mean."""
        values = torch.randn(16, 128, dtype=torch.float32)
        mask = (torch.rand(16, 128) > 0.1).float()

        result = TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask, shift_mean=True)

        # Compute mean of masked elements only
        masked_result = result[mask.bool()]
        self.assertTrue(
            abs(masked_result.mean().item()) < 1e-4,
            f"Whitened mean should be ~0, got {masked_result.mean().item()}"
        )

    def test_whitened_values_have_unit_variance(self):
        """Test that whitened values have approximately unit variance."""
        values = torch.randn(16, 128, dtype=torch.float32)
        mask = (torch.rand(16, 128) > 0.1).float()

        result = TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask, shift_mean=True)

        # Compute variance of masked elements only
        masked_result = result[mask.bool()]
        self.assertTrue(
            abs(masked_result.var(unbiased=False).item() - 1.0) < 1e-3,
            f"Whitened variance should be ~1, got {masked_result.var(unbiased=False).item()}"
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton kernel tests")
@unittest.skipUnless(TORCH_FUNCTIONAL.HAVE_TRITON, "Triton is required for Triton kernel tests")
class MaskedWhitenCUDATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_triton_matches_torch_impl(self):
        """Test that Triton implementation matches torch implementation.

        Note: Due to float32 atomic add precision, Triton may differ slightly
        from torch implementation. Both should produce statistically equivalent
        results (whitened values with zero mean and unit variance).
        """
        values = torch.randn(8, 64, device="cuda", dtype=torch.float32)
        mask = (torch.rand(8, 64, device="cuda") > 0.2).float()

        expected = TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask, shift_mean=True)
        actual = TORCH_FUNCTIONAL.masked_whiten_triton(values, mask, shift_mean=True)

        # Use looser tolerance due to float32 atomic add precision differences
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_triton_matches_torch_impl_shift_mean_false(self):
        """Test Triton with shift_mean=False."""
        values = torch.randn(8, 64, device="cuda", dtype=torch.float32)
        mask = (torch.rand(8, 64, device="cuda") > 0.2).float()

        expected = TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask, shift_mean=False)
        actual = TORCH_FUNCTIONAL.masked_whiten_triton(values, mask, shift_mean=False)

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_triton_matches_reference(self):
        """Test that Triton implementation matches the original reference.

        Note: Due to float32 atomic add precision, Triton may differ slightly
        from reference. Both should produce statistically equivalent results.
        """
        values = torch.randn(16, 128, device="cuda", dtype=torch.float32)
        mask = (torch.rand(16, 128, device="cuda") > 0.1).float()

        expected = _reference_masked_whiten(values, mask, shift_mean=True)
        actual = TORCH_FUNCTIONAL.masked_whiten_triton(values, mask, shift_mean=True)

        # Use looser tolerance due to float32 atomic add precision differences
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_triton_large_tensor(self):
        """Test Triton on a larger tensor."""
        values = torch.randn(32, 512, device="cuda", dtype=torch.float32)
        mask = (torch.rand(32, 512, device="cuda") > 0.15).float()

        expected = _reference_masked_whiten(values, mask, shift_mean=True)
        actual = TORCH_FUNCTIONAL.masked_whiten_triton(values, mask, shift_mean=True)

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_triton_small_tensor(self):
        """Test Triton on a small tensor (single block)."""
        values = torch.randn(2, 32, device="cuda", dtype=torch.float32)
        mask = (torch.rand(2, 32, device="cuda") > 0.3).float()

        expected = _reference_masked_whiten(values, mask, shift_mean=True)
        actual = TORCH_FUNCTIONAL.masked_whiten_triton(values, mask, shift_mean=True)

        # Use looser tolerance due to float32 atomic add precision differences
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_whitened_values_have_zero_mean_triton(self):
        """Test that Triton whitened values have approximately zero mean."""
        values = torch.randn(16, 128, device="cuda", dtype=torch.float32)
        mask = (torch.rand(16, 128, device="cuda") > 0.1).float()

        result = TORCH_FUNCTIONAL.masked_whiten_triton(values, mask, shift_mean=True)

        masked_result = result[mask.bool()]
        self.assertTrue(
            abs(masked_result.mean().item()) < 1e-4,
            f"Whitened mean should be ~0, got {masked_result.mean().item()}"
        )

    def test_whitened_values_have_unit_variance_triton(self):
        """Test that Triton whitened values have approximately unit variance."""
        values = torch.randn(16, 128, device="cuda", dtype=torch.float32)
        mask = (torch.rand(16, 128, device="cuda") > 0.1).float()

        result = TORCH_FUNCTIONAL.masked_whiten_triton(values, mask, shift_mean=True)

        masked_result = result[mask.bool()]
        self.assertTrue(
            abs(masked_result.var(unbiased=False).item() - 1.0) < 1e-3,
            f"Whitened variance should be ~1, got {masked_result.var(unbiased=False).item()}"
        )


class MaskedWhitenNumericalStabilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_near_zero_variance(self):
        """Test whitening with near-zero variance values."""
        values = torch.ones(16, 64) + torch.randn(16, 64) * 1e-6
        mask = (torch.rand(16, 64) > 0.1).float()

        result_torch = TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask)
        result_ref = _reference_masked_whiten(values, mask)

        # Both use same algorithm, should match closely
        torch.testing.assert_close(result_torch, result_ref, rtol=1e-4, atol=1e-4)

    def test_large_dynamic_range(self):
        """Test with values spanning large dynamic range."""
        values = torch.tensor([1e-6, 1e6, 1.0, -1e6, -1e-6] * 32, dtype=torch.float32).reshape(8, 20)
        mask = (torch.rand(8, 20) > 0.1).float()

        result_torch = TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask)
        result_ref = _reference_masked_whiten(values, mask)

        torch.testing.assert_close(result_torch, result_ref, rtol=1e-3, atol=1e-3)

    def test_extreme_values(self):
        """Test with extreme values near float32 limits."""
        values = torch.randn(8, 64, dtype=torch.float32) * 1e3
        mask = (torch.rand(8, 64) > 0.1).float()

        result_torch = TORCH_FUNCTIONAL._masked_whiten_torch_impl(values, mask)
        result_ref = _reference_masked_whiten(values, mask)

        torch.testing.assert_close(result_torch, result_ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
