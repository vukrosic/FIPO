from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


def _load_future_kl_module():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "kernel" / "future_kl.py"
    spec = importlib.util.spec_from_file_location("future_kl_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FUTURE_KL = _load_future_kl_module()


def _reference_future_kl(kl_response: torch.Tensor, gamma: float) -> torch.Tensor:
    future_kl = torch.zeros_like(kl_response)
    carry = torch.zeros(kl_response.shape[0], device=kl_response.device, dtype=kl_response.dtype)
    gamma_t = torch.tensor(gamma, device=kl_response.device, dtype=kl_response.dtype)
    for col in range(kl_response.shape[1] - 1, -1, -1):
        carry = kl_response[:, col] + gamma_t * carry
        future_kl[:, col] = carry
    return future_kl


def _reference_influence_weights(
    future_kl: torch.Tensor,
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    clip_ratio: float,
    clip_high_only: bool,
    safe_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    raw = torch.exp(future_kl)
    if clip_ratio != 0.0:
        upper_bound = 1.0 + clip_ratio
        if clip_high_only:
            lower_bound = 1.0
            clipped = torch.clamp(raw, min=1.0, max=upper_bound)
        else:
            lower_bound = 1.0 - clip_ratio
            clipped = torch.clamp(raw, min=lower_bound, max=upper_bound)
    else:
        upper_bound = 10.0
        lower_bound = 0.0
        clipped = torch.clamp(raw, max=upper_bound)

    safe_mask = (advantages < 0) & (ratio > safe_threshold)
    clipped = torch.where(safe_mask, torch.clamp(clipped, min=0.8, max=1.0), clipped)
    return raw, clipped, lower_bound, upper_bound


class FutureKLKernelCPUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_future_kl_reference_matches_reverse_loop(self):
        kl_response = torch.randn(5, 17, dtype=torch.float32)
        gamma = 2 ** (-1.0 / 32.0)

        expected = _reference_future_kl(kl_response, gamma)
        actual = FUTURE_KL.compute_future_kl_chunked_reference(kl_response, gamma, chunk_size=4)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_future_kl_auto_falls_back_to_torch_on_cpu(self):
        kl_response = torch.randn(3, 11, dtype=torch.float32)
        gamma = 0.97

        expected = FUTURE_KL.compute_future_kl(kl_response, gamma, impl="torch", chunk_size=3)
        actual = FUTURE_KL.compute_future_kl(kl_response, gamma, impl="auto", chunk_size=3)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_influence_weights_torch_matches_reference_formula(self):
        future_kl = torch.randn(4, 9, dtype=torch.float32)
        advantages = torch.randn(4, 9, dtype=torch.float32)
        ratio = torch.exp(torch.randn(4, 9, dtype=torch.float32) * 0.1)

        expected = _reference_influence_weights(
            future_kl=future_kl,
            advantages=advantages,
            ratio=ratio,
            clip_ratio=0.2,
            clip_high_only=True,
            safe_threshold=4.0,
        )
        actual = FUTURE_KL.compute_influence_weights_torch(
            future_kl=future_kl,
            advantages=advantages,
            ratio=ratio,
            clip_ratio=0.2,
            clip_high_only=True,
            safe_threshold=4.0,
        )

        for got, want in zip(actual[:2], expected[:2], strict=True):
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
        self.assertEqual(actual[2:], expected[2:])

    def test_masked_mean_torch_matches_manual(self):
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        expected = torch.tensor((1.0 + 3.0 + 4.0 + 5.0) / 4.0)
        actual = FUTURE_KL.compute_masked_mean_torch(values, mask)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton kernel tests")
@unittest.skipUnless(FUTURE_KL.HAVE_TRITON, "Triton is required for Triton kernel tests")
class FutureKLKernelCUDATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_triton_future_kl_matches_reference(self):
        kl_response = torch.randn(8, 257, device="cuda", dtype=torch.float32)
        gamma = 2 ** (-1.0 / 32.0)

        expected = FUTURE_KL.compute_future_kl_chunked_reference(kl_response, gamma, chunk_size=32)
        actual = FUTURE_KL.compute_future_kl(kl_response, gamma, impl="triton", chunk_size=32)

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_triton_influence_weights_matches_torch(self):
        future_kl = torch.randn(8, 257, device="cuda", dtype=torch.float32)
        advantages = torch.randn(8, 257, device="cuda", dtype=torch.float32)
        ratio = torch.exp(0.1 * torch.randn(8, 257, device="cuda", dtype=torch.float32))

        expected = FUTURE_KL.compute_influence_weights_torch(
            future_kl=future_kl,
            advantages=advantages,
            ratio=ratio,
            clip_ratio=0.2,
            clip_high_only=True,
            safe_threshold=10.0,
        )
        actual = FUTURE_KL.compute_influence_weights(
            future_kl=future_kl,
            advantages=advantages,
            ratio=ratio,
            clip_ratio=0.2,
            clip_high_only=True,
            safe_threshold=10.0,
            impl="triton",
        )

        for got, want in zip(actual[:2], expected[:2], strict=True):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        self.assertEqual(actual[2:], expected[2:])

    def test_triton_masked_mean_matches_torch(self):
        values = torch.randn(16, 129, device="cuda", dtype=torch.float32)
        mask = (torch.rand(16, 129, device="cuda") > 0.2).float()

        expected = FUTURE_KL.compute_masked_mean(values, mask, impl="torch")
        actual = FUTURE_KL.compute_masked_mean(values, mask, impl="triton")

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_bfloat16_inputs_can_be_upcast_before_triton_future_kl(self):
        kl_response = torch.randn(8, 129, device="cuda", dtype=torch.bfloat16)
        gamma = 0.97

        expected = FUTURE_KL.compute_future_kl(kl_response.float(), gamma, impl="torch", chunk_size=16)
        actual = FUTURE_KL.compute_future_kl(kl_response.float(), gamma, impl="triton", chunk_size=16)

        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_triton_fused_ppo_loss_matches_torch(self):
        advantages = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        ratio = torch.exp(0.1 * torch.randn(16, 257, device="cuda", dtype=torch.float32))
        response_mask = (torch.rand(16, 257, device="cuda") > 0.1).float()

        expected_loss, expected_clipfrac, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            impl="torch"
        )
        actual_loss, actual_clipfrac, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            impl="triton"
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4)

    def test_compute_ratio_metrics_matches_original_pattern(self):
        """Verify compute_ratio_metrics produces same results as original scalar metrics pattern."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        batch_size, response_len = 16, 257
        clip_ratio_c = 3.0
        ratio = torch.exp(0.1 * torch.randn(batch_size, response_len, device="cuda", dtype=torch.float32))
        advantages = (torch.rand(batch_size, response_len, device="cuda") - 0.5) * 2.0
        response_mask = (torch.rand(batch_size, response_len, device="cuda") > 0.1).float()

        # Original pattern from core_algos.py
        def original_metrics(ratio, advantages, response_mask):
            is_negative_adv = (advantages < 0)
            neg_valid = ratio[(advantages < 0) & response_mask.bool()]
            pos_valid = ratio[(advantages > 0) & response_mask.bool()]

            result = {}
            if neg_valid.numel() > 0:
                result["neg_is_max"] = neg_valid.max()
                result["neg_is_p75"] = torch.quantile(neg_valid, 0.75)
                result["neg_is_p995"] = torch.quantile(neg_valid, 0.995)
                result["neg_is_p999"] = torch.quantile(neg_valid, 0.999)
            else:
                result["neg_is_max"] = torch.tensor(0.0, device=ratio.device)
                result["neg_is_p995"] = torch.tensor(0.0, device=ratio.device)
                result["neg_is_p999"] = torch.tensor(0.0, device=ratio.device)
                result["neg_is_p75"] = torch.tensor(0.0, device=ratio.device)

            if pos_valid.numel() > 0:
                result["pos_is_max"] = pos_valid.max()
                result["pos_is_p25"] = torch.quantile(pos_valid, 0.25)
                result["pos_is_median"] = torch.quantile(pos_valid, 0.5)
                result["pos_is_p75"] = torch.quantile(pos_valid, 0.75)
                result["pos_is_p995"] = torch.quantile(pos_valid, 0.995)
                result["pos_is_p999"] = torch.quantile(pos_valid, 0.999)
                result["pos_is_min"] = pos_valid.min()
            else:
                result["pos_is_p25"] = torch.tensor(0.0, device=ratio.device)
                result["pos_is_max"] = torch.tensor(0.0, device=ratio.device)
                result["pos_is_median"] = torch.tensor(0.0, device=ratio.device)
                result["pos_is_p75"] = torch.tensor(0.0, device=ratio.device)
                result["pos_is_p995"] = torch.tensor(0.0, device=ratio.device)
                result["pos_is_p999"] = torch.tensor(0.0, device=ratio.device)
                result["pos_is_min"] = torch.tensor(0.0, device=ratio.device)

            result["neg_ratio_2_3"] = (((ratio >= 2.0) & (ratio < 3.0) & is_negative_adv).float() * response_mask).sum() / (response_mask.sum() + 1e-8)
            result["neg_ratio_3_4"] = (((ratio >= 3.0) & (ratio < 4.0) & is_negative_adv).float() * response_mask).sum() / (response_mask.sum() + 1e-8)
            result["neg_ratio_4_10"] = (((ratio >= 4.0) & (ratio < clip_ratio_c) & is_negative_adv).float() * response_mask).sum() / (response_mask.sum() + 1e-8)
            result["pos_mini_frac"] = (((ratio < 1e-3) & (advantages > 0)).float() * response_mask).sum() / (response_mask.sum() + 1e-8)
            return result

        orig = original_metrics(ratio, advantages, response_mask)
        fused = FUTURE_KL.compute_ratio_metrics(ratio, advantages, response_mask, clip_ratio_c=clip_ratio_c)

        for key in orig:
            orig_val = orig[key].item()
            fused_val = fused[key].item()
            if abs(orig_val) > 1e-6 or abs(fused_val) > 1e-6:
                rel_diff = abs(orig_val - fused_val) / (abs(orig_val) + 1e-8)
                self.assertLess(rel_diff, 1e-4, f"{key}: orig={orig_val}, fused={fused_val}")


if __name__ == "__main__":
    unittest.main()
