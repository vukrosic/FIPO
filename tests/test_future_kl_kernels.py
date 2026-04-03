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

        expected_loss, expected_clipfrac, expected_clipfrac_lower, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            impl="torch"
        )
        actual_loss, actual_clipfrac, actual_clipfrac_lower, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            impl="triton"
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac_lower, expected_clipfrac_lower, rtol=1e-4, atol=1e-4)

    def test_triton_fused_ppo_loss_seq_mean_token_sum_matches_torch(self):
        """Test that Triton fused PPO loss matches torch for seq-mean-token-sum mode."""
        advantages = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        ratio = torch.exp(0.1 * torch.randn(16, 257, device="cuda", dtype=torch.float32))
        response_mask = (torch.rand(16, 257, device="cuda") > 0.1).float()

        expected_loss, expected_clipfrac, expected_clipfrac_lower, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            loss_agg_mode="seq-mean-token-sum",
            impl="torch"
        )
        actual_loss, actual_clipfrac, actual_clipfrac_lower, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            loss_agg_mode="seq-mean-token-sum",
            impl="triton"
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac_lower, expected_clipfrac_lower, rtol=1e-4, atol=1e-4)

    def test_triton_fused_ppo_loss_seq_mean_token_mean_matches_torch(self):
        """Test that Triton fused PPO loss matches torch for seq-mean-token-mean mode."""
        advantages = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        ratio = torch.exp(0.1 * torch.randn(16, 257, device="cuda", dtype=torch.float32))
        response_mask = (torch.rand(16, 257, device="cuda") > 0.1).float()

        expected_loss, expected_clipfrac, expected_clipfrac_lower, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            loss_agg_mode="seq-mean-token-mean",
            impl="torch"
        )
        actual_loss, actual_clipfrac, actual_clipfrac_lower, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            loss_agg_mode="seq-mean-token-mean",
            impl="triton"
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac_lower, expected_clipfrac_lower, rtol=1e-4, atol=1e-4)

    def test_triton_fused_ppo_loss_seq_mean_token_sum_norm_matches_torch(self):
        """Test that Triton fused PPO loss matches torch for seq-mean-token-sum-norm mode."""
        advantages = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        ratio = torch.exp(0.1 * torch.randn(16, 257, device="cuda", dtype=torch.float32))
        response_mask = (torch.rand(16, 257, device="cuda") > 0.1).float()

        expected_loss, expected_clipfrac, expected_clipfrac_lower, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            loss_agg_mode="seq-mean-token-sum-norm",
            impl="torch"
        )
        actual_loss, actual_clipfrac, actual_clipfrac_lower, _, _ = FUTURE_KL.compute_fused_ppo_loss(
            advantages, ratio, response_mask,
            cliprange_low=0.2, cliprange_high=0.2, clip_ratio_c=3.0,
            loss_agg_mode="seq-mean-token-sum-norm",
            impl="triton"
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac_lower, expected_clipfrac_lower, rtol=1e-4, atol=1e-4)

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


# =============================================================================
# Tests for Fused Value Loss Kernel
# =============================================================================


class ValueLossKernelCPUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_value_loss_torch_token_mean(self):
        """Test value loss torch implementation with token-mean mode."""
        vpreds = torch.randn(8, 129, dtype=torch.float32)
        values = torch.randn(8, 129, dtype=torch.float32)
        returns = torch.randn(8, 129, dtype=torch.float32)
        response_mask = (torch.rand(8, 129) > 0.1).float()

        vf_loss, vf_clipfrac = FUTURE_KL.compute_value_loss_torch(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=0.2,
            loss_agg_mode="token-mean",
        )

        # Manually compute expected values
        vpreds_clipped = torch.clamp(vpreds, values - 0.2, values + 0.2)
        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpreds_clipped - returns) ** 2
        clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
        expected_vf_loss = 0.5 * ((clipped_vf_losses * response_mask).sum() / (response_mask.sum() + 1e-8))
        expected_vf_clipfrac = ((vf_losses2 > vf_losses1).float() * response_mask).sum() / (response_mask.sum() + 1e-8)

        torch.testing.assert_close(vf_loss, expected_vf_loss, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(vf_clipfrac, expected_vf_clipfrac, rtol=1e-5, atol=1e-5)

    def test_value_loss_torch_seq_mean_token_sum(self):
        """Test value loss torch implementation with seq-mean-token-sum mode."""
        vpreds = torch.randn(8, 129, dtype=torch.float32)
        values = torch.randn(8, 129, dtype=torch.float32)
        returns = torch.randn(8, 129, dtype=torch.float32)
        response_mask = (torch.rand(8, 129) > 0.1).float()

        vf_loss, vf_clipfrac = FUTURE_KL.compute_value_loss_torch(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=0.2,
            loss_agg_mode="seq-mean-token-sum",
        )

        # Manually compute expected values
        vpreds_clipped = torch.clamp(vpreds, values - 0.2, values + 0.2)
        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpreds_clipped - returns) ** 2
        clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
        seq_losses = torch.sum(clipped_vf_losses * response_mask, dim=-1)
        expected_vf_loss = 0.5 * torch.mean(seq_losses)

        torch.testing.assert_close(vf_loss, expected_vf_loss, rtol=1e-5, atol=1e-5)

    def test_value_loss_torch_seq_mean_token_mean(self):
        """Test value loss torch implementation with seq-mean-token-mean mode."""
        vpreds = torch.randn(8, 129, dtype=torch.float32)
        values = torch.randn(8, 129, dtype=torch.float32)
        returns = torch.randn(8, 129, dtype=torch.float32)
        response_mask = (torch.rand(8, 129) > 0.1).float()

        vf_loss, vf_clipfrac = FUTURE_KL.compute_value_loss_torch(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=0.2,
            loss_agg_mode="seq-mean-token-mean",
        )

        # Manually compute expected values
        vpreds_clipped = torch.clamp(vpreds, values - 0.2, values + 0.2)
        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpreds_clipped - returns) ** 2
        clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
        seq_losses = torch.sum(clipped_vf_losses * response_mask, dim=-1) / torch.sum(response_mask, dim=-1)
        expected_vf_loss = 0.5 * torch.mean(seq_losses)

        torch.testing.assert_close(vf_loss, expected_vf_loss, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton kernel tests")
@unittest.skipUnless(FUTURE_KL.HAVE_TRITON, "Triton is required for Triton kernel tests")
class ValueLossKernelCUDATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_triton_value_loss_matches_torch_token_mean(self):
        """Test that Triton value loss matches torch for token-mean mode."""
        vpreds = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        values = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        returns = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        response_mask = (torch.rand(16, 257, device="cuda") > 0.1).float()

        expected_loss, expected_clipfrac = FUTURE_KL.compute_value_loss_torch(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=0.2,
            loss_agg_mode="token-mean",
        )
        actual_loss, actual_clipfrac = FUTURE_KL.compute_value_loss(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=0.2,
            loss_agg_mode="token-mean",
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4)

    def test_triton_value_loss_auto_falls_back_to_torch_on_cpu(self):
        """Test that auto impl falls back to torch on CPU tensors."""
        vpreds = torch.randn(8, 129, dtype=torch.float32)
        values = torch.randn(8, 129, dtype=torch.float32)
        returns = torch.randn(8, 129, dtype=torch.float32)
        response_mask = (torch.rand(8, 129) > 0.1).float()

        expected_loss, expected_clipfrac = FUTURE_KL.compute_value_loss_torch(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=0.2,
            loss_agg_mode="token-mean",
        )
        actual_loss, actual_clipfrac = FUTURE_KL.compute_value_loss(
            vpreds=vpreds,
            values=values,
            returns=returns,
            response_mask=response_mask,
            cliprange_value=0.2,
            loss_agg_mode="token-mean",
            impl="auto",
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-5, atol=1e-5)

    def test_triton_value_loss_with_different_batch_sizes(self):
        """Test value loss kernel with various batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            vpreds = torch.randn(batch_size, 257, device="cuda", dtype=torch.float32)
            values = torch.randn(batch_size, 257, device="cuda", dtype=torch.float32)
            returns = torch.randn(batch_size, 257, device="cuda", dtype=torch.float32)
            response_mask = (torch.rand(batch_size, 257, device="cuda") > 0.1).float()

            expected_loss, expected_clipfrac = FUTURE_KL.compute_value_loss_torch(
                vpreds=vpreds,
                values=values,
                returns=returns,
                response_mask=response_mask,
                cliprange_value=0.2,
                loss_agg_mode="token-mean",
            )
            actual_loss, actual_clipfrac = FUTURE_KL.compute_value_loss(
                vpreds=vpreds,
                values=values,
                returns=returns,
                response_mask=response_mask,
                cliprange_value=0.2,
                loss_agg_mode="token-mean",
            )

            torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4, msg=f"batch_size={batch_size}")
            torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4, msg=f"batch_size={batch_size}")

    def test_triton_value_loss_with_different_cliprange(self):
        """Test value loss kernel with various clip ranges."""
        vpreds = torch.randn(8, 257, device="cuda", dtype=torch.float32)
        values = torch.randn(8, 257, device="cuda", dtype=torch.float32)
        returns = torch.randn(8, 257, device="cuda", dtype=torch.float32)
        response_mask = (torch.rand(8, 257, device="cuda") > 0.1).float()

        for cliprange in [0.05, 0.1, 0.2, 0.3, 0.5]:
            expected_loss, expected_clipfrac = FUTURE_KL.compute_value_loss_torch(
                vpreds=vpreds,
                values=values,
                returns=returns,
                response_mask=response_mask,
                cliprange_value=cliprange,
                loss_agg_mode="token-mean",
            )
            actual_loss, actual_clipfrac = FUTURE_KL.compute_value_loss(
                vpreds=vpreds,
                values=values,
                returns=returns,
                response_mask=response_mask,
                cliprange_value=cliprange,
                loss_agg_mode="token-mean",
            )

            torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4, msg=f"cliprange={cliprange}")
            torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4, msg=f"cliprange={cliprange}")


# =============================================================================
# Tests for Fused KL Loss Kernel
# =============================================================================


class KLLossKernelCPUTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_kl_loss_torch_matches_original_two_step(self):
        """Test that torch implementation matches original kl_penalty + agg_loss pattern."""
        logprob = torch.randn(8, 129, dtype=torch.float32)
        ref_logprob = torch.randn(8, 129, dtype=torch.float32)
        response_mask = (torch.rand(8, 129) > 0.1).float()

        # Original two-step pattern
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20.0, max=20.0)
        ratio = torch.exp(kl)
        kld = ratio - kl - 1.0
        kld = torch.clamp(kld, min=-10.0, max=10.0)
        kld = kld * response_mask
        expected_loss = kld.sum() / (response_mask.sum() + 1e-8)

        # clipfrac: tokens where kld < -10 before clamping
        kld_preclamp = ratio - kl - 1.0
        was_clamped = kld_preclamp < -10.0
        expected_clipfrac = (was_clamped.float() * response_mask).sum() / (response_mask.sum() + 1e-8)

        # Fused implementation
        actual_loss, actual_clipfrac = FUTURE_KL.compute_kl_loss_torch(
            logprob=logprob,
            ref_logprob=ref_logprob,
            response_mask=response_mask,
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-5, atol=1e-5)

    def test_kl_loss_auto_falls_back_to_torch_on_cpu(self):
        """Test that auto impl falls back to torch on CPU tensors."""
        logprob = torch.randn(8, 129, dtype=torch.float32)
        ref_logprob = torch.randn(8, 129, dtype=torch.float32)
        response_mask = (torch.rand(8, 129) > 0.1).float()

        expected_loss, expected_clipfrac = FUTURE_KL.compute_kl_loss(
            logprob=logprob,
            ref_logprob=ref_logprob,
            response_mask=response_mask,
            impl="torch",
        )
        actual_loss, actual_clipfrac = FUTURE_KL.compute_kl_loss(
            logprob=logprob,
            ref_logprob=ref_logprob,
            response_mask=response_mask,
            impl="auto",
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton kernel tests")
@unittest.skipUnless(FUTURE_KL.HAVE_TRITON, "Triton is required for Triton kernel tests")
class KLLossKernelCUDATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    def test_triton_kl_loss_matches_torch(self):
        """Test that Triton KL loss matches torch implementation."""
        logprob = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        ref_logprob = torch.randn(16, 257, device="cuda", dtype=torch.float32)
        response_mask = (torch.rand(16, 257, device="cuda") > 0.1).float()

        expected_loss, expected_clipfrac = FUTURE_KL.compute_kl_loss(
            logprob=logprob,
            ref_logprob=ref_logprob,
            response_mask=response_mask,
            impl="torch",
        )
        actual_loss, actual_clipfrac = FUTURE_KL.compute_kl_loss(
            logprob=logprob,
            ref_logprob=ref_logprob,
            response_mask=response_mask,
            impl="triton",
        )

        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4)

    def test_triton_kl_loss_with_different_batch_sizes(self):
        """Test KL loss kernel with various batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            logprob = torch.randn(batch_size, 257, device="cuda", dtype=torch.float32)
            ref_logprob = torch.randn(batch_size, 257, device="cuda", dtype=torch.float32)
            response_mask = (torch.rand(batch_size, 257, device="cuda") > 0.1).float()

            expected_loss, expected_clipfrac = FUTURE_KL.compute_kl_loss(
                logprob=logprob,
                ref_logprob=ref_logprob,
                response_mask=response_mask,
                impl="torch",
            )
            actual_loss, actual_clipfrac = FUTURE_KL.compute_kl_loss(
                logprob=logprob,
                ref_logprob=ref_logprob,
                response_mask=response_mask,
                impl="triton",
            )

            torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4, msg=f"batch_size={batch_size}")
            torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4, msg=f"batch_size={batch_size}")

    def test_triton_kl_loss_parity_with_original_actor_code(self):
        """Verify compute_kl_loss produces same results as actor code path."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        batch_size, response_len = 16, 512
        logprob = torch.randn(batch_size, response_len, device="cuda", dtype=torch.float32)
        ref_logprob = torch.randn(batch_size, response_len, device="cuda", dtype=torch.float32)
        response_mask = (torch.rand(batch_size, response_len, device="cuda") > 0.1).float()

        # Original actor code pattern (from dp_actor.py / megatron_actor.py):
        # kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
        # kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
        # where kl_loss_type="low_var_kl" and loss_agg_mode="token-mean"

        # kl_penalty (low_var_kl mode)
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20.0, max=20.0)
        ratio = torch.exp(kl)
        kld = ratio - kl - 1.0
        kld = torch.clamp(kld, min=-10.0, max=10.0)

        # agg_loss (token-mean mode)
        expected_kl_loss = (kld * response_mask).sum() / (response_mask.sum() + 1e-8)

        # clipfrac calculation (same as in compute_kl_loss_torch)
        kld_preclamp = ratio - kl - 1.0
        was_clamped = kld_preclamp < -10.0
        expected_clipfrac = (was_clamped.float() * response_mask).sum() / (response_mask.sum() + 1e-8)

        # Fused implementation
        actual_loss, actual_clipfrac = FUTURE_KL.compute_kl_loss(
            logprob=logprob,
            ref_logprob=ref_logprob,
            response_mask=response_mask,
            impl="triton",
        )

        torch.testing.assert_close(actual_loss, expected_kl_loss, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual_clipfrac, expected_clipfrac, rtol=1e-4, atol=1e-4)
