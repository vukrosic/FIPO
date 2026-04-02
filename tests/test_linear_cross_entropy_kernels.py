from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ensure_namespace(module_name: str, module_path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    module = types.ModuleType(module_name)
    module.__path__ = [str(module_path)]
    sys.modules[module_name] = module
    return module


def _load_linear_ce_modules():
    verl_root = REPO_ROOT / "verl"
    utils_root = verl_root / "utils"
    kernel_root = utils_root / "kernel"

    _ensure_namespace("verl", verl_root)
    _ensure_namespace("verl.utils", utils_root)
    kernel_pkg = _ensure_namespace("verl.utils.kernel", kernel_root)

    device_mod = _load_module("verl.utils.device", utils_root / "device.py")
    kernels_mod = _load_module("verl.utils.kernel.kernels", kernel_root / "kernels.py")
    linear_ce_mod = _load_module("verl.utils.kernel.linear_cross_entropy", kernel_root / "linear_cross_entropy.py")

    sys.modules["verl.utils"].device = device_mod
    kernel_pkg.kernels = kernels_mod
    kernel_pkg.linear_cross_entropy = linear_ce_mod
    return kernels_mod, linear_ce_mod


KERNELS, LINEAR_CE = _load_linear_ce_modules()


def _reference_linear_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    reduction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = hidden.float() @ weight.float().transpose(0, 1)
    logits = logits / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    nll = -log_probs.gather(1, labels[:, None]).squeeze(1)
    entropy = -(probs * log_probs).sum(dim=-1)

    if reduction == "none":
        return nll, entropy
    if reduction == "sum":
        return nll.sum(), entropy.sum()
    if reduction == "mean":
        return nll.mean(), entropy.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for linear CE kernel tests")
@unittest.skipUnless(KERNELS.HAVE_TRITON, "Triton is required for linear CE kernel tests")
class LinearCrossEntropyKernelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        KERNELS.set_backward_method(KERNELS.BackwardEnum._Split_Dlogits_N)
        KERNELS.set_forward_vocab_per_split(4096)
        KERNELS.set_backward_vocab_per_split(16384)

    def _run_forward_backward_case(self, *, dtype: torch.dtype, reduction: str):
        num_tokens = 64
        hidden_size = 128
        vocab_size = 512
        temperature = 0.9

        hidden_kernel = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        weight_kernel = torch.randn(vocab_size, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
        labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda", dtype=torch.long)

        hidden_ref = hidden_kernel.detach().clone().requires_grad_(True)
        weight_ref = weight_kernel.detach().clone().requires_grad_(True)

        kernel_nll, kernel_entropy = LINEAR_CE.linear_cross_entropy(
            hidden_kernel,
            weight_kernel,
            labels,
            temperature,
            reduction,
            None,
        )
        ref_nll, ref_entropy = _reference_linear_ce(
            hidden_ref,
            weight_ref,
            labels,
            temperature=temperature,
            reduction=reduction,
        )

        if reduction == "none":
            grad_nll = torch.randn_like(kernel_nll)
            grad_entropy = torch.randn_like(kernel_entropy)
            torch.autograd.backward((kernel_nll, kernel_entropy), (grad_nll, grad_entropy))
            torch.autograd.backward((ref_nll, ref_entropy), (grad_nll, grad_entropy))
        else:
            grad_nll = torch.randn((), device="cuda", dtype=torch.float32)
            grad_entropy = torch.randn((), device="cuda", dtype=torch.float32)
            torch.autograd.backward((kernel_nll, kernel_entropy), (grad_nll, grad_entropy))
            torch.autograd.backward((ref_nll, ref_entropy), (grad_nll, grad_entropy))

        out_rtol = 5e-4 if dtype == torch.float32 else 3e-2
        out_atol = 5e-4 if dtype == torch.float32 else 3e-2
        grad_rtol = 1e-3 if dtype == torch.float32 else 5e-2
        grad_atol = 1e-3 if dtype == torch.float32 else 5e-2

        torch.testing.assert_close(kernel_nll.float(), ref_nll.float(), rtol=out_rtol, atol=out_atol)
        torch.testing.assert_close(kernel_entropy.float(), ref_entropy.float(), rtol=out_rtol, atol=out_atol)
        torch.testing.assert_close(hidden_kernel.grad.float(), hidden_ref.grad.float(), rtol=grad_rtol, atol=grad_atol)
        torch.testing.assert_close(weight_kernel.grad.float(), weight_ref.grad.float(), rtol=grad_rtol, atol=grad_atol)

    def test_forward_backward_matches_reference_float32(self):
        self._run_forward_backward_case(dtype=torch.float32, reduction="none")

    def test_forward_backward_matches_reference_bfloat16(self):
        self._run_forward_backward_case(dtype=torch.bfloat16, reduction="none")

    def test_mean_reduction_matches_reference(self):
        self._run_forward_backward_case(dtype=torch.float32, reduction="mean")

    def test_split_size_setters_allow_alternate_configs(self):
        KERNELS.set_forward_vocab_per_split(2048)
        KERNELS.set_backward_vocab_per_split(8192)

        hidden = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(1024, 128, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, 1024, (32,), device="cuda", dtype=torch.long)

        nll, entropy = LINEAR_CE.linear_cross_entropy(hidden, weight, labels, 1.0, "none", None)

        self.assertTrue(torch.isfinite(nll).all().item())
        self.assertTrue(torch.isfinite(entropy).all().item())


if __name__ == "__main__":
    unittest.main()
