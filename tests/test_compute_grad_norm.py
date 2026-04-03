"""Tests for compute_grad_norm function in verl.utils.torch_functional."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


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


def _load_torch_functional_modules():
    verl_root = REPO_ROOT / "verl"
    utils_root = verl_root / "utils"

    _ensure_namespace("verl", verl_root)
    _ensure_namespace("verl.utils", utils_root)

    torch_functional_mod = _load_module("verl.utils.torch_functional", utils_root / "torch_functional.py")
    sys.modules["verl.utils"].torch_functional = torch_functional_mod
    return torch_functional_mod


TORCH_FUNCTIONAL = _load_torch_functional_modules()


def compute_grad_norm_reference(model: nn.Module):
    """Reference implementation with .item() calls for correctness comparison.

    This is the original implementation that uses Python scalar accumulation
    and CPU-GPU synchronization on each parameter.
    """
    total_grad_square = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_square += torch.sum(torch.square(param.grad.detach())).item()
    return total_grad_square


class TestComputeGradNorm(unittest.TestCase):
    """Test cases for compute_grad_norm optimization."""

    def test_basic_functionality(self):
        """Test basic gradient norm computation with a simple model."""
        model = nn.Linear(512, 256)
        x = torch.randn(32, 512)
        loss = model(x).sum()
        loss.backward()

        result = TORCH_FUNCTIONAL.compute_grad_norm(model)
        expected = compute_grad_norm_reference(model)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)
        # Relative tolerance due to floating point accumulation differences
        self.assertAlmostEqual(result, expected, delta=expected * 1e-5)

    def test_no_gradients(self):
        """Test with model that has no gradients (no backward pass)."""
        model = nn.Linear(512, 256)
        # Don't call backward

        result = TORCH_FUNCTIONAL.compute_grad_norm(model)
        expected = compute_grad_norm_reference(model)

        self.assertEqual(result, 0.0)
        self.assertEqual(result, expected)

    def test_partial_gradients(self):
        """Test with some parameters having no gradients."""
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 128)
        )
        x = torch.randn(32, 512)
        loss = model(x).sum()
        loss.backward()

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        result = TORCH_FUNCTIONAL.compute_grad_norm(model)
        expected = compute_grad_norm_reference(model)

        self.assertGreater(result, 0.0)
        # Relative tolerance due to floating point accumulation differences
        self.assertAlmostEqual(result, expected, delta=expected * 1e-5)

    def test_larger_model(self):
        """Test with a larger model."""
        model = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )
        x = torch.randn(64, 512)
        loss = model(x).sum()
        loss.backward()

        result = TORCH_FUNCTIONAL.compute_grad_norm(model)
        expected = compute_grad_norm_reference(model)

        # Relative tolerance for larger values
        self.assertAlmostEqual(result, expected, delta=expected * 1e-5)

    def test_zero_gradients(self):
        """Test with zero gradients."""
        model = nn.Linear(512, 256)
        x = torch.randn(32, 512)
        loss = model(x).sum()
        loss.backward()

        # Zero out gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()

        result = TORCH_FUNCTIONAL.compute_grad_norm(model)
        expected = compute_grad_norm_reference(model)

        self.assertEqual(result, 0.0)
        self.assertEqual(result, expected)

    def test_nested_model(self):
        """Test with nested model architecture."""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(256, 128)
                self.layer2 = nn.Linear(128, 64)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                x = self.layer1(x)
                x = self.dropout(x)
                x = self.layer2(x)
                return x

        model = NestedModel()
        x = torch.randn(32, 256)
        loss = model(x).sum()
        loss.backward()

        result = TORCH_FUNCTIONAL.compute_grad_norm(model)
        expected = compute_grad_norm_reference(model)

        self.assertGreater(result, 0.0)
        self.assertAlmostEqual(result, expected, delta=expected * 1e-5)


class TestComputeGradNormGPU(unittest.TestCase):
    """GPU-specific tests for compute_grad_norm."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_gpu_computation(self):
        """Test that GPU tensors work correctly."""
        model = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        ).cuda()

        x = torch.randn(64, 512).cuda()
        loss = model(x).sum()
        loss.backward()

        result = TORCH_FUNCTIONAL.compute_grad_norm(model)
        expected = compute_grad_norm_reference(model)

        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)
        self.assertAlmostEqual(result, expected, delta=expected * 1e-5)

    def test_gpu_empty_grad(self):
        """Test with model on GPU but no gradients."""
        model = nn.Linear(512, 256).cuda()
        # Don't call backward

        result = TORCH_FUNCTIONAL.compute_grad_norm(model)
        expected = compute_grad_norm_reference(model)

        self.assertEqual(result, 0.0)
        self.assertEqual(result, expected)


class TestComputeGradNormBenchmark(unittest.TestCase):
    """Benchmark tests for compute_grad_norm optimization."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_benchmark_small_model(self):
        """Benchmark with small model."""
        model = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        ).cuda()

        x = torch.randn(32, 512).cuda()
        loss = model(x).sum()
        loss.backward()

        # Warm up
        for _ in range(100):
            _ = TORCH_FUNCTIONAL.compute_grad_norm(model)

        import time
        num_iterations = 5000

        # Benchmark optimized version
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            result_opt = TORCH_FUNCTIONAL.compute_grad_norm(model)
        torch.cuda.synchronize()
        optimized_time = time.perf_counter() - start

        # Benchmark reference (with .item() calls)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            result_ref = compute_grad_norm_reference(model)
        torch.cuda.synchronize()
        reference_time = time.perf_counter() - start

        print(f"\nBenchmark Small Model:")
        print(f"  Optimized: {optimized_time * 1000:.2f} ms for {num_iterations} iterations "
              f"({optimized_time / num_iterations * 1000:.4f} ms/iter)")
        print(f"  Reference: {reference_time * 1000:.2f} ms for {num_iterations} iterations "
              f"({reference_time / num_iterations * 1000:.4f} ms/iter)")
        print(f"  Speedup:   {reference_time / optimized_time:.2f}x")

        # Verify correctness
        self.assertAlmostEqual(result_opt, result_ref, delta=result_ref * 1e-5)

    def test_benchmark_large_model(self):
        """Benchmark with large model (more parameters = more .item() calls)."""
        model = nn.Sequential(
            nn.Linear(4096, 16384),
            nn.ReLU(),
            nn.Linear(16384, 16384),
            nn.ReLU(),
            nn.Linear(16384, 4096)
        ).cuda()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nLarge model parameters: {total_params:,}")

        x = torch.randn(64, 4096).cuda()
        loss = model(x).sum()
        loss.backward()

        # Warm up
        for _ in range(50):
            _ = TORCH_FUNCTIONAL.compute_grad_norm(model)

        import time
        num_iterations = 2000

        # Benchmark optimized version
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            result_opt = TORCH_FUNCTIONAL.compute_grad_norm(model)
        torch.cuda.synchronize()
        optimized_time = time.perf_counter() - start

        # Benchmark reference (with .item() calls)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            result_ref = compute_grad_norm_reference(model)
        torch.cuda.synchronize()
        reference_time = time.perf_counter() - start

        print(f"Benchmark Large Model:")
        print(f"  Optimized: {optimized_time * 1000:.2f} ms for {num_iterations} iterations "
              f"({optimized_time / num_iterations * 1000:.4f} ms/iter)")
        print(f"  Reference: {reference_time * 1000:.2f} ms for {num_iterations} iterations "
              f"({reference_time / num_iterations * 1000:.4f} ms/iter)")
        print(f"  Speedup:   {reference_time / optimized_time:.2f}x")

        # Verify correctness
        self.assertAlmostEqual(result_opt, result_ref, delta=result_ref * 1e-5)


if __name__ == "__main__":
    unittest.main()
