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


def _load_torch_functional_modules():
    verl_root = REPO_ROOT / "verl"
    utils_root = verl_root / "utils"

    _ensure_namespace("verl", verl_root)
    _ensure_namespace("verl.utils", utils_root)

    torch_functional_mod = _load_module("verl.utils.torch_functional", utils_root / "torch_functional.py")
    sys.modules["verl.utils"].torch_functional = torch_functional_mod
    return torch_functional_mod


TORCH_FUNCTIONAL = _load_torch_functional_modules()


def remove_pad_token_reference(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Reference implementation with Python loop for correctness comparison."""
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask, strict=True):
        no_padding_batch.append((ids[len(ids) - mask.sum() :]).cpu().numpy().tolist())
    return no_padding_batch


class TestRemovePadToken(unittest.TestCase):
    """Test cases for remove_pad_token vectorized implementation."""

    def test_basic_padding(self):
        """Test with basic right-side padding."""
        # Batch of 3 sequences, each with length 8
        # Sequence 0: [pad, pad, pad, 3, 4, 5, 6, 7] -> valid: [3, 4, 5, 6, 7]
        # Sequence 1: [pad, pad, 2, 3, 4, 5, 6, 7] -> valid: [2, 3, 4, 5, 6, 7]
        # Sequence 2: [1, 2, 3, 4, 5, 6, 7, 8] -> valid: [1, 2, 3, 4, 5, 6, 7, 8]
        input_ids = torch.tensor([
            [0, 0, 0, 3, 4, 5, 6, 7],
            [0, 0, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ])
        attention_mask = torch.tensor([
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])

        result = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)
        expected = remove_pad_token_reference(input_ids, attention_mask)

        self.assertEqual(result, expected)
        self.assertEqual(result[0], [3, 4, 5, 6, 7])
        self.assertEqual(result[1], [2, 3, 4, 5, 6, 7])
        self.assertEqual(result[2], [1, 2, 3, 4, 5, 6, 7, 8])

    def test_all_padding(self):
        """Test with all padding (edge case)."""
        input_ids = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        attention_mask = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        result = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)
        expected = remove_pad_token_reference(input_ids, attention_mask)

        self.assertEqual(result, expected)
        self.assertEqual(result[0], [])
        self.assertEqual(result[1], [])

    def test_no_padding(self):
        """Test with no padding at all."""
        input_ids = torch.tensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ])

        result = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)
        expected = remove_pad_token_reference(input_ids, attention_mask)

        self.assertEqual(result, expected)
        self.assertEqual(result[0], [1, 2, 3, 4])
        self.assertEqual(result[1], [5, 6, 7, 8])

    def test_variable_length_sequences(self):
        """Test with variable length valid sequences."""
        input_ids = torch.tensor([
            [0, 0, 0, 1],       # 1 valid token
            [0, 0, 1, 2],       # 2 valid tokens
            [0, 1, 2, 3],       # 3 valid tokens
            [1, 2, 3, 4],       # 4 valid tokens
        ])
        attention_mask = torch.tensor([
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ])

        result = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)
        expected = remove_pad_token_reference(input_ids, attention_mask)

        self.assertEqual(result, expected)
        self.assertEqual(result[0], [1])
        self.assertEqual(result[1], [1, 2])
        self.assertEqual(result[2], [1, 2, 3])
        self.assertEqual(result[3], [1, 2, 3, 4])

    def test_single_sequence(self):
        """Test with batch size of 1."""
        input_ids = torch.tensor([[0, 0, 1, 2, 3]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]])

        result = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)
        expected = remove_pad_token_reference(input_ids, attention_mask)

        self.assertEqual(result, expected)
        self.assertEqual(result[0], [1, 2, 3])

    def test_large_batch(self):
        """Test with larger batch size."""
        batch_size = 100
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.randint(0, 2, (batch_size, seq_len))

        result = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)
        expected = remove_pad_token_reference(input_ids, attention_mask)

        self.assertEqual(result, expected)

    def test_gpu(self):
        """Test that GPU tensors work correctly."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        input_ids = torch.tensor([
            [0, 0, 0, 3, 4, 5, 6, 7],
            [0, 0, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]).cuda()
        attention_mask = torch.tensor([
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]).cuda()

        result = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)
        expected = remove_pad_token_reference(input_ids.cpu(), attention_mask.cpu())

        self.assertEqual(result, expected)


class TestRemovePadTokenBenchmark(unittest.TestCase):
    """Benchmark tests for remove_pad_token vectorized implementation."""

    def test_benchmark_small(self):
        """Benchmark with small batch."""
        batch_size = 32
        seq_len = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.randint(0, 2, (batch_size, seq_len))

        # Warm up
        for _ in range(10):
            _ = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)

        # Benchmark vectorized
        import time
        num_iterations = 1000
        start = time.perf_counter()
        for _ in range(num_iterations):
            result = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)
        vectorized_time = time.perf_counter() - start

        # Benchmark reference
        start = time.perf_counter()
        for _ in range(num_iterations):
            result_ref = remove_pad_token_reference(input_ids, attention_mask)
        reference_time = time.perf_counter() - start

        print(f"\nBenchmark Small Batch (bs={batch_size}, seq_len={seq_len}):")
        print(f"  Vectorized: {vectorized_time * 1000:.2f} ms for {num_iterations} iterations ({vectorized_time / num_iterations * 1000:.4f} ms/iter)")
        print(f"  Reference:  {reference_time * 1000:.2f} ms for {num_iterations} iterations ({reference_time / num_iterations * 1000:.4f} ms/iter)")
        print(f"  Speedup:    {reference_time / vectorized_time:.2f}x")

        # Vectorized should be faster or at least comparable
        self.assertLess(vectorized_time, reference_time * 1.5)  # At least as fast

    def test_benchmark_large(self):
        """Benchmark with large batch."""
        batch_size = 256
        seq_len = 512
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.randint(0, 2, (batch_size, seq_len))

        # Warm up
        for _ in range(10):
            _ = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)

        # Benchmark vectorized
        import time
        num_iterations = 100
        start = time.perf_counter()
        for _ in range(num_iterations):
            result = TORCH_FUNCTIONAL.remove_pad_token(input_ids, attention_mask)
        vectorized_time = time.perf_counter() - start

        # Benchmark reference
        start = time.perf_counter()
        for _ in range(num_iterations):
            result_ref = remove_pad_token_reference(input_ids, attention_mask)
        reference_time = time.perf_counter() - start

        print(f"\nBenchmark Large Batch (bs={batch_size}, seq_len={seq_len}):")
        print(f"  Vectorized: {vectorized_time * 1000:.2f} ms for {num_iterations} iterations ({vectorized_time / num_iterations * 1000:.4f} ms/iter)")
        print(f"  Reference:  {reference_time * 1000:.2f} ms for {num_iterations} iterations ({reference_time / num_iterations * 1000:.4f} ms/iter)")
        print(f"  Speedup:    {reference_time / vectorized_time:.2f}x")

        # Vectorized should be significantly faster for larger batches
        self.assertLess(vectorized_time, reference_time)


if __name__ == "__main__":
    unittest.main()