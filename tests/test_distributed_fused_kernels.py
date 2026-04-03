"""Tests for fused distributed communication operations.

These tests verify that the optimized distributed operations produce the same
results as the original implementations.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch
import torch.distributed as dist


def _load_torch_functional_module():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "torch_functional.py"
    spec = importlib.util.spec_from_file_location("torch_functional_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TORCH_FUNCTIONAL = _load_torch_functional_module()


def _run_distributed_test_worker(rank, world_size, test_func, results_dict):
    """Worker function to run distributed tests."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        result = test_func(rank, world_size)
        results_dict[rank] = ("success", result)
    except Exception as e:
        results_dict[rank] = ("error", str(e))
    finally:
        dist.destroy_process_group()


def _run_distributed_test(test_func, world_size=2):
    """Run a distributed test across multiple processes using spawn."""
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    manager = multiprocessing.Manager()
    results_dict = manager.dict()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=_run_distributed_test_worker, args=(rank, world_size, test_func, results_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Check all ranks succeeded
    errors = []
    for rank in range(world_size):
        status, data = results_dict[rank]
        if status == "error":
            errors.append(f"Rank {rank}: {data}")

    if errors:
        raise RuntimeError("Test failed:\n" + "\n".join(errors))

    # Return results from all ranks
    return [results_dict[rank][1] for rank in range(world_size)]


# Tests for distributed_mean_max_min_std


def _test_distributed_mean_max_min_std_basic(rank, world_size):
    """Test basic functionality of distributed_mean_max_min_std."""
    torch.manual_seed(42 + rank)

    # Create local tensor with different values per rank
    local_tensor = torch.randn(4, 5, dtype=torch.float32)

    # All ranks compute
    mean, max_val, min_val, std = TORCH_FUNCTIONAL.distributed_mean_max_min_std(
        local_tensor, compute_max=True, compute_min=True, compute_std=True
    )

    # Gather all local tensors to compute expected values
    all_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors, local_tensor)
    all_tensors = torch.stack(all_tensors, dim=0)

    # Compute expected values
    expected_mean = all_tensors.mean()
    expected_max = all_tensors.max()
    expected_min = all_tensors.min()
    expected_std = all_tensors.std(unbiased=True)

    # Check results match
    torch.testing.assert_close(mean, expected_mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(max_val, expected_max, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(min_val, expected_min, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(std, expected_std, rtol=1e-5, atol=1e-5)

    return True


def _test_distributed_mean_max_min_std_partial(rank, world_size):
    """Test with compute_max=False, compute_min=False, compute_std=False combinations."""
    torch.manual_seed(42 + rank)
    local_tensor = torch.randn(3, 4, dtype=torch.float32)

    # Test with only mean
    mean, max_val, min_val, std = TORCH_FUNCTIONAL.distributed_mean_max_min_std(
        local_tensor, compute_max=False, compute_min=False, compute_std=False
    )
    assert max_val is None
    assert min_val is None
    assert std is None

    all_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors, local_tensor)
    all_tensors = torch.stack(all_tensors, dim=0)
    expected_mean = all_tensors.mean()
    torch.testing.assert_close(mean, expected_mean, rtol=1e-5, atol=1e-5)

    # Test with max only
    mean, max_val, min_val, std = TORCH_FUNCTIONAL.distributed_mean_max_min_std(
        local_tensor, compute_max=True, compute_min=False, compute_std=False
    )
    assert min_val is None
    assert std is None
    expected_max = all_tensors.max()
    torch.testing.assert_close(max_val, expected_max, rtol=1e-5, atol=1e-5)

    return True


def _test_distributed_mean_max_min_std_single_element(rank, world_size):
    """Test with single-element tensors."""
    torch.manual_seed(42 + rank)
    local_tensor = torch.tensor([rank * 1.0], dtype=torch.float32)

    mean, max_val, min_val, std = TORCH_FUNCTIONAL.distributed_mean_max_min_std(
        local_tensor, compute_max=True, compute_min=True, compute_std=True
    )

    all_values = torch.tensor([float(i) for i in range(world_size)])
    expected_mean = all_values.mean()
    expected_max = all_values.max()
    expected_min = all_values.min()
    # Std with ddof=1 for single-element per rank
    expected_std = all_values.std(unbiased=True)

    torch.testing.assert_close(mean, expected_mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(max_val, expected_max, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(min_val, expected_min, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(std, expected_std, rtol=1e-5, atol=1e-5)

    return True


# Tests for broadcast_dict_tensor


def _test_broadcast_dict_tensor(rank, world_size):
    """Test broadcast_dict_tensor functionality."""
    torch.manual_seed(42)

    # Create a TensorDict-like object
    class SimpleSortedKeys:
        def __init__(self, d):
            self._d = d
            self.sorted_keys = sorted(d.keys())

        def __getitem__(self, key):
            return self._d[key]

        def __setitem__(self, key, value):
            self._d[key] = value

    # Use same seed on all ranks so we know expected values
    torch.manual_seed(42)
    if rank == 0:
        tensors = SimpleSortedKeys({
            "a": torch.randn(2, 3),
            "b": torch.randn(3, 4),
            "c": torch.randn(4, 5),
        })
    else:
        tensors = SimpleSortedKeys({
            "a": torch.zeros(2, 3),
            "b": torch.zeros(3, 4),
            "c": torch.zeros(4, 5),
        })

    # Broadcast from rank 0
    group = dist.group.WORLD
    TORCH_FUNCTIONAL.broadcast_dict_tensor(tensors, src=0, group=group)

    # Now all ranks should have the same values as rank 0's original
    torch.manual_seed(42)
    expected = {
        "a": torch.randn(2, 3),
        "b": torch.randn(3, 4),
        "c": torch.randn(4, 5),
    }

    for key in tensors.sorted_keys:
        torch.testing.assert_close(tensors[key], expected[key])

    return True


# Tests for allgather_dict_tensors


def _test_allgather_dict_tensors_same_shape(rank, world_size):
    """Test allgather_dict_tensors with same-shaped tensors."""
    torch.manual_seed(42 + rank)

    # All tensors have same shape
    tensors = {
        "a": torch.randn(2, 3, dtype=torch.float32),
        "b": torch.randn(2, 3, dtype=torch.float32),
        "c": torch.randn(2, 3, dtype=torch.float32),
    }

    group = dist.group.WORLD
    result = TORCH_FUNCTIONAL.allgather_dict_tensors(tensors, size=world_size, group=group, dim=0)

    # Verify result shape
    assert result["a"].shape == (world_size * 2, 3)
    assert result["b"].shape == (world_size * 2, 3)
    assert result["c"].shape == (world_size * 2, 3)

    # Gather expected manually
    expected = {}
    for key in tensors:
        all_vals = [torch.zeros_like(tensors[key]) for _ in range(world_size)]
        dist.all_gather(all_vals, tensors[key])
        expected[key] = torch.cat(all_vals, dim=0)

    for key in tensors:
        torch.testing.assert_close(result[key], expected[key])

    return True


def _test_allgather_dict_tensors_different_shapes(rank, world_size):
    """Test allgather_dict_tensors with different-shaped tensors."""
    torch.manual_seed(42 + rank)

    # Different shapes
    tensors = {
        "small": torch.randn(2, dtype=torch.float32),
        "medium": torch.randn(3, 4, dtype=torch.float32),
        "large": torch.randn(5, 6, 7, dtype=torch.float32),
    }

    group = dist.group.WORLD
    result = TORCH_FUNCTIONAL.allgather_dict_tensors(tensors, size=world_size, group=group, dim=0)

    # Verify shapes
    assert result["small"].shape == (world_size * 2,)
    assert result["medium"].shape == (world_size * 3, 4)
    assert result["large"].shape == (world_size * 5, 6, 7)

    # Gather expected manually
    expected = {}
    for key in tensors:
        all_vals = [torch.zeros_like(tensors[key]) for _ in range(world_size)]
        dist.all_gather(all_vals, tensors[key])
        expected[key] = torch.cat(all_vals, dim=0)

    for key in tensors:
        torch.testing.assert_close(result[key], expected[key])

    return True


def _test_allgather_dict_tensors_with_tensordict(rank, world_size):
    """Test allgather_dict_tensors with TensorDict input."""
    from tensordict import TensorDict

    torch.manual_seed(42 + rank)

    # Use consistent batch size (3) for all tensors
    tensors = TensorDict(
        source={
            "x": torch.randn(3, 4, dtype=torch.float32),
            "y": torch.randn(3, dtype=torch.float32),  # Fixed: batch dim is 3, not 5
        },
        batch_size=[3],
    )

    group = dist.group.WORLD
    result = TORCH_FUNCTIONAL.allgather_dict_tensors(tensors, size=world_size, group=group, dim=0)

    # Check result is TensorDict
    assert isinstance(result, TensorDict)
    assert result.batch_size[0] == world_size * 3
    assert "x" in result
    assert "y" in result

    # Verify values
    for key in ["x", "y"]:
        all_vals = [torch.zeros_like(tensors[key]) for _ in range(world_size)]
        dist.all_gather(all_vals, tensors[key])
        expected = torch.cat(all_vals, dim=0)
        torch.testing.assert_close(result[key], expected)

    return True


# Test classes


class TestDistributedMeanMaxMinStd(unittest.TestCase):
    """Tests for distributed_mean_max_min_std."""

    def test_basic(self):
        """Test basic functionality with all metrics enabled."""
        _run_distributed_test(_test_distributed_mean_max_min_std_basic, world_size=2)

    def test_partial_computation(self):
        """Test with various compute_* flags disabled."""
        _run_distributed_test(_test_distributed_mean_max_min_std_partial, world_size=2)

    def test_single_element_tensors(self):
        """Test with single-element tensors."""
        _run_distributed_test(_test_distributed_mean_max_min_std_single_element, world_size=2)


class TestBroadcastDictTensor(unittest.TestCase):
    """Tests for broadcast_dict_tensor."""

    def test_broadcast(self):
        """Test broadcasting from rank 0."""
        _run_distributed_test(_test_broadcast_dict_tensor, world_size=2)


class TestAllgatherDictTensors(unittest.TestCase):
    """Tests for allgather_dict_tensors."""

    def test_same_shape(self):
        """Test with all tensors having the same shape."""
        _run_distributed_test(_test_allgather_dict_tensors_same_shape, world_size=2)

    def test_different_shapes(self):
        """Test with tensors having different shapes."""
        _run_distributed_test(_test_allgather_dict_tensors_different_shapes, world_size=2)

    def test_with_tensordict(self):
        """Test with TensorDict input."""
        _run_distributed_test(_test_allgather_dict_tensors_with_tensordict, world_size=2)


if __name__ == "__main__":
    unittest.main()
