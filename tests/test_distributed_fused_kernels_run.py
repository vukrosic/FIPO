"""Standalone test runner for distributed fused kernels."""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_distributed_mean_max_min_std_basic(rank, world_size, results):
    """Test basic functionality of distributed_mean_max_min_std."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'

    # Add project root to path for spawned processes
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    import importlib.util
    module_path = Path(__file__).resolve().parents[1] / 'verl' / 'utils' / 'torch_functional.py'
    spec = importlib.util.spec_from_file_location('tf', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(42 + rank)
    tensor = torch.randn(4, 5, dtype=torch.float32)

    mean, max_v, min_v, std = module.distributed_mean_max_min_std(
        tensor, compute_max=True, compute_min=True, compute_std=True
    )

    # Gather all tensors to verify
    all_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors, tensor)
    all_tensors = torch.stack(all_tensors, dim=0)

    expected_mean = all_tensors.mean()
    expected_max = all_tensors.max()
    expected_min = all_tensors.min()
    expected_std = all_tensors.std(unbiased=True)

    try:
        torch.testing.assert_close(mean, expected_mean, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(max_v, expected_max, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(min_v, expected_min, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(std, expected_std, rtol=1e-5, atol=1e-5)
        results[rank] = 'success'
    except Exception as e:
        results[rank] = f'error: {e}'

    dist.destroy_process_group()


def test_broadcast_dict_tensor(rank, world_size, results):
    """Test broadcast_dict_tensor functionality."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'

    # Add project root to path for spawned processes
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    import importlib.util
    module_path = Path(__file__).resolve().parents[1] / 'verl' / 'utils' / 'torch_functional.py'
    spec = importlib.util.spec_from_file_location('tf', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
    module.broadcast_dict_tensor(tensors, src=0, group=group)

    # Now all ranks should have the same values as rank 0's original
    torch.manual_seed(42)
    expected = {
        "a": torch.randn(2, 3),
        "b": torch.randn(3, 4),
        "c": torch.randn(4, 5),
    }

    try:
        for key in tensors.sorted_keys:
            torch.testing.assert_close(tensors[key], expected[key])
        results[rank] = 'success'
    except Exception as e:
        results[rank] = f'error: {e}'

    dist.destroy_process_group()


def test_allgather_dict_tensors_same_shape(rank, world_size, results):
    """Test allgather_dict_tensors with same-shaped tensors."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'

    # Add project root to path for spawned processes
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    import importlib.util
    module_path = Path(__file__).resolve().parents[1] / 'verl' / 'utils' / 'torch_functional.py'
    spec = importlib.util.spec_from_file_location('tf', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(42 + rank)

    # All tensors have same shape
    tensors = {
        "a": torch.randn(2, 3, dtype=torch.float32),
        "b": torch.randn(2, 3, dtype=torch.float32),
        "c": torch.randn(2, 3, dtype=torch.float32),
    }

    group = dist.group.WORLD
    result = module.allgather_dict_tensors(tensors, size=world_size, group=group, dim=0)

    # Verify shapes
    assert result["a"].shape == (world_size * 2, 3), f"a: {result['a'].shape} != {(world_size * 2, 3)}"
    assert result["b"].shape == (world_size * 2, 3)
    assert result["c"].shape == (world_size * 2, 3)

    # Gather expected manually
    expected = {}
    for key in tensors:
        all_vals = [torch.zeros_like(tensors[key]) for _ in range(world_size)]
        dist.all_gather(all_vals, tensors[key])
        expected[key] = torch.cat(all_vals, dim=0)

    try:
        for key in tensors:
            torch.testing.assert_close(result[key], expected[key])
        results[rank] = 'success'
    except Exception as e:
        results[rank] = f'error: {e}'

    dist.destroy_process_group()


def run_test(test_func, world_size=2):
    """Run a distributed test across multiple processes using spawn."""
    ctx = mp.get_context('spawn')
    manager = mp.Manager()
    results = manager.dict()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=test_func, args=(rank, world_size, results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Check all ranks succeeded
    errors = []
    for rank in range(world_size):
        status = results.get(rank, 'missing')
        if status != 'success':
            errors.append(f"Rank {rank}: {status}")

    if errors:
        raise RuntimeError("Test failed:\n" + "\n".join(errors))

    print(f"{test_func.__name__}: PASSED")


if __name__ == '__main__':
    print("Running distributed fused kernel tests...")
    print()

    print("Test 1: distributed_mean_max_min_std basic")
    run_test(test_distributed_mean_max_min_std_basic)

    print("Test 2: broadcast_dict_tensor")
    run_test(test_broadcast_dict_tensor)

    print("Test 3: allgather_dict_tensors same shape")
    run_test(test_allgather_dict_tensors_same_shape)

    print()
    print("All tests passed!")
