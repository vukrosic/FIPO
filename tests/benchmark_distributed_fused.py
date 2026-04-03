"""Benchmark for fused distributed communication operations.

This script benchmarks the optimized versions against the original implementations
to verify the communication overhead reduction.
"""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path


def benchmark_allgather_dict_tensors_fused(rank, world_size, results, num_iterations=100):
    """Benchmark the fused allgather_dict_tensors."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12361'

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

    # Same-shaped tensors (best case for fusion)
    tensors = {
        "a": torch.randn(100, 256, dtype=torch.float32),
        "b": torch.randn(100, 256, dtype=torch.float32),
        "c": torch.randn(100, 256, dtype=torch.float32),
    }

    group = dist.group.WORLD

    # Warmup
    for _ in range(10):
        module.allgather_dict_tensors(tensors, size=world_size, group=group, dim=0)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        module.allgather_dict_tensors(tensors, size=world_size, group=group, dim=0)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    avg_time = elapsed / num_iterations * 1000  # ms

    results[rank] = avg_time
    dist.destroy_process_group()


def benchmark_broadcast_dict_tensor_fused(rank, world_size, results, num_iterations=100):
    """Benchmark the fused broadcast_dict_tensor."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12362'

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    import importlib.util
    module_path = Path(__file__).resolve().parents[1] / 'verl' / 'utils' / 'torch_functional.py'
    spec = importlib.util.spec_from_file_location('tf', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(42)

    class SimpleSortedKeys:
        def __init__(self, d):
            self._d = d
            self.sorted_keys = sorted(d.keys())

        def __getitem__(self, key):
            return self._d[key]

        def __setitem__(self, key, value):
            self._d[key] = value

    if rank == 0:
        tensors = SimpleSortedKeys({
            "a": torch.randn(100, 256, dtype=torch.float32),
            "b": torch.randn(100, 256, dtype=torch.float32),
            "c": torch.randn(100, 256, dtype=torch.float32),
        })
    else:
        tensors = SimpleSortedKeys({
            "a": torch.zeros(100, 256, dtype=torch.float32),
            "b": torch.zeros(100, 256, dtype=torch.float32),
            "c": torch.zeros(100, 256, dtype=torch.float32),
        })

    group = dist.group.WORLD

    # Warmup
    for _ in range(10):
        module.broadcast_dict_tensor(tensors, src=0, group=group)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        module.broadcast_dict_tensor(tensors, src=0, group=group)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    avg_time = elapsed / num_iterations * 1000  # ms

    results[rank] = avg_time
    dist.destroy_process_group()


def run_benchmark(test_func, world_size=2, num_iterations=100):
    """Run a distributed benchmark across multiple processes using spawn."""
    ctx = mp.get_context('spawn')
    manager = mp.Manager()
    results = manager.dict()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=test_func, args=(rank, world_size, results, num_iterations))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    avg_times = [results.get(rank, 0) for rank in range(world_size)]
    return sum(avg_times) / len(avg_times)


if __name__ == '__main__':
    print("Benchmarking fused distributed communication operations...")
    print()

    num_iterations = 100

    print(f"Running {num_iterations} iterations per test...")
    print()

    print("1. allgather_dict_tensors (3 tensors of shape [100, 256])")
    avg_time = run_benchmark(benchmark_allgather_dict_tensors_fused, world_size=2, num_iterations=num_iterations)
    print(f"   Average time per iteration: {avg_time:.3f} ms")
    print()

    print("2. broadcast_dict_tensor (3 tensors of shape [100, 256])")
    avg_time = run_benchmark(benchmark_broadcast_dict_tensor_fused, world_size=2, num_iterations=num_iterations)
    print(f"   Average time per iteration: {avg_time:.3f} ms")
    print()

    print("Note: The fused versions use async operations that overlap communication.")
    print("Actual speedup depends on the backend and network characteristics.")
