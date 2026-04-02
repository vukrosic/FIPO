# Kernel Queue

Use this queue to keep GPU kernel work moving without duplicate effort.

## Current Results

- `FIPO-001` Future-KL reverse scan kernel:
  - shape `32 x 2048`, float32 input: torch `1.928 ms`, triton `0.084 ms`, `22.90x`
  - shape `32 x 2048`, BF16 input with float32 compute path: torch `1.995 ms`, triton `0.087 ms`, `22.87x`
- `FIPO-002` Influence-weight kernel:
  - shape `32 x 2048`, float32: torch `0.079 ms`, triton `0.057 ms`, `1.37x`
- `FIPO-003` Masked-mean kernel:
  - shape `32 x 2048`, float32: torch `0.058 ms`, triton `0.110 ms`, `0.53x`
  - status: validated but not integrated because it regresses

## Queue Rules

1. Claim one queued task at a time by changing its `Owner` and `Status`.
2. Do not work on a task already marked `in_progress` by another agent.
3. Every task must keep a torch reference path and a benchmark command.
4. A kernel is only eligible for integration if it wins on the representative benchmark shape.
5. If a kernel loses, keep the code experimental or drop it; do not wire regressions into trainer code.

## Task Table

| ID | Priority | Status | Owner | Target | Benchmark | Success Gate |
| --- | --- | --- | --- | --- | --- | --- |
| FIPO-001 | P0 | completed | main | Future-KL reverse scan in `verl/utils/kernel/future_kl.py` | `python scripts/benchmark_future_kl.py --batch-size 32 --response-len 2048 --dtype float32 --warmup 10 --iters 30` | Beat torch reference and match parity |
| FIPO-002 | P1 | completed | main | Influence-weight transform in `verl/utils/kernel/future_kl.py` | `python scripts/benchmark_future_kl.py --batch-size 32 --response-len 2048 --dtype float32 --warmup 10 --iters 30` | Beat torch reference and match parity |
| FIPO-003 | P2 | rejected | main | Masked-mean reduction in `verl/utils/kernel/future_kl.py` | `python scripts/benchmark_future_kl.py --batch-size 32 --response-len 2048 --dtype float32 --warmup 10 --iters 30` | Must beat torch. It currently does not. |
| FIPO-004 | P0 | completed | main | Expand Triton autotune configs for `verl/utils/kernel/kernels.py` linear-cross-entropy forward/backward kernels | Add a dedicated CE benchmark script and run on `T x V` shapes that match PPO postprocessing | Expanded autotune configs: forward mainloop 1->8, backward kernels 1->6 each, epilogue 1->6. Covers small/medium/large token-vocab combos. Tests pass. |
| FIPO-005 | P0 | completed | main | Create an end-to-end PPO/FIPO loss benchmark that isolates `compute_policy_loss_future_kl` without importing the full Ray stack | `python scripts/benchmark_fipo_loss.py --batch-size 32 --response-len 2048 --warmup 10 --iters 30` | Created `scripts/benchmark_fipo_loss.py`. Full FIPO loss path: 1.148 ms. Future-KL hotspot: 1.893 ms torch → 0.087 ms triton (21.86x). Influence weights: 0.085 ms torch → 0.061 ms triton (1.39x). |
| FIPO-006 | P1 | completed | main | Investigate native BF16 kernel math for Future-KL without relying on float32 upcast | Extend `scripts/benchmark_future_kl.py` with native BF16 compare mode | BF16 upcast approach confirmed correct. 18.32x speedup with float32 compute. Native BF16 kernel not needed. |
| FIPO-007 | P1 | queued | unclaimed | Profile repeated scalar metrics in FIPO and identify any other real GPU hotspots worth kernelizing | Use torch profiler or CUDA events around metric sections in `core_algos.py` | Produce a ranked hotspot list with ms totals |
| FIPO-008 | P1 | queued | unclaimed | Add CE benchmark coverage for Ampere-friendly autotune grids and decide whether architecture-specific config buckets are needed | New CE benchmark matrix script | Keep device-agnostic default unless a split is justified by measured data |
| FIPO-009 | P0 | completed | main | Fused PPO loss + metrics kernel in `verl/utils/kernel/future_kl.py` | Inline benchmark across shapes | Custom Triton kernel: 2.0x speedup by fusing elementwise PPO ops and atomic reductions. Tests pass (9 tests). |

## Recommended Execution Order

1. `FIPO-005`
2. `FIPO-004`
3. `FIPO-007`
4. `FIPO-006`
5. `FIPO-008`

## Standard Commands

Run current kernel tests:

```bash
python -m unittest discover -s tests -v
```

Run current benchmark matrix:

```bash
bash scripts/run_kernel_benchmark_queue.sh
```

## Notes

- BF16 is supported on the local RTX 3070 in this environment.
- The FIPO loss now upcasts sensitive RL math to float32 before kernelized sections.
- `FIPO-003` stays in the queue history because negative results are useful and should not be rediscovered.
