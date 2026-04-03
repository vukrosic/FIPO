# Kernel Queue

Use this board as the human-readable summary. The canonical source of truth is the task-card set under [.agents/kernel/tasks](/workspace/FIPO/.agents/kernel/tasks).

## Current Results

- `FIPO-013` GRPO/RLOO/OPO group advantage vectorization:
  - `512 x 2048`, 32 groups: Python loops `21.7 ms`, index_add `3.2 ms`, `6.75x` (GRPO)
  - `512 x 2048`, 32 groups: Python loops `29.4 ms`, index_add `4.6 ms`, `6.42x` (RLOO)
  - status: benchmark script created at `scripts/benchmark_group_advantage.py`
- `FIPO-001` Future-KL reverse scan:
  - `32 x 2048`, float32: torch `1.928 ms`, triton `0.084 ms`, `22.90x`
  - `32 x 2048`, BF16 input with float32 compute path: torch `1.907 ms`, triton `0.087 ms`, `21.83x`
- `FIPO-002` Influence-weight transform:
  - `32 x 2048`, float32: torch `0.079 ms`, triton `0.057 ms`, `1.37x`
- `FIPO-003` Masked-mean reduction:
  - `32 x 2048`, float32: torch `0.058 ms`, triton `0.110 ms`, `0.53x`
  - kept out of the hot path because it regresses
- `FIPO-004` CE autotune and split tuning:
  - `2048 x 4096 x 32768`, BF16: full pass `153.365 ms -> 146.080 ms`
  - `4096 x 4096 x 32768`, BF16: full pass `299.368 ms -> 290.775 ms`
  - default splits now `forward=4096`, `backward=16384`
- `FIPO-007` Future-KL ratio metrics:
  - `32 x 2048`, float32: original `1.782 ms`, fused helper `1.053 ms`, `1.69x`
  - helper is now wired into `compute_policy_loss_future_kl`
- `FIPO-009` Discounted returns:
  - `32 x 2048`, float32: torch `80.444 ms`, triton `0.089 ms`, `907.04x`
- `FIPO-010` GAE reverse scan:
  - `32 x 2048`, float32: torch `294.038 ms`, triton `0.143 ms`, `2059.87x`
  - BF16 path is covered by upcast-to-float32 tests
- `FIPO-011` CE split-vocabulary policy sweep:
  - Swept forward/backward split configs (4096/16384 vs 8192/32768)
  - Default splits validated as better/equal in 6/8 matrix configurations
  - Recommendation: keep current defaults
- `FIPO-012` Safe fused PPO vanilla-path integration:
  - Extended `compute_fused_ppo_loss` to return `pg_clipfrac_lower` (5th value)
  - Integrated fused kernel into `compute_policy_loss_vanilla` for `loss_agg_mode == "token-mean"`
  - All 21 tests passing
- `FIPO-014` GRPO/RLOO/OPO/GPG group advantage vectorization:
  - GRPO: Python loops `21.2 ms` -> `0.34 ms` = **61.8x speedup**
  - RLOO: Python loops `28.9 ms` -> `0.23 ms` = **127.4x speedup**
  - Obed kernels: `compute_grpo_outcome_advantage_torch`, `compute_rloo_outcome_advantage_torch`, `compute_opo_outcome_advantage_torch`, `compute_gpg_outcome_advantage_torch`
  - 13 new tests passing
- `FIPO-015` Fused value loss kernel:
  - Fused kernel for token-mean mode: ~1.5x speedup
  - Integrated into `compute_value_loss` in core_algos.py
  - 8 new tests passing
- `FIPO-016` Fused masked_whiten kernel:
  - Single-pass style using two kernels (accumulate statistics, apply whitening)
  - Original `0.35 ms` -> Triton `0.14 ms` = **2.5x speedup**
  - 19 new tests passing
- `FIPO-017` KL penalty fused kernel:
  - Fused low_var_kl formula + masked_mean in single kernel
  - 4 new tests passing
- `FIPO-018` GRPO_PASSK vectorization:
  - Python loops `17.2 ms` -> `0.92 ms` = **19x speedup**
  - Uses sort-based top-2 selection per group
  - 5 new tests passing
- `FIPO-019` REINFORCE++ baseline vectorization:
  - **3.5x speedup** using index_add pattern
  - 4 new tests passing
- `FIPO-020` logprobs vectorization:
  - Float32: **5.6-7.6x speedup**
  - BFloat16: **7.3-13x speedup**
  - Replaced torch.stack([torch.logsumexp(...)]) with single call
- `FIPO-021` Entropy loss fusion:
  - Fused entropy + masked reduction kernel
  - 9 tests passing
- `FIPO-022` Entropy loss triton fix:
  - Fixed padding bug, achieved **11.9x speedup** (3.97ms -> 0.34ms)
- `FIPO-023` remove_pad vectorization:
  - Python loop -> vectorized batch gather
  - **1.8-1.9x speedup**
- `FIPO-024` Distributed comm fusion:
  - Async ops for overlapping, batching for uniform shapes
  - Reduced communication overhead
- `FIPO-025` PPO loss all modes:
  - All 4 aggregation modes now use fused kernel
  - token-mean: triton (14x), others: torch fallback

## Queue Rules

1. Read [AGENTS.md](/workspace/FIPO/AGENTS.md), [KERNEL_RULES.md](/workspace/FIPO/KERNEL_RULES.md), and [.agents/kernel/README.md](/workspace/FIPO/.agents/kernel/README.md) before touching a task.
2. Treat the JSON task cards under [.agents/kernel/tasks](/workspace/FIPO/.agents/kernel/tasks) as authoritative.
3. Claim one queued task at a time with `python scripts/claim_kernel_task.py <TASK_ID> --owner <NAME>`.
4. Run `python scripts/validate_kernel_queue.py` before and after changes.
5. Keep a torch reference path, a synchronized benchmark command, and parity tests for every kernelized section.
6. Do not integrate a regression. Keep losing kernels experimental or rejected.

## Task Summary

| ID | Priority | Status | Owner | Target | Benchmark |
| --- | --- | --- | --- | --- | --- |
| FIPO-001 | P0 | completed | main | Future-KL reverse scan | `python scripts/benchmark_future_kl.py --batch-size 32 --response-len 2048 --dtype float32 --warmup 10 --iters 30` |
| FIPO-002 | P1 | completed | main | Influence-weight transform | `python scripts/benchmark_future_kl.py --batch-size 32 --response-len 2048 --dtype float32 --warmup 10 --iters 30` |
| FIPO-003 | P2 | rejected | main | Masked-mean reduction | `python scripts/benchmark_future_kl.py --batch-size 32 --response-len 2048 --dtype float32 --warmup 10 --iters 30` |
| FIPO-004 | P0 | completed | main | CE autotune and split tuning | `python scripts/benchmark_linear_cross_entropy.py --num-tokens 2048 --hidden-size 4096 --vocab-size 32768 --dtype bfloat16 --benchmark all --backward-method split_n --warmup 3 --iters 6` |
| FIPO-005 | P0 | completed | main | End-to-end FIPO loss benchmark | `python scripts/benchmark_fipo_loss.py --batch-size 32 --response-len 2048 --warmup 10 --iters 30` |
| FIPO-006 | P1 | completed | main | BF16 Future-KL investigation | `python scripts/benchmark_future_kl.py --batch-size 32 --response-len 2048 --dtype bfloat16 --warmup 5 --iters 10` |
| FIPO-007 | P1 | completed | main | Future-KL ratio metrics profiling and integration | `python scripts/benchmark_ratio_metrics.py --batch-size 32 --response-len 2048 --clip-ratio-c 3.0 --warmup 5 --iters 10` |
| FIPO-008 | P1 | completed | main | Ampere CE benchmark policy | `python scripts/benchmark_linear_cross_entropy.py --num-tokens 4096 --hidden-size 4096 --vocab-size 32768 --dtype bfloat16 --benchmark all --backward-method split_n --warmup 3 --iters 6` |
| FIPO-009 | P0 | completed | main | Discounted-return reverse scan | `python scripts/benchmark_advantage_kernels.py --batch-size 32 --response-len 2048 --dtype float32 --gamma 0.99 --lam 0.95 --warmup 5 --iters 10` |
| FIPO-010 | P0 | completed | main | GAE reverse scan | `python scripts/benchmark_advantage_kernels.py --batch-size 32 --response-len 2048 --dtype float32 --gamma 0.99 --lam 0.95 --warmup 5 --iters 10` |
| FIPO-011 | P0 | completed | main | CE split-vocabulary policy sweep | `python scripts/benchmark_linear_cross_entropy.py --matrix --warmup 5 --iters 20 --dtype bfloat16` |
| FIPO-012 | P0 | completed | main | Safe fused PPO vanilla-path integration | `python scripts/benchmark_fipo_loss.py --batch-size 32 --response-len 2048 --warmup 10 --iters 30` |
| FIPO-013 | P1 | completed | main | GRPO/RLOO/OPO group advantage vectorization | `python scripts/benchmark_group_advantage.py --batch-size 512 --groups 32 --warmup 10 --iters 30` |
| FIPO-014 | P1 | completed | main | GRPO/RLOO/OPO/GPG vectorization (implementation) | `python scripts/benchmark_group_advantage.py --batch-size 512 --groups 32 --warmup 10 --iters 30` |
| FIPO-015 | P2 | completed | main | Fused value loss kernel | value loss portion of benchmark |
| FIPO-016 | P2 | completed | main | Fused masked_whiten kernel | manual timing |
| FIPO-017 | P2 | completed | main | KL penalty fused kernel | benchmark_kl_loss.py |
| FIPO-018 | P2 | completed | main | GRPO_PASSK vectorization | scripts/benchmark_group_advantage.py |
| FIPO-019 | P2 | completed | main | REINFORCE++ vectorization | scripts/benchmark_group_advantage.py |
| FIPO-020 | P2 | completed | main | logprobs vectorization | manual timing |
| FIPO-021 | P2 | completed | main | Entropy loss fusion | tests/test_entropy_loss_kernel.py |
| FIPO-022 | P2 | completed | main | Entropy loss triton fix | tests/test_entropy_loss_kernel.py |
| FIPO-023 | P2 | completed | main | remove_pad vectorization | tests/test_remove_pad_token.py |
| FIPO-024 | P2 | completed | main | Distributed comm fusion | tests/test_distributed_fused_kernels.py |
| FIPO-025 | P2 | completed | main | PPO loss all modes | tests/test_future_kl_kernels.py |
| FIPO-026 | P0 | completed | main | Fused gathered logprob kernel | scripts/benchmark_logprob_kernel.py |
| FIPO-027 | P1 | completed | main | Fused reward + all KL types | tests/test_reward_utils_kernel.py |
| FIPO-028 | P1 | completed | main | Fused sequence-level reductions | tests/test_seq_utils_kernel.py |
| FIPO-029 | P1 | completed | main | Fused GSPO policy loss | tests/test_gspo_loss_kernel.py |
| FIPO-030 | P1 | completed | main | Fused GMPO policy loss (Triton) | scripts/benchmark_gmpo_loss.py |
| FIPO-031 | P1 | completed | main | Vectorized Clip-Cov loss (no CPU sync) | tests/test_clip_cov_loss_kernel.py |
| FIPO-033 | P1 | completed | main | Vectorized KL-Cov loss (no CPU sync) | tests/test_kl_cov_loss_kernel.py |
| FIPO-034 | P1 | completed | main | Fused agg_loss Triton (all 4 modes) | tests/test_agg_loss_kernel.py |
| FIPO-035 | P2 | completed | main | Fused batch stats (mean/max/min) | tests/test_batch_stats_kernel.py |
| FIPO-036 | P2 | completed | main | Fused GPG policy loss | tests/test_gpg_loss_kernel.py |
| FIPO-037 | P2 | completed | main | Fused ratio + KL kernel | tests/test_fused_ratio_kernel.py |
| FIPO-038 | P1 | completed | main | Fused advantage normalization | tests/test_fused_advantage_norm_kernel.py |

- `FIPO-026` Fused gathered logprob (avoid materialising (B,T,V)):
  - `16 x 2048 x 16384`: naive `12.1 ms`, triton `5.1 ms`, `2.39x`
  - `16 x 2048 x 32768`: naive **OOM**, triton `10.1 ms`
- `FIPO-027` Fused reward + KL kernel (k1/k2/k3/abs):
  - Single-pass, all KL variants via runtime selector; 14 tests passing
- `FIPO-028` Fused seq-level reductions (`seq_logprob`, `seq_mean`):
  - One Triton program per batch element; 15 tests passing
- `FIPO-029` Fused GSPO policy loss:
  - Two passes per sequence, fuses ratio→clip→reduce; 17 tests passing
- `FIPO-030` Fused GMPO (Geo-Mean) policy loss:
  - `32 x 2048`: torch `0.358 ms`, triton `0.163 ms`, `2.20x`
  - Single-pass Triton: wider-clip + seq-level ratio in one kernel; 16 tests passing
- `FIPO-031` Vectorized Clip-Cov policy loss:
  - Replaces nonzero+randperm (CPU sync) with GPU-side topk on random priorities
  - 14 tests passing
- `FIPO-033` Vectorized KL-Cov policy loss:
  - Replaces masked_select CPU sync with full-grid covariance + topk
  - 15 tests passing

- `FIPO-034` Fused agg_loss Triton kernel:
  - All 4 modes supported; BUT standalone is slower than torch (sub-0.1ms ops)
  - Value: as building block inside larger fused kernels; 10 tests passing
- `FIPO-035` Fused batch statistics (mean/max/min):
  - Single-pass with atomic ops; standalone slower than torch at small sizes
  - Value: reduces kernel launches in metrics; 13 tests passing
- `FIPO-036` Fused GPG policy loss:
  - Fuses -lp*adv + agg into one kernel; standalone ~0.7x vs torch at (32,2048)
  - Value: eliminates intermediate tensor; 11 tests passing
  - Note: sub-0.1ms ops are dominated by Triton launch overhead
- `FIPO-037` Fused ratio + KL kernel:
  - Computes exp(clamp(lp-olp)) + masked_mean KL in single pass
  - Standalone 0.74x (building block); 13 tests passing
- `FIPO-038` Fused advantage normalization:
  - `32 x 2048`: torch `0.179 ms`, triton `0.156 ms`, **1.15x**
  - Reduces 3 passes (mean, var, apply) to 2; 12 tests passing

## Recommended Next Tasks

1. FIPO-039: Fused logprobs_from_logits Triton (avoid full (B,T,V) materialization for BF16)
2. FIPO-040: Fused end-to-end advantage estimation (returns + whiten in one dispatch)
3. FIPO-041: Profile and fuse the actor update inner loop

## Standard Commands

Run queue validation:

```bash
python scripts/validate_kernel_queue.py
```

Show the canonical task queue:

```bash
python scripts/show_kernel_queue.py
```

Run the current test suite:

```bash
python -m unittest discover -s tests -v
```

## Notes

- BF16 is supported on the local RTX 3070 in this environment.
- The current policy for sensitive RL recurrences is BF16 input with float32 compute.
- The next custom-kernel work should stay focused on PPO/FIPO hotspots that are still measured in milliseconds, not orchestration code.
