# Agent Critique And Instructions

This file is the current handoff for long-running kernel work in this repo.

## Current State

- Queue validation: `python scripts/validate_kernel_queue.py` passes.
- Queue summary: `12` task cards total, `9` completed, `2` queued, `1` rejected.
- No active agents are running right now.
- Current queue view: `python scripts/show_kernel_queue.py`
- Current full test suite status: `python -m unittest discover -s tests -v` passed with `21` tests.
- Local GPU: RTX 3070, BF16 supported, Triton available.

## What We Have

Completed and measured kernel wins:

- Future-KL reverse scan:
  - `32 x 2048`, float32: `1.928 ms -> 0.084 ms`, `22.90x`
- Influence weights:
  - `32 x 2048`, float32: `0.079 ms -> 0.057 ms`, `1.37x`
- Discounted returns for REINFORCE++ and REMAX:
  - `32 x 2048`, float32: `80.444 ms -> 0.089 ms`, `907.04x`
- GAE reverse scan:
  - `32 x 2048`, float32: `294.038 ms -> 0.143 ms`, `2059.87x`
- Future-KL ratio metrics helper:
  - `32 x 2048`, float32: `1.782 ms -> 1.053 ms`, `1.69x`
- CE tuning:
  - `2048 x 4096 x 32768`, BF16 full pass: `153.365 ms -> 146.080 ms`
  - `4096 x 4096 x 32768`, BF16 full pass: `299.368 ms -> 290.775 ms`
  - current split defaults: `forward=4096`, `backward=16384`

Rejected work:

- Masked-mean Triton kernel lost on the representative shape and is not integrated.

## What The Agents Were Doing

- `Hegel` did read-only hotspot analysis and identified the ratio-metrics integration path as the next low-risk win.
- That recommendation was applied. `compute_ratio_metrics` is now parameterized by `clip_ratio_c` and wired into `compute_policy_loss_future_kl`.
- `Tesla` was asked to refresh queue docs, but that work was superseded and the agent was closed to avoid conflicting edits.
- Current active agent count: `0`.

## Critique

The good:

- The work is finally biased toward measured hotspots instead of speculative kernel writing.
- The repo now has real benchmark entrypoints and actual kernel parity tests.
- The task-card system is usable and validated.
- The biggest RL recurrences are no longer Python loops on the hot path.

The weak points:

- The repo still mixes experimental helpers, integrated kernels, and benchmark-only code in the same modules.
- `verl/utils/kernel/future_kl.py` is becoming a grab-bag. It needs discipline or it will turn into an unmaintainable dumping ground.
- End-to-end benchmarking is still thinner than microbenchmark coverage. We have stronger helper timing than full trainer-path timing.
- The package import path is still awkward because top-level imports drag in `ray`. That blocks simple benchmark scripts for some paths.
- The repo still tracks `.pyc` and `__pycache__` noise, which makes worktree review worse than it should be.
- There is still no serious automation for “keep the GPU busy for hours.” Right now that requires an agent following the queue instead of a persistent runner.

The main strategic risk:

- Writing more kernels without first proving an end-to-end trainer-path win will eventually waste time.
- The next kernels need to be chosen from measured PPO/FIPO bottlenecks, not from whatever math looks kernel-friendly.

## Current Priorities

These are the next queued tasks:

1. `FIPO-011`: CE split-vocabulary policy sweep
2. `FIPO-012`: Safe fused PPO vanilla-path integration

Why this order:

- CE is still measured in tens to hundreds of milliseconds, so even modest gains matter.
- The fused PPO helper already exists, but it is not safe to integrate until it preserves vanilla PPO semantics and aggregation behavior.

## Instructions For Any Agent

Read these first:

1. [AGENTS.md](/workspace/FIPO/AGENTS.md)
2. [KERNEL_RULES.md](/workspace/FIPO/KERNEL_RULES.md)
3. [.agents/kernel/README.md](/workspace/FIPO/.agents/kernel/README.md)

Then run:

```bash
python scripts/validate_kernel_queue.py
python scripts/show_kernel_queue.py
```

Claim exactly one queued task:

```bash
python scripts/claim_kernel_task.py <TASK_ID> --owner <NAME>
```

Do not touch files outside the claimed `scope_files`.

## Instructions For FIPO-011

Goal:

- Sweep CE split thresholds on a wider shape matrix and either keep `4096/16384` or replace them with a measured improvement.

Allowed scope:

- `verl/utils/kernel/kernels.py`
- `scripts/benchmark_linear_cross_entropy.py`
- `tests/test_linear_cross_entropy_kernels.py`

Required commands:

```bash
python scripts/benchmark_linear_cross_entropy.py --matrix --warmup 5 --iters 20 --dtype bfloat16
python -m unittest tests.test_linear_cross_entropy_kernels -v
```

Success condition:

- Better measured CE times on the representative matrix with parity intact.

Failure condition:

- If the sweep does not beat the current defaults consistently, keep `4096/16384`.

## Instructions For FIPO-012

Goal:

- Extend the fused PPO helper only if it can preserve vanilla PPO semantics.

Required semantic checks:

- `pg_clipfrac_lower` must still match.
- `loss_agg_mode` behavior must still match.
- Detached scalar metrics must still match.
- Do not regress the existing Future-KL path while improving vanilla PPO.

Allowed scope:

- `verl/utils/kernel/future_kl.py`
- `verl/trainer/ppo/core_algos.py`
- `tests/test_future_kl_kernels.py`

Required commands:

```bash
python scripts/benchmark_fipo_loss.py --batch-size 32 --response-len 2048 --warmup 10 --iters 30
python -m unittest discover -s tests -v
```

Success condition:

- End-to-end benchmark win with parity preserved.

Failure condition:

- If semantics drift or the end-to-end path does not win, keep the fused helper experimental.

## BF16 Rule

- BF16 is supported on this GPU.
- For sensitive RL recurrences, the current policy is: BF16 input, float32 compute.
- Do not introduce native BF16 recurrence math unless it is both numerically stable and faster in measured trainer-path benchmarks.

## Coordination Rule

- One GPU-exclusive task at a time.
- One owner per task.
- Update the task card first.
- Update [KERNEL_QUEUE.md](/workspace/FIPO/KERNEL_QUEUE.md) second.
- `queue.json` is a legacy snapshot. Keep it aligned, but do not treat it as the source of truth.

## If You Need To Keep The GPU Busy For Hours

Run tasks serially in this order:

1. `FIPO-011` matrix sweep
2. `FIPO-012` end-to-end PPO integration attempt
3. Re-profile PPO/FIPO hotspots after those two land

Do not fill idle time by inventing new kernels.
Use the next measured bottleneck from the profiler results.
