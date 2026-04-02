# Kernel Rules

This repo now has a repeatable pattern for GPU kernel work. Follow it.

## Core Rules

1. Start from a correct Torch reference.
2. Benchmark the reference with synchronized CUDA timings before changing code.
3. Keep a runtime switch: `auto | torch | triton`.
4. Preserve a non-Triton fallback for CPU, unsupported CUDA setups, and debugging.
5. Accumulate in `float32` unless there is a strong reason not to.
6. Do not enable low-precision paths by default without parity checks.
7. Prefer forward-only kernels when the optimized value is detached or has no custom backward need.
8. Add multiple autotune configs; one config is not real autotuning.
9. Benchmark representative shapes, not just one toy case.
10. Measure warm runs with `torch.cuda.Event`, and report milliseconds.

## Integration Rules

1. Isolate kernel logic in a small helper module under `verl/utils/kernel/`.
2. Keep policy code readable; call a helper instead of embedding Triton logic in trainer code.
3. Add an `impl` config knob close to the feature being optimized.
4. If model outputs may be `bf16/fp16`, upcast sensitive RL math to `float32` locally.
5. Reuse computed intermediates instead of recomputing expensive ops like `exp`.

## Validation Rules

1. Compare Triton output to the Torch reference on CUDA.
2. Check more than one shape and at least one long-sequence case.
3. Use explicit tolerances and write down which dtype they apply to.
4. If parity only holds after an upcast, document that and make the upcast explicit in code.
5. Do not switch `auto` to the Triton path on a dtype you have not validated.

## Current Pattern In This Repo

1. [future_kl.py](/workspace/FIPO/verl/utils/kernel/future_kl.py) contains the reference and Triton implementations.
2. [core_algos.py](/workspace/FIPO/verl/trainer/ppo/core_algos.py) selects the implementation with `policy_loss.future_kl_impl`.
3. [benchmark_future_kl.py](/workspace/FIPO/scripts/benchmark_future_kl.py) records before/after timings in milliseconds.
4. `compute_policy_loss_future_kl` keeps its RL math in float32 so BF16 model outputs can safely use the optimized path.

## Next Kernel Candidates

1. Broader autotune coverage for the existing linear-cross-entropy Triton kernels in `verl/utils/kernel/kernels.py`.
2. A fused metric-reduction kernel only if it beats plain torch reductions on representative PPO shapes.
3. End-to-end PPO-step profiling to confirm hotspot wins survive trainer overhead.
