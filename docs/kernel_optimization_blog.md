# Kernel Optimization Tutorial: How We Turned FIPO Hot Paths Into Fused Kernels

This repo started as a research codebase for Future-KL influenced policy optimization. The work I focused on here was different: it was an optimization fork for the training stack, aimed at finding real hot paths and replacing repeated Python-side tensor work with fused CUDA/Triton kernels.

The rule was simple: if a kernel did not beat the torch path on representative shapes, it stayed as a reference implementation. If it did win, I added a dispatcher, parity tests, and an integration path in the trainer.

## The workflow

1. Start from the queue.
   The repo already has a kernel task queue, so I used it as the source of truth. Each task has a scope, a benchmark command, and a test command. That keeps the work bounded and prevents overlapping edits.

2. Preserve the torch reference.
   Every fused path kept a torch implementation beside it. That mattered for two reasons: correctness checks and a fallback when the CUDA path was not a win.

3. Measure before integrating.
   I used synchronized CUDA timings and compared the kernel against the existing baseline on representative shapes. The gate was not "does it work"; the gate was "does it actually win".

4. Add tests before closing the task.
   Each task got parity coverage, edge cases, and integration tests where the kernel touched trainer code.

## What changed

The biggest wins came from removing repeated work and reducing intermediate tensors.

- `Future-KL` stayed the main end-to-end hotspot, so the original fused reverse-scan kernel remained the backbone of the FIPO loss path.
- `returns + whitening` was fused into a single path for REINFORCE++ outcome advantages, which removed extra passes over the same `(B, T)` tensor.
- `logprob + entropy` was fused so the actor path could read logits once instead of computing log-probabilities and entropy separately.
- `logprobs_from_logits` and `entropy_from_logits` were generalized to use the streaming kernels for CUDA fallback paths, including remove-padding style `(N, V)` inputs.
- `geo_mean` policy loss was wired to the existing GMPO helper so the trainer can use the fused path directly.

The representative benchmark numbers from this fork were:

- `returns + whitening`: `1.153 ms -> 0.352 ms` on `32x2048 float32`
- `logprob + entropy`: `10.776 ms -> 4.750 ms` on padded `16x2048x8192 float32`
- `logprobs_from_logits` dispatch: `19.080 ms -> 2.563 ms` on padded `16x2048x8192 float32`
- `entropy_from_logits` dispatch: `42.334 ms -> 5.600 ms` on padded `16x2048x8192 float32`
- `geo_mean` / GMPO: `0.339 ms -> 0.156 ms` on `32x2048`

## Why the gains showed up

Two patterns kept repeating:

- Repeated work was the first thing to remove.
  The biggest savings came from collapsing multiple passes into one pass over the same data, especially where the old path did separate reductions, gathers, or softmax-related computations.

- Moving less data mattered as much as raw math.
  Several of the wins came from avoiding intermediate `(B, T, V)` tensors or from keeping the computation on GPU instead of materializing filtered arrays and then reducing them.

That is why the largest changes were not just "faster math". They were structural changes to the execution path.

## How to apply the pattern

If you want to repeat this process on another hotspot:

1. Identify a real hotspot with a benchmark, not intuition.
2. Write or preserve a torch reference first.
3. Build the Triton or fused implementation.
4. Benchmark the exact representative shape you care about.
5. Add tests for correctness, dtype handling, and integration.
6. Only flip the trainer default if the fused path wins.

That sequence sounds boring, but it is what kept the fork honest. Most "optimizations" are just moving cost around. The useful ones are the ones that remove work altogether.
