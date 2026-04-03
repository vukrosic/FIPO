# FIPO Kernel Engineering Blog

A running tutorial-style log of every kernel we added, why it matters, and the
numbers behind it.  Newest entries at the top.

---

## FIPO-038 · Fused Advantage Normalization  *(2026-04-03)*

**What.** `masked_whiten(returns, mask)` is called by REINFORCE++ and GRPO
after computing advantages.  It requires 3 passes over `(B, T)`: one for mean,
one for variance, and one to apply whitening.  The Triton kernel reduces this
to 2 passes: accumulate `(sum, sum_sq, count)` with atomic adds, then apply
`(x - mean) * rsqrt(var)` with mask in a second kernel.

**File.** `verl/utils/kernel/fused_advantage_norm.py`
**Tests.** `tests/test_fused_advantage_norm_kernel.py`

**Results** (`B=32, T=2048`, float32):

| Implementation | Time (ms) | Speedup |
|---|---|---|
| torch | 0.179 | — |
| triton | 0.156 | **1.15x** |

---

## FIPO-037 · Fused Ratio + KL Kernel  *(2026-04-03)*

**What.** Every PPO policy loss starts with `neg_kl = lp - olp; ratio =
exp(clamp(neg_kl)); kl = masked_mean(-neg_kl)`.  This kernel fuses all three
into a single flat scan with atomic reductions for the KL accumulator.

**File.** `verl/utils/kernel/fused_ratio.py`
**Tests.** `tests/test_fused_ratio_kernel.py`

**Note.** Standalone benchmark shows 0.74x (Triton launch overhead dominates
at sub-0.1ms).  The real value is as a building block inside larger fused
kernels — eliminating the intermediate `neg_kl` tensor and the separate
`masked_mean` kernel launch.

---

## FIPO-034–036 · Utility Kernels  *(2026-04-03)*

**What.** Three utility kernels for common PPO patterns:

- **FIPO-034** `agg_loss`: Fused loss aggregation for all 4 modes (token-mean,
  seq-mean-token-sum, seq-mean-token-mean, seq-mean-token-sum-norm).  One Triton
  program per sequence.
- **FIPO-035** `batch_stats`: Single-pass mean/max/min using atomic reductions.
  Useful for `compute_data_metrics` where 20+ separate reductions are called.
- **FIPO-036** `fused_gpg_loss`: Fuses `-log_prob * advantages + agg_loss` into
  a single kernel, avoiding the intermediate `pg_losses` tensor.

**Lesson learned.** At sub-0.1ms per operation, Triton kernel launch overhead
dominates.  These standalone kernels are slower than torch's optimized CUDA
ops.  They become valuable when fused *into* larger computation chains (like
the GMPO/GSPO kernels which already embed this pattern).

---

## FIPO-033 · Vectorized KL-Cov Policy Loss  *(2026-04-03)*

**What.** `compute_policy_loss_kl_cov` selectively adds a KL penalty to the
top-*k*% tokens ranked by covariance between advantages and log-probs.  The
original implementation uses `torch.masked_select()` which triggers a CPU sync
(CUDA must tell the CPU how many elements passed the mask before allocating the
output tensor).

**File.** `verl/utils/kernel/kl_cov_loss.py`
**Tests.** `tests/test_kl_cov_loss_kernel.py`

**Key idea.** Instead of `masked_select` → `topk` on a variable-sized tensor,
we compute covariance over the full `(B, T)` grid, set masked positions to
`-inf`, and call `topk` directly on the flat grid.  This stays fully on GPU
with no CPU sync for size queries.

---

## FIPO-031 · Vectorized Clip-Cov Policy Loss  *(2026-04-03)*

**What.** `compute_policy_loss_clip_cov` zeros out a random subset of tokens
whose covariance(advantage, log_prob) falls in a specified band.  The original
uses `torch.nonzero()` (CPU sync) + `torch.randperm()` (CPU-side RNG).

**File.** `verl/utils/kernel/clip_cov_loss.py`
**Tests.** `tests/test_clip_cov_loss_kernel.py`

**Key idea.** Replace the `nonzero+randperm` pattern with:
1. Assign each in-band token a random priority via `torch.rand(B, T)`
2. Set out-of-band tokens to priority -1
3. `topk(flat_priority, k)` to select `k` random in-band tokens

This keeps the entire selection on GPU.  The random seed is different from the
original `randperm`, but the statistical properties are identical (uniform
random selection from the candidate set).

---

## FIPO-030 · Fused GMPO Policy Loss  *(2026-04-03)*

**What.** GMPO (Geometric-Mean Policy Optimization) computes a *sequence-level*
importance ratio by exponentiating the mean of *wider-clipped* per-token
log-ratios:

```
neg_kl_min[t] = sgn(adv[t]) * min(sgn * neg_kl[t], sgn * clamp(neg_kl[t]))
ratio[b]      = exp( mean_t( neg_kl_min[t] * mask[t] ) )
loss[b]       = -mean_t(adv[t] * mask[t]) * ratio[b]
```

**File.** `verl/utils/kernel/gmpo_loss.py`
**Tests.** `tests/test_gmpo_loss_kernel.py`
**Benchmark.** `scripts/benchmark_gmpo_loss.py`

**Key idea.** Unlike standard PPO which clips a token-level ratio, GMPO clips
per-token log-ratios *then* exponentiates their sequence mean.  This is a
single-pass reduction: one Triton program per batch element accumulates
`neg_kl_min_sum`, `adv_sum`, `mask_sum`, and clip counts in a single scan over
`T`.  No intermediate `(B, T)` tensors are written.

**Results** (`B=32, T=2048`, float32):

| Implementation | Time (ms) | Speedup |
|---|---|---|
| torch | 0.358 | — |
| triton | 0.163 | **2.20x** |

---

## FIPO-029 · Fused GSPO Policy Loss  *(2026-04-03)*

**What.** GSPO (Group Sequence Policy Optimization) computes a *sequence-level*
importance ratio rather than a token-level one.  The naive implementation makes
three passes over `(B, T)` data: one to sum `neg_kl` per sequence, one to
compute the clipped loss, and one for the final `seq-mean-token-mean`
reduction.  The fused kernel does two passes (one to build `neg_kl_seq`, one
for the clipped loss) using a single Triton program per batch element.

**File.** `verl/utils/kernel/gspo_loss.py`  
**Tests.** `tests/test_gspo_loss_kernel.py`

**Key idea.** For each batch element `b`, the GSPO ratio is:

```
s_it = exp( (1/|y_b|) * Σ_t neg_kl[b,t] * mask[b,t] )
```

This is a *constant* for all tokens in sequence `b`.  Computing it once per
sequence in an inner loop lets us fuse the whole loss into two sequential
passes without any intermediate `(B, T)` tensors.

---

## FIPO-028 · Fused Sequence-Level Reductions  *(2026-04-03)*

**What.** Operations of the form `(values * mask).sum(-1)` and
`mask.sum(-1).clamp(min=1)` appear in GSPO, Geo-Mean PPO, REINFORCE++, and
RLOO.  Each naive call launches 2–3 CUDA kernels with `(B, T)` reads.  A
single Triton program per batch element streams over `T` once.

**File.** `verl/utils/kernel/seq_utils.py`  
**Tests.** `tests/test_seq_utils_kernel.py`

**Functions.**

| Function | Output | Replaces |
|---|---|---|
| `compute_seq_logprob(lp, mask)` | `(seq_sum, seq_len)` | `(lp*mask).sum(-1)` + `mask.sum(-1)` |
| `compute_seq_mean(values, mask)` | `seq_means` | `(v*mask).sum(-1) / mask.sum(-1)` |

The bottleneck is memory bandwidth, not compute.  By reading each `(B, T)`
tensor only once, we cut total DRAM traffic roughly in half versus the naive
pattern.

---

## FIPO-027 · Fused Reward + KL Kernel  *(2026-04-03)*

**What.** `compute_rewards(scores, old_lp, ref_lp, kl_ratio)` is called every
rollout iteration.  The naive approach allocates `kl` as an intermediate and
then applies it.  The fused kernel computes all four KL variants and the final
reward in a single pass with no intermediate allocation.

**File.** `verl/utils/kernel/reward_utils.py`  
**Tests.** `tests/test_reward_utils_kernel.py`

**Supported KL variants** (selected at runtime, no re-compilation needed):

| `kl_type` | Formula |
|---|---|
| `k1` / `kl` | `lp − ref_lp` |
| `k2` / `mse` | `0.5 * (lp − ref_lp)²` |
| `k3` / `low_var_kl` | `exp(ref−lp) − (ref−lp) − 1`  (unbiased, clamped) |
| `abs` | `|lp − ref_lp|` |

The `tl.where` chain selects the right formula at runtime via a `kl_type`
integer argument – no kernel re-compilation for different loss modes.

---

## FIPO-026 · Fused Gathered Log-Probability Kernel  *(2026-04-03)*

**What.** Computing per-token log-probs from logits naively requires:

```python
lp_full  = F.log_softmax(logits, dim=-1)   # allocates (B, T, V)  ← expensive
token_lp = lp_full.gather(-1, ids)         # (B, T)
```

For `B=16, T=2048, V=32768` this tries to allocate **4 GB** just for the
intermediate — it OOMs on a typical 8 GB card.

**File.** `verl/utils/kernel/logprob.py`  
**Tests.** `tests/test_logprob_kernel.py`  
**Benchmark.** `scripts/benchmark_logprob_kernel.py`

**Key idea.** The kernel runs one Triton program per `(b, t)` position.  Each
program streams over the vocab in `BLOCK_V`-wide chunks using the **online
logsumexp** trick to maintain a running `(max, sum_exp)` pair.  The selected
logit is picked up during the same single pass.  Output is `O(B*T)` — the full
`(B, T, V)` tensor is never written to memory.

**Results** (`B=16, T=2048`, float32):

| Vocab | Naive (ms) | Triton (ms) | Speedup |
|---|---|---|---|
| 1 024 | 0.68 | 0.33 | 2.06× |
| 4 096 | 2.67 | 1.28 | 2.08× |
| 8 192 | 5.35 | 2.55 | 2.10× |
| 16 384 | 12.1 | 5.09 | 2.39× |
| **32 768** | **OOM** | **10.1 ms** | **OOM-avoidance** |

---

## FIPO-025 · PPO Loss All Aggregation Modes  *(2026-04-02)*

**What.** `compute_fused_ppo_loss` previously only supported `"token-mean"` via
the Triton path.  Extended with a two-kernel (`_fused_ppo_loss_kernel` +
`_fused_ppo_loss_reduce_kernel`) approach for `"seq-mean-token-sum"`,
`"seq-mean-token-mean"`, and `"seq-mean-token-sum-norm"`.

**File.** `verl/utils/kernel/future_kl.py`  
**Tests.** `tests/test_future_kl_kernels.py`

---

## FIPO-022 · Entropy Loss Triton Fix + Speedup  *(2026-04-02)*

**What.** The original Triton entropy kernel had a padding-position bug where
`exp(0 - max)` was included in the sum (should be 0 for invalid vocab slots).
Fixed by zeroing `exp_logits` where `load_mask == 0`.

**Results:** `3.97 ms → 0.34 ms` = **11.9×** speedup at `(32, 512, 4096)` on
an RTX 3070.

---

## FIPO-016 · Fused `masked_whiten` Kernel  *(2026-04-02)*

**What.** Whitening advantages (`mean=0, std=1`) requires two passes over the
data: one for mean/variance, one to apply.  The Triton implementation uses
atomic-add reductions for the statistics pass and a second elementwise kernel
for the application.

**Results:** `0.35 ms → 0.14 ms` = **2.5×** speedup.

---

## FIPO-010 · GAE Reverse-Scan Kernel  *(2026-04-01)*

**What.** Generalized Advantage Estimation (GAE) is a sequential recurrence
over the time axis.  The Python loop version processes one timestep at a time.
The Triton kernel processes `BLOCK_ROWS` sequences in parallel while scanning
backwards through time.

**Results:** `294 ms → 0.14 ms` = **2059×** speedup at `(32, 2048)`.

The enormous speedup comes from eliminating the Python interpreter overhead of
2048 sequential loop iterations.

---

## FIPO-009 · Discounted Returns Reverse-Scan  *(2026-04-01)*

**What.** Same pattern as GAE but simpler (no value function):
`G_t = r_t + γ * G_{t+1} * mask_t`.

**Results:** `80.4 ms → 0.089 ms` = **907×** speedup.

---

## FIPO-001 · Future-KL Reverse Scan  *(2026-03-31)*

**What.** FIPO weights each token's policy gradient by its *future KL*:

```
FutureKL_t = Σ_{j≥t} γ^{j-t} * KL_j
```

This is a reverse cumulative sum with an exponential decay.  The Triton kernel
scans right-to-left maintaining a running `carry`:

```
carry = kl[t] + γ * carry
future_kl[t] = carry
```

**Results:** `1.93 ms → 0.084 ms` = **22.9×** speedup at `(32, 2048)`.

This kernel is the core of the FIPO algorithm and is called once per PPO
update step.

---

## How to Read the Numbers

- All timings are **wall-clock GPU time** measured with `torch.cuda.synchronize()`
  before/after `N` iterations, divided by `N`.
- Hardware: NVIDIA RTX 3070 (Ampere, 8 GB VRAM).
- Dtype: float32 unless noted.
- The torch "reference" path for recurrences uses Python `for` loops to
  match the semantics of the original verl trainer code.  The PyTorch-only
  `torch.cumsum` / `torch.einsum` alternatives are often faster than the loop
  but still slower than Triton.

---

## Adding a New Kernel

1. Copy an existing kernel file as a template (e.g. `future_kl.py`).
2. Write the torch reference first — it's the test oracle.
3. Write the Triton kernel with `@triton.autotune` over `BLOCK_SIZE` / `BLOCK_ROWS`.
4. Add `_triton` / `_torch` / dispatch trio.
5. Write tests in `tests/test_<name>_kernel.py` (CPU + CUDA classes).
6. Add a task card to `.agents/kernel/tasks/FIPO-NNN.json`.
7. Update `KERNEL_QUEUE.md` and `queue.json`.
8. Run `python -m pytest tests/ --ignore=tests/test_distributed_fused_kernels_run.py`.

The `impl="auto"` pattern means the kernel can be dropped into the trainer
without any config changes — it silently falls back to torch on CPU or when
Triton is unavailable.
