# FIPO Kernel Engineering Blog

A running tutorial-style log of every kernel we added, why it matters, and the
numbers behind it.  Newest entries at the top.

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
