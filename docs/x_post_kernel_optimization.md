I found optimizations to `FIPO`

- the biggest gains came from removing repeated work around logits, entropy, and advantage weighting, then using fused kernels to combine multiple operations into one pass
- a lot of speed came from moving less data around and keeping CUDA fallback paths on GPU instead of materializing extra intermediates

- 3.28x speedup on returns + whitening on `32x2048 float32`
- 2.27x speedup on logprob + entropy on padded `16x2048x8192 float32`
- 7.45x speedup on `logprobs_from_logits` dispatch on padded `16x2048x8192 float32`
- 7.56x speedup on `entropy_from_logits` dispatch on padded `16x2048x8192 float32`
- 2.17x speedup on `geo_mean` / GMPO on `32x2048`
