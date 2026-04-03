# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contain small torch utilities
"""

import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizer

from verl.utils.device import get_device_name, get_torch_device

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


try:
    import torch_npu

    NPU_CROSS_ENTROPY_LOSS_AVAILABLE = hasattr(torch_npu, "npu_cross_entropy_loss")
except ImportError:
    NPU_CROSS_ENTROPY_LOSS_AVAILABLE = False

try:
    from verl.utils.kernel.entropy_from_logits import compute_entropy_from_logits as _compute_entropy_from_logits_kernel
except ImportError:
    _compute_entropy_from_logits_kernel = None


def gather_from_labels(data, label):
    """Gather the label from data. The value in label should be [0, vocab_size)

    Args:
        data: (..., vocab_size)
        label (torch.IntTensor) : (...,)

    Returns:

    """

    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    return output


def logprobs_from_logits(logits, labels, inplace_backward=True):
    """
    Compute per-token log-probabilities for the given labels.

    Uses a Flash-Attention–based cross-entropy (if available) for efficient backward,
    otherwise falls back to a standard log-softmax+gather approach.

    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

    Args:
        logits (Tensor): Model outputs of shape (..., vocab_size).
        labels (LongTensor): True class indices of shape matching logits[..., :-1].
        inplace_backward (bool): If True and Flash-Attn is available, perform backward in-place.

    Returns:
        Tensor: Log-probabilities of the target labels, shape logits.shape[:-1].
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels, inplace_backward=inplace_backward)
        output = output.view(*batch_dim)
    elif NPU_CROSS_ENTROPY_LOSS_AVAILABLE:
        output = logprobs_from_logits_torch_npu(logits, labels)
    elif logits.is_cuda:
        from verl.utils.kernel.logprob import compute_token_logprob

        if logits.dim() == 2:
            output = compute_token_logprob(
                logits.unsqueeze(0),
                labels.unsqueeze(0),
                impl="auto",
            ).squeeze(0)
        else:
            output = compute_token_logprob(logits, labels, impl="auto")
        if logits.dtype in {torch.float16, torch.bfloat16}:
            output = output.to(logits.dtype)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels, inplace_backward=True):
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]


def logprobs_from_logits_torch_npu(logits, labels):
    batch_dim = logits.shape[:-1]
    logits = logits.reshape(-1, logits.shape[-1])
    loss, _, _, _ = torch_npu.npu_cross_entropy_loss(logits, labels.reshape(-1), reduction="none")
    return -loss.view(*batch_dim)


def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy


def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(logits, dim=-1)
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        # Vectorized: apply log_softmax on full tensor then gather
        log_softmax_all = F.log_softmax(logits, dim=-1)
        logprobs_labels = log_softmax_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logprobs_labels


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def _entropy_from_logits_cuda(logits: torch.Tensor) -> torch.Tensor:
    if _compute_entropy_from_logits_kernel is None:
        raise RuntimeError("entropy_from_logits kernel module is unavailable")

    entropy = _compute_entropy_from_logits_kernel(logits, impl="auto")
    if logits.dtype in {torch.float16, torch.bfloat16}:
        entropy = entropy.to(logits.dtype)
    return entropy


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    if logits.is_cuda:
        return _entropy_from_logits_cuda(logits)
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def entropy_from_logits_with_chunking(logits: torch.Tensor, chunk_size: int = 2048):
    """Memory-efficient entropy calculation with chunking."""
    if logits.is_cuda:
        return _entropy_from_logits_cuda(logits)
    entropy = torch.zeros(logits.shape[0], device=logits.device)
    for i in range(0, logits.shape[0], chunk_size):
        logits_chunk = logits[i : i + chunk_size].float()
        pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
        entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(pd_chunk * logits_chunk, dim=-1)
        entropy[i : i + chunk_size] = entropy_chunk
    return entropy


def masked_sum(values, mask, axis=None):
    """Compute sum of tensor with masked values."""
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    valid_values = torch.where(mask.bool(), values, 0.0)
    return valid_values.sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """
    Whiten `values` by normalizing with mean and variance computed over `mask`.

    Args:
        values (torch.Tensor): Input tensor.
        mask (torch.Tensor): Boolean tensor of same shape, selects elements for stats.
        shift_mean (bool): If True (default), output is zero-mean;
                           if False, the original mean is re-added after scaling.

    Returns:
        torch.Tensor: Whitened tensor of same shape as `values`.
    """
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


# =============================================================================
# Fused Masked Whitening Kernel
# =============================================================================
# Computes mean + variance + whitened output using two GPU kernels:
# 1. First kernel: accumulates sum, sum_sq, count using atomic adds
# 2. Second kernel: applies whitening using pre-computed statistics
#
# This reduces data reads from 3 (original) to 2 (fused) and avoids
# materializing the centered_values intermediate tensor.
#
# Numerical stability is addressed via Bessel's correction for variance.


if HAVE_TRITON:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        ],
        key=["numel"],
    )
    @triton.jit
    def _masked_whiten_accum_kernel(
        values_ptr,
        mask_ptr,
        sum_ptr,
        sum_sq_ptr,
        count_ptr,
        numel,
        BLOCK_SIZE: tl.constexpr,
    ):
        """First pass: accumulate sum and sum_sq using atomic adds.

        This is the first pass that computes:
        - sum of values * mask
        - sum of values^2 * mask
        - count of mask

        Note: Uses sum_sq/n - mean^2 formula for variance. This is numerically
        equivalent to mean((x - mean)^2) but can suffer from catastrophic
        cancellation when variance is near zero. For near-zero variance inputs,
        prefer using the torch implementation.
        """
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_flags = offsets < numel

        # Load values and mask
        values = tl.load(values_ptr + offsets, mask=mask_flags, other=0.0).to(tl.float32)
        mask_val = tl.load(mask_ptr + offsets, mask=mask_flags, other=0.0).to(tl.float32)

        # Accumulate weighted values and squared values
        weighted_values = values * mask_val
        weighted_sq = values * values * mask_val

        # Atomic add to accumulate
        tl.atomic_add(sum_ptr, tl.sum(weighted_values, axis=0))
        tl.atomic_add(sum_sq_ptr, tl.sum(weighted_sq, axis=0))
        tl.atomic_add(count_ptr, tl.sum(mask_val, axis=0))

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        ],
        key=["numel"],
    )
    @triton.jit
    def _masked_whiten_apply_kernel(
        values_ptr,
        output_ptr,
        numel,
        mean: tl.float32,
        var: tl.float32,
        shift_mean: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Second pass: apply whitening using pre-computed mean and variance.

        This is the second pass that reads data and applies:
        whitened = (values - mean) * rsqrt(var + eps)
        if not shift_mean:
            whitened += mean
        """
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_flags = offsets < numel

        # Load values (mask is not needed for apply phase)
        values = tl.load(values_ptr + offsets, mask=mask_flags, other=0.0).to(tl.float32)

        # Apply whitening
        whitened = (values - mean) * tl.rsqrt(var + 1e-8)
        if shift_mean == 0:
            whitened = whitened + mean

        # Store result (mask not applied to output - follows original masked_whiten behavior)
        tl.store(output_ptr + offsets, whitened, mask=mask_flags)


def _masked_whiten_torch_impl(values, mask, shift_mean=True):
    """Torch reference implementation for fused masked whitening.

    Uses the original two-pass algorithm for numerical stability:
    1. First pass: compute mean via masked_mean
    2. Second pass: compute variance via centered values
    3. Apply whitening

    This matches the original masked_whiten behavior exactly.
    """
    # Compute statistics using the same algorithm as the original
    mask_sum = mask.sum()
    if mask_sum == 0:
        raise ValueError("At least one element in the mask has to be 1.")
    if mask_sum == 1:
        raise ValueError("The sum of the mask is one, which can cause a division by zero.")

    # First pass: masked_mean
    mean = (values * mask).sum() / (mask_sum + 1e-8)

    # Second pass: masked_var using centered values (numerically stable)
    centered_values = values - mean
    variance = (centered_values * centered_values * mask).sum() / (mask_sum + 1e-8)

    # Bessel's correction
    bessel_correction = mask_sum / (mask_sum - 1)
    variance = variance * bessel_correction

    # Apply whitening
    whitened = (values - mean) * torch.rsqrt(variance + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


def masked_whiten_triton(values, mask, shift_mean=True):
    """Fused masked whitening using Triton kernels.

    Two-kernel approach:
    1. First kernel: accumulates sum, sum_sq, count using atomic adds
    2. Second kernel: applies whitening using pre-computed statistics

    This reduces data reads from 3 (original) to 2 (fused) and avoids
    materializing the centered_values intermediate tensor.

    Args:
        values (torch.Tensor): Input tensor, must be float32 CUDA tensor.
        mask (torch.Tensor): Boolean or numeric mask of same shape, must be float32 CUDA tensor.
        shift_mean (bool): If True (default), output is zero-mean;
                           if False, the original mean is re-added after scaling.

    Returns:
        torch.Tensor: Whitened tensor of same shape as `values`.
    """
    if not HAVE_TRITON:
        raise RuntimeError("Triton is not installed.")
    if not values.is_cuda or not mask.is_cuda:
        raise RuntimeError("Triton masked_whiten requires CUDA tensors.")
    if values.dtype != torch.float32 or mask.dtype != torch.float32:
        raise RuntimeError("Triton masked_whiten currently supports float32 inputs only.")

    values_flat = values.reshape(-1)
    mask_flat = mask.reshape(-1)
    numel = values_flat.numel()

    # Allocate accumulation buffers
    sum_out = torch.zeros((), device=values.device, dtype=torch.float32)
    sum_sq_out = torch.zeros((), device=values.device, dtype=torch.float32)
    count_out = torch.zeros((), device=values.device, dtype=torch.float32)

    # First kernel: accumulate
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _masked_whiten_accum_kernel[grid](
        values_flat,
        mask_flat,
        sum_out,
        sum_sq_out,
        count_out,
        numel,
    )

    # Compute final statistics on host
    mask_sum = count_out.item()
    if mask_sum == 0:
        raise ValueError("At least one element in the mask has to be 1.")

    weighted_sum = sum_out.item()
    weighted_sq_sum = sum_sq_out.item()

    mean = weighted_sum / (mask_sum + 1e-8)
    var = (weighted_sq_sum / (mask_sum + 1e-8)) - (mean * mean)
    # Bessel's correction
    var = var * (mask_sum / (mask_sum - 1)) if mask_sum > 1 else (var + 1e-8)

    # Allocate output and run second kernel
    output = torch.empty_like(values_flat)
    _masked_whiten_apply_kernel[grid](
        values_flat,
        output,
        numel,
        mean,
        var,
        int(shift_mean),
    )

    return output.reshape_as(values)


def get_response_mask(response_id: torch.Tensor, eos_token: int | list[int] = 2, dtype=torch.int64):
    """
    end of sentence token can be int or list: 1 or [1, 2]
    e.g.
    response_id = torch.tensor([[20, 10, 34, 1, 0, 0, 0],
                                [78, 0, 76, 2, 1, 0, 0],
                                [23, 98, 1, 0, 0, 0, 0],
                                [33, 3, 98, 45, 1, 0, 0]])
    #eos_token=1
    response_mask:  tensor([[1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0]])
    #eos_token=[1,2]
    response_mask:  tensor([[1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0]])
    """
    eos_mask = torch.isin(response_id, torch.tensor(eos_token, device=response_id.device)).int()
    return (eos_mask.cumsum(dim=1) - eos_mask).eq(0).to(dtype)


def compute_grad_norm(model: nn.Module):
    # Vectorized gradient norm computation - accumulates on GPU without loop
    device = next(model.parameters()).device
    grad_sq_list = []
    for param in model.parameters():
        if param.grad is not None:
            grad_sq_list.append(torch.sum(torch.square(param.grad.detach())))
    if grad_sq_list:
        total_grad_square = torch.stack(grad_sq_list).sum()
    else:
        total_grad_square = torch.tensor(0.0, device=device)
    return total_grad_square.item()


def broadcast_dict_tensor(tensors: dict[str, torch.Tensor] | TensorDict, src, group):
    """Broadcast a dictionary of tensors from src to all processes.

    Uses async operations to issue all broadcasts without waiting, allowing
    the communication to be overlapped by the backend.

    Args:
        tensors: Dictionary of tensors to broadcast
        src: Source rank
        group: Process group

    Returns:
        None (in-place operation)
    """
    handles = []
    for key in tensors.sorted_keys:
        handle = torch.distributed.broadcast(tensors[key], src=src, group=group, async_op=True)
        handles.append(handle)

    # Wait for all broadcasts to complete
    for handle in handles:
        handle.wait()


def allgather_dict_tensors(tensors: dict[str, torch.Tensor] | TensorDict, size, group, dim=0):
    """Gather tensors from all processes and concatenate along specified dimension.

    Optimized version that uses a single all-gather by packing all tensors into
    a contiguous buffer when possible, or async operations for general tensors.

    Args:
        tensors: Dictionary of tensors to gather
        size: World size (number of processes)
        group: Process group
        dim: Dimension along which to concatenate results

    Returns:
        Dictionary with gathered and concatenated tensors
    """
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False

    sorted_keys = sorted(tensors_as_dict.keys())
    vals = [tensors_as_dict[k] for k in sorted_keys]

    # Try to use single all-gather by stacking tensors
    # This works when all tensors have the same shape
    try:
        # Stack all tensors: shape (num_keys, *val.shape)
        stacked = torch.stack(vals, dim=0)
        num_keys = len(vals)

        # All-gather the stacked tensor
        output_stacked = [torch.empty_like(stacked) for _ in range(size)]
        torch.distributed.all_gather(output_stacked, stacked, group=group, async_op=False)

        # Concatenate along dim=0 to get (size * num_keys, *val.shape)
        combined = torch.cat(output_stacked, dim=0)

        # Reshape to (size, num_keys, *val.shape)
        combined = combined.reshape(size, num_keys, *vals[0].shape[1:])

        # Split back into dict
        output = {}
        for i, key in enumerate(sorted_keys):
            # Select all ranks' contributions for this key: (size, *val.shape)
            key_data = combined[:, i]
            output[key] = torch.cat([key_data[j] for j in range(size)], dim=dim)
    except (RuntimeError, ValueError):
        # Fall back to async per-key all-gather if shapes differ
        output = {}
        handles = []
        for key in sorted_keys:
            val = tensors_as_dict[key]
            output[key] = [torch.empty_like(val) for _ in range(size)]
            handle = torch.distributed.all_gather(output[key], val, group=group, async_op=True)
            handles.append((key, handle))

        # Wait for all to complete
        for key, handle in handles:
            handle.wait()

        # Concatenate along dim
        for key in sorted_keys:
            output[key] = torch.cat(output[key], dim=dim)

    if is_tensor_dict:
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    return output


def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> list[TensorDict]:
    assert tensors.batch_size[0] % batch_size == 0, (
        f"input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}"
    )
    return tensors.split(batch_size)


def pad_2d_list_to_length(response, pad_token_id, max_length=None):
    """
    pad a 2D list (e.g. responses, logprobs) to a 2D tensor.
    """
    response_length = max(len(sub_list) for sub_list in response)
    target_length = max_length if max_length is not None and max_length > response_length else response_length

    # Vectorized padding using numpy
    import numpy as np
    num_rows = len(response)
    padded = np.full((num_rows, target_length), pad_token_id, dtype=np.int64)
    for i, sub_list in enumerate(response):
        padded[i, :len(sub_list)] = sub_list
    return torch.from_numpy(padded)


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    # (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)


def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad=True,
    truncation="error",
):
    """Process tokenizer outputs to consistent shapes via padding/truncation.

    Args:
        input_ids: Token indices [batch_size, seq_len]
        attention_mask: Mask [batch_size, seq_len]
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: "left", "right", "middle" or "error"

    Returns:
        (input_ids, attention_mask) padded/truncated to max_length
    """
    assert truncation in ["left", "right", "middle", "error"]
    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > max_length:
        if truncation == "left":
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            input_ids = torch.cat([input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1)
            attention_mask = torch.cat([attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1)
        elif truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}")

    return input_ids, attention_mask


def tokenize_and_postprocess_data(
    prompt: str, tokenizer: PreTrainedTokenizer, max_length: int, pad_token_id: int, left_pad=True, truncation="error"
):
    """Tokenize text and process outputs to consistent tensor shapes.

    Args:
        prompt: Input text to tokenize
        tokenizer: HuggingFace tokenizer instance
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: Truncation strategy ("left"/"right"/"error")

    Returns:
        Tuple of (input_ids, attention_mask) from postprocess_data
    """
    input_data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]

    return postprocess_data(input_ids, attention_mask, max_length, pad_token_id, left_pad, truncation)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token.

    Vectorized implementation using torch operations.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[List[int]]): contains the rmpad token ids per query.
    """
    bs, seq_len = input_ids.shape

    # Number of valid tokens per sequence
    valid_counts = attention_mask.sum(dim=1)  # [bs]

    # Maximum number of valid tokens in the batch
    max_valid = valid_counts.max().item()

    # For each sequence, the starting index is seq_len - valid_count[i]
    # This is because valid tokens are right-aligned (last mask.sum() elements)
    start_indices = seq_len - valid_counts  # [bs]

    # Create indices [0, 1, 2, ..., max_valid-1]
    indices = torch.arange(max_valid, device=input_ids.device)  # [max_valid]

    # For each sequence i, positions[i] = start_indices[i] + indices
    # positions[i][j] = start_indices[i] + j
    positions = start_indices.unsqueeze(1) + indices  # [bs, max_valid]

    # Create mask for valid positions: indices < valid_counts
    # For position j in sequence i, it's valid if j < valid_counts[i]
    # This is because valid positions are start_indices[i] + j for j in [0, valid_counts[i])
    mask = indices.unsqueeze(0) < valid_counts.unsqueeze(1)  # [bs, max_valid]

    # Clamp positions to valid range to avoid index out of bounds in gather
    positions_clamped = positions.clamp(max=seq_len - 1)

    # Gather tokens: for each (i, j), get input_ids[i][positions_clamped[i][j]]
    gathered = input_ids.gather(dim=1, index=positions_clamped)  # [bs, max_valid]

    # Zero out invalid positions
    gathered = gathered.masked_fill(~mask, 0)

    # Convert to list of lists with only valid tokens
    # Batch the .item() calls - get all valid lengths at once
    valid_lens = valid_counts.tolist()  # Single GPU sync
    no_padding_batch = []
    for i in range(bs):
        no_padding_batch.append(gathered[i][:valid_lens[i]].cpu().numpy().tolist())
    return no_padding_batch


def log_probs_from_logits_response(input_ids, logits, response_length):
    """Compute the response log_probs from full logits. Note that logits = model(input_ids)

    Args:
        input_ids: [batch_size, seqlen]
        logits: [batch_size, seqlen, vocab_size]

    Returns:
        response_log_prob:
    """
    response_logits = logits[:, -response_length - 1 : -1]
    response = input_ids[:, -response_length:]
    response_log_prob = logprobs_from_logits(logits=response_logits, labels=response)
    return response_log_prob


def log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """Compute the log_probs from logits with rmpad logits and pad input. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size]
        response_length: int
    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    return output


def log_probs_from_logits_all_rmpad(input_ids_rmpad, logits_rmpad, indices, batch_size, seqlen, response_length):
    """Compute the log_probs from logits with rmpad input_ids and logits. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size

    Args:
        input_ids_rmpad: [1, total_nnz]
        logits_rmpad: [total_nnz, vocab_size]
        indices: [total_nnz]
        batch_size: int
        seqlen: int
        response_length: int
    """
    from flash_attn.bert_padding import pad_input

    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # transpose back to [total_nnz, 1]
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    return output


def post_process_logits(input_ids, logits, temperature, top_k, top_p):
    if temperature != 1.0:
        logits = logits.div_(temperature)  # inplace operation to avoid OOM
    # TODO: add them back
    # if top_k is not None and top_k > 0:
    #     logits = TopKLogitsWarper(top_k=top_k)(input_ids, logits)
    # if top_p is not None and top_p < 1.0 and top_p > 0.0:
    #     logits = TopPLogitsWarper(top_p=top_p)(input_ids, logits)
    return logits


"""
Optimizer related
"""


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * (float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    """
    Create a constant LR schedule with a linear warmup phase.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_warmup_steps (int): Number of steps to ramp up the LR from 0 to initial value.
        last_epoch (int, optional): The index of the last epoch when resuming training. Defaults to -1.

    Returns:
        LambdaLR: Scheduler that increases LR linearly during warmup, then holds it constant.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def get_wsd_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    stable_ratio: float = 0.9,
):
    """
    Create a Warmup-Stable-Decay learning rate scheduler.

    The schedule follows three phases:
    1. Warmup: Learning rate increases linearly from 0 to the initial LR
    2. Stable: Learning rate remains constant at the initial LR
    3. Decay: Learning rate decreases following a cosine curve to min_lr_ratio * initial LR

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum learning rate ratio w.r.t the initial learning rate.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule during decay phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        stable_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The ratio of non-warmup steps that should maintain a constant learning rate.
            Set to 0.0 to behave exactly like cosine schedule.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    remaining_steps = max(0, num_training_steps - num_warmup_steps)
    num_stable_steps = int(remaining_steps * stable_ratio)
    num_decay_steps = remaining_steps - num_stable_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        if current_step < num_training_steps:
            progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
            value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
            return (1.0 - min_lr_ratio) * value + min_lr_ratio
        return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@contextmanager
def check_device_is_available():
    """
    Some modules must be imported after CUDA is initialized. Such as sglang's sharding manager.

    This context manager checks if CUDA is available and raises an error if it is not.
    """
    if not get_torch_device().is_available():
        raise RuntimeError("Device {} must be initialized before importing this module.".format(get_device_name()))

    yield


def distributed_mean_max_min_std(local_tensor, compute_max=True, compute_min=True, compute_std=True):
    """Compute distributed statistics across all processes.

    This is a fused version that reduces communication overhead by:
    - Using a single all-reduce for SUM operations (sum, num, sum_of_squares packed)
    - Using async all-reduce for max/min to overlap with SUM operation

    Args:
        local_tensor: Tensor containing local values
        compute_max: Include maximum value calculation
        compute_min: Include minimum value calculation
        compute_std: Include standard deviation calculation

    Returns:
        Tuple containing (mean, max, min, std) in this order. None for disabled metrics.
    """
    # Compute local statistics
    local_sum = torch.sum(local_tensor)
    local_num = torch.tensor(torch.numel(local_tensor), device=local_tensor.device, dtype=torch.float32)
    local_sum_of_squares = torch.sum(torch.pow(local_tensor, 2))

    # Launch async all-reduce for max/min (they must use separate ops)
    handles = []
    if compute_max:
        local_max = torch.max(local_tensor)
        handle = torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX, async_op=True)
        handles.append(('max', local_max, handle))
    else:
        local_max = None

    if compute_min:
        local_min = torch.min(local_tensor)
        handle = torch.distributed.all_reduce(local_min, op=torch.distributed.ReduceOp.MIN, async_op=True)
        handles.append(('min', local_min, handle))
    else:
        local_min = None

    # Pack sum, num, sum_of_squares into one tensor and reduce together
    packed_stats = torch.stack([local_sum, local_num, local_sum_of_squares])
    torch.distributed.all_reduce(packed_stats, op=torch.distributed.ReduceOp.SUM)

    global_sum, global_num, global_sum_of_squares = packed_stats
    global_mean = global_sum / global_num

    # Wait for max/min operations to complete
    for name, tensor, handle in handles:
        handle.wait()
        if name == 'max':
            local_max = tensor
        else:
            local_min = tensor

    # Compute std from sum_of_squares: Var = E[X^2] - E[X]^2
    # Using Bessel's correction: std = sqrt(Var * n / (n-1))
    if compute_std:
        global_e_x2 = global_sum_of_squares / global_num
        global_var = global_e_x2 - global_mean.pow(2)
        global_std = torch.sqrt(global_var * global_num / (global_num - 1))
    else:
        global_std = None

    return global_mean, local_max, local_min, global_std


def distributed_masked_mean(local_tensor, local_mask):
    """Compute global mean of non-masked elements across distributed processes.

    Args:
        local_tensor (torch.Tensor): Input tensor with local values
        local_mask (torch.Tensor): Binary mask (1=valid, 0=ignore) matching local_tensor shape

    Returns:
        torch.Tensor: Global mean of all valid elements across processes
    """
    local_tensor = local_tensor * local_mask

    local_sum = torch.sum(local_tensor)
    local_num = torch.sum(local_mask)

    torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_num, op=torch.distributed.ReduceOp.SUM)

    global_mean = local_sum / local_num
    return global_mean
