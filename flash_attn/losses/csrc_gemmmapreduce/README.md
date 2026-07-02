# GeMMMapReduce-cuda Integration

This directory contains the custom JIT-autotuned, block-reduction Cross-Entropy kernels integrated natively from the `GeMMMapReduce-cuda` codebase.

## Overview

We have successfully migrated the standalone `GeMMMapReduce-cuda` codebase natively into this `flash-attention` fork. It is now 100% self-sufficient and operates without any PyTorch SDPA fallbacks.

The primary entry point is the `GeMMMrXEntropyLoss` module found in `flash_attn/losses/cross_entropy.py`, which is dynamically compiled at runtime via the JIT autotuner (`gemmmr_autotune.py`).

## Key Modifications and Bug Fixes

During integration, we implemented several critical patches to ensure stability and compatibility, particularly for consumer/client GPUs (e.g., L4, Ada):

1. **JIT Autotuner Restraints for L4**:
   - The autotuner logic was updated to strictly return `[16]` for the candidate tile size on GPUs with `< 150 KB` shared memory (like the L4). This prevents an aggressive 32-tile fallback attempt that would throw a `cudaFuncSetAttribute` failure and irreversibly dirty the global CUDA Context error state with `cudaErrorInvalidValue`.

2. **OOM Fixes in Forward and Backward Kernels**:
   - We slashed the dynamic shared memory allocation by 50% by redirecting the prediction matrix (`pred`) reads to leverage the L2 cache directly from global memory instead of pulling it into shared memory.
   - We safely wrapped the `pred` global memory reads in an `if (m < M)` bounds check, while keeping the multi-warp synchronization operations (`__shfl_down_sync`) safely outside the block to prevent warp divergence deadlocks. This fixed out-of-bounds `cudaErrorIllegalAddress` segmentation faults caused by padding threads.

3. **MAX_D Hardcoded Bound Increased**:
   - The original `GeMMMapReduce-cuda` repository hardcoded `#define MAX_D 256` for the thread-local gradient accumulator array. Since our models use a projection space dimension `D` of `1024`, the backward kernel silently overflowed this array, corrupting the GPU stack memory. We dynamically bumped `#define MAX_D` to `4096` in `xentropy_backward_kernel.cu` to seamlessly accommodate massive 1024-dimensional feature projections!

## Usage

Simply import and use the integrated fused loss module:

```python
from flash_attn.losses.cross_entropy import GeMMMrXEntropyLoss

# Execute Attention Pass
out = flash_attn_func(q, k, v, causal=True)

# Generate Projection Weights
weight = torch.randn(vocab_size, nheads * headdim, device=device, dtype=torch.bfloat16)
targets = torch.randint(0, vocab_size, (batch, seqlen), device=device)

# Fused GeMMMapReduce Cross-Entropy Pass
loss_fn = GeMMMrXEntropyLoss()
loss = loss_fn(out.view(-1, nheads * headdim), weight, targets.view(-1))
loss.backward()
```
