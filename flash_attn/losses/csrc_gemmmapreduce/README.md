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

## Benchmarks

We benchmarked the GeMMMapReduce fused kernel against a standard PyTorch baseline (`F.linear` followed by `F.cross_entropy`) on an NVIDIA L4 GPU. The baseline must materialize the massive `[batch * seqlen, vocab_size]` logits tensor in global memory before applying cross-entropy, representing an $O(N \times V)$ memory complexity. Our fused kernel computes the loss block-by-block, reducing the memory complexity to $O(N)$.

### Hardware
* **GPU:** NVIDIA L4
* **Shared Memory limit per block:** 48 KB

### Settings
* **Batch Size:** 1
* **Heads:** 8
* **Head Dim:** 128
* **Hidden Dim:** 1024
* **Vocab Size:** 32,000

### Results

| Seq Len | PyTorch Time (ms) | GeMMMr Time (ms) | PyTorch Mem (MB) | GeMMMr Mem (MB) | Mem Reduction |
|---------|-------------------|------------------|------------------|-----------------|---------------|
| 2048    | 18.17             | 30375.95         | 460.77           | 149.52          | **3.1x**      |
| 4096    | 16.94             | 58358.43         | 836.78           | 157.55          | **5.3x**      |
| 8192    | 36.12             | 90876.05         | 1594.81          | 173.60          | **9.2x**      |
| 16384   | 81.96             | 180939.18        | 3110.88          | 206.71          | **15.0x**     |
| 32768   | 151.53            | 359274.08        | 6143.00          | 269.93          | **22.8x**     |

*Note: The dramatic 22.8x memory reduction demonstrates the $O(N)$ efficiency. The custom kernel's current execution latency is bottlenecked by unoptimized scalar loops for matrix multiplication operations; integration of SM89/SM90 Tensor Core MMA instructions is required to reach runtime parity with cuBLAS.*
