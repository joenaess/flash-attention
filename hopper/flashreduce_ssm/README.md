# Mamba Affine State Updates: Specialized Blelloch Suffix Scan

This directory contains a highly optimized, custom CUDA implementation of Mamba-style non-commutative sequential scan operations and their analytical gradients based on **Amaru Cuba Gyllensten's "Flashreduce Theory"**. 

---

## 1. Mathematical Formulation

### Forward Pass
The forward pass is defined by a non-commutative sequential recurrence relation:
$$x_t = a_t x_{t-1} + b_t \quad \text{for } t = 1 \dots T, \text{ with } x_0 = 0$$

To parallelize this recurrence, each step is mapped to an element of the **Affine Function Monoid** $(a_t, b_t)$. 
The composition of two transformations $f_1 = (a_1, b_1)$ and $f_2 = (a_2, b_2)$ is:
$$f_1 \circ f_2 \implies (a_1 a_2, a_1 b_2 + b_1)$$

By executing an associative parallel prefix scan with the operator:
$$scan\_op(A, B) = compose(B, A) = (B.a \cdot A.a, B.a \cdot A.b + B.b)$$
we compute all intermediate state vectors $x_t = P_t.b$ in $O(\log T)$ time instead of $O(T)$ sequential steps.

### Backward Pass (Specialized Blelloch Suffix Scan)
Using the **Finalize-Embed Factorization**, the gradients with respect to parameters $a_t$ and $b_t$ are computed via:
$$e_t = \frac{\partial L}{\partial b_t} = a_{t+1} e_{t+1} + g_t \quad \text{for } t = T-1 \dots 1, \text{ with } e_T = g_T$$
$$\nabla b_t = e_t, \quad \nabla a_t = e_t x_{t-1}$$

This backward pass is a **suffix scan** (right-to-left scan) over the gradient accumulator object $e_t$ using the exact same composition operator.

---

## 2. Hardware Mapping & Optimization

### Two-Level Parallelism
1. **Grid Level**: We launch $Batch \times Dim$ completely independent thread blocks. Each block handles a single 1D sequence of length $T$ for one channel, avoiding slow and complex inter-block synchronizations or cooperative grid launches.
2. **Block Level**: Inside each block of size `BLOCK_T`, we perform hierarchical scans using warp shuffles (`__shfl_up_sync` and `__shfl_down_sync`) to communicate state prefixes/suffixes within registers, only using Shared Memory to bridge warp boundaries.

### Local Gradient Theorem and Prefix Materialization
To satisfy **Corollary 3.3**, we write the intermediate states $x_t$ and the running block-level prefixes `BlockPrefixes` during the forward pass to global memory. In the backward pass, each block reads `g_x` ($x_{t-1}$) and performs the specialized Blelloch suffix scan entirely in local registers and warp registers, computing local gradients $\nabla a$ and $\nabla b$ on-chip before writing them out in a coalesced format.

---

## 3. Directory Layout

- [affine_monoid.hpp](affine_monoid.hpp): Monoid struct definitions and composition rules.
- [mamba_fwd_kernel.cuh](mamba_fwd_kernel.cuh): Forward prefix scan kernel with prefix materialization.
- [mamba_bwd_kernel.cuh](mamba_bwd_kernel.cuh): Backward specialized Blelloch suffix scan.
- [flashreduce_kernels.cu](flashreduce_kernels.cu): Launch wrappers and CUDA type dispatches.
- [flashreduce_api.cpp](flashreduce_api.cpp): Pybind11 binding entrypoint.
