#pragma once

#include <cute/tensor.hpp>
#include "affine_monoid.hpp"

namespace flashreduce {

using namespace cute;

template <typename T, int BLOCK_T, int N_THREADS>
__global__ void mamba_fwd_kernel(
    const T* __restrict__ g_a,       // (batch, seqlen, dim)
    const T* __restrict__ g_b,       // (batch, seqlen, dim)
    T* __restrict__ g_x,             // (batch, seqlen, dim)
    float* __restrict__ g_block_prefixes_a, // (batch, num_blocks, dim)
    float* __restrict__ g_block_prefixes_b, // (batch, num_blocks, dim)
    int seqlen,
    int dim,
    int stride_b,                    // Stride for batch
    int stride_s                     // Stride for seqlen
) {
    // Each block processes a single (batch, dim) channel
    int d_idx = blockIdx.x;
    int b_idx = blockIdx.y;

    if (d_idx >= dim) return;

    // Shared memory for warp synchronization inside block scan
    __shared__ AffineState<float> smem_scan[N_THREADS / 32];

    // Initialize running prefix state
    AffineState<float> running_prefix(1.0f, 0.0f);

    constexpr int ITEMS_PER_THREAD = BLOCK_T / N_THREADS;
    int num_blocks = (seqlen + BLOCK_T - 1) / BLOCK_T;

    // Loop over sequence in blocks of size BLOCK_T
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // Load data into local registers
        AffineState<float> local_states[ITEMS_PER_THREAD];

        #pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            int t = block_idx * BLOCK_T + threadIdx.x * ITEMS_PER_THREAD + j;
            if (t < seqlen) {
                int offset = b_idx * stride_b + t * stride_s + d_idx;
                float a_val = static_cast<float>(g_a[offset]);
                float b_val = static_cast<float>(g_b[offset]);
                local_states[j] = AffineState<float>(a_val, b_val);
            } else {
                local_states[j] = AffineState<float>(1.0f, 0.0f); // Identity
            }
        }

        // 1. Thread-local sequential inclusive scan
        for (int j = 1; j < ITEMS_PER_THREAD; ++j) {
            local_states[j] = scan_op(local_states[j - 1], local_states[j]);
        }
        AffineState<float> thread_sum = local_states[ITEMS_PER_THREAD - 1];

        // 2. Warp-level scan on thread_sum
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;

        AffineState<float> val = thread_sum;
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            float remote_a = __shfl_up_sync(0xffffffff, val.a, offset);
            float remote_b = __shfl_up_sync(0xffffffff, val.b, offset);
            if (lane >= offset) {
                val = compose(val, AffineState<float>(remote_a, remote_b));
            }
        }

        // Write warp aggregates to shared memory
        if (lane == 31) {
            smem_scan[warp_id] = val;
        }
        __syncthreads();

        // Warp 0 scans warp aggregates
        if (warp_id == 0) {
            AffineState<float> warp_val = (threadIdx.x < (N_THREADS / 32)) ? smem_scan[threadIdx.x] : AffineState<float>();
            #pragma unroll
            for (int offset = 1; offset < 32; offset *= 2) {
                float remote_a = __shfl_up_sync(0xffffffff, warp_val.a, offset);
                float remote_b = __shfl_up_sync(0xffffffff, warp_val.b, offset);
                if (lane >= offset) {
                    warp_val = compose(warp_val, AffineState<float>(remote_a, remote_b));
                }
            }
            if (threadIdx.x < (N_THREADS / 32)) {
                smem_scan[threadIdx.x] = warp_val;
            }
        }
        __syncthreads();

        // Broadcast previous warp sum to current warp
        AffineState<float> block_prefix_for_thread;
        if (warp_id > 0) {
            block_prefix_for_thread = smem_scan[warp_id - 1];
        }

        // Unconditional warp shuffle to fetch previous lane's inclusive prefix
        float prev_lane_a = __shfl_up_sync(0xffffffff, val.a, 1);
        float prev_lane_b = __shfl_up_sync(0xffffffff, val.b, 1);

        AffineState<float> warp_prefix_within_warp;
        if (lane > 0) {
            warp_prefix_within_warp = AffineState<float>(prev_lane_a, prev_lane_b);
        } else {
            warp_prefix_within_warp = AffineState<float>(1.0f, 0.0f);
        }

        AffineState<float> warp_prefix = warp_prefix_within_warp;
        if (warp_id > 0) {
            warp_prefix = compose(warp_prefix, block_prefix_for_thread);
        }

        // Apply running prefix and warp prefix to thread's local states
        #pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            int t = block_idx * BLOCK_T + threadIdx.x * ITEMS_PER_THREAD + j;
            if (t < seqlen) {
                AffineState<float> scan_val = compose(local_states[j], warp_prefix);
                AffineState<float> final_state = compose(scan_val, running_prefix);

                // Write output back to global memory
                int offset = b_idx * stride_b + t * stride_s + d_idx;
                g_x[offset] = static_cast<T>(final_state.b);
            }
        }

        // Update running_prefix with total block sum
        AffineState<float> total_block_sum = smem_scan[N_THREADS / 32 - 1];
        running_prefix = compose(total_block_sum, running_prefix);

        // At the end of every block's computation, write the block prefix to global memory
        if (threadIdx.x == N_THREADS - 1) {
            int prefix_offset = b_idx * (num_blocks * dim) + block_idx * dim + d_idx;
            g_block_prefixes_a[prefix_offset] = running_prefix.a;
            g_block_prefixes_b[prefix_offset] = running_prefix.b;
        }
        __syncthreads();
    }
}

} // namespace flashreduce
