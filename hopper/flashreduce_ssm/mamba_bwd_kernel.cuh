#pragma once

#include <cute/tensor.hpp>
#include "affine_monoid.hpp"

namespace flashreduce {

using namespace cute;

template <typename T>
__device__ __forceinline__ AffineState<T> warp_scan_suffix(AffineState<T> val) {
    int lane = threadIdx.x % 32;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float remote_a = __shfl_down_sync(0xffffffff, val.a, offset);
        float remote_b = __shfl_down_sync(0xffffffff, val.b, offset);
        if (lane < 32 - offset) {
            val = compose(val, AffineState<T>(remote_a, remote_b));
        }
    }
    return val;
}

template <typename T, int BLOCK_T, int N_THREADS>
__global__ void mamba_bwd_kernel(
    const T* __restrict__ g_a,       // (batch, seqlen, dim)
    const T* __restrict__ g_x,       // (batch, seqlen, dim) - forward states
    const T* __restrict__ g_dy,      // (batch, seqlen, dim) - grad_output
    T* __restrict__ g_da,            // (batch, seqlen, dim) - grad_a
    T* __restrict__ g_db,            // (batch, seqlen, dim) - grad_b
    int seqlen,
    int dim,
    int stride_b,                    // Stride for batch
    int stride_s                     // Stride for seqlen
) {
    // Each block processes a single (batch, dim) channel in reverse
    int d_idx = blockIdx.x;
    int b_idx = blockIdx.y;

    if (d_idx >= dim) return;

    // Shared memory for warp synchronization inside block suffix scan
    __shared__ AffineState<float> smem_scan[N_THREADS / 32];

    // Initialize running suffix state (0 gradient from subsequent steps)
    AffineState<float> running_suffix(1.0f, 0.0f);

    constexpr int ITEMS_PER_THREAD = BLOCK_T / N_THREADS;
    int num_blocks = (seqlen + BLOCK_T - 1) / BLOCK_T;

    // Loop over sequence blocks in REVERSE order
    for (int block_idx = num_blocks - 1; block_idx >= 0; --block_idx) {
        AffineState<float> local_states[ITEMS_PER_THREAD];
        float local_x_prev[ITEMS_PER_THREAD];

        #pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
            int t = block_idx * BLOCK_T + threadIdx.x * ITEMS_PER_THREAD + j;
            if (t < seqlen) {
                int offset = b_idx * stride_b + t * stride_s + d_idx;
                // Scale factor for sequential update:
                // e_t = a_{t+1} * e_{t+1} + g_t.
                // So the monoid element at step t is (a_{t+1}, g_t).
                float a_next = (t + 1 < seqlen) ? static_cast<float>(g_a[b_idx * stride_b + (t + 1) * stride_s + d_idx]) : 0.0f;
                float dy_val = static_cast<float>(g_dy[offset]);
                local_states[j] = AffineState<float>(a_next, dy_val);

                // Load x_{t-1} for computing grad_a
                local_x_prev[j] = (t > 0) ? static_cast<float>(g_x[b_idx * stride_b + (t - 1) * stride_s + d_idx]) : 0.0f;
            } else {
                local_states[j] = AffineState<float>(1.0f, 0.0f); // Identity
                local_x_prev[j] = 0.0f;
            }
        }

        // 1. Thread-local sequential suffix scan (inclusive)
        for (int j = ITEMS_PER_THREAD - 2; j >= 0; --j) {
            local_states[j] = compose(local_states[j], local_states[j + 1]);
        }
        AffineState<float> thread_sum = local_states[0];

        // 2. Warp-level suffix scan on thread_sum
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;

        AffineState<float> val = thread_sum;
        val = warp_scan_suffix(val);

        // Write warp aggregates to shared memory (lane 0 has warp sum)
        if (lane == 0) {
            smem_scan[warp_id] = val;
        }
        __syncthreads();

        // Warp 0 suffix-scans warp aggregates
        if (warp_id == 0) {
            AffineState<float> warp_val = (threadIdx.x < (N_THREADS / 32)) ? smem_scan[threadIdx.x] : AffineState<float>();
            warp_val = warp_scan_suffix(warp_val);
            if (threadIdx.x < (N_THREADS / 32)) {
                smem_scan[threadIdx.x] = warp_val;
            }
        }
        __syncthreads();

        // Broadcast next warp sum to current warp
        AffineState<float> block_suffix_for_thread;
        if (warp_id < (N_THREADS / 32) - 1) {
            block_suffix_for_thread = smem_scan[warp_id + 1];
        }

        // Unconditional warp shuffle to fetch next lane's inclusive suffix scan
        float next_lane_a = __shfl_down_sync(0xffffffff, val.a, 1);
        float next_lane_b = __shfl_down_sync(0xffffffff, val.b, 1);

        AffineState<float> warp_suffix_within_warp;
        if (lane < 31) {
            warp_suffix_within_warp = AffineState<float>(next_lane_a, next_lane_b);
        } else {
            warp_suffix_within_warp = AffineState<float>(1.0f, 0.0f);
        }

        AffineState<float> warp_suffix = warp_suffix_within_warp;
        if (warp_id < (N_THREADS / 32) - 1) {
            warp_suffix = compose(warp_suffix, block_suffix_for_thread);
        }

        // Apply running suffix and warp suffix to thread's local states
        #pragma unroll
        for (int j = ITEMS_PER_THREAD - 1; j >= 0; --j) {
            int t = block_idx * BLOCK_T + threadIdx.x * ITEMS_PER_THREAD + j;
            if (t < seqlen) {
                AffineState<float> scan_val = compose(local_states[j], warp_suffix);
                AffineState<float> final_suffix = compose(scan_val, running_suffix);

                // Compute updates: grad_b = e_t, grad_a = e_t * x_{t-1}
                float grad_b = final_suffix.b;
                float grad_a = grad_b * local_x_prev[j];

                // Write outputs back to global memory
                int offset = b_idx * stride_b + t * stride_s + d_idx;
                g_db[offset] = static_cast<T>(grad_b);
                g_da[offset] = static_cast<T>(grad_a);
            }
        }

        // Update running_suffix with total block sum
        AffineState<float> total_block_sum = smem_scan[0];
        running_suffix = compose(total_block_sum, running_suffix);
        __syncthreads();
    }
}

} // namespace flashreduce
