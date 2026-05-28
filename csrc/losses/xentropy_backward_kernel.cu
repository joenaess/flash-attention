#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>
#include <ATen/cuda/Atomic.cuh>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
#define MAX_D 256

template <typename scalar_t>
__global__ void xentropy_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_p,
    const scalar_t* __restrict__ grad_n,
    const scalar_t* __restrict__ pred,
    const scalar_t* __restrict__ trg,
    const int64_t* __restrict__ truth,
    const int64_t* __restrict__ tixs,
    const scalar_t* __restrict__ p_out,
    scalar_t* __restrict__ grad_pred,
    scalar_t* __restrict__ grad_trg,
    size_t M,
    size_t N,
    size_t D) {

    using acc_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;

    extern __shared__ char shared_mem[];
    scalar_t* s_pred = (scalar_t*)shared_mem;
    scalar_t* s_trg = (scalar_t*)(shared_mem + TILE_SIZE * D * sizeof(scalar_t));
    int64_t* s_tixs = (int64_t*)(shared_mem + 2 * TILE_SIZE * D * sizeof(scalar_t));
    __shared__ acc_t cross_warp_staging[32];

    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    
    acc_t local_grad_pred[MAX_D] = {0};
    
    acc_t g_p = 0;
    acc_t g_n = 0;
    acc_t p_o = 0;
    int64_t truth_m = -1;

    // Load pred tile into shared memory and pre-load scalars
    if (m < M) {
        g_p = (acc_t)grad_p[m];
        g_n = (acc_t)grad_n[m];
        p_o = (acc_t)p_out[m];
        truth_m = truth[m];
        for (size_t d = 0; d < D; ++d) {
            s_pred[threadIdx.x * D + d] = pred[m * D + d];
        }
    } else {
        for (size_t d = 0; d < D; ++d) {
            s_pred[threadIdx.x * D + d] = 0;
        }
    }

    for (size_t n_start = 0; n_start < N; n_start += TILE_SIZE) {
        size_t trg_n = n_start + threadIdx.x;

        // Load trg and tixs tiles cooperatively
        if (trg_n < N) {
            for (size_t d = 0; d < D; ++d) {
                s_trg[threadIdx.x * D + d] = trg[trg_n * D + d];
            }
            s_tixs[threadIdx.x] = tixs[trg_n];
        }
        
        __syncthreads(); // Wait for loads to complete

        // Inner Math Loop (All threads participate to avoid __shfl_down_sync deadlock)
        for (size_t n_step = 0; n_step < TILE_SIZE; ++n_step) {
            size_t actual_n = n_start + n_step;
            if (actual_n >= N) break;

            acc_t h_val = 0;
            for (size_t d = 0; d < D; ++d) {
                h_val += (acc_t)s_pred[threadIdx.x * D + d] * (acc_t)s_trg[n_step * D + d];
            }

            acc_t dh = 0;
            if (m < M) {
                dh = g_p * exp(h_val - p_o);
                if (truth_m == s_tixs[n_step]) {
                    dh += g_n;
                }
            }

            for (size_t d = 0; d < D; ++d) {
                // Accumulate grad_pred entirely in local registers
                if (m < M) {
                    local_grad_pred[d] += dh * (acc_t)s_trg[n_step * D + d];
                }
                
                // Multi-warp Block-Level Shared Memory Reduction Tree
                int lane_id = threadIdx.x % 32;
                int warp_id = threadIdx.x / 32;
                int num_warps = blockDim.x / 32;

                acc_t local_grad_trg = 0;
                if (m < M) {
                    local_grad_trg = dh * (acc_t)s_pred[threadIdx.x * D + d];
                }
                
                // Phase 1 (Warp-Local Reduction)
                acc_t val = local_grad_trg;
                for (int offset = 16; offset > 0; offset /= 2) {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                
                // Phase 2 (Shared Staging Array write by warp leaders)
                if (lane_id == 0) {
                    cross_warp_staging[warp_id] = val;
                }
                __syncthreads(); // Barrier to ensure all warps checked in
                
                // Phase 3 (Inter-Warp Final Sweep by Warp 0)
                if (warp_id == 0) {
                    acc_t warp_val = (lane_id < num_warps) ? cross_warp_staging[lane_id] : 0.0;
                    for (int offset = 16; offset > 0; offset /= 2) {
                        warp_val += __shfl_down_sync(0xffffffff, warp_val, offset);
                    }
                    
                    // Phase 4 (Mathematical Global Write strictly gated by absolute block leader)
                    if (threadIdx.x == 0) {
                        gpuAtomicAdd(&grad_trg[actual_n * D + d], (scalar_t)warp_val);
                    }
                }
                __syncthreads(); // Barrier to ensure staging is not overwritten in next iteration
            }
        }
        
        __syncthreads(); // Synchronize before overwriting shared memory with next tile
    }

    // Write final accumulated grad_pred to global memory
    if (m < M) {
        for (size_t d = 0; d < D; ++d) {
            grad_pred[m * D + d] = (scalar_t)local_grad_pred[d];
        }
    }
}

void launch_xentropy_backward_kernel(
    torch::Tensor grad_p,
    torch::Tensor grad_n,
    torch::Tensor pred,
    torch::Tensor trg,
    torch::Tensor truth,
    torch::Tensor tixs,
    torch::Tensor p_out,
    torch::Tensor grad_pred,
    torch::Tensor grad_trg,
    size_t M,
    size_t N,
    size_t D) {

    int threads = TILE_SIZE;
    int blocks = (M + threads - 1) / threads;
    size_t shared_mem_size = 2 * TILE_SIZE * D * pred.element_size() + TILE_SIZE * sizeof(int64_t);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, pred.scalar_type(), "xentropy_cuda_backward_kernel", ([&] {
        cudaFuncSetAttribute(xentropy_cuda_backward_kernel<scalar_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        xentropy_cuda_backward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            grad_p.data_ptr<scalar_t>(),
            grad_n.data_ptr<scalar_t>(),
            pred.data_ptr<scalar_t>(),
            trg.data_ptr<scalar_t>(),
            truth.data_ptr<int64_t>(),
            tixs.data_ptr<int64_t>(),
            p_out.data_ptr<scalar_t>(),
            grad_pred.data_ptr<scalar_t>(),
            grad_trg.data_ptr<scalar_t>(),
            M, N, D
        );
    }));
}
