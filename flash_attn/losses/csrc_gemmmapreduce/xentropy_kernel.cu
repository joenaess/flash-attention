#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

template <typename scalar_t>
__global__ void xentropy_cuda_kernel(
    const scalar_t* __restrict__ pred,
    const scalar_t* __restrict__ trg,
    const int64_t* __restrict__ truth,
    const int64_t* __restrict__ tixs,
    scalar_t* __restrict__ p_out,
    scalar_t* __restrict__ n_out,
    size_t M,
    size_t N,
    size_t D) {

    using acc_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;

    extern __shared__ char shared_mem[];
    scalar_t* s_trg = (scalar_t*)shared_mem;

    size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    


    acc_t p_acc = -INFINITY;
    acc_t n_acc = 0;

    int64_t truth_val = -1;
    if (m < M) {
        truth_val = truth[m];
    }

    // Iterate over trg matrix in steps of TILE_SIZE
    for (size_t n_start = 0; n_start < N; n_start += TILE_SIZE) {
        // Load trg tile cooperatively and contiguously
        size_t total_trg_elements = TILE_SIZE * D;
        for (size_t i = threadIdx.x; i < total_trg_elements; i += blockDim.x) {
            size_t r = i / D;
            size_t c = i % D;
            size_t global_n = n_start + r;
            if (global_n < N) {
                s_trg[r * D + c] = trg[global_n * D + c];
            } else {
                s_trg[r * D + c] = 0.0;
            }
        }
        
        __syncthreads(); // Wait for all loads to complete

        // Only compute if this thread has a valid row 'm'
        if (m < M) {
            // Inner Math Loop: calculate against all elements in the trg tile
            for (size_t n_step = 0; n_step < TILE_SIZE; ++n_step) {
                size_t actual_n = n_start + n_step;
                if (actual_n >= N) break;

                acc_t h_val = 0;
                for (size_t d = 0; d < D; ++d) {
                    h_val += (acc_t)pred[m * D + d] * (acc_t)s_trg[n_step * D + d];
                }

                if (truth_val == tixs[actual_n]) {
                    n_acc += h_val;
                }

                if (p_acc <= -INFINITY) {
                    p_acc = h_val;
                } else {
                    acc_t diff = p_acc - h_val;
                    acc_t max_val = p_acc > h_val ? p_acc : h_val;
                    acc_t abs_diff = diff > 0 ? diff : -diff;
                    p_acc = max_val + log1p(exp(-abs_diff));
                }
            }
        }
        
        __syncthreads(); // Wait before overwriting shared memory with the next tile
    }

    if (m < M) {
        p_out[m] = (scalar_t)p_acc;
        n_out[m] = (scalar_t)n_acc;
    }
}

void launch_xentropy_kernel(
    torch::Tensor pred,
    torch::Tensor trg,
    torch::Tensor truth,
    torch::Tensor tixs,
    torch::Tensor p_out,
    torch::Tensor n_out,
    size_t M,
    size_t N,
    size_t D) {

    int threads = TILE_SIZE;
    int blocks = (M + threads - 1) / threads;
    size_t shared_mem_size = TILE_SIZE * D * pred.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, pred.scalar_type(), "xentropy_cuda_kernel", ([&] {
        xentropy_cuda_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            pred.data_ptr<scalar_t>(),
            trg.data_ptr<scalar_t>(),
            truth.data_ptr<int64_t>(),
            tixs.data_ptr<int64_t>(),
            p_out.data_ptr<scalar_t>(),
            n_out.data_ptr<scalar_t>(),
            M, N, D
        );
    }));
}
