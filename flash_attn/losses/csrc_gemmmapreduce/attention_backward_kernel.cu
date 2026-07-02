#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>
#include <ATen/cuda/Atomic.cuh>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

template <typename scalar_t>
__global__ void attention_backward_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ O,
    const scalar_t* __restrict__ dO,
    const scalar_t* __restrict__ l,
    const scalar_t* __restrict__ m_in,
    scalar_t* __restrict__ dQ,
    scalar_t* __restrict__ dK,
    scalar_t* __restrict__ dV,
    size_t M,
    size_t N,
    size_t F,
    size_t D) {

    using acc_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;

    size_t F_stride = F + 1;
    size_t D_stride = D + 1;

    extern __shared__ char shared_mem[];
    acc_t* s_Q = (acc_t*)shared_mem;
    acc_t* s_dO = (acc_t*)(shared_mem + TILE_SIZE * F_stride * sizeof(acc_t));
    acc_t* s_K = (acc_t*)(shared_mem + TILE_SIZE * (F_stride + D_stride) * sizeof(acc_t));
    acc_t* s_V = (acc_t*)(shared_mem + TILE_SIZE * (2 * F_stride + D_stride) * sizeof(acc_t));
    acc_t* s_dK = (acc_t*)(shared_mem + TILE_SIZE * (2 * F_stride + 2 * D_stride) * sizeof(acc_t));
    acc_t* s_dV = (acc_t*)(shared_mem + TILE_SIZE * (3 * F_stride + 2 * D_stride) * sizeof(acc_t));
    acc_t* s_P = (acc_t*)(shared_mem + TILE_SIZE * (3 * F_stride + 3 * D_stride) * sizeof(acc_t));
    acc_t* s_dS = (acc_t*)(shared_mem + (TILE_SIZE * (3 * F_stride + 3 * D_stride) + TILE_SIZE * TILE_SIZE) * sizeof(acc_t));

    int m = blockIdx.x * blockDim.x + threadIdx.x;

    acc_t delta = 0.0;
    if (m < M) {
        for (size_t d = 0; d < D; ++d) {
            delta += (acc_t)dO[m * D + d] * (acc_t)O[m * D + d];
        }
    }

    // Load Q and dO cooperatively and contiguously into shared memory
    size_t total_Q_elements = TILE_SIZE * F;
    for (size_t i = threadIdx.x; i < total_Q_elements; i += blockDim.x) {
        size_t r = i / F;
        size_t c = i % F;
        size_t global_m = blockIdx.x * TILE_SIZE + r;
        if (global_m < M) {
            s_Q[r * F_stride + c] = (acc_t)Q[global_m * F + c];
        } else {
            s_Q[r * F_stride + c] = 0.0;
        }
    }

    size_t total_dO_elements = TILE_SIZE * D;
    for (size_t i = threadIdx.x; i < total_dO_elements; i += blockDim.x) {
        size_t r = i / D;
        size_t c = i % D;
        size_t global_m = blockIdx.x * TILE_SIZE + r;
        if (global_m < M) {
            s_dO[r * D_stride + c] = (acc_t)dO[global_m * D + c];
        } else {
            s_dO[r * D_stride + c] = 0.0;
        }
    }
    __syncthreads();

    for (size_t j_start = 0; j_start < N; j_start += TILE_SIZE) {
        // Load K and V cooperatively and contiguously into shared memory
        size_t total_K_elements = TILE_SIZE * F;
        for (size_t i = threadIdx.x; i < total_K_elements; i += blockDim.x) {
            size_t r = i / F;
            size_t c = i % F;
            size_t global_j = j_start + r;
            if (global_j < N) {
                s_K[r * F_stride + c] = (acc_t)K[global_j * F + c];
            } else {
                s_K[r * F_stride + c] = 0.0;
            }
        }

        size_t total_V_elements = TILE_SIZE * D;
        for (size_t i = threadIdx.x; i < total_V_elements; i += blockDim.x) {
            size_t r = i / D;
            size_t c = i % D;
            size_t global_j = j_start + r;
            if (global_j < N) {
                s_V[r * D_stride + c] = (acc_t)V[global_j * D + c];
            } else {
                s_V[r * D_stride + c] = 0.0;
            }
        }
        __syncthreads();

        // Step A: Compute s_P and s_dS cooperatively for this key/value tile
        for (size_t j = 0; j < TILE_SIZE; ++j) {
            acc_t S_ij = 0.0;
            if (m < M && (j_start + j) < N) {
                for (size_t f = 0; f < F; ++f) {
                    S_ij += s_Q[threadIdx.x * F_stride + f] * s_K[j * F_stride + f];
                }
                s_P[threadIdx.x * TILE_SIZE + j] = exp(S_ij - (acc_t)m_in[m]) / (acc_t)l[m];
            } else {
                s_P[threadIdx.x * TILE_SIZE + j] = 0.0;
            }
        }
        __syncthreads();

        for (size_t j = 0; j < TILE_SIZE; ++j) {
            acc_t dP_ij = 0.0;
            if (m < M && (j_start + j) < N) {
                for (size_t d = 0; d < D; ++d) {
                    dP_ij += s_dO[threadIdx.x * D_stride + d] * s_V[j * D_stride + d];
                }
                s_dS[threadIdx.x * TILE_SIZE + j] = s_P[threadIdx.x * TILE_SIZE + j] * (dP_ij - delta);
            } else {
                s_dS[threadIdx.x * TILE_SIZE + j] = 0.0;
            }
        }
        __syncthreads();

        // Step B: Accumulate dQ and dK
        // Initialize s_dK to 0 cooperatively
        size_t total_dK_elements = TILE_SIZE * F;
        for (size_t i = threadIdx.x; i < total_dK_elements; i += blockDim.x) {
            size_t r = i / F;
            size_t c = i % F;
            s_dK[r * F_stride + c] = 0.0;
        }
        __syncthreads();

        // 1. Accumulate dQ directly to global (no contention between different rows m)
        for (size_t f = 0; f < F; ++f) {
            acc_t val = 0.0;
            for (size_t j = 0; j < TILE_SIZE; ++j) {
                size_t actual_n = j_start + j;
                if (actual_n < N) {
                    val += s_dS[threadIdx.x * TILE_SIZE + j] * s_K[j * F_stride + f];
                }
            }
            if (m < M) {
                gpuAtomicAdd(&dQ[m * F + f], (scalar_t)val);
            }
        }

        // 2. Accumulate dK to shared memory s_dK
        for (size_t f = 0; f < F; ++f) {
            acc_t q_val = s_Q[threadIdx.x * F_stride + f];
            for (size_t j = 0; j < TILE_SIZE; ++j) {
                size_t actual_n = j_start + j;
                if (actual_n < N) {
                    acc_t val = s_dS[threadIdx.x * TILE_SIZE + j] * q_val;
                    atomicAdd(&s_dK[j * F_stride + f], val);
                }
            }
        }
        __syncthreads();

        // Write s_dK to global dK cooperatively
        for (size_t i = threadIdx.x; i < total_dK_elements; i += blockDim.x) {
            size_t r = i / F;
            size_t c = i % F;
            size_t actual_n = j_start + r;
            if (actual_n < N) {
                gpuAtomicAdd(&dK[actual_n * F + c], (scalar_t)s_dK[r * F_stride + c]);
            }
        }
        __syncthreads();

        // Step C: Accumulate dV
        // Initialize s_dV to 0 cooperatively
        size_t total_dV_elements = TILE_SIZE * D;
        for (size_t i = threadIdx.x; i < total_dV_elements; i += blockDim.x) {
            size_t r = i / D;
            size_t c = i % D;
            s_dV[r * D_stride + c] = 0.0;
        }
        __syncthreads();

        // Accumulate dV to shared
        for (size_t d = 0; d < D; ++d) {
            acc_t dO_val = s_dO[threadIdx.x * D_stride + d];
            for (size_t j = 0; j < TILE_SIZE; ++j) {
                size_t actual_n = j_start + j;
                if (actual_n < N) {
                    acc_t val = s_P[threadIdx.x * TILE_SIZE + j] * dO_val;
                    atomicAdd(&s_dV[j * D_stride + d], val);
                }
            }
        }
        __syncthreads();

        // Write s_dV to global dV cooperatively
        for (size_t i = threadIdx.x; i < total_dV_elements; i += blockDim.x) {
            size_t r = i / D;
            size_t c = i % D;
            size_t actual_n = j_start + r;
            if (actual_n < N) {
                gpuAtomicAdd(&dV[actual_n * D + c], (scalar_t)s_dV[r * D_stride + c]);
            }
        }
        __syncthreads();
    }
}

void launch_attention_backward_kernel(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor l,
    torch::Tensor m_in,
    torch::Tensor dQ,
    torch::Tensor dK,
    torch::Tensor dV,
    size_t M,
    size_t N,
    size_t F,
    size_t D) {

    int threads = TILE_SIZE;
    int blocks = (M + threads - 1) / threads;

    size_t F_stride = F + 1;
    size_t D_stride = D + 1;
    size_t element_size = (Q.scalar_type() == at::ScalarType::Double) ? sizeof(double) : sizeof(float);
    size_t shared_mem_size = (TILE_SIZE * (3 * F_stride + 3 * D_stride) + 2 * TILE_SIZE * TILE_SIZE) * element_size;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, Q.scalar_type(), "attention_backward_kernel", ([&] {
        cudaError_t err = cudaFuncSetAttribute((const void*)attention_backward_kernel<scalar_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));
        }
        attention_backward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            O.data_ptr<scalar_t>(),
            dO.data_ptr<scalar_t>(),
            l.data_ptr<scalar_t>(),
            m_in.data_ptr<scalar_t>(),
            dQ.data_ptr<scalar_t>(),
            dK.data_ptr<scalar_t>(),
            dV.data_ptr<scalar_t>(),
            M, N, F, D
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            TORCH_CHECK(false, "Kernel launch failed: ", cudaGetErrorString(err));
        }
    }));
}
