#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
#define MAX_D 256
#define MAX_F 256

template <typename scalar_t>
__global__ void attention_forward_kernel(
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    scalar_t* __restrict__ O,
    scalar_t* __restrict__ l,
    scalar_t* __restrict__ m_out,
    size_t M,
    size_t N,
    size_t F,
    size_t D) {

    using acc_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;

    extern __shared__ char shared_mem[];
    scalar_t* s_K = (scalar_t*)shared_mem;
    scalar_t* s_V = (scalar_t*)(shared_mem + TILE_SIZE * F * sizeof(scalar_t));

    int m = blockIdx.x * blockDim.x + threadIdx.x;

    acc_t m_i = -INFINITY;
    acc_t l_i = 0.0;
    acc_t O_i[MAX_D] = {0.0};
    acc_t q_i[MAX_F] = {0.0};

    if (m < M) {
        for (size_t f = 0; f < F; ++f) {
            q_i[f] = (acc_t)Q[m * F + f];
        }
    }

    for (size_t j_start = 0; j_start < N; j_start += TILE_SIZE) {
        // Load K cooperatively and contiguously using 128-bit vectorized loads
        constexpr size_t VEC_SIZE_K = 16 / sizeof(scalar_t);
        size_t total_K_vec = (TILE_SIZE * F) / VEC_SIZE_K;
        const float4* K_vec = reinterpret_cast<const float4*>(K);
        float4* s_K_vec = reinterpret_cast<float4*>(s_K);

        for (size_t i = threadIdx.x; i < total_K_vec; i += blockDim.x) {
            size_t scalar_idx = i * VEC_SIZE_K;
            size_t r = scalar_idx / F;
            size_t c = scalar_idx % F;
            size_t global_j = j_start + r;
            if (global_j < N) {
                s_K_vec[i] = K_vec[(global_j * F + c) / VEC_SIZE_K];
            } else {
                s_K_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // Load V cooperatively and contiguously using 128-bit vectorized loads
        constexpr size_t VEC_SIZE_V = 16 / sizeof(scalar_t);
        size_t total_V_vec = (TILE_SIZE * D) / VEC_SIZE_V;
        const float4* V_vec = reinterpret_cast<const float4*>(V);
        float4* s_V_vec = reinterpret_cast<float4*>(s_V);

        for (size_t i = threadIdx.x; i < total_V_vec; i += blockDim.x) {
            size_t scalar_idx = i * VEC_SIZE_V;
            size_t r = scalar_idx / D;
            size_t c = scalar_idx % D;
            size_t global_j = j_start + r;
            if (global_j < N) {
                s_V_vec[i] = V_vec[(global_j * D + c) / VEC_SIZE_V];
            } else {
                s_V_vec[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        if (m < M) {
            // Find max score in tile
            acc_t scores[TILE_SIZE] = {0};
            acc_t m_tile = -INFINITY;
            
            for (size_t j = 0; j < TILE_SIZE; ++j) {
                if (j_start + j < N) {
                    acc_t score = 0;
                    for (size_t f = 0; f < F; ++f) {
                        score += q_i[f] * (acc_t)s_K[j * F + f];
                    }
                    scores[j] = score;
                    m_tile = (score > m_tile) ? score : m_tile;
                }
            }

            // Monoid Fold: Update Global Max and Scale History
            acc_t m_new = (m_i > m_tile) ? m_i : m_tile;
            acc_t exp_scale = exp(m_i - m_new);
            
            l_i *= exp_scale;
            for (size_t d = 0; d < D; ++d) {
                O_i[d] *= exp_scale;
            }

            // Accumulate new tile
            for (size_t j = 0; j < TILE_SIZE; ++j) {
                if (j_start + j < N) {
                    acc_t exp_s = exp(scores[j] - m_new);
                    l_i += exp_s;
                    for (size_t d = 0; d < D; ++d) {
                        O_i[d] += exp_s * (acc_t)s_V[j * D + d];
                    }
                }
            }
            m_i = m_new;
        }
        __syncthreads();
    }

    if (m < M) {
        // Finalize O
        for (size_t d = 0; d < D; ++d) {
            O[m * D + d] = (scalar_t)(O_i[d] / l_i);
        }
        l[m] = (scalar_t)l_i;
        m_out[m] = (scalar_t)m_i;
    }
}

void launch_attention_forward_kernel(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor l,
    torch::Tensor m_out,
    size_t M,
    size_t N,
    size_t F,
    size_t D) {

    int threads = TILE_SIZE;
    int blocks = (M + threads - 1) / threads;
    size_t shared_mem_size = TILE_SIZE * (F + D) * Q.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, Q.scalar_type(), "attention_forward_kernel", ([&] {
        cudaFuncSetAttribute(attention_forward_kernel<scalar_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        attention_forward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            O.data_ptr<scalar_t>(),
            l.data_ptr<scalar_t>(),
            m_out.data_ptr<scalar_t>(),
            M, N, F, D
        );
    }));
}
