#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>
#include <cute/tensor.hpp>

#ifndef BLK_M
#define BLK_M 64
#define BLK_N 64
#define BLK_K 128
#endif

using namespace cute;

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

    using Element = std::conditional_t<
        std::is_same_v<scalar_t, at::Half>, cutlass::half_t,
        std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, cutlass::bfloat16_t, scalar_t>>;
    using ElementCompute = float;
    
    const Element* __restrict__ pred_elem = reinterpret_cast<const Element*>(pred);
    const Element* __restrict__ trg_elem = reinterpret_cast<const Element*>(trg);

    using mma_op = std::conditional_t<
        std::is_same_v<scalar_t, at::BFloat16> || std::is_same_v<scalar_t, cutlass::bfloat16_t>,
        SM80_16x8x16_F32BF16BF16F32_TN,
        SM80_16x8x16_F32F16F16F32_TN>;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using TiledMma = decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{})));
    TiledMma tiled_mma;

    int thread_idx = threadIdx.x;
    int m_block = blockIdx.x;
    size_t m_start = m_block * BLK_M;

    extern __shared__ char shared_mem[];
    Element* sA_ptr = (Element*)shared_mem;
    Element* sB_ptr = sA_ptr + BLK_M * BLK_K;

    // sA is Row-Major (BLK_M, BLK_K)
    Tensor sA = make_tensor(make_smem_ptr(sA_ptr), make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{});
    // B must be logically (BLK_N, BLK_K). Since memory is Row-Major, we use LayoutRight.
    Tensor sB = make_tensor(make_smem_ptr(sB_ptr), make_shape(Int<BLK_N>{}, Int<BLK_K>{}), LayoutRight{});

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);

    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);

    Tensor gC_fake = make_tensor(make_gmem_ptr((ElementCompute*)nullptr), make_shape(Int<BLK_M>{}, Int<BLK_N>{}), LayoutRight{});
    Tensor tCrC = thr_mma.partition_fragment_C(gC_fake);
    Tensor tCcC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_N>{})));

    float local_p_acc[BLK_M];
    float local_n_acc[BLK_M];
    for (int i = 0; i < BLK_M; ++i) {
        local_p_acc[i] = -INFINITY;
        local_n_acc[i] = 0;
    }

    // Allocate small shared memory for n_acc reduction
    __shared__ float s_n_acc[BLK_M];
    if (threadIdx.x == 0) {
        for (int i = 0; i < BLK_M; ++i) {
            s_n_acc[i] = 0;
        }
    }
    __syncthreads();

    // Iterate over trg matrix in steps of BLK_N
    for (size_t n_start = 0; n_start < N; n_start += BLK_N) {
        
        clear(tCrC);

        for (size_t k_start = 0; k_start < D; k_start += BLK_K) {
            
            // Load sA (pred)
            int total_A = BLK_M * BLK_K;
            for (int i = thread_idx; i < total_A; i += blockDim.x) {
                int r = i / BLK_K;
                int c = i % BLK_K;
                size_t g_m = m_start + r;
                size_t g_k = k_start + c;
                if (g_m < M && g_k < D) {
                    sA_ptr[i] = pred_elem[g_m * D + g_k];
                } else {
                    sA_ptr[i] = (Element)0.0f;
                }
            }

            // Load sB (trg)
            int total_B = BLK_N * BLK_K;
            for (int i = thread_idx; i < total_B; i += blockDim.x) {
                int c = i / BLK_K;
                int r = i % BLK_K;
                size_t g_n = n_start + c;
                size_t g_k = k_start + r;
                if (g_n < N && g_k < D) {
                    sB_ptr[i] = trg_elem[g_n * D + g_k];
                } else {
                    sB_ptr[i] = (Element)0.0f;
                }
            }
            __syncthreads();

            // GEMM
            int K_MAX = size<2>(tCrA);
            for (int k = 0; k < K_MAX; ++k) {
                cute::copy(tCsA(_, _, k), tCrA(_, _, k));
                cute::copy(tCsB(_, _, k), tCrB(_, _, k));
                cute::gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), tCrC);
            }
            __syncthreads();
        }

        // Epilogue
        for (int m = 0; m < size<1>(tCrC); ++m) {
            for (int n = 0; n < size<2>(tCrC); ++n) {
                for (int i = 0; i < size<0>(tCrC); ++i) {
                    auto coord = tCcC(i, m, n);
                    int row = get<0>(coord);
                    int col = get<1>(coord);
                    size_t global_m = m_start + row;
                    size_t global_n = n_start + col;
                    
                    float val = tCrC(i, m, n);
                    
                    if (global_m < M && global_n < N) {
                        if (tixs[global_n] == truth[global_m]) {
                            atomicAdd(&s_n_acc[row], val);
                        }
                        
                        float prev_p = local_p_acc[row];
                        if (prev_p <= -INFINITY) {
                            local_p_acc[row] = val;
                        } else {
                            float diff = prev_p - val;
                            float max_val = prev_p > val ? prev_p : val;
                            float abs_diff = diff > 0 ? diff : -diff;
                            local_p_acc[row] = max_val + log1p(exp(-abs_diff));
                        }
                    }
                }
            }
        }
    }

    __syncthreads();

    // Use shared_mem as scratch space for reduction
    float* smem_p = (float*)shared_mem;
    for (int r = 0; r < BLK_M; ++r) {
        smem_p[r * blockDim.x + thread_idx] = local_p_acc[r];
    }
    __syncthreads();

    if (thread_idx < BLK_M) {
        float p = -INFINITY;
        for (int t = 0; t < blockDim.x; ++t) {
            float val = smem_p[thread_idx * blockDim.x + t];
            if (val > -INFINITY) {
                if (p <= -INFINITY) {
                    p = val;
                } else {
                    float diff = p - val;
                    float max_val = p > val ? p : val;
                    float abs_diff = diff > 0 ? diff : -diff;
                    p = max_val + log1p(exp(-abs_diff));
                }
            }
        }
        size_t global_m = m_start + thread_idx;
        if (global_m < M) {
            p_out[global_m] = (scalar_t)p;
            n_out[global_m] = (scalar_t)s_n_acc[thread_idx];
        }
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

    int threads = 128;
    int blocks = (M + BLK_M - 1) / BLK_M;
    size_t shared_mem_size = BLK_M * BLK_K * pred.element_size() + BLK_N * BLK_K * pred.element_size();

    // Ensure we have enough shared memory for the reduction
    size_t reduction_smem = BLK_M * threads * sizeof(float);
    if (reduction_smem > shared_mem_size) {
        shared_mem_size = reduction_smem;
    }

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
