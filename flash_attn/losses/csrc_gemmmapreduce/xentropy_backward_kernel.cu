#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>
#include <ATen/cuda/Atomic.cuh>
#include <cute/tensor.hpp>

#ifndef BLK_M
#define BLK_M 64
#define BLK_N 64
#define BLK_K 128
#endif

#include <cutlass/numeric_types.h>

using namespace cute;

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
    size_t M, size_t N, size_t D) 
{
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
    int n_block = blockIdx.y;
    
    size_t m_start = m_block * BLK_M;
    size_t n_start = n_block * BLK_N;

    if (m_start >= M || n_start >= N) return;

    extern __shared__ char shared_mem[];
    Element* sA_ptr = (Element*)shared_mem;
    Element* sB_ptr = sA_ptr + BLK_M * BLK_K;
    Element* sC_ptr = sB_ptr + BLK_N * BLK_K;
    
    // sA is Row-Major (BLK_M, BLK_K)
    Tensor sA = make_tensor(make_smem_ptr(sA_ptr), make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{});
    // sB is Col-Major (BLK_K, BLK_N) for B in GEMM (to match trg Row-Major)
    Tensor sB = make_tensor(make_smem_ptr(sB_ptr), make_shape(Int<BLK_N>{}, Int<BLK_K>{}), LayoutLeft{});
    // sC is Row-Major (BLK_M, BLK_N) to hold dh
    Tensor sC = make_tensor(make_smem_ptr(sC_ptr), make_shape(Int<BLK_M>{}, Int<BLK_N>{}), LayoutRight{});

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);

    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);

    Tensor gC_fake = make_tensor(make_gmem_ptr((ElementCompute*)nullptr), make_shape(Int<BLK_M>{}, Int<BLK_N>{}), LayoutRight{});
    Tensor tCrC_logits = thr_mma.partition_fragment_C(gC_fake);
    clear(tCrC_logits);

    // ==========================================
    // 1. Compute Logits (h_val)
    // ==========================================
    for (size_t k_start = 0; k_start < D; k_start += BLK_K) {
        
        // Load sA (pred) vectorized
        int total_A_vec = (BLK_M * BLK_K) / 8;
        auto* sA_vec = reinterpret_cast<uint4*>(sA_ptr);
        auto* pred_vec = reinterpret_cast<const uint4*>(pred);
        for (int i = thread_idx; i < total_A_vec; i += blockDim.x) {
            int r = i / (BLK_K / 8);
            int c_vec = i % (BLK_K / 8);
            size_t g_m = m_start + r;
            size_t g_k_vec = (k_start / 8) + c_vec;
            if (g_m < M && g_k_vec < (D / 8)) {
                sA_vec[i] = pred_vec[g_m * (D / 8) + g_k_vec];
            } else {
                sA_vec[i] = {0, 0, 0, 0};
            }
        }

        // Load sB (trg) vectorized
        int total_B_vec = (BLK_N * BLK_K) / 8;
        auto* sB_vec = reinterpret_cast<uint4*>(sB_ptr);
        auto* trg_vec = reinterpret_cast<const uint4*>(trg);
        for (int i = thread_idx; i < total_B_vec; i += blockDim.x) {
            int c = i / (BLK_K / 8);
            int r_vec = i % (BLK_K / 8);
            size_t g_n = n_start + c;
            size_t g_k_vec = (k_start / 8) + r_vec;
            if (g_n < N && g_k_vec < (D / 8)) {
                sB_vec[i] = trg_vec[g_n * (D / 8) + g_k_vec];
            } else {
                sB_vec[i] = {0, 0, 0, 0};
            }
        }
        __syncthreads();

        int K_MAX = size<2>(tCrA);
        for (int k = 0; k < K_MAX; ++k) {
            cute::copy(tCsA(_, _, k), tCrA(_, _, k));
            cute::copy(tCsB(_, _, k), tCrB(_, _, k));
            cute::gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), tCrC_logits);
        }
        __syncthreads();
    }

    // ==========================================
    // 2. Compute dh and write to sC
    // ==========================================
    Tensor tCcC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_N>{})));
    
    for (int m = 0; m < size<1>(tCrC_logits); ++m) {
        for (int n = 0; n < size<2>(tCrC_logits); ++n) {
            for (int i = 0; i < size<0>(tCrC_logits); ++i) {
                auto coord = tCcC(i, m, n);
                int row = get<0>(coord);
                int col = get<1>(coord);
                size_t global_m = m_start + row;
                size_t global_n = n_start + col;
                
                float dh = 0;
                if (global_m < M && global_n < N) {
                    float h_val = tCrC_logits(i, m, n);
                    float g_p = (float)grad_p[global_m];
                    float p_o = (float)p_out[global_m];
                    dh = g_p * exp(h_val - p_o);
                    
                    if (tixs[global_n] == truth[global_m]) {
                        dh += (float)grad_n[global_m];
                    }
                }
                
                // Write dh directly to shared memory using the logical coordinate!
                // Because sC is Row-Major (BLK_M, BLK_N), sC(row, col) is mapped to row * BLK_N + col
                sC_ptr[row * BLK_N + col] = (Element)dh;
            }
        }
    }
    __syncthreads();

    // ==========================================
    // 3. Compute grad_pred and grad_trg
    // ==========================================
    // For grad_pred += dh @ trg:
    // A = dh (sC, Row-Major BLK_M x BLK_N) -> matches sA format (M x K where K=BLK_N)
    // B = trg (global) -> load into sB (Col-Major BLK_N x BLK_K where K=BLK_K is now N=BLK_N)
    
    // Actually, wait! dh is (BLK_M x BLK_N). trg is (BLK_N x BLK_K).
    // The inner dimension for GEMM is BLK_N!
    // But BLK_N is 64. Our tiled_mma expects the inner dimension to be partitioned.
    // If we just use scalar loops for the backward gradients, it might be simpler, 
    // OR we can map a new TiledMMA!
    // But since `dh` is already in shared memory, we can just use simple manual thread-level math 
    // for `grad_pred` and `grad_trg` if we don't want to instantiate another `TiledMMA`.
    // Wait, the user said "Issue a reverse cute::gemm to compute the gradients"
    // To do this, we need `tiled_mma` where K=BLK_N. 
    // But `make_tiled_mma` doesn't strictly depend on BLK_K, it just defines the atom!
    
    // grad_pred tile:
    Tensor sA_dh = make_tensor(make_smem_ptr(sC_ptr), make_shape(Int<BLK_M>{}, Int<BLK_N>{}), LayoutRight{});
    
    TiledMMA tiled_mma_gp = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                           Layout<Shape<_2, _2, _1>>{},
                                           Tile<Int<BLK_M>, Int<BLK_K>, Int<BLK_N>>{});
    auto thr_mma_gp = tiled_mma_gp.get_thread_slice(thread_idx);
    
    for (size_t k_start = 0; k_start < D; k_start += BLK_K) {
        
        // We already have dh in sA_dh. We need to load trg into sB_trg.
        // trg is (BLK_N, BLK_K).
        // B must be partitioned as (N_gemm, K_gemm) = (BLK_K, BLK_N).
        Tensor sB_trg = make_tensor(make_smem_ptr(sB_ptr), make_shape(Int<BLK_K>{}, Int<BLK_N>{}), LayoutLeft{});
        
        int total_B_vec_trg = (BLK_N * BLK_K) / 8;
        auto* sB_vec_trg = reinterpret_cast<uint4*>(sB_ptr);
        auto* trg_vec_bwd = reinterpret_cast<const uint4*>(trg);
        for (int i = thread_idx; i < total_B_vec_trg; i += blockDim.x) {
            int c = i / (BLK_K / 8);
            int r_vec = i % (BLK_K / 8);
            size_t g_n = n_start + c;
            size_t g_k_vec = (k_start / 8) + r_vec;
            if (g_n < N && g_k_vec < (D / 8)) {
                sB_vec_trg[i] = trg_vec_bwd[g_n * (D / 8) + g_k_vec];
            } else {
                sB_vec_trg[i] = {0, 0, 0, 0};
            }
        }
        __syncthreads();

        Tensor tCsA_dh = thr_mma_gp.partition_A(sA_dh);
        Tensor tCrA_dh = thr_mma_gp.partition_fragment_A(sA_dh);
        Tensor tCsB_trg = thr_mma_gp.partition_B(sB_trg);
        Tensor tCrB_trg = thr_mma_gp.partition_fragment_B(sB_trg);
        
        Tensor gC_grad_pred = make_tensor(make_gmem_ptr((ElementCompute*)nullptr), make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{});
        Tensor tCrC_grad_pred = thr_mma_gp.partition_fragment_C(gC_grad_pred);
        clear(tCrC_grad_pred);

        int K_MAX_dh = size<2>(tCrA_dh);
        for (int k = 0; k < K_MAX_dh; ++k) {
            cute::copy(tCsA_dh(_, _, k), tCrA_dh(_, _, k));
            cute::copy(tCsB_trg(_, _, k), tCrB_trg(_, _, k));
            cute::gemm(tiled_mma_gp, tCrA_dh(_, _, k), tCrB_trg(_, _, k), tCrC_grad_pred);
        }
        __syncthreads();

        // Write to global grad_pred
        Tensor tCcC_gp = thr_mma_gp.partition_C(make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_K>{})));
        for (int m = 0; m < size<1>(tCrC_grad_pred); ++m) {
            for (int n = 0; n < size<2>(tCrC_grad_pred); ++n) {
                for (int i = 0; i < size<0>(tCrC_grad_pred); ++i) {
                    auto coord = tCcC_gp(i, m, n);
                    int row = get<0>(coord);
                    int col = get<1>(coord);
                    size_t global_m = m_start + row;
                    size_t global_k = k_start + col;
                    if (global_m < M && global_k < D) {
                        gpuAtomicAdd(&grad_pred[global_m * D + global_k], (scalar_t)tCrC_grad_pred(i, m, n));
                    }
                }
            }
        }
        
        // ==========================================
        // Now grad_trg += dh^T @ pred
        // ==========================================
        // dh is (BLK_M, BLK_N). dh^T is (BLK_N, BLK_M).
        // pred is (BLK_M, BLK_K).
        // sA for dh^T: we can view sC as Col-Major (BLK_M, BLK_N), which makes it (BLK_N, BLK_M) Row-Major?
        // Wait, if sC is Row-Major (BLK_M, BLK_N), its transpose is Col-Major (BLK_M, BLK_N).
        // But TiledMMA TN expects A to be Row-Major!
        // We can just load dh^T into sA_dh_T properly!
        Tensor sA_dh_T = make_tensor(make_smem_ptr(sC_ptr), make_shape(Int<BLK_N>{}, Int<BLK_M>{}), LayoutLeft{});
        
        // Load pred into sB_pred. pred is (BLK_M, BLK_K) Row-Major.
        // We can reuse sA_ptr because it is no longer needed in the second pass.
        Element* sB_pred_ptr = sA_ptr;
        Tensor sB_pred = make_tensor(make_smem_ptr(sB_pred_ptr), make_shape(Int<BLK_K>{}, Int<BLK_M>{}), LayoutLeft{});
        
        TiledMMA tiled_mma_gt = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                               Layout<Shape<_2, _2, _1>>{},
                                               Tile<Int<BLK_N>, Int<BLK_K>, Int<BLK_M>>{});
        auto thr_mma_gt = tiled_mma_gt.get_thread_slice(thread_idx);
        
        int total_A_pred_vec = (BLK_M * BLK_K) / 8;
        auto* sA_pred_vec = reinterpret_cast<uint4*>(sB_pred_ptr);
        auto* pred_vec_bwd = reinterpret_cast<const uint4*>(pred);
        for (int i = thread_idx; i < total_A_pred_vec; i += blockDim.x) {
            int c = i / (BLK_K / 8);
            int r_vec = i % (BLK_K / 8);
            size_t g_m = m_start + c;
            size_t g_k_vec = (k_start / 8) + r_vec;
            if (g_m < M && g_k_vec < (D / 8)) {
                sA_pred_vec[i] = pred_vec_bwd[g_m * (D / 8) + g_k_vec];
            } else {
                sA_pred_vec[i] = {0, 0, 0, 0};
            }
        }
        __syncthreads();

        Tensor tCsA_dh_T = thr_mma_gt.partition_A(sA_dh_T);
        Tensor tCrA_dh_T = thr_mma_gt.partition_fragment_A(sA_dh_T);
        Tensor tCsB_pred = thr_mma_gt.partition_B(sB_pred);
        Tensor tCrB_pred = thr_mma_gt.partition_fragment_B(sB_pred);
        
        Tensor gC_grad_trg = make_tensor(make_gmem_ptr((ElementCompute*)nullptr), make_shape(Int<BLK_N>{}, Int<BLK_K>{}), LayoutRight{});
        Tensor tCrC_grad_trg = thr_mma_gt.partition_fragment_C(gC_grad_trg);
        clear(tCrC_grad_trg);

        int K_MAX_dh_T = size<2>(tCrA_dh_T);
        for (int k = 0; k < K_MAX_dh_T; ++k) {
            cute::copy(tCsA_dh_T(_, _, k), tCrA_dh_T(_, _, k));
            cute::copy(tCsB_pred(_, _, k), tCrB_pred(_, _, k));
            cute::gemm(tiled_mma_gt, tCrA_dh_T(_, _, k), tCrB_pred(_, _, k), tCrC_grad_trg);
        }
        __syncthreads();

        // Write to global grad_trg
        Tensor tCcC_gt = thr_mma_gt.partition_C(make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{})));
        for (int m = 0; m < size<1>(tCrC_grad_trg); ++m) {
            for (int n = 0; n < size<2>(tCrC_grad_trg); ++n) {
                for (int i = 0; i < size<0>(tCrC_grad_trg); ++i) {
                    auto coord = tCcC_gt(i, m, n);
                    int row = get<0>(coord);
                    int col = get<1>(coord);
                    size_t global_n = n_start + row;
                    size_t global_k = k_start + col;
                    if (global_n < N && global_k < D) {
                        gpuAtomicAdd(&grad_trg[global_n * D + global_k], (scalar_t)tCrC_grad_trg(i, m, n));
                    }
                }
            }
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

    // Ensure we initialize grad_pred and grad_trg to 0 since we use atomicAdd
    grad_pred.zero_();
    grad_trg.zero_();

    int threads = 128;
    dim3 blocks((M + BLK_M - 1) / BLK_M, (N + BLK_N - 1) / BLK_N);
    
    // sA + sB + sC
    size_t shared_mem_size = BLK_M * BLK_K * pred.element_size() + 
                             BLK_N * BLK_K * pred.element_size() + 
                             BLK_M * BLK_N * pred.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, pred.scalar_type(), "xentropy_cuda_backward_kernel", ([&] {
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
