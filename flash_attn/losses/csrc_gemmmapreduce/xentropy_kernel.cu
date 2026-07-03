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
    Element* sB_ptr = sA_ptr + 2 * BLK_M * BLK_K;

    // SMEM Swizzling
    // We apply Swizzle<3,3,3> to XOR-scramble the physical addresses.
    // This perfectly eliminates 2-way and 4-way shared memory bank conflicts
    // when using ldmatrix to pull tiles from SMEM into the L1 register file.
    auto sA_layout = composition(Swizzle<3,3,3>{}, make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{}));
    auto sB_layout = composition(Swizzle<3,3,3>{}, make_layout(make_shape(Int<BLK_N>{}, Int<BLK_K>{}), LayoutRight{}));

    auto sA0 = make_tensor(make_smem_ptr(sA_ptr), sA_layout);
    auto sA1 = make_tensor(make_smem_ptr(sA_ptr + BLK_M * BLK_K), sA_layout);
    //


    };
    auto sB0 = make_tensor(make_smem_ptr(sB_ptr), sB_layout);
    auto sB1 = make_tensor(make_smem_ptr(sB_ptr + BLK_N * BLK_K), sB_layout);
    //


    };

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA0 = thr_mma.partition_A(sA0);
    Tensor tCrA0 = thr_mma.partition_fragment_A(sA0);
    Tensor tCsA1 = thr_mma.partition_A(sA1);
    Tensor tCrA1 = thr_mma.partition_fragment_A(sA1);

    Tensor tCsB0 = thr_mma.partition_B(sB0);
    Tensor tCrB0 = thr_mma.partition_fragment_B(sB0);
    Tensor tCsB1 = thr_mma.partition_B(sB1);
    Tensor tCrB1 = thr_mma.partition_fragment_B(sB1);

    Tensor gC_fake = make_tensor(make_gmem_ptr((ElementCompute*)nullptr), make_shape(Int<BLK_M>{}, Int<BLK_N>{}), LayoutRight{});
    Tensor tCrC = thr_mma.partition_fragment_C(gC_fake);
    Tensor tCcC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_N>{})));

    // Tiled copy for Asynchronous GMEM -> SMEM pipeline
    // We use a 128-bit (16-byte) copy atom utilizing Ampere's cp.async.cg
    // This allows background loading without stalling the math pipeline.
    using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>;
    using TiledCopy = decltype(make_tiled_copy(CopyAtom{},
                                               make_layout(make_shape(Int<32>{}, Int<4>{}), LayoutRight{}),
                                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
    TiledCopy tiled_copy;
    auto thr_copy = tiled_copy.get_thread_slice(thread_idx);

    Tensor tAsA0 = thr_copy.partition_D(sA0);
    Tensor tAsA1 = thr_copy.partition_D(sA1);
    Tensor tBsB0 = thr_copy.partition_D(sB0);
    Tensor tBsB1 = thr_copy.partition_D(sB1);

    Tensor gA = make_tensor(make_gmem_ptr(pred_elem), make_shape(M, D), make_stride(D, _1{}));
    Tensor gB = make_tensor(make_gmem_ptr(trg_elem), make_shape(N, D), make_stride(D, _1{}));

    Tensor cA = make_identity_tensor(make_shape(M, D));
    Tensor cB = make_identity_tensor(make_shape(N, D));

    Tensor gA_blk = local_tile(gA, make_tile(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(m_block, _));
    Tensor cA_blk = local_tile(cA, make_tile(Int<BLK_M>{}, Int<BLK_K>{}), make_coord(m_block, _));

    float local_p_acc[BLK_M];
    float local_n_acc[BLK_M];
    for (int i = 0; i < BLK_M; ++i) {
        local_p_acc[i] = -INFINITY;
        local_n_acc[i] = 0;
    }

    // Iterate over trg matrix in steps of BLK_N
    for (size_t n_start = 0; n_start < N; n_start += BLK_N) {
        
        clear(tCrC);

        Tensor gB_blk = local_tile(gB, make_tile(Int<BLK_N>{}, Int<BLK_K>{}), make_coord(n_start / BLK_N, _));
        Tensor cB_blk = local_tile(cB, make_tile(Int<BLK_N>{}, Int<BLK_K>{}), make_coord(n_start / BLK_N, _));

        int K_TILES = (D + BLK_K - 1) / BLK_K;
        int k_tile = 0;

        // Prologue
        if (k_tile < K_TILES) {
            Tensor tAgA = thr_copy.partition_S(gA_blk(_, _, k_tile));
            Tensor tAcA = thr_copy.partition_S(cA_blk(_, _, k_tile));
            #pragma unroll
            for (int m = 0; m < size<1>(tAgA); ++m) {
                for (int k = 0; k < size<2>(tAgA); ++k) {
                    bool valid = get<0>(tAcA(0, m, k)) < M && get<1>(tAcA(0, m, k)) < D;
                    if (valid) cute::copy(tiled_copy, tAgA(_, m, k), tAsA0(_, m, k));
                    else       cute::clear(tAsA0(_, m, k));
                }
            }

            Tensor tBgB = thr_copy.partition_S(gB_blk(_, _, k_tile));
            Tensor tBcB = thr_copy.partition_S(cB_blk(_, _, k_tile));
            #pragma unroll
            for (int m = 0; m < size<1>(tBgB); ++m) {
                for (int k = 0; k < size<2>(tBgB); ++k) {
                    bool valid = get<0>(tBcB(0, m, k)) < N && get<1>(tBcB(0, m, k)) < D;
                    if (valid) cute::copy(tiled_copy, tBgB(_, m, k), tBsB0(_, m, k));
                    else       cute::clear(tBsB0(_, m, k));
                }
            }
            cute::cp_async_fence();
        }

        for (; k_tile < K_TILES; ++k_tile) {
            cute::cp_async_wait<0>();
            __syncthreads();

            int next_k_tile = k_tile + 1;
            if (next_k_tile < K_TILES) {
                Tensor tAgA = thr_copy.partition_S(gA_blk(_, _, next_k_tile));
                Tensor tAcA = thr_copy.partition_S(cA_blk(_, _, next_k_tile));
                #pragma unroll
                for (int m = 0; m < size<1>(tAgA); ++m) {
                    for (int k = 0; k < size<2>(tAgA); ++k) {
                        bool valid = get<0>(tAcA(0, m, k)) < M && get<1>(tAcA(0, m, k)) < D;
                        if (next_k_tile % 2 == 0) {
                            if (valid) cute::copy(tiled_copy, tAgA(_, m, k), tAsA0(_, m, k));
                            else       cute::clear(tAsA0(_, m, k));
                        } else {
                            if (valid) cute::copy(tiled_copy, tAgA(_, m, k), tAsA1(_, m, k));
                            else       cute::clear(tAsA1(_, m, k));
                        }
                    }
                }

                Tensor tBgB = thr_copy.partition_S(gB_blk(_, _, next_k_tile));
                Tensor tBcB = thr_copy.partition_S(cB_blk(_, _, next_k_tile));
                #pragma unroll
                for (int m = 0; m < size<1>(tBgB); ++m) {
                    for (int k = 0; k < size<2>(tBgB); ++k) {
                        bool valid = get<0>(tBcB(0, m, k)) < N && get<1>(tBcB(0, m, k)) < D;
                        if (next_k_tile % 2 == 0) {
                            if (valid) cute::copy(tiled_copy, tBgB(_, m, k), tBsB0(_, m, k));
                            else       cute::clear(tBsB0(_, m, k));
                        } else {
                            if (valid) cute::copy(tiled_copy, tBgB(_, m, k), tBsB1(_, m, k));
                            else       cute::clear(tBsB1(_, m, k));
                        }
                    }
                }
                cute::cp_async_fence();
            }

            int stage = k_tile % 2;
            if (stage == 0) {
                int K_MAX = size<2>(tCrA0);
                for (int k = 0; k < K_MAX; ++k) {
                    cute::copy(tCsA0(_, _, k), tCrA0(_, _, k));
                    cute::copy(tCsB0(_, _, k), tCrB0(_, _, k));
                    cute::gemm(tiled_mma, tCrA0(_, _, k), tCrB0(_, _, k), tCrC);
                }
            } else {
                int K_MAX = size<2>(tCrA1);
                for (int k = 0; k < K_MAX; ++k) {
                    cute::copy(tCsA1(_, _, k), tCrA1(_, _, k));
                    cute::copy(tCsB1(_, _, k), tCrB1(_, _, k));
                    cute::gemm(tiled_mma, tCrA1(_, _, k), tCrB1(_, _, k), tCrC);
                }
            }
        }
        
        // Epilogue Factorization: Fast Local Accumulation
        // Instead of atomicAdd into shared memory on every element (which causes severe contention),
        // we hold the running values (local_n_acc, local_p_acc) entirely within L1 thread registers.
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
                            local_n_acc[row] += val;
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

    // Epilogue Factorization: Block-wide parallel reduction
    // Use shared_mem as scratch space to reduce the L1 thread-local registers (local_p_acc, local_n_acc)
    // back into a final coalesced output vector.
    float* smem_p = (float*)shared_mem;
    float* smem_n = smem_p + blockDim.x; 

    for (int r = 0; r < BLK_M; ++r) {
        __syncthreads();
        smem_p[thread_idx] = local_p_acc[r];
        smem_n[thread_idx] = local_n_acc[r];
        __syncthreads();

        if (thread_idx == 0) {
            float p = -INFINITY;
            float sum_n = 0;
            for (int t = 0; t < blockDim.x; ++t) {
                sum_n += smem_n[t];
                float val = smem_p[t];
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
            size_t global_m = m_start + r;
            if (global_m < M && p > -INFINITY) {
                p_out[global_m] = (scalar_t)p;
                n_out[global_m] = (scalar_t)sum_n;
            }
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
    size_t shared_mem_size = 2 * (BLK_M * BLK_K * pred.element_size() + BLK_N * BLK_K * pred.element_size());

    // Ensure we have enough shared memory for the reduction
    size_t reduction_smem = 2 * threads * sizeof(float);
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
