#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>
#include <ATen/cuda/Atomic.cuh>
#include <cute/tensor.hpp>

// Additional headers for TMA and WGMMA
#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

#ifndef XENTROPY_BLK_M
#define XENTROPY_BLK_M 64
#define XENTROPY_BLK_N 64
#define XENTROPY_BLK_K 128
#endif

#include <cutlass/numeric_types.h>

using namespace cute;

template <typename Element>
struct KernelTmaTypes {
    using Shape_MK = Shape<int32_t, int32_t>;
    using Stride_MK = Stride<int64_t, _1>;
    
    using SmemLayoutA = decltype(tile_to_shape(
        cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, Element, Int<XENTROPY_BLK_M>, Int<XENTROPY_BLK_K>>(),
        make_shape(Int<XENTROPY_BLK_M>{}, Int<XENTROPY_BLK_K>{})));

    using SmemLayoutB = decltype(tile_to_shape(
        cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, Element, Int<XENTROPY_BLK_N>, Int<XENTROPY_BLK_K>>(),
        make_shape(Int<XENTROPY_BLK_N>{}, Int<XENTROPY_BLK_K>{})));
        
    using SmemLayoutC = decltype(tile_to_shape(
        cutlass::gemm::collective::detail::ss_smem_selector<cute::GMMA::Major::K, Element, Int<XENTROPY_BLK_M>, Int<XENTROPY_BLK_N>>(),
        make_shape(Int<XENTROPY_BLK_M>{}, Int<XENTROPY_BLK_N>{})));

    using TmaA = decltype(make_tma_copy_A_sm90(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), Shape_MK{}, Stride_MK{}),
        SmemLayoutA{},
        Shape<Int<XENTROPY_BLK_M>, Int<XENTROPY_BLK_N>, Int<XENTROPY_BLK_K>>{}, 
        Shape<_1, _1, _1>{}));

    using TmaB = decltype(make_tma_copy_B_sm90(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), Shape_MK{}, Stride_MK{}),
        SmemLayoutB{},
        Shape<Int<XENTROPY_BLK_M>, Int<XENTROPY_BLK_N>, Int<XENTROPY_BLK_K>>{}, 
        Shape<_1, _1, _1>{}));
};

template <typename scalar_t>
struct BackwardParams {
    using Element = std::conditional_t<
        std::is_same_v<scalar_t, at::Half>, cutlass::half_t,
        std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, cutlass::bfloat16_t, scalar_t>>;
    using TmaTypes = KernelTmaTypes<Element>;
    
    const scalar_t* __restrict__ grad_p;
    const scalar_t* __restrict__ grad_n;
    const scalar_t* __restrict__ pred;
    const scalar_t* __restrict__ trg;
    const int64_t* __restrict__ truth;
    const int64_t* __restrict__ tixs;
    const scalar_t* __restrict__ p_out;
    scalar_t* __restrict__ grad_pred;
    scalar_t* __restrict__ grad_trg;
    
    typename TmaTypes::TmaA tma_load_pred;
    typename TmaTypes::TmaB tma_load_trg;
    
    int32_t M, N, D;
};

template <typename scalar_t>
__global__ void __launch_bounds__(128) xentropy_cuda_backward_kernel(BackwardParams<scalar_t> params) {
    using Element = typename BackwardParams<scalar_t>::Element;
    using ElementCompute = float;
    using TmaTypes = KernelTmaTypes<Element>;
    
    // WGMMA setup
    using TileShape_MNK = Shape<Int<XENTROPY_BLK_M>, Int<XENTROPY_BLK_N>, Int<XENTROPY_BLK_K>>;
    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementCompute, TileShape_MNK>(),
        Layout<Shape<_1, _1, _1>>{}));
    TiledMma tiled_mma;
    
    int thread_idx = threadIdx.x;
    int m_block = blockIdx.x;
    int n_block = blockIdx.y;
    
    size_t m_start = m_block * XENTROPY_BLK_M;
    size_t n_start = n_block * XENTROPY_BLK_N;

    if (m_start >= params.M || n_start >= params.N) return;

    extern __shared__ char shared_mem[];
    Element* sA_ptr = (Element*)shared_mem;
    Element* sB_ptr = sA_ptr + cute::cosize_v<typename TmaTypes::SmemLayoutA>;
    Element* sC_ptr = sB_ptr + cute::cosize_v<typename TmaTypes::SmemLayoutB>;
    
    Tensor sA = make_tensor(make_smem_ptr(sA_ptr), typename TmaTypes::SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(sB_ptr), typename TmaTypes::SmemLayoutB{});
    Tensor sC = make_tensor(make_smem_ptr(sC_ptr), typename TmaTypes::SmemLayoutC{});
    
    // Mbarrier for TMA
    using SharedStorage = cute::uint64_t;
    SharedStorage* mbarrier_ptr = reinterpret_cast<SharedStorage*>(sC_ptr + cute::cosize_v<typename TmaTypes::SmemLayoutC>);
    __shared__ typename TmaTypes::TmaA smem_tma_pred;
    __shared__ typename TmaTypes::TmaB smem_tma_trg;
    
    if (threadIdx.x == 0) {
        cute::initialize_barrier(*mbarrier_ptr, blockDim.x);
        smem_tma_pred = params.tma_load_pred;
        smem_tma_trg = params.tma_load_trg;
    }
    __syncthreads();
    
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrC_logits = thr_mma.partition_fragment_C(make_identity_tensor(make_shape(Int<XENTROPY_BLK_M>{}, Int<XENTROPY_BLK_N>{})));
    clear(tCrC_logits);

    Tensor mPred = smem_tma_pred.get_tma_tensor(make_shape(params.M, params.D));
    Tensor mTrg = smem_tma_trg.get_tma_tensor(make_shape(params.N, params.D));
    
    Tensor gPred = local_tile(mPred, make_tile(Int<XENTROPY_BLK_M>{}, Int<XENTROPY_BLK_K>{}), make_coord(m_block, _));
    Tensor gTrg = local_tile(mTrg, make_tile(Int<XENTROPY_BLK_N>{}, Int<XENTROPY_BLK_K>{}), make_coord(n_block, _));
    
    auto cta_tma_pred = smem_tma_pred.get_slice(Int<0>{});
    Tensor tAgA = cta_tma_pred.partition_S(gPred);
    Tensor tAsA = cta_tma_pred.partition_D(sA);
    
    auto cta_tma_trg = smem_tma_trg.get_slice(Int<0>{});
    Tensor tBgB = cta_tma_trg.partition_S(gTrg);
    Tensor tBsB = cta_tma_trg.partition_D(sB);
    
    int lane_predicate = cute::elect_one_sync();

    // 1. Compute Logits (h_val) via WGMMA
    int K_TILES = (params.D + XENTROPY_BLK_K - 1) / XENTROPY_BLK_K;
    int phase = 0;
    
    for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
        
        clear(tCrC_logits);
        for (int k_tile_logits = 0; k_tile_logits < K_TILES; ++k_tile_logits) {
            if (thread_idx == 0) {
                cute::set_barrier_transaction_bytes(*mbarrier_ptr, 
                                                size(gPred(_, _, k_tile_logits)) * sizeof(Element) + 
                                                size(gTrg(_, _, k_tile_logits)) * sizeof(Element));
                cute::copy(smem_tma_pred.with(*mbarrier_ptr, 0), tAgA(_, _, _, k_tile_logits), tAsA);
                cute::copy(smem_tma_trg.with(*mbarrier_ptr, 0), tBgB(_, _, _, k_tile_logits), tBsB);
            }
            cute::wait_barrier(*mbarrier_ptr, phase); 
            phase ^= 1;
            __syncthreads();
            
            cute::gemm(tiled_mma, tCsA, tCsB, tCrC_logits);
        }
        
        // 2. MapReduce: Compute probabilities and gradients dh
        Tensor tCcC = thr_mma.partition_C(make_identity_tensor(make_shape(Int<XENTROPY_BLK_M>{}, Int<XENTROPY_BLK_N>{})));
        
        for (int m = 0; m < size<1>(tCrC_logits); ++m) {
            for (int n = 0; n < size<2>(tCrC_logits); ++n) {
                for (int i = 0; i < size<0>(tCrC_logits); ++i) {
                    auto coord = tCcC(i, m, n);
                    int row = get<0>(coord);
                    int col = get<1>(coord);
                    size_t global_m = m_start + row;
                    size_t global_n = n_start + col;
                    
                    float dh = 0;
                    if (global_m < params.M && global_n < params.N) {
                        float h_val = tCrC_logits(i, m, n);
                        float g_p = (float)params.grad_p[global_m];
                        float p_o = (float)params.p_out[global_m];
                        dh = g_p * exp(h_val - p_o);
                        
                        if (params.tixs[global_n] == params.truth[global_m]) {
                            dh += (float)params.grad_n[global_m];
                        }
                    }
                    
                    sC(row, col) = (Element)dh;
                }
            }
        }
        __syncthreads();

        // 3. Compute grad_pred = dh @ trg and grad_trg = dh^T @ pred
        using TiledMma_GP = decltype(cute::make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementCompute, Shape<Int<XENTROPY_BLK_M>, Int<XENTROPY_BLK_K>, Int<XENTROPY_BLK_N>>, cute::GMMA::Major::K, cute::GMMA::Major::MN>(),
            Layout<Shape<_1, _1, _1>>{}));
        TiledMma_GP tiled_mma_gp;
        auto thr_mma_gp = tiled_mma_gp.get_thread_slice(thread_idx);
        
        using TiledMma_GT = decltype(cute::make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementCompute, Shape<Int<XENTROPY_BLK_N>, Int<XENTROPY_BLK_K>, Int<XENTROPY_BLK_M>>, cute::GMMA::Major::MN, cute::GMMA::Major::MN>(),
            Layout<Shape<_1, _1, _1>>{}));
        TiledMma_GT tiled_mma_gt;
        auto thr_mma_gt = tiled_mma_gt.get_thread_slice(thread_idx);
        
        Tensor sA_T = make_tensor(sA.data(), make_layout(make_shape(get<1>(sA.layout().shape()), get<0>(sA.layout().shape())), make_stride(get<1>(sA.layout().stride()), get<0>(sA.layout().stride()))));
        Tensor sB_T = make_tensor(sB.data(), make_layout(make_shape(get<1>(sB.layout().shape()), get<0>(sB.layout().shape())), make_stride(get<1>(sB.layout().stride()), get<0>(sB.layout().stride()))));
        Tensor sC_T = make_tensor(sC.data(), make_layout(make_shape(get<1>(sC.layout().shape()), get<0>(sC.layout().shape())), make_stride(get<1>(sC.layout().stride()), get<0>(sC.layout().stride()))));
        
        Tensor tCsA_dh = thr_mma_gp.partition_A(sC);
        Tensor tCsB_trg = thr_mma_gp.partition_B(sB_T);
        Tensor tCrC_grad_pred = thr_mma_gp.partition_fragment_C(make_identity_tensor(make_shape(Int<XENTROPY_BLK_M>{}, Int<XENTROPY_BLK_K>{})));
        Tensor tCcC_gp = thr_mma_gp.partition_C(make_identity_tensor(make_shape(Int<XENTROPY_BLK_M>{}, Int<XENTROPY_BLK_K>{})));

        Tensor tCsA_dh_T = thr_mma_gt.partition_A(sC_T);
        Tensor tCsB_pred = thr_mma_gt.partition_B(sA_T);
        Tensor tCrC_grad_trg = thr_mma_gt.partition_fragment_C(make_identity_tensor(make_shape(Int<XENTROPY_BLK_N>{}, Int<XENTROPY_BLK_K>{})));
        Tensor tCcC_gt = thr_mma_gt.partition_C(make_identity_tensor(make_shape(Int<XENTROPY_BLK_N>{}, Int<XENTROPY_BLK_K>{})));
        
        clear(tCrC_grad_pred);
        clear(tCrC_grad_trg);
        
        if (thread_idx == 0) {
            cute::set_barrier_transaction_bytes(*mbarrier_ptr, 
                                            size(gPred(_, _, k_tile)) * sizeof(Element) + 
                                            size(gTrg(_, _, k_tile)) * sizeof(Element));
            cute::copy(smem_tma_pred.with(*mbarrier_ptr, 0), tAgA(_, _, _, k_tile), tAsA);
            cute::copy(smem_tma_trg.with(*mbarrier_ptr, 0), tBgB(_, _, _, k_tile), tBsB);
        }
        cute::wait_barrier(*mbarrier_ptr, phase); 
        phase ^= 1;
        __syncthreads();
        
        cute::gemm(tiled_mma_gp, tCsA_dh, tCsB_trg, tCrC_grad_pred);
        cute::gemm(tiled_mma_gt, tCsA_dh_T, tCsB_pred, tCrC_grad_trg);
        
        cute::warpgroup_wait<0>();
        
        for (int m = 0; m < size<1>(tCrC_grad_pred); ++m) {
            for (int n = 0; n < size<2>(tCrC_grad_pred); ++n) {
                for (int i = 0; i < size<0>(tCrC_grad_pred); ++i) {
                    auto coord = tCcC_gp(i, m, n);
                    int row = get<0>(coord);
                    int col = get<1>(coord);
                    size_t global_m = m_start + row;
                    size_t global_k = k_tile * XENTROPY_BLK_K + col;
                    if (global_m < params.M && global_k < params.D) {
                        gpuAtomicAdd(&params.grad_pred[global_m * params.D + global_k], (scalar_t)tCrC_grad_pred(i, m, n));
                    }
                }
            }
        }
        
        for (int m = 0; m < size<1>(tCrC_grad_trg); ++m) {
            for (int n = 0; n < size<2>(tCrC_grad_trg); ++n) {
                for (int i = 0; i < size<0>(tCrC_grad_trg); ++i) {
                    auto coord = tCcC_gt(i, m, n);
                    int row = get<0>(coord);
                    int col = get<1>(coord);
                    size_t global_n = n_start + row;
                    size_t global_k = k_tile * XENTROPY_BLK_K + col;
                    if (global_n < params.N && global_k < params.D) {
                        gpuAtomicAdd(&params.grad_trg[global_n * params.D + global_k], (scalar_t)tCrC_grad_trg(i, m, n));
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename Element>
void dispatch_xentropy_backward(
    at::Tensor grad_p, at::Tensor grad_n, at::Tensor pred, at::Tensor trg,
    at::Tensor truth, at::Tensor tixs, at::Tensor p_out,
    at::Tensor grad_pred, at::Tensor grad_trg,
    int M, int N, int D, dim3 blocks, int threads)
{
    using TmaTypes = KernelTmaTypes<Element>;
        
    typename TmaTypes::TmaA tma_load_pred = make_tma_copy_A_sm90(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(pred.data_ptr<scalar_t>())), 
                    make_shape(static_cast<int32_t>(M), static_cast<int32_t>(D)), 
                    make_stride(static_cast<int64_t>(D), _1{})),
        typename TmaTypes::SmemLayoutA{},
        Shape<Int<XENTROPY_BLK_M>, Int<XENTROPY_BLK_N>, Int<XENTROPY_BLK_K>>{}, 
        Shape<_1, _1, _1>{});

    typename TmaTypes::TmaB tma_load_trg = make_tma_copy_B_sm90(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(trg.data_ptr<scalar_t>())), 
                    make_shape(static_cast<int32_t>(N), static_cast<int32_t>(D)), 
                    make_stride(static_cast<int64_t>(D), _1{})),
        typename TmaTypes::SmemLayoutB{},
        Shape<Int<XENTROPY_BLK_M>, Int<XENTROPY_BLK_N>, Int<XENTROPY_BLK_K>>{}, 
        Shape<_1, _1, _1>{});
        
    BackwardParams<scalar_t> params{
        grad_p.data_ptr<scalar_t>(),
        grad_n.data_ptr<scalar_t>(),
        pred.data_ptr<scalar_t>(),
        trg.data_ptr<scalar_t>(),
        truth.data_ptr<int64_t>(),
        tixs.data_ptr<int64_t>(),
        p_out.data_ptr<scalar_t>(),
        grad_pred.data_ptr<scalar_t>(),
        grad_trg.data_ptr<scalar_t>(),
        tma_load_pred,
        tma_load_trg,
        static_cast<int32_t>(M),
        static_cast<int32_t>(N),
        static_cast<int32_t>(D)
    };
    
    size_t shared_mem_size = sizeof(Element) * (
        cute::cosize_v<typename TmaTypes::SmemLayoutA> + 
        cute::cosize_v<typename TmaTypes::SmemLayoutB> + 
        cute::cosize_v<typename TmaTypes::SmemLayoutC>
    ) + sizeof(cute::uint64_t); // for mbarrier
    
    // Ensure shared memory is configured correctly for Hopper
    cudaFuncSetAttribute(
        xentropy_cuda_backward_kernel<scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size);
        
    xentropy_cuda_backward_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(params);
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
    size_t M_,
    size_t N_,
    size_t D_) 
{
    grad_pred.zero_();
    grad_trg.zero_();
    int M = pred.size(0);
    int D = pred.size(1);
    int N = trg.size(0);
    
    int threads = 128; // 1 Warpgroup
    dim3 blocks((M + XENTROPY_BLK_M - 1) / XENTROPY_BLK_M, (N + XENTROPY_BLK_N - 1) / XENTROPY_BLK_N);
    
    if (pred.scalar_type() == at::ScalarType::Half) {
        dispatch_xentropy_backward<c10::Half, cutlass::half_t>(
            grad_p, grad_n, pred, trg, truth, tixs, p_out, grad_pred, grad_trg, M, N, D, blocks, threads);
    } else if (pred.scalar_type() == at::ScalarType::BFloat16) {
        dispatch_xentropy_backward<c10::BFloat16, cutlass::bfloat16_t>(
            grad_p, grad_n, pred, trg, truth, tixs, p_out, grad_pred, grad_trg, M, N, D, blocks, threads);
    } else {
        TORCH_CHECK(false, "xentropy_cuda_backward_kernel only supports Half and BFloat16 on SM90");
    }
}
