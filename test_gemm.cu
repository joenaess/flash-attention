#include <cute/tensor.hpp>
#include <iostream>
using namespace cute;

template <typename T>
__global__ void kernel() {
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using TiledMma = decltype(make_tiled_mma(MMA_Atom<MMA_Traits<mma_op>>{}, make_layout(Shape<_2, _2, _1>{})));
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor sA = make_tensor(make_smem_ptr((T*)nullptr), make_shape(Int<64>{}, Int<128>{}), LayoutRight{});
    Tensor sB = make_tensor(make_smem_ptr((T*)nullptr), make_shape(Int<128>{}, Int<64>{}), LayoutLeft{});
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCrC = thr_mma.partition_fragment_C(make_shape(Int<64>{}, Int<64>{}));
    
    cute::copy(tCsA, tCrA);
    cute::copy(tCsB, tCrB);
    cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
}

int main() {
    std::cout << "Compiles!" << std::endl;
    return 0;
}
