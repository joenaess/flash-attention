#include <cute/tensor.hpp>
#include <iostream>
using namespace cute;

template <int BLK_M, int BLK_N, int BLK_K>
__global__ void xentropy_cuda_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ trg,
    float* __restrict__ p_out,
    size_t M, size_t N, size_t D) {
    
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using TiledMma = decltype(make_tiled_mma(mma_atom{}, make_layout(Shape<_2, _2, _1>{})));
    TiledMma tiled_mma;
}

int main() {
    std::cout << "Compiles!" << std::endl;
    return 0;
}
