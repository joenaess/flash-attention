#include <cute/tensor.hpp>
#include <iostream>
using namespace cute;
int main() {
    using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;
    using TiledMma = decltype(make_tiled_mma(mma_op{}));
    std::cout << "CuTe compiles!" << std::endl;
    return 0;
}
