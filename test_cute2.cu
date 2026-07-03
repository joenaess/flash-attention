#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>

template <typename T>
void do_something() {
    using mma_op = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
                                      cute::SM80_16x8x16_F32BF16BF16F32_TN,
                                      cute::SM80_16x8x16_F32F16F16F32_TN>;
    using TiledMma = decltype(cute::make_tiled_mma(cute::MMA_Atom<cute::MMA_Traits<mma_op>>{}, cute::make_layout(cute::Shape<cute::_2, cute::_2, cute::_1>{})));
    std::cout << "OK" << std::endl;
}
int main() {
    do_something<cutlass::bfloat16_t>();
    return 0;
}
