#pragma once

#include <cute/tensor.hpp>

namespace flashreduce {

using namespace cute;

template <typename T>
struct AffineState {
    T a; // Scale factor
    T b; // Bias/value factor

    __device__ __forceinline__ AffineState() : a(static_cast<T>(1)), b(static_cast<T>(0)) {}
    __device__ __forceinline__ AffineState(T scale, T bias) : a(scale), b(bias) {}
};

// compose: non-commutative composition rule defined in Section 8.5
// compose((a1, b1), (a2, b2)) -> (a1 * a2, a1 * b2 + b1)
template <typename T>
__device__ __forceinline__ AffineState<T> compose(const AffineState<T>& lhs, const AffineState<T>& rhs) {
    return AffineState<T>(lhs.a * rhs.a, lhs.a * rhs.b + lhs.b);
}

// scan_op: associative binary operator for sequential prefix scan
// Since x_t = a_t * x_{t-1} + b_t, we compose later states onto earlier ones:
// scan_op(earlier, later) = compose(later, earlier) -> (later.a * earlier.a, later.a * earlier.b + later.b)
template <typename T>
__device__ __forceinline__ AffineState<T> scan_op(const AffineState<T>& earlier, const AffineState<T>& later) {
    return compose(later, earlier);
}

} // namespace flashreduce
