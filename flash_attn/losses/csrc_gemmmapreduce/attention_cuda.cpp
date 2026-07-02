#include <torch/extension.h>

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
    size_t D);

void launch_attention_backward_kernel(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor l,
    torch::Tensor m_in,
    torch::Tensor dQ,
    torch::Tensor dK,
    torch::Tensor dV,
    size_t M,
    size_t N,
    size_t F,
    size_t D);

void attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor l,
    torch::Tensor m_out) {

    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    size_t M = Q.size(0);
    size_t F = Q.size(1);
    size_t N = K.size(0);
    size_t D = V.size(1);

    launch_attention_forward_kernel(Q, K, V, O, l, m_out, M, N, F, D);
}

void attention_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor l,
    torch::Tensor m_in,
    torch::Tensor dQ,
    torch::Tensor dK,
    torch::Tensor dV) {

    size_t M = Q.size(0);
    size_t F = Q.size(1);
    size_t N = K.size(0);
    size_t D = V.size(1);

    launch_attention_backward_kernel(Q, K, V, O, dO, l, m_in, dQ, dK, dV, M, N, F, D);
}

