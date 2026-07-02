#include <torch/extension.h>
#include <vector>

// Declarations from attention_cuda.cpp
void attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor l,
    torch::Tensor m_out);

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
    torch::Tensor dV);

// Declarations from xentropy_cuda.cpp
std::vector<torch::Tensor> xentropy_cuda_forward(
    torch::Tensor pred,
    torch::Tensor trg,
    torch::Tensor truth,
    torch::Tensor tixs);

std::vector<torch::Tensor> xentropy_cuda_backward(
    torch::Tensor grad_p,
    torch::Tensor grad_n,
    torch::Tensor pred,
    torch::Tensor trg,
    torch::Tensor truth,
    torch::Tensor tixs,
    torch::Tensor p_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "Attention forward pass (CUDA)");
    m.def("attention_backward", &attention_backward, "Attention backward pass (CUDA)");
    m.def("xentropy_forward", &xentropy_cuda_forward, "GeMMMapReduce XEntropy Forward (CUDA)");
    m.def("xentropy_backward", &xentropy_cuda_backward, "GeMMMapReduce XEntropy Backward (CUDA)");
}
