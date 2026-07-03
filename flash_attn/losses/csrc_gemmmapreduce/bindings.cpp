#include <torch/extension.h>
#include <vector>

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
    m.def("xentropy_forward", &xentropy_cuda_forward, "GeMMMapReduce XEntropy Forward (CUDA)");
    m.def("xentropy_backward", &xentropy_cuda_backward, "GeMMMapReduce XEntropy Backward (CUDA)");
}
