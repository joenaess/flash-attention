#include <torch/extension.h>
#include <vector>

// Forward declaration of the kernel launcher
void launch_xentropy_kernel(
    torch::Tensor pred,
    torch::Tensor trg,
    torch::Tensor truth,
    torch::Tensor tixs,
    torch::Tensor p_out,
    torch::Tensor n_out,
    size_t M,
    size_t N,
    size_t D);

std::vector<torch::Tensor> xentropy_cuda_forward(
    torch::Tensor pred,
    torch::Tensor trg,
    torch::Tensor truth,
    torch::Tensor tixs) {

    TORCH_CHECK(pred.device().is_cuda(), "pred must be a CUDA tensor");
    TORCH_CHECK(trg.device().is_cuda(), "trg must be a CUDA tensor");
    TORCH_CHECK(truth.device().is_cuda(), "truth must be a CUDA tensor");
    TORCH_CHECK(tixs.device().is_cuda(), "tixs must be a CUDA tensor");

    TORCH_CHECK(pred.is_contiguous(), "pred must be contiguous");
    TORCH_CHECK(trg.is_contiguous(), "trg must be contiguous");
    TORCH_CHECK(truth.is_contiguous(), "truth must be contiguous");
    TORCH_CHECK(tixs.is_contiguous(), "tixs must be contiguous");

    size_t M = pred.size(0);
    size_t D = pred.size(1);
    size_t N = trg.size(0);

    auto p_out = torch::zeros({(int64_t)M}, pred.options());
    auto n_out = torch::zeros({(int64_t)M}, pred.options());

    launch_xentropy_kernel(pred, trg, truth, tixs, p_out, n_out, M, N, D);

    return {p_out, n_out};
}

// Forward declaration of the backward kernel launcher
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
    size_t M,
    size_t N,
    size_t D);

std::vector<torch::Tensor> xentropy_cuda_backward(
    torch::Tensor grad_p,
    torch::Tensor grad_n,
    torch::Tensor pred,
    torch::Tensor trg,
    torch::Tensor truth,
    torch::Tensor tixs,
    torch::Tensor p_out) {

    TORCH_CHECK(grad_p.device().is_cuda(), "grad_p must be a CUDA tensor");
    TORCH_CHECK(grad_n.device().is_cuda(), "grad_n must be a CUDA tensor");
    
    size_t M = pred.size(0);
    size_t D = pred.size(1);
    size_t N = trg.size(0);

    auto grad_pred = torch::zeros_like(pred);
    auto grad_trg = torch::zeros_like(trg);

    launch_xentropy_backward_kernel(grad_p, grad_n, pred, trg, truth, tixs, p_out, grad_pred, grad_trg, M, N, D);

    return {grad_pred, grad_trg};
}

