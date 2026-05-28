#pragma once
#include <torch/extension.h>

std::vector<at::Tensor> mamba_fwd_cuda(const at::Tensor& a, const at::Tensor& b);
std::vector<at::Tensor> mamba_bwd_cuda(const at::Tensor& a, const at::Tensor& x, const at::Tensor& dy);
