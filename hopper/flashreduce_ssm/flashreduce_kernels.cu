#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "mamba_fwd_kernel.cuh"
#include "mamba_bwd_kernel.cuh"

std::vector<at::Tensor> mamba_fwd_cuda(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    int batch = a.size(0);
    int seqlen = a.size(1);
    int dim = a.size(2);

    int stride_b = seqlen * dim;
    int stride_s = dim;

    auto x = torch::empty_like(b);

    // Setup block tiling configuration
    constexpr int BLOCK_T = 256;
    constexpr int N_THREADS = 128;
    int num_blocks = (seqlen + BLOCK_T - 1) / BLOCK_T;

    auto block_prefixes_a = torch::empty({batch, num_blocks, dim}, a.options().dtype(torch::kFloat32));
    auto block_prefixes_b = torch::empty({batch, num_blocks, dim}, b.options().dtype(torch::kFloat32));

    // Launch configuration
    dim3 grid(dim, batch);
    dim3 block(N_THREADS);

    at::cuda::CUDAGuard device_guard{a.device()};
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "mamba_fwd_cuda", ([&] {
        flashreduce::mamba_fwd_kernel<scalar_t, BLOCK_T, N_THREADS><<<grid, block, 0, stream>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            block_prefixes_a.data_ptr<float>(),
            block_prefixes_b.data_ptr<float>(),
            seqlen,
            dim,
            stride_b,
            stride_s
        );
    }));

    return {x, block_prefixes_a, block_prefixes_b};
}

std::vector<at::Tensor> mamba_bwd_cuda(const at::Tensor& a, const at::Tensor& x, const at::Tensor& dy) {
    TORCH_CHECK(a.is_cuda() && x.is_cuda() && dy.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && x.is_contiguous() && dy.is_contiguous(), "Inputs must be contiguous");

    int batch = a.size(0);
    int seqlen = a.size(1);
    int dim = a.size(2);

    int stride_b = seqlen * dim;
    int stride_s = dim;

    auto da = torch::empty_like(a);
    auto db = torch::empty_like(a);

    constexpr int BLOCK_T = 256;
    constexpr int N_THREADS = 128;

    dim3 grid(dim, batch);
    dim3 block(N_THREADS);

    at::cuda::CUDAGuard device_guard{a.device()};
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "mamba_bwd_cuda", ([&] {
        flashreduce::mamba_bwd_kernel<scalar_t, BLOCK_T, N_THREADS><<<grid, block, 0, stream>>>(
            a.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            dy.data_ptr<scalar_t>(),
            da.data_ptr<scalar_t>(),
            db.data_ptr<scalar_t>(),
            seqlen,
            dim,
            stride_b,
            stride_s
        );
    }));

    return {da, db};
}
