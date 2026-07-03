with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_kernel.cu", "r") as f:
    content = f.read()

content = content.replace("AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, pred.scalar_type(), \"xentropy_cuda_kernel\", ([&] {",
"""AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, pred.scalar_type(), "xentropy_cuda_kernel", ([&] {
        cudaFuncSetAttribute(
            xentropy_cuda_kernel<scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_size);""")

with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_kernel.cu", "w") as f:
    f.write(content)
