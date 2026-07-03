with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "r") as f:
    content = f.read()

content = content.replace("void xentropy_cuda_backward(\n    at::Tensor grad_p,\n    at::Tensor grad_n,\n    at::Tensor pred,\n    at::Tensor trg,\n    at::Tensor truth,\n    at::Tensor tixs,\n    at::Tensor p_out,\n    at::Tensor grad_pred,\n    at::Tensor grad_trg) \n{", 
"""void launch_xentropy_backward_kernel(
    torch::Tensor grad_p,
    torch::Tensor grad_n,
    torch::Tensor pred,
    torch::Tensor trg,
    torch::Tensor truth,
    torch::Tensor tixs,
    torch::Tensor p_out,
    torch::Tensor grad_pred,
    torch::Tensor grad_trg,
    size_t M_,
    size_t N_,
    size_t D_) 
{
    grad_pred.zero_();
    grad_trg.zero_();""")

with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "w") as f:
    f.write(content)
