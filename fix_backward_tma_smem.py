with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "r") as f:
    content = f.read()

content = content.replace("""    if (thread_idx == 0) {
        cute::mbarrier_init(*mbarrier_ptr, threads);
    }
    __syncthreads();""",
"""    __shared__ typename TmaTypes::TmaA smem_tma_pred;
    __shared__ typename TmaTypes::TmaB smem_tma_trg;
    if (thread_idx == 0) {
        cute::mbarrier_init(*mbarrier_ptr, threads);
        smem_tma_pred = params.tma_load_pred;
        smem_tma_trg = params.tma_load_trg;
    }
    __syncthreads();""")

content = content.replace("params.tma_load_pred.with(*mbarrier_ptr, 0)", "smem_tma_pred.with(*mbarrier_ptr, 0)")
content = content.replace("params.tma_load_trg.with(*mbarrier_ptr, 0)", "smem_tma_trg.with(*mbarrier_ptr, 0)")
content = content.replace("params.tma_load_pred.get_slice(Int<0>{});", "smem_tma_pred.get_slice(Int<0>{});")
content = content.replace("params.tma_load_trg.get_slice(Int<0>{});", "smem_tma_trg.get_slice(Int<0>{});")
content = content.replace("params.tma_load_trg.get_tma_tensor", "smem_tma_trg.get_tma_tensor")
content = content.replace("params.tma_load_pred.get_tma_tensor", "smem_tma_pred.get_tma_tensor")

with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "w") as f:
    f.write(content)
