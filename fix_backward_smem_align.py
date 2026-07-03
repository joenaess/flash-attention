with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "r") as f:
    content = f.read()

content = content.replace("""    // Initialize mbarrier
    uint64_t* mbarrier_ptr = reinterpret_cast<uint64_t*>(shared_mem);
    if (thread_idx == 0) {
        cute::mbarrier_init(*mbarrier_ptr, threads);
    }
    __syncthreads();
    
    Element* sC_ptr = reinterpret_cast<Element*>(mbarrier_ptr + 1);
    Element* sA_ptr = sC_ptr + cute::cosize_v<typename TmaTypes::SmemLayoutC>;
    Element* sB_ptr = sA_ptr + cute::cosize_v<typename TmaTypes::SmemLayoutA>;""",
"""    // Allocate pointers (ensure 16-byte alignment for TMA by putting mbarrier at the end)
    Element* sA_ptr = reinterpret_cast<Element*>(shared_mem);
    Element* sB_ptr = sA_ptr + cute::cosize_v<typename TmaTypes::SmemLayoutA>;
    Element* sC_ptr = sB_ptr + cute::cosize_v<typename TmaTypes::SmemLayoutB>;
    
    // mbarrier can be at any 8-byte aligned address
    uint64_t* mbarrier_ptr = reinterpret_cast<uint64_t*>(sC_ptr + cute::cosize_v<typename TmaTypes::SmemLayoutC>);
    
    if (thread_idx == 0) {
        cute::mbarrier_init(*mbarrier_ptr, threads);
    }
    __syncthreads();""")

with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "w") as f:
    f.write(content)
