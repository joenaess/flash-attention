with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "r") as f:
    content = f.read()

content = content.replace("cute::wait_barrier(*mbarrier_ptr, (K_TILES + k_tile) % 2);", "cute::wait_barrier(*mbarrier_ptr, k_tile % 2);")
content = content.replace("gpuAtomicAdd(&params.grad_trg[global_n * params.D + global_k], (scalar_t)tCrC_grad_trg(i, m, n));\n                    }\n                }\n            }\n        }\n    }", 
"gpuAtomicAdd(&params.grad_trg[global_n * params.D + global_k], (scalar_t)tCrC_grad_trg(i, m, n));\n                    }\n                }\n            }\n        }\n        __syncthreads();\n    }")

with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "w") as f:
    f.write(content)
