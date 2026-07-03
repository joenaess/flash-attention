with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "r") as f:
    content = f.read()

content = content.replace(
    "cute::GMMA::Major::K, cute::GMMA::Major::K>(),",
    "cute::GMMA::Major::K, cute::GMMA::Major::MN>(),"
)

with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu", "w") as f:
    f.write(content)
