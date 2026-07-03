with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_kernel.cu", "r") as f:
    content = f.read()

content = content.replace("//\n\n\n    };\n", "")

with open("flash_attn/losses/csrc_gemmmapreduce/xentropy_kernel.cu", "w") as f:
    f.write(content)
