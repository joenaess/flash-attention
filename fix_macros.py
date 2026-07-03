import os

filepath = 'flash_attn/losses/csrc_gemmmapreduce/xentropy_backward_kernel.cu'
with open(filepath, 'r') as f:
    content = f.read()

# Replace any remaining BLK_M that isn't XENTROPY_BLK_M
# We can just do a regex replace to ensure we only replace isolated BLK_M
import re
content = re.sub(r'\b(?<!XENTROPY_)BLK_M\b', 'XENTROPY_BLK_M', content)
content = re.sub(r'\b(?<!XENTROPY_)BLK_N\b', 'XENTROPY_BLK_N', content)
content = re.sub(r'\b(?<!XENTROPY_)BLK_K\b', 'XENTROPY_BLK_K', content)

with open(filepath, 'w') as f:
    f.write(content)
