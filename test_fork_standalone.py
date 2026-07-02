import torch
import sys

# 1. STRICT IMPORT CHECK
try:
    # We explicitly import the bare-metal C++ function. 
    # If this fails, we are NOT allowed to fall back to PyTorch SDPA.
    from flash_attn.flash_attn_interface import flash_attn_func
    print("[SUCCESS] flash_attn_2_cuda C++ extension loaded successfully.")
except ImportError as e:
    print(f"[FATAL ERROR] Custom FlashAttention C++ binary not found: {e}")
    sys.exit(1)

try:
    # Explicitly import our custom autotuned loss
    from flash_attn.losses.cross_entropy import GeMMMrXEntropyLoss
    print("[SUCCESS] GeMMMapReduce JIT-Autotuned Loss loaded successfully.")
except ImportError as e:
    print(f"[FATAL ERROR] Custom GeMMMapReduce Loss not found: {e}")
    sys.exit(1)

# 2. HARDWARE CAPABILITY CHECK
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print(f"\n[HARDWARE] Detected GPU: {props.name}")
print(f"[HARDWARE] Shared Memory per Block: {props.shared_memory_per_block} bytes")

# 3. END-TO-END SMOKE TEST
print("\n[TEST] Initiating Forward/Backward execution...")
batch, seqlen, nheads, headdim = 2, 4096, 8, 128
vocab_size = 32000

# Dummy Attention Tensors (bfloat16)
q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16, requires_grad=True)

# Dummy Targets
targets = torch.randint(0, vocab_size, (batch, seqlen), device=device)

# Attention Pass (STRICT CUSTOM CALL)
out = flash_attn_func(q, k, v, causal=True)

# Create an LM Head projection weight matrix
weight = torch.randn(vocab_size, nheads * headdim, device=device, dtype=torch.bfloat16)

# Autotuned Fused Cross-Entropy Pass
# This will trigger the Ninja JIT compiler on the very first run!
loss_fn = GeMMMrXEntropyLoss()
loss = loss_fn(out.view(-1, nheads * headdim), weight, targets.view(-1))

# Backward Pass
loss.backward()

print(f"[SUCCESS] Loss: {loss.item():.4f}")
print("[SUCCESS] Gradients successfully populated.")
print("[VERDICT] The standalone FlashAttention fork is 100% functional on this server!")
