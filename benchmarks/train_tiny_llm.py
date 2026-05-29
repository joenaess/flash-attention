import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our autotuned cross entropy wrapper
from flash_attn.losses.cross_entropy_cuda_wrapper import gemmmr_xentropy_cuda

# 1. Load and Tokenize real data (Character-level TinyShakespeare)
if not os.path.exists("input.txt"):
    print("[Dataset] 'input.txt' not found, writing a small mock real text sample for test run...")
    mock_data = "To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles,\nAnd by opposing end them? To die: to sleep;\nNo more; and by a sleep to say we end\nThe heart-ache and the thousand natural shocks\nThat flesh is heir to, 'tis a consummation\nDevoutly to be wish'd. To die, to sleep;\nTo sleep: perchance to dream: ay, there's the rub;\nFor in that sleep of death what dreams may come\nWhen we have shuffled off this mortal coil,\nMust give us pause. There's the respect\nThat makes calamity of so long life;\n" * 100
    with open("input.txt", "w", encoding="utf-8") as f:
        f.write(mock_data)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Simple encode/decode
encode = lambda s: [char_to_ix[c] for c in s]
decode = lambda l: "".join([ix_to_char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n_train = int(0.9 * len(data))
train_data = data[:n_train]

def get_batch(batch_size, block_size):
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i : i + block_size] for i in ix])
    y = torch.stack([train_data[i + 1 : i + block_size + 1] for i in ix])
    return x.cuda(), y.cuda()

from flash_attn.cute import flash_attn_func

# 2. Define a simple nanoGPT Transformer Block
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to standard layout: (batch, seqlen, num_heads, head_dim)
        # flash_attn_func natively expects exactly this layout!
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        # Explicitly use our local compiled FlashAttention kernel
        y, _ = flash_attn_func(q, k, v, softmax_scale=None, causal=True)
        
        # Collapse heads back down to standard embedding dimension
        y = y.contiguous().view(B, T, C)
        
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=2048):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            h = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying (classic GPT design)
        self.transformer.wte.weight = self.lm_head.weight

    def forward_eager(self, idx, targets):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Project representation to full vocab logits size: [B * T, VocabSize]
        logits = self.lm_head(x.view(-1, x.size(-1)))
        loss = F.cross_entropy(logits, targets.view(-1))
        return loss

    def forward_fused(self, idx, targets):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Flatten sequence: [M, D] where M = B * T
        pred = x.view(-1, x.size(-1))
        # The weight Matrix is of shape [VocabSize, D]
        trg = self.lm_head.weight
        
        # Invoke our dynamic, autotuned fused cross-entropy
        loss = gemmmr_xentropy_cuda(pred, trg, targets.view(-1))
        return loss.mean()

# 3. Benchmark Execution Loop
def run_benchmark():
    batch_size = 8
    block_size = 1024
    steps = 15
    dtype = torch.bfloat16
    vocab_size_large = 32768
    
    print(f"\nInitializing TinyLLM with Vocab Size = {vocab_size_large}...")
    model = TinyLLM(vocab_size=vocab_size_large, block_size=block_size).cuda().to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Generate realistic training data batches for large vocab size
    x = torch.randint(vocab_size_large, (batch_size, block_size), device="cuda")
    y = torch.randint(vocab_size_large, (batch_size, block_size), device="cuda")
    
    # ================= BASELINE: EAGER CROSS ENTROPY =================
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    
    for step in range(steps):
        optimizer.zero_grad()
        loss = model.forward_eager(x, y)
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print(f"[Eager Baseline] Step {step:2d} | Loss: {loss.item():.4f}")
            
    torch.cuda.synchronize()
    eager_time = (time.time() - start_time) / steps * 1000.0
    eager_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Reset model parameters
    model = TinyLLM(vocab_size=vocab_size_large, block_size=block_size).cuda().to(dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # ================= OPTIMIZED: FUSED AUTOTUNED KERNEL =============
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()
    
    for step in range(steps):
        optimizer.zero_grad()
        loss = model.forward_fused(x, y)
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            print(f"[Fused Mode]     Step {step:2d} | Loss: {loss.item():.4f}")
            
    torch.cuda.synchronize()
    fused_time = (time.time() - start_time) / steps * 1000.0
    fused_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    print("\n" + "="*50)
    print("           TINY LLM COMPARISON SUMMARY")
    print("="*50)
    print(f"Eager Mode Baseline: {eager_time:.2f} ms/step | Peak Mem: {eager_mem:.2f} MB")
    print(f"Fused Autotuned Mode: {fused_time:.2f} ms/step | Peak Mem: {fused_mem:.2f} MB")
    print(f"Memory reduction factor: {eager_mem / fused_mem:.2f}x!")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_benchmark()
