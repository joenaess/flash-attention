import torch
import time
from flash_attn.flashreduce_interface import flashreduce_affine_scan

def benchmark_scan(batch, seqlen, dim, dtype=torch.bfloat16, num_warmups=10, num_iters=100):
    device = "cuda"
    a = torch.randn(batch, seqlen, dim, device=device, dtype=dtype).clamp(-0.9, 0.9).requires_grad_(True)
    b = torch.randn(batch, seqlen, dim, device=device, dtype=dtype).requires_grad_(True)
    dy = torch.randn(batch, seqlen, dim, device=device, dtype=dtype)
    
    # Warmup runs
    for _ in range(num_warmups):
        x = flashreduce_affine_scan(a, b)
        loss = (x * dy).sum()
        loss.backward()
        a.grad.zero_()
        b.grad.zero_()
        
    torch.cuda.synchronize()
    
    # Benchmark Forward pass
    t0 = time.time()
    for _ in range(num_iters):
        x = flashreduce_affine_scan(a, b)
    torch.cuda.synchronize()
    fwd_time = (time.time() - t0) / num_iters * 1000.0 # ms
    
    # Benchmark Forward + Backward passes
    t0 = time.time()
    for _ in range(num_iters):
        x = flashreduce_affine_scan(a, b)
        loss = (x * dy).sum()
        loss.backward()
        a.grad.zero_()
        b.grad.zero_()
    torch.cuda.synchronize()
    total_time = (time.time() - t0) / num_iters * 1000.0 # ms
    bwd_time = total_time - fwd_time
    
    # Measure peak memory allocation
    torch.cuda.reset_peak_memory_stats()
    x = flashreduce_affine_scan(a, b)
    loss = (x * dy).sum()
    loss.backward()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) # MB
    
    return fwd_time, bwd_time, peak_mem

if __name__ == "__main__":
    print(f"Benchmarking Mamba Specialized Blelloch Suffix Scan...")
    print(f"{'SeqLen':<10} | {'Fwd Time (ms)':<15} | {'Bwd Time (ms)':<15} | {'Peak Mem (MB)':<15}")
    print("-" * 65)
    
    for seqlen in [4096, 8192, 16384, 32768]:
        try:
            fwd, bwd, mem = benchmark_scan(batch=8, seqlen=seqlen, dim=64)
            print(f"{seqlen:<10} | {fwd:<15.3f} | {bwd:<15.3f} | {mem:<15.2f}")
        except Exception as e:
            print(f"{seqlen:<10} | Error: {e}")
