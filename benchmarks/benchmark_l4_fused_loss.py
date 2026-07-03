import torch
import torch.nn.functional as F
import sys
from flash_attn.losses.cross_entropy import GeMMMrXEntropyLoss

def benchmark_forward_backward(func, *args, num_warmup=1, num_iters=2):
    # Warmup
    for _ in range(num_warmup):
        loss = func(*args)
        loss.backward()
        # Clear gradients to prevent accumulation impact
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.grad is not None:
                arg.grad = None
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
    for i in range(num_iters):
        start_events[i].record()
        loss = func(*args)
        loss.backward()
        end_events[i].record()
        
        # Clear gradients
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.grad is not None:
                arg.grad = None

    torch.cuda.synchronize()
    
    # Calculate metrics
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / num_iters
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert to MB
    
    return avg_time, peak_mem

def run_pytorch_baseline(out, weight, targets):
    logits = F.linear(out, weight)
    loss = F.cross_entropy(logits, targets)
    return loss

def run_gemmmr_fused(out, weight, targets, loss_fn):
    return loss_fn(out, weight, targets)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"--- FUSED LM HEAD + CROSS ENTROPY BENCHMARK ---")
    print(f"Hardware: {props.name} | Shared Mem: {props.shared_memory_per_block // 1024} KB\n")
    
    # Static parameters based on LLM standards
    batch = 1
    nheads = 8
    headdim = 128
    hidden_dim = nheads * headdim # 1024
    vocab_size = 32000
    
    seqlens = [2048, 4096, 8192, 16384, 32768]
    
    print(f"| Seq Len | PyTorch Time (ms) | GeMMMr Time (ms) | PyTorch Mem (MB) | GeMMMr Mem (MB) | Mem Reduction |")
    print(f"|---------|-------------------|------------------|------------------|-----------------|---------------|")
    
    gemmmr_fn = GeMMMrXEntropyLoss()

    for seqlen in seqlens:
        try:
            # Inputs
            out = torch.randn(batch * seqlen, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            weight = torch.randn(vocab_size, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
            targets = torch.randint(0, vocab_size, (batch * seqlen,), device=device)
            
            # Run Baseline
            pt_time, pt_mem = benchmark_forward_backward(run_pytorch_baseline, out, weight, targets)
            
            # Run Custom Fused Kernel
            # Note: We pass gemmmr_fn via lambda to maintain signature
            gemm_time, gemm_mem = benchmark_forward_backward(
                lambda o, w, t: run_gemmmr_fused(o, w, t, gemmmr_fn), 
                out, weight, targets
            )
            
            mem_reduction = pt_mem / gemm_mem if gemm_mem > 0 else 0
            
            print(f"| {seqlen:<7} | {pt_time:<17.2f} | {gemm_time:<16.2f} | {pt_mem:<16.2f} | {gemm_mem:<15.2f} | {mem_reduction:<12.1f}x |")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"| {seqlen:<7} | OOM                 | -                | OOM              | -               | -             |")
            else:
                print(f"Error at seqlen {seqlen}: {e}")
            torch.cuda.empty_cache()
