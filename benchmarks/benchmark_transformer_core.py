import time
import torch
import gc
from flash_attn.losses.cross_entropy_cuda_wrapper import gemmmr_xentropy_cuda

def measure_memory():
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def benchmark_transformer_core():
    device = "cuda"
    dtype = torch.bfloat16
    D = 128
    vocab_size = 32768
    
    seq_lens = [4096, 8192, 16384, 32768]
    
    print("=" * 90)
    print("   HIGH-PERFORMANCE FUSED CROSS-ENTROPY BENCHMARK (D=128, VOCAB=32768, BF16)")
    print("=" * 90)
    print(f"{'SeqLen':<10} | {'Method':<10} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Total (ms)':<10} | {'Peak Mem (MB)':<15}")
    print("-" * 90)
    
    results_markdown = [
        "| Seq Length | Method | Forward (ms) | Backward (ms) | Total (ms) | Peak Memory (MB) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for seq_len in seq_lens:
        M = seq_len
        
        # 1. Benchmark PyTorch Eager (Logits + Loss)
        try:
            measure_memory() # reset memory
            pred_ref = torch.randn(M, D, dtype=dtype, device=device).requires_grad_(True)
            trg_ref = torch.randn(vocab_size, D, dtype=dtype, device=device).requires_grad_(True)
            truth = torch.randint(vocab_size, (M,), device=device)
            
            # Warmup
            logits = pred_ref @ trg_ref.T
            loss = torch.nn.functional.cross_entropy(logits, truth)
            loss.backward()
            
            start_mem = measure_memory()
            
            start_fwd = torch.cuda.Event(enable_timing=True)
            end_fwd = torch.cuda.Event(enable_timing=True)
            start_bwd = torch.cuda.Event(enable_timing=True)
            end_bwd = torch.cuda.Event(enable_timing=True)
            
            # Forward timing
            torch.cuda.synchronize()
            start_fwd.record()
            logits = pred_ref @ trg_ref.T
            loss = torch.nn.functional.cross_entropy(logits, truth)
            end_fwd.record()
            torch.cuda.synchronize()
            fwd_time_ref = start_fwd.elapsed_time(end_fwd)
            
            # Backward timing
            torch.cuda.synchronize()
            start_bwd.record()
            loss.backward()
            end_bwd.record()
            torch.cuda.synchronize()
            bwd_time_ref = start_bwd.elapsed_time(end_bwd)
            
            peak_mem_ref = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            # Clean up reference tensors
            del logits, loss, pred_ref, trg_ref
            gc.collect()
            torch.cuda.empty_cache()
            
            eager_str = f"{fwd_time_ref:.2f} ms | {bwd_time_ref:.2f} ms | {fwd_time_ref + bwd_time_ref:.2f} ms | {peak_mem_ref:.2f} MB"
            print(f"{seq_len:<10} | {'Eager':<10} | {fwd_time_ref:<10.2f} | {bwd_time_ref:<10.2f} | {fwd_time_ref + bwd_time_ref:<10.2f} | {peak_mem_ref:<15.2f}")
            results_markdown.append(f"| {seq_len} | Eager | {fwd_time_ref:.2f} | {bwd_time_ref:.2f} | {fwd_time_ref + bwd_time_ref:.2f} | {peak_mem_ref:.2f} |")
        except RuntimeError as e:
            # Handle out of memory for standard eager at high sequence lengths
            eager_str = "OOM"
            print(f"{seq_len:<10} | {'Eager':<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<15}")
            results_markdown.append(f"| {seq_len} | Eager | OOM | OOM | OOM | OOM |")
        
        # 2. Benchmark Fused Custom Fused Kernel
        measure_memory() # reset memory
        pred_cust = torch.randn(M, D, dtype=dtype, device=device).requires_grad_(True)
        trg_cust = torch.randn(vocab_size, D, dtype=dtype, device=device).requires_grad_(True)
        
        # Warmup and Autotuning (run once to trigger JIT search and caching)
        loss_cust = gemmmr_xentropy_cuda(pred_cust, trg_cust, truth)
        loss_cust.backward(torch.ones_like(loss_cust))
        
        measure_memory()
        
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)
        
        # Forward timing
        torch.cuda.synchronize()
        start_fwd.record()
        loss_cust = gemmmr_xentropy_cuda(pred_cust, trg_cust, truth)
        end_fwd.record()
        torch.cuda.synchronize()
        fwd_time_cust = start_fwd.elapsed_time(end_fwd)
        
        dy = torch.ones_like(loss_cust)
        
        # Backward timing
        torch.cuda.synchronize()
        start_bwd.record()
        loss_cust.backward(dy)
        end_bwd.record()
        torch.cuda.synchronize()
        bwd_time_cust = start_bwd.elapsed_time(end_bwd)
        
        peak_mem_cust = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Clean up custom tensors
        del loss_cust, pred_cust, trg_cust, dy
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"{seq_len:<10} | {'Fused':<10} | {fwd_time_cust:<10.2f} | {bwd_time_cust:<10.2f} | {fwd_time_cust + bwd_time_cust:<10.2f} | {peak_mem_cust:<15.2f}")
        results_markdown.append(f"| {seq_len} | **Fused (Ours)** | **{fwd_time_cust:.2f}** | **{bwd_time_cust:.2f}** | **{fwd_time_cust + bwd_time_cust:.2f}** | **{peak_mem_cust:.2f}** |")
        print("-" * 90)
    
    print("\nBenchmark completed successfully.")
    
    # Save the markdown results table as an artifact
    markdown_content = "\n".join(results_markdown)
    return markdown_content

if __name__ == "__main__":
    benchmark_transformer_core()
