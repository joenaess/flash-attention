import os
import time
import torch
from torch.utils.cpp_extension import load

# Global cache for the best compiled module
_BEST_MODULE = None


def autotune_attention_forward(Q, K, V):
    global _BEST_MODULE
    if _BEST_MODULE is not None:
        return _BEST_MODULE

    M, F = Q.shape
    N, D = V.shape

    def get_smart_candidate_tiles():
        """
        Queries the hardware properties of the current GPU to dynamically 
        select viable block tile candidates based on physical shared memory ceilings.
        """
        try:
            device_props = torch.cuda.get_device_properties(0)
            max_shared_mem = device_props.shared_memory_per_block
            print(f"[Autotuner] Detected max shared memory per block: {max_shared_mem} bytes")
        except Exception as e:
            print(f"[Autotuner] Warning: Failed to query GPU properties ({e}). Falling back to baseline.")
            return [32]

        # Explicit threshold separating consumer/client GPUs from enterprise accelerators
        # Consumer limits (Ada 4080, L4, Ampere 3090): ~49,152 to 101,376 bytes
        # Datacenter limits (Hopper H100/H200, Blackwell B200): 233,472+ bytes
        # Use a broader set of candidate tile sizes to ensure backward pass fits in shared memory
        if max_shared_mem < 150000:
            return [16]
        else:
            return [16, 32, 64, 128]

    candidate_tile_sizes = get_smart_candidate_tiles()

    best_time = float("inf")
    best_module = None
    best_size = None

    # Resolve paths to source files in csrc_gemmmapreduce/ directory
    # Since autotune.py is inside flash_attn/losses/, csrc_gemmmapreduce/ is in the same directory.
    csrc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "csrc_gemmmapreduce"))

    sources = [
        os.path.join(csrc_dir, "xentropy_cuda.cpp"),
        os.path.join(csrc_dir, "bindings.cpp"),
        os.path.join(csrc_dir, "xentropy_kernel.cu"),
        os.path.join(csrc_dir, "xentropy_backward_kernel.cu"),
        os.path.join(csrc_dir, "attention_cuda.cpp"),
        os.path.join(csrc_dir, "attention_kernel.cu"),
        os.path.join(csrc_dir, "attention_backward_kernel.cu"),
    ]

    print(
        f"\n[Autotuner] Initiating autotuning search across candidate tile sizes: {candidate_tile_sizes}"
    )

    for size in candidate_tile_sizes:
        print(f"[Autotuner] Dynamic JIT-compiling candidate with -DTILE_SIZE={size}...")
        try:
            # Dynamically compile the module with -DTILE_SIZE={size}
            module = load(
                name=f"gemmmapreduce_cuda_autotune_{size}",
                sources=sources,
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-std=c++20",
                    f"-DTILE_SIZE={size}",
                ],
                verbose=False,
            )

            # Warmup phase (3 iterations)
            O = torch.zeros((M, D), device=Q.device, dtype=Q.dtype)
            l = torch.zeros(M, device=Q.device, dtype=Q.dtype)
            m_out = torch.zeros(M, device=Q.device, dtype=Q.dtype)
            
            dO = torch.zeros((M, D), device=Q.device, dtype=Q.dtype)
            m_in = torch.zeros(M, device=Q.device, dtype=Q.dtype)
            dQ = torch.zeros((M, F), device=Q.device, dtype=Q.dtype)
            dK = torch.zeros((N, F), device=Q.device, dtype=Q.dtype)
            dV = torch.zeros((N, D), device=Q.device, dtype=Q.dtype)

            for _ in range(3):
                module.attention_forward(Q, K, V, O, l, m_out)
                module.attention_backward(Q, K, V, O, dO, l, m_in, dQ, dK, dV)

            # Measurement phase (using torch.cuda.Event)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            for _ in range(10):
                module.attention_forward(Q, K, V, O, l, m_out)
                module.attention_backward(Q, K, V, O, dO, l, m_in, dQ, dK, dV)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event) / 10.0
            print(
                f"[Autotuner] Candidate TILE_SIZE={size} executed in average: {elapsed_time_ms:.4f} ms"
            )

            if elapsed_time_ms < best_time:
                best_time = elapsed_time_ms
                best_module = module
                best_size = size
        except Exception as e:
            print(f"[Autotuner] Skipping candidate TILE_SIZE={size} due to error: {e}")

    if best_module is None:
        raise RuntimeError(
            "[Autotuner] Failed to compile any candidate tile size configurations."
        )

    print(
        f"[Autotuner] Selection complete: optimal config is TILE_SIZE={best_size} ({best_time:.4f} ms)\n"
    )
    _BEST_MODULE = best_module
    return _BEST_MODULE
