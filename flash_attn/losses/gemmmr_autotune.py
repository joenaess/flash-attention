import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
import os
import time
import torch
from torch.utils.cpp_extension import load

# Global cache for the best compiled module
_BEST_MODULE = None


def autotune_xentropy_forward(pred, trg, truth, tixs):
    global _BEST_MODULE
    if _BEST_MODULE is not None:
        return _BEST_MODULE

    M, D = pred.shape
    N = trg.shape[0]

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
            return [(64, 64, 128), (128, 64, 64)] # L4 Safe
        else:
            return [(64, 128, 128), (128, 128, 128)] # Hopper Safe

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
    ]

    print(
        f"\n[Autotuner] Initiating autotuning search across candidate tile sizes: {candidate_tile_sizes}"
    )

    for size in candidate_tile_sizes:
        print(f"[Autotuner] Dynamic JIT-compiling candidate with BLK_M={size[0]}, BLK_N={size[1]}, BLK_K={size[2]}...")
        try:
            # Dynamically compile the module with -DXENTROPY_BLK_M={size[0]} -DXENTROPY_BLK_N={size[1]} -DXENTROPY_BLK_K={size[2]}
            module = load(
                name=f"gemmmapreduce_cuda_autotune_{size[0]}_{size[1]}_{size[2]}",
                sources=sources,
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math", "-lineinfo",
                    "-std=c++20",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    f"-DXENTROPY_BLK_M={size[0]}",
                    f"-DXENTROPY_BLK_N={size[1]}",
                    f"-DXENTROPY_BLK_K={size[2]}",
                    "-I/home/ubuntu/jonas/flash-attention/csrc/cutlass/include",
                    "-I/home/ubuntu/jonas/flash-attention/csrc/cutlass/tools/util/include"
                ],
                verbose=False,
            )

            # Warmup phase (3 iterations)
            grad_p = torch.zeros(M, device=pred.device, dtype=pred.dtype)
            grad_n = torch.zeros(M, device=pred.device, dtype=pred.dtype)

            for _ in range(3):
                p_out, n_out = module.xentropy_forward(pred, trg, truth, tixs)
                module.xentropy_backward

            # Measurement phase (using torch.cuda.Event)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            for _ in range(10):
                p_out, n_out = module.xentropy_forward(pred, trg, truth, tixs)
                module.xentropy_backward
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
