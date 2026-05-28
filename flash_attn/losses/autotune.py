import os
import torch
from torch.utils.cpp_extension import load

# Global cache for the best compiled cross-entropy module
_BEST_XENTROPY_MODULE = None
_BEST_XENTROPY_TILE_SIZE = None

def get_smart_candidate_tiles():
    """
    Queries GPU hardware properties to dynamically select viable candidate tile sizes
    based on physical shared memory ceilings.
    """
    try:
        device_props = torch.cuda.get_device_properties(0)
        max_shared_mem = device_props.shared_memory_per_block
        print(f"[Autotuner] Detected max shared memory per block: {max_shared_mem} bytes")
    except Exception as e:
        print(f"[Autotuner] Warning: Codebase falling back to baseline: {e}")
        return [32]

    # Safeguard consumer architectures while exposing massive enterprise SM spaces
    # Ada 4080, L4: ~49,152 to 101,376 bytes
    # Datacenter limits (H100, B200): 233,472+ bytes
    if max_shared_mem < 150000:
        return [32]  # Safety cap for RTX 4080 / L4
    else:
        return [32, 64, 128]  # Unlocked target for H100 / H200 / B200

def autotune_xentropy(pred, trg, truth, tixs):
    """
    Dynamically autotunes and JIT-compiles the fused cross-entropy kernel using Ninja
    with specialized -DTILE_SIZE={size} compile definitions.
    """
    global _BEST_XENTROPY_MODULE, _BEST_XENTROPY_TILE_SIZE
    if _BEST_XENTROPY_MODULE is not None:
        return _BEST_XENTROPY_MODULE

    candidate_tile_sizes = get_smart_candidate_tiles()
    best_time = float("inf")
    best_module = None
    best_size = None

    M = pred.size(0)
    D = pred.size(1)
    N = trg.size(0)

    # Resolve paths to source files in csrc/losses/
    losses_csrc_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "csrc", "losses")
    )

    sources = [
        os.path.join(losses_csrc_dir, "xentropy_cuda.cpp"),
        os.path.join(losses_csrc_dir, "bindings.cpp"),
        os.path.join(losses_csrc_dir, "xentropy_kernel.cu"),
        os.path.join(losses_csrc_dir, "xentropy_backward_kernel.cu"),
    ]

    print(f"\n[Autotuner] Initiating Cross-Entropy search across candidate tile sizes: {candidate_tile_sizes}")

    for size in candidate_tile_sizes:
        print(f"[Autotuner] Dynamic JIT-compiling candidate with -DTILE_SIZE={size}...")
        try:
            # JIT compile the module with Ninja and the tile size specialized compile flag
            module = load(
                name=f"flash_attn_xentropy_autotune_{size}",
                sources=sources,
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    f"-DTILE_SIZE={size}",
                ],
                verbose=False,
            )

            # Warmup phase (3 iterations)
            p_out = torch.zeros(M, device=pred.device, dtype=pred.dtype)
            n_out = torch.zeros(M, device=pred.device, dtype=pred.dtype)

            for _ in range(3):
                module.xentropy_forward(pred, trg, truth, tixs)

            # Measurement phase (using torch.cuda.Event)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()
            for _ in range(10):
                module.xentropy_forward(pred, trg, truth, tixs)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event) / 10.0
            print(f"[Autotuner] Candidate TILE_SIZE={size} executed in average: {elapsed_time_ms:.4f} ms")

            if elapsed_time_ms < best_time:
                best_time = elapsed_time_ms
                best_module = module
                best_size = size
        except Exception as e:
            print(f"[Autotuner] Skipping candidate TILE_SIZE={size} due to compilation/runtime error: {e}")

    if best_module is None:
        raise RuntimeError("[Autotuner] Failed to compile any candidate tile size configurations.")

    print(f"[Autotuner] Selection complete: optimal config is TILE_SIZE={best_size} ({best_time:.4f} ms)\n")
    _BEST_XENTROPY_MODULE = best_module
    _BEST_XENTROPY_TILE_SIZE = best_size
    return _BEST_XENTROPY_MODULE
