import os
import pytest
import torch
from torch.utils.cpp_extension import load
from flash_attn.losses.cross_entropy_cuda_wrapper import gemmmr_xentropy_cuda

def compile_custom_xentropy(tile_size):
    """
    Helper to JIT compile the custom cross-entropy module with an explicit TILE_SIZE.
    """
    losses_csrc_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "csrc", "losses")
    )
    sources = [
        os.path.join(losses_csrc_dir, "xentropy_cuda.cpp"),
        os.path.join(losses_csrc_dir, "bindings.cpp"),
        os.path.join(losses_csrc_dir, "xentropy_kernel.cu"),
        os.path.join(losses_csrc_dir, "xentropy_backward_kernel.cu"),
    ]
    # Unique name for each tile size to avoid caching conflicts
    module_name = f"test_flash_attn_xentropy_{tile_size}"
    module = load(
        name=module_name,
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            f"-DTILE_SIZE={tile_size}",
        ],
        verbose=False,
    )
    return module

class ExplicitGeMMMrXEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, trg, truth, tixs, module):
        ctx.module = module
        pred_c = pred.contiguous()
        trg_c = trg.contiguous()
        truth_c = truth.contiguous()
        tixs_c = tixs.contiguous()
        
        p_out, n_out = module.xentropy_forward(pred_c, trg_c, truth_c, tixs_c)
        ctx.save_for_backward(pred_c, trg_c, truth_c, tixs_c, p_out)
        return p_out, n_out

    @staticmethod
    def backward(ctx, grad_p, grad_n):
        pred, trg, truth, tixs, p_out = ctx.saved_tensors
        module = ctx.module
        
        grad_p_c = grad_p.contiguous()
        grad_n_c = grad_n.contiguous()
        
        grad_pred, grad_trg = module.xentropy_backward(
            grad_p_c, grad_n_c, pred, trg, truth, tixs, p_out
        )
        return grad_pred, grad_trg, None, None, None

def run_explicit_xentropy(pred, trg, truth, tixs, module):
    p, n = ExplicitGeMMMrXEntropy.apply(pred, trg, truth, tixs, module)
    return p - n

def test_xentropy_autotune_gradcheck():
    """
    Phase 4: Run standard torch.autograd.gradcheck on double precision (float64)
    to mathematically verify the backward analytical gradients of the autotuned module.
    """
    device = "cuda"
    M, N, D = 16, 16, 8
    
    # FP64 is required for gradcheck finite differences comparisons
    pred = torch.randn(M, D, dtype=torch.float64, device=device, requires_grad=True)
    trg = torch.randn(N, D, dtype=torch.float64, device=device, requires_grad=True)
    truth = torch.randint(N, (M,), device=device)
    tixs = torch.arange(N, device=device)

    def func_to_test(p, t):
        return gemmmr_xentropy_cuda(p, t, truth, tixs)

    # Validate autograd compliance
    print("\n[Test] Running torch.autograd.gradcheck on autotuned loss...")
    test_passed = torch.autograd.gradcheck(func_to_test, (pred, trg), eps=1e-6, atol=1e-4)
    assert test_passed, "Gradcheck failed for autotuned loss!"
    print("[Test] Gradcheck passed successfully.")

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tile_size", [64, 128])
def test_xentropy_correctness(tile_size, dtype):
    """
    Phase 4: Assert that for float32 and bfloat16 inputs at multi-warp scales (TILE_SIZE=64, 128),
    analytical gradients match native PyTorch eager cross-entropy.
    """
    device = "cuda"
    M, N, D = 128, 64, 32
    
    # 1. Compile custom module explicitly for this tile size
    module = compile_custom_xentropy(tile_size)

    # 2. Initialize inputs
    pred_custom = torch.randn(M, D, dtype=dtype, device=device).requires_grad_(True)
    trg_custom = torch.randn(N, D, dtype=dtype, device=device).requires_grad_(True)
    
    truth = torch.randint(N, (M,), device=device)
    tixs = torch.arange(N, device=device)
    
    # 3. Custom forward and backward passes
    loss_custom = run_explicit_xentropy(pred_custom, trg_custom, truth, tixs, module)
    dy = torch.randn_like(loss_custom)
    
    loss_custom_sum = (loss_custom * dy).sum()
    loss_custom_sum.backward()
    
    grad_pred_custom = pred_custom.grad.clone()
    grad_trg_custom = trg_custom.grad.clone()

    # 4. Reference PyTorch eager cross-entropy
    pred_ref = pred_custom.detach().clone().requires_grad_(True)
    trg_ref = trg_custom.detach().clone().requires_grad_(True)
    
    # eager logit matrix = pred @ trg.T
    logits = pred_ref @ trg_ref.T
    loss_ref = torch.nn.functional.cross_entropy(logits, truth, reduction="none")
    
    loss_ref_sum = (loss_ref * dy).sum()
    loss_ref_sum.backward()
    
    grad_pred_ref = pred_ref.grad.clone()
    grad_trg_ref = trg_ref.grad.clone()

    # 5. Verify correctness using dtype-dependent tolerances
    if dtype == torch.float32:
        atol_fwd, rtol_fwd = 1e-4, 1e-4
        atol_bwd, rtol_bwd = 1e-3, 1e-3
    else:  # bfloat16
        atol_fwd, rtol_fwd = 2e-1, 2e-1
        atol_bwd, rtol_bwd = 3e-1, 3e-1

    # Check forward outputs match standard loss
    assert torch.allclose(loss_custom, loss_ref, atol=atol_fwd, rtol=rtol_fwd), f"Forward outputs mismatch at TILE_SIZE={tile_size}!"
    
    # Check backward analytical gradients match eager perfectly
    assert torch.allclose(grad_pred_custom, grad_pred_ref, atol=atol_bwd, rtol=rtol_bwd), f"grad_pred does not match native eager at TILE_SIZE={tile_size}!"
    assert torch.allclose(grad_trg_custom, grad_trg_ref, atol=atol_bwd, rtol=rtol_bwd), f"grad_trg does not match native eager at TILE_SIZE={tile_size}!"
    
    print(f"\n[Test] SUCCESS: Correctness & analytical gradients match eager at TILE_SIZE={tile_size} ({dtype})!")
