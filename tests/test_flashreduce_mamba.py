import torch
import pytest
from flash_attn.flashreduce_interface import flashreduce_affine_scan

def reference_mamba_scan(a, b):
    # a: (batch, seqlen, dim)
    # b: (batch, seqlen, dim)
    batch, seqlen, dim = a.shape
    x = torch.zeros_like(b)
    curr = torch.zeros(batch, dim, device=a.device, dtype=a.dtype)
    for t in range(seqlen):
        curr = a[:, t, :] * curr + b[:, t, :]
        x[:, t, :] = curr
    return x

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch, seqlen, dim", [
    (2, 512, 64),
    (4, 1024, 32),
    (8, 2048, 16),
])
def test_flashreduce_mamba_forward(dtype, batch, seqlen, dim):
    device = "cuda"
    # Clamp scale values between -0.9 and 0.9 to ensure stability
    a = torch.randn(batch, seqlen, dim, device=device, dtype=dtype).clamp(-0.9, 0.9)
    b = torch.randn(batch, seqlen, dim, device=device, dtype=dtype)
    
    # Run optimized kernel
    x_opt = flashreduce_affine_scan(a, b)
    
    # Run reference sequential scan
    x_ref = reference_mamba_scan(a, b)
    
    # Check correctness
    atol = 1e-1 if dtype == torch.bfloat16 else 1e-5
    rtol = 1e-1 if dtype == torch.bfloat16 else 1e-5
    assert torch.allclose(x_opt, x_ref, atol=atol, rtol=rtol), "Forward outputs do not match!"

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch, seqlen, dim", [
    (2, 256, 32),
    (4, 128, 16),
])
def test_flashreduce_mamba_backward(dtype, batch, seqlen, dim):
    device = "cuda"
    # Set scale factor 'a' between -0.8 and 0.8
    a_raw = torch.randn(batch, seqlen, dim, device=device, dtype=dtype)
    a = (a_raw.clamp(-0.8, 0.8)).detach().requires_grad_(True)
    b = torch.randn(batch, seqlen, dim, device=device, dtype=dtype, requires_grad=True)
    
    # Forward pass
    x_opt = flashreduce_affine_scan(a, b)
    
    # Reference forward pass
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    x_ref = reference_mamba_scan(a_ref, b_ref)
    
    # Incoming gradient dy
    dy = torch.randn_like(x_opt)
    
    # Reference backward pass
    loss_ref = (x_ref * dy).sum()
    loss_ref.backward()
    da_ref, db_ref = a_ref.grad, b_ref.grad
    
    # Optimized backward pass
    loss_opt = (x_opt * dy).sum()
    loss_opt.backward()
    da_opt, db_opt = a.grad, b.grad
    
    # Check correctness of gradients
    atol = 1e-1 if dtype == torch.bfloat16 else 1e-4
    rtol = 1e-1 if dtype == torch.bfloat16 else 1e-4
    assert torch.allclose(da_opt, da_ref, atol=atol, rtol=rtol), "grad_a does not match reference!"
    assert torch.allclose(db_opt, db_ref, atol=atol, rtol=rtol), "grad_b does not match reference!"
