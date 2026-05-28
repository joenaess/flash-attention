import torch
from flash_attn.losses.autotune import autotune_xentropy

class GeMMMrXEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, trg, truth, tixs):
        # Dynamically JIT-compiles (once) and finds the optimal TILE_SIZE config
        best_module = autotune_xentropy(pred, trg, truth, tixs)
        
        pred_c = pred.contiguous()
        trg_c = trg.contiguous()
        truth_c = truth.contiguous()
        tixs_c = tixs.contiguous()
        
        p_out, n_out = best_module.xentropy_forward(pred_c, trg_c, truth_c, tixs_c)
        ctx.save_for_backward(pred_c, trg_c, truth_c, tixs_c, p_out)
        return p_out, n_out

    @staticmethod
    def backward(ctx, grad_p, grad_n):
        pred, trg, truth, tixs, p_out = ctx.saved_tensors
        
        # We need the JIT autotuned module to call the backward kernel
        # Since it is cached globally, autotune_xentropy returns instantly.
        best_module = autotune_xentropy(pred, trg, truth, tixs)
        
        grad_p_c = grad_p.contiguous()
        grad_n_c = grad_n.contiguous()
        
        grad_pred, grad_trg = best_module.xentropy_backward(
            grad_p_c, grad_n_c, pred, trg, truth, tixs, p_out
        )
        
        return grad_pred, grad_trg, None, None

def gemmmr_xentropy_cuda(pred, trg, truth, tixs=None):
    """
    User-facing PyTorch function matching the interface of standard Cross-Entropy
    but utilizing the dynamically-autotuned, hardware-aware fused CUDA kernel.
    """
    if tixs is None:
        tixs = torch.arange(trg.size(0), device=trg.device, dtype=torch.int64)
    p, n = GeMMMrXEntropy.apply(pred, trg, truth, tixs)
    return p - n
