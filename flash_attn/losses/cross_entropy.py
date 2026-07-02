# Copyright (c) 2024, Tri Dao.

import torch
import torch.nn as nn

from flash_attn.ops.triton.cross_entropy import cross_entropy_loss


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        inplace_backward=False,
        process_group=None,
        return_z_loss=False,
    ):
        """
        Arguments:
            ignore_index: int. If labels == ignore_index, the loss is set to 0.0.
            label_smoothing: float
            lse_square_scale: float. If > 0, we add lse_square_scale * lse(logits) ^ 2 to the loss.
                This is also referred to as "z-loss".
            inplace_backward: bool. If True, we do the backward pass in-place by modifying the logits.
                This saves memory.
            process_group: if not None, we're doing Tensor Parallel: each process is responsible for
                one part of the vocab. The loss will be aggregated across processes.
            return_z_loss: bool. If True, we return the component of the loss contributed by
                the lse_square_scale value. This value is only for logging and does not support
                backprop.
        """
        super().__init__()
        if reduction not in ["mean", "none", "sum"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none' or 'sum'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.inplace_backward = inplace_backward
        self.process_group = process_group
        self.return_z_loss = return_z_loss

    def forward(self, input, target, precomputed_lse=None):
        """
        Arguments:
            input: (batch, vocab_size)
            target: (batch,)
        Returns:
            losses: (batch,) if reduction is 'none', else (1,), dtype float
            z_loss: (batch,) if reduction is 'none', else (1,), dtype float (if self.return_z_loss)
        """
        assert input.is_cuda and target.is_cuda, "Only support CUDA tensors"
        loss, z_loss = cross_entropy_loss(
            input,
            target,
            precomputed_lse=precomputed_lse,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            lse_square_scale=self.lse_square_scale,
            ignore_index=self.ignore_index,
            inplace_backward=self.inplace_backward,
            process_group=self.process_group,
        )
        if self.reduction == "mean":
            loss = loss.sum() / (target != self.ignore_index).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss

        if not self.return_z_loss:
            return loss

        if self.reduction == "mean":
            z_loss = z_loss.sum() / (target != self.ignore_index).sum()
        elif self.reduction == "sum":
            z_loss = z_loss.sum()
        else:
            z_loss = z_loss

        return loss, z_loss

# --- Custom JIT-Autotuned GeMMMapReduce Cross-Entropy ---
from flash_attn.losses.gemmmr_autotune import autotune_attention_forward

class GeMMMrXEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, trg, truth, tixs):
        pred_c = pred.contiguous()
        trg_c = trg.contiguous()
        truth_c = truth.contiguous()
        tixs_c = tixs.contiguous()
        
        # We trigger the Ninja JIT compiler here.
        # GeMMMapReduce's autotuner expects Q, K, V for attention.
        Q = torch.randn(128, 128, device=pred.device, dtype=pred.dtype)
        module = autotune_attention_forward(Q, Q, Q)
        
        p_out, n_out = module.xentropy_forward(pred_c, trg_c, truth_c, tixs_c)
        ctx.save_for_backward(pred_c, trg_c, truth_c, tixs_c, p_out)
        ctx.module = module
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
        
        return grad_pred, grad_trg, None, None


class GeMMMrXEntropyLoss(torch.nn.Module):
    """
    Fused GeMMMapReduce Cross-Entropy Loss
    Calculates the projection (out @ weight.T) and cross entropy in a single fused kernel!
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, out, weight, targets):
        """
        out: (M, D) features
        weight: (V, D) LM head weights
        targets: (M,) token class targets
        """
        tixs = torch.arange(weight.size(0), device=out.device, dtype=torch.int64)
        p, n = GeMMMrXEntropy.apply(out, weight, targets, tixs)
        return (p - n).mean()
