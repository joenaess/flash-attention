import torch
import flashreduce_cuda

class FlashreduceAffineScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        """
        Forward pass for the optimized Mamba-style Affine State scan.
        x_t = a_t * x_{t-1} + b_t
        
        Args:
            a: (batch, seqlen, dim) scale tensor
            b: (batch, seqlen, dim) bias/input tensor
            
        Returns:
            x: (batch, seqlen, dim) output tensor
        """
        # Ensure inputs are contiguous float32/float16/bfloat16 CUDA tensors
        a_contig = a.contiguous()
        b_contig = b.contiguous()
        
        x, block_prefixes_a, block_prefixes_b = flashreduce_cuda.mamba_forward(a_contig, b_contig)
        
        # Save tensors needed for backward pass:
        # - a: needed for prefix/suffix coefficients
        # - x: needed to compute grad_a = e_t * x_{t-1}
        ctx.save_for_backward(a_contig, x)
        return x

    @staticmethod
    def backward(ctx, dy):
        """
        Backward pass using specialized Blelloch suffix scan.
        
        Args:
            dy: (batch, seqlen, dim) grad_output
            
        Returns:
            da: (batch, seqlen, dim) grad_a
            db: (batch, seqlen, dim) grad_b
        """
        a, x = ctx.saved_tensors
        dy_contig = dy.contiguous()
        
        da, db = flashreduce_cuda.mamba_backward(a, x, dy_contig)
        return da, db

def flashreduce_affine_scan(a, b):
    return FlashreduceAffineScan.apply(a, b)
