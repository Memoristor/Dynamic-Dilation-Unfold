import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try:
    from . import _ext as _backend
except ImportError:
    raise ImportError("Could not import CUDA extension. Please build the extension first.")


class DynamicDilationUnfoldFunction(Function):
    @staticmethod
    def forward(ctx, input, dilation_map, kernel_size, stride=1, padding=0, dilation=1):
        """
        Args:
            input: (B, C, H, W)
            dilation_map: (B, 1, H_out, W_out) - dilation value for each output position
            kernel_size: int or tuple
            stride: int or tuple
            padding: int or tuple
            dilation: int or tuple (base dilation multiplier)
        
        Returns:
            output: (B, C*kH*kW, H_out*W_out)
        """
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size
        
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
        
        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h, pad_w = padding
        
        if isinstance(dilation, int):
            dilation_h = dilation_w = dilation
        else:
            dilation_h, dilation_w = dilation
        
        # Ensure dilation_map has correct shape
        batch_size = input.size(0)
        height_out = (input.size(2) + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
        width_out = (input.size(3) + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
        
        if dilation_map.dim() == 4:
            assert dilation_map.size(1) == 1, "dilation_map should have 1 channel"
            dilation_map = dilation_map.squeeze(1)  # (B, H_out, W_out)
        
        assert dilation_map.size(0) == batch_size
        assert dilation_map.size(1) == height_out
        assert dilation_map.size(2) == width_out
        
        # Flatten dilation_map for CUDA kernel
        dilation_map_flat = dilation_map.contiguous().view(batch_size, -1)  # (B, H_out*W_out)
        
        output = _backend.forward(
            input.contiguous(),
            dilation_map_flat.contiguous(),
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w
        )
        
        ctx.save_for_backward(input, dilation_map_flat)
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        
        return output
    
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, dilation_map = ctx.saved_tensors
        
        grad_input, grad_dilation_map = _backend.backward(
            grad_output.contiguous(),
            input,
            dilation_map,
            ctx.kernel_h, ctx.kernel_w,
            ctx.stride_h, ctx.stride_w,
            ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w
        )
        
        # Reshape grad_dilation_map back to original shape
        batch_size = input.size(0)
        height_out = (input.size(2) + 2 * ctx.pad_h - ctx.dilation_h * (ctx.kernel_h - 1) - 1) // ctx.stride_h + 1
        width_out = (input.size(3) + 2 * ctx.pad_w - ctx.dilation_w * (ctx.kernel_w - 1) - 1) // ctx.stride_w + 1
        grad_dilation_map = grad_dilation_map.view(batch_size, height_out, width_out).unsqueeze(1)
        
        return grad_input, grad_dilation_map, None, None, None, None


def dynamic_dilation_unfold(input, kernel_size, dilation_map, stride=1, padding=0, dilation=1):
    """
    Dynamic dilation unfold operation.
    
    Args:
        input: Input tensor of shape (B, C, H, W)
        kernel_size: Size of the sliding blocks (int or tuple)
        dilation_map: Dilation map of shape (B, 1, H_out, W_out)
        stride: Stride of the sliding blocks (int or tuple)
        padding: Implicit zero padding (int or tuple)
        dilation: Base dilation multiplier (int or tuple)
    
    Returns:
        output: Tensor of shape (B, C*kH*kW, H_out*W_out)
    
    Example:
        >>> input = torch.randn(2, 3, 32, 32).cuda()
        >>> dilation_map = torch.ones(2, 1, 30, 30).cuda()
        >>> output = dynamic_dilation_unfold(input, kernel_size=3, dilation_map=dilation_map)
        >>> print(output.shape)  # torch.Size([2, 27, 900])
    """
    return DynamicDilationUnfoldFunction.apply(
        input, dilation_map, kernel_size, stride, padding, dilation
    )