import torch
import torch.nn as nn
from .functions import dynamic_dilation_unfold


class DynamicDilationUnfold(nn.Module):
    """
    PyTorch module for dynamic dilation unfold operation.
    
    Args:
        kernel_size: Size of the sliding blocks
        stride: Stride of the sliding blocks
        padding: Implicit zero padding
        dilation: Base dilation multiplier
    
    Example:
        >>> unfold = DynamicDilationUnfold(kernel_size=3, stride=1, padding=1).cuda()
        >>> input = torch.randn(2, 3, 32, 32).cuda()
        >>> dilation_map = torch.ones(2, 1, 32, 32).cuda()
        >>> output = unfold(input, dilation_map)
    """
    
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super(DynamicDilationUnfold, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def forward(self, input, dilation_map):
        """
        Args:
            input: (B, C, H, W)
            dilation_map: (B, 1, H_out, W_out)
        
        Returns:
            output: (B, C*kH*kW, H_out*W_out)
        """
        return dynamic_dilation_unfold(
            input, self.kernel_size, dilation_map,
            self.stride, self.padding, self.dilation
        )
    
    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, ' \
               f'padding={self.padding}, dilation={self.dilation}'