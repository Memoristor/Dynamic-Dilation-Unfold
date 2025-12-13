import torch
import torch.nn as nn
from .functions import dynamic_dilation_unfold


class DynamicDilationUnfold(nn.Module):
    """
    PyTorch module for dynamic dilation unfold operation with optional grouping.
    
    Args:
        kernel_size: Size of the sliding blocks
        stride: Stride of the sliding blocks
        padding: Implicit zero padding
        dilation: Base dilation multiplier
        groups: Number of groups for grouped dynamic dilation unfold
               (Input channels must be divisible by groups)
    
    Example:
        >>> unfold = DynamicDilationUnfold(kernel_size=3, stride=1, padding=1, groups=2).cuda()
        >>> input = torch.randn(2, 6, 32, 32).cuda()  # 6 channels divisible by 2 groups
        >>> dilation_map = torch.ones(2, 2, 32, 32).cuda()  # 2 groups
        >>> output = unfold(input, dilation_map)
    """
    
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(DynamicDilationUnfold, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
    def forward(self, input, dilation_map):
        """
        Args:
            input: (B, C, H, W), where C must be divisible by self.groups
            dilation_map: (B, G, H_out, W_out), where G is the number of groups
        
        Returns:
            output: (B, C*kH*kW, H_out*W_out)
        """
        return dynamic_dilation_unfold(
            input, self.kernel_size, dilation_map,
            self.stride, self.padding, self.dilation, self.groups
        )
    
    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, ' \
               f'padding={self.padding}, dilation={self.dilation}, groups={self.groups}'