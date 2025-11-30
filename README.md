# Dynamic Dilation Unfold

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/pytorch-1.8+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/cuda-10.2+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 📖 Overview

**Dynamic Dilation Unfold** is a PyTorch extension that implements spatially-varying dilation rates for the unfold (im2col) operation with high-performance CUDA acceleration. Unlike standard `F.unfold` which uses a fixed dilation rate across the entire feature map, this implementation allows each spatial position to have its own dilation rate, enabling more flexible and adaptive feature extraction.

This is inspired by Deformable Convolution (DCNv2), but focuses specifically on the unfold operation, making it useful for:
- Attention mechanisms with adaptive receptive fields
- Multi-scale feature extraction
- Deformable sampling in transformers
- Custom pooling operations with varying receptive fields

## ✨ Key Features

- **🔄 Spatially-Varying Dilation**: Each output position can have different dilation rates
- **⚡ CUDA Acceleration**: Optimized CUDA kernels for maximum performance
- **🎓 Full Autograd Support**: Differentiable with respect to both input and dilation_map
- **🔧 Easy Integration**: Drop-in replacement compatible with `torch.nn.Unfold` API
- **💪 Mixed Precision**: Supports FP16, FP32, and FP64
- **🎯 Bilinear Interpolation**: Smooth sampling for sub-pixel positions
- **📦 Simple Installation**: One-line pip installation

## 🚀 Installation

### Prerequisites

```bash
# Check your environment
python --version  # >= 3.7
python -c "import torch; print(torch.__version__)"  # >= 1.8.0
python -c "import torch; print(torch.version.cuda)"  # Check CUDA version
nvcc --version  # Should match PyTorch CUDA version
```

### Install from Source

```bash
git clone https://github.com/Memoristor/Dynamic-Dilation-Unfold.git
cd dynamic_dilation_unfold
pip install -e .
```

If you encounter compilation errors:

```bash
# Clean previous builds
rm -rf build/ *.egg-info
python setup.py clean --all

# Reinstall with verbose output
pip install -e . -v
```

## 📚 Quick Start

### Basic Usage

```python
import torch
from dynamic_dilation_unfold import dynamic_dilation_unfold

# Create input tensor (B, C, H, W)
x = torch.randn(2, 3, 32, 32).cuda()

# Create dilation map (B, 1, H_out, W_out)
# Each value represents the dilation rate at that position
dilation_map = torch.ones(2, 1, 32, 32).cuda()

# Apply dynamic dilation unfold
output = dynamic_dilation_unfold(
    input=x,
    kernel_size=3,
    dilation_map=dilation_map,
    stride=1,
    padding=1,
    dilation=1  # Base dilation multiplier
)

print(f"Input shape:  {x.shape}")           # [2, 3, 32, 32]
print(f"Output shape: {output.shape}")      # [2, 27, 1024]
# Output: [B, C*kH*kW, H_out*W_out]
```

### Module Interface

```python
from dynamic_dilation_unfold import DynamicDilationUnfold

# Create reusable module
unfold = DynamicDilationUnfold(
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1
).cuda()

# Use in forward pass
output = unfold(x, dilation_map)
```

### Advanced: Spatially-Varying Dilation

```python
import torch
import matplotlib.pyplot as plt

# Create input
x = torch.randn(1, 3, 64, 64).cuda()

# Create spatially-varying dilation map
dilation_map = torch.ones(1, 1, 64, 64).cuda()

# Top-left: small receptive field (dilation=0.5)
dilation_map[:, :, :32, :32] = 0.5

# Top-right: medium receptive field (dilation=1.0)
dilation_map[:, :, :32, 32:] = 1.0

# Bottom-left: large receptive field (dilation=2.0)
dilation_map[:, :, 32:, :32] = 2.0

# Bottom-right: very large receptive field (dilation=3.0)
dilation_map[:, :, 32:, 32:] = 3.0

# Apply unfold
output = dynamic_dilation_unfold(x, kernel_size=3, dilation_map=dilation_map)

# Visualize dilation map
plt.imshow(dilation_map[0, 0].cpu(), cmap='viridis')
plt.colorbar(label='Dilation Rate')
plt.title('Spatially-Varying Dilation Map')
plt.show()
```

## 🔧 API Reference

### Function API

```python
dynamic_dilation_unfold(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    dilation_map: torch.Tensor,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1
) -> torch.Tensor
```

**Parameters:**

- **input** (*torch.Tensor*): Input tensor of shape $(B, C, H, W)$
- **kernel_size** (*int or tuple*): Size of the sliding window. Can be a single int or tuple $(k_H, k_W)$
- **dilation_map** (*torch.Tensor*): Dilation map of shape $(B, 1, H_{out}, W_{out})$. Each value $\geq 0$ indicates the dilation rate at that spatial position
- **stride** (*int or tuple, optional*): Stride of the sliding window. Default: 1
- **padding** (*int or tuple, optional*): Implicit zero padding. Default: 0
- **dilation** (*int or tuple, optional*): Base dilation multiplier. Default: 1

**Returns:**

- **output** (*torch.Tensor*): Output tensor of shape $(B, C \times k_H \times k_W, H_{out} \times W_{out})$

**Output Shape Calculation:**

$$H_{out} = \left\lfloor \frac{H + 2 \times \text{padding} - \text{dilation} \times (k_H - 1) - 1}{\text{stride}} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W + 2 \times \text{padding} - \text{dilation} \times (k_W - 1) - 1}{\text{stride}} \right\rfloor + 1$$

### Module API

```python
class DynamicDilationUnfold(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1
    )
    
    def forward(
        self,
        input: torch.Tensor,
        dilation_map: torch.Tensor
    ) -> torch.Tensor
```

## 📐 Mathematical Formulation

### Standard Unfold vs. Dynamic Dilation Unfold

The key difference between standard unfold and dynamic dilation unfold lies in how the sampling positions are computed.

**Standard Unfold** (`torch.nn.functional.unfold`):

For output position $(i, j)$ and kernel position $(m, n)$ where $m \in [0, k_H-1]$ and $n \in [0, k_W-1]$, the sampling coordinates use **fixed dilation**:

$$h_{in} = i \cdot s_h - p_h + m \cdot d_h$$

$$w_{in} = j \cdot s_w - p_w + n \cdot d_w$$

where $s_h, s_w$ are strides, $p_h, p_w$ are paddings, and $d_h, d_w$ are **fixed** dilation rates that apply uniformly across all spatial positions.

**Dynamic Dilation Unfold** (this implementation):

For output position $(i, j)$ and kernel position $(m, n)$, the sampling coordinates use **spatially-varying dilation**:

$$h_{in} = i \cdot s_h - p_h + \left(m - \frac{k_H - 1}{2}\right) \cdot d_h \cdot \mathbf{D}_{b,0,i,j}$$

$$w_{in} = j \cdot s_w - p_w + \left(n - \frac{k_W - 1}{2}\right) \cdot d_w \cdot \mathbf{D}_{b,0,i,j}$$

where $\mathbf{D}_{b,0,i,j}$ is the **spatially-varying** dilation rate from the dilation map at position $(i, j)$ for batch $b$.

**Note**: The term $(m - \frac{k_H - 1}{2})$ centers the kernel coordinates around the middle position, making the dilation expansion symmetric around the center. This is similar to the offset convention used in Deformable Convolution.

### Example: 3×3 Kernel

For a $3 \times 3$ kernel ($k_H = k_W = 3$), the kernel positions and their offsets are:

**Standard Unfold:**
- $(m, n) = (0, 0)$: offset $= (0, 0)$
- $(m, n) = (1, 1)$: offset $= (d_h, d_w)$ 
- $(m, n) = (2, 2)$: offset $= (2d_h, 2d_w)$

**Dynamic Dilation Unfold:**
- $(m, n) = (0, 0)$: offset $= (-d_h \cdot \mathbf{D}, -d_w \cdot \mathbf{D})$
- $(m, n) = (1, 1)$: offset $= (0, 0)$ (center remains fixed)
- $(m, n) = (2, 2)$: offset $= (d_h \cdot \mathbf{D}, d_w \cdot \mathbf{D})$

This makes the receptive field expand/contract symmetrically around each output position based on the local dilation value $\mathbf{D}_{b,0,i,j}$.

## 💡 Usage Example: Gradient Flow Analysis

```python
import torch
from dynamic_dilation_unfold import dynamic_dilation_unfold

# Enable gradient tracking
x = torch.randn(2, 3, 16, 16, requires_grad=True).cuda()
dilation_map = torch.ones(2, 1, 16, 16, requires_grad=True).cuda() * 1.5

# Forward pass
output = dynamic_dilation_unfold(x, kernel_size=3, dilation_map=dilation_map, padding=1)

# Backward pass
loss = output.mean()
loss.backward()

# Analyze gradients
print("Input gradient:")
print(f"  Shape: {x.grad.shape}")
print(f"  Mean: {x.grad.mean().item():.6f}")
print(f"  Std: {x.grad.std().item():.6f}")
print(f"  Has NaN: {torch.isnan(x.grad).any()}")

print("\nDilation map gradient:")
print(f"  Shape: {dilation_map.grad.shape}")
print(f"  Mean: {dilation_map.grad.mean().item():.6f}")
print(f"  Std: {dilation_map.grad.std().item():.6f}")
print(f"  Has NaN: {torch.isnan(dilation_map.grad).any()}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd tests
python test_dynamic_unfold.py
```

**Test Coverage:**

- ✅ Basic forward pass
- ✅ Gradient computation (input & dilation_map)
- ✅ Numerical gradient verification
- ✅ Different dilation values
- ✅ Spatially-varying dilation
- ✅ Module interface
- ✅ Edge cases (zero/large dilations)
- ✅ Mixed precision (FP16/FP32/FP64)

## 📊 Performance Benchmark

### Speed Comparison

```python
import torch
import time
from dynamic_dilation_unfold import dynamic_dilation_unfold

def benchmark(func, *args, warmup=10, iterations=100):
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        func(*args)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed / iterations * 1000  # ms

# Setup
B, C, H, W = 4, 64, 128, 128
x = torch.randn(B, C, H, W).cuda()
dilation_map = torch.ones(B, 1, H, W).cuda()

# Benchmark
time_dynamic = benchmark(lambda: dynamic_dilation_unfold(x, 3, dilation_map, padding=1))

print(f"Dynamic Dilation Unfold: {time_dynamic:.2f} ms")
print(f"Input shape: {x.shape}")
```

### Memory Usage

```python
import torch
from dynamic_dilation_unfold import dynamic_dilation_unfold

torch.cuda.reset_peak_memory_stats()

x = torch.randn(4, 64, 128, 128).cuda()
dilation_map = torch.ones(4, 1, 128, 128).cuda()

output = dynamic_dilation_unfold(x, kernel_size=3, dilation_map=dilation_map, padding=1)

memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"Peak memory usage: {memory_mb:.2f} MB")
```

## 🤔 FAQ

**Q: What's the difference from standard `F.unfold`?**

A: Standard `F.unfold` uses a fixed dilation rate for all positions. Dynamic Dilation Unfold allows each spatial location to have its own dilation rate specified by `dilation_map`.

**Q: Can I use this with convolutional layers?**

A: This is specifically for the unfold operation. For dynamic convolutions, consider using Deformable Convolution (DCNv2). However, you can combine this unfold with manual matrix multiplication to achieve similar effects.

**Q: What happens if dilation is 0?**

A: When dilation is 0, all kernel positions sample from the center point. This creates a "collapsed" receptive field, which might be useful for certain applications.

**Q: Is this differentiable?**

A: Yes! Both the input and dilation_map have full gradient support through bilinear interpolation.

**Q: Can I use fractional dilation values?**

A: Yes! The dilation_map supports any non-negative floating-point values, enabling smooth, continuous control over receptive field sizes.

**Q: What's the computational overhead?**

A: The main overhead comes from bilinear interpolation (4 samples per kernel position) and atomic operations in the backward pass. Typically 2-3x slower than standard unfold for similar configurations.

**Q: Does it support mixed precision training?**

A: Yes! The implementation supports FP16 (half), FP32 (float), and FP64 (double) precision.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

This project is inspired by:
- [Deformable Convolution V2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) by Chengdazhi
- [DCNv2](https://github.com/CharlesShang/DCNv2) by CharlesShang