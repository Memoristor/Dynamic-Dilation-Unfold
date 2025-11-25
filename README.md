# Dynamic Dilation Unfold

<div align="center">

[English](#english) | [中文](#chinese)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.8+](https://img.shields.io/badge/pytorch-1.8+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/cuda-10.2+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

<a name="english"></a>
## English Documentation

### 📖 Overview

**Dynamic Dilation Unfold** is a PyTorch extension that implements spatially-varying dilation rates for the unfold (im2col) operation with high-performance CUDA acceleration. Unlike standard `F.unfold` which uses a fixed dilation rate across the entire feature map, this implementation allows each spatial position to have its own dilation rate, enabling more flexible and adaptive feature extraction.

This is inspired by Deformable Convolution (DCNv2), but focuses specifically on the unfold operation, making it useful for:
- Attention mechanisms with adaptive receptive fields
- Multi-scale feature extraction
- Deformable sampling in transformers
- Custom pooling operations with varying receptive fields

### ✨ Key Features

- **🔄 Spatially-Varying Dilation**: Each output position can have different dilation rates
- **⚡ CUDA Acceleration**: Optimized CUDA kernels for maximum performance
- **🎓 Full Autograd Support**: Differentiable with respect to both input and dilation_map
- **🔧 Easy Integration**: Drop-in replacement compatible with `torch.nn.Unfold` API
- **💪 Mixed Precision**: Supports FP16, FP32, and FP64
- **🎯 Bilinear Interpolation**: Smooth sampling for sub-pixel positions
- **📦 Simple Installation**: One-line pip installation

### 🚀 Installation

#### Prerequisites

```bash
# Check your environment
python --version  # >= 3.7
python -c "import torch; print(torch.__version__)"  # >= 1.8.0
python -c "import torch; print(torch.version.cuda)"  # Check CUDA version
nvcc --version  # Should match PyTorch CUDA version
```

#### Install from Source

```bash
git clone https://github.com/yourusername/dynamic_dilation_unfold.git
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

### 📚 Quick Start

#### Basic Usage

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

#### Module Interface

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

#### Advanced: Spatially-Varying Dilation

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

### 🔧 API Reference

#### Function API

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

- **input** (*torch.Tensor*): Input tensor of shape `(B, C, H, W)`
- **kernel_size** (*int or tuple*): Size of the sliding window. Can be a single int or tuple `(kH, kW)`
- **dilation_map** (*torch.Tensor*): Dilation map of shape `(B, 1, H_out, W_out)`. Each value >= 0 indicates the dilation rate at that spatial position
- **stride** (*int or tuple, optional*): Stride of the sliding window. Default: 1
- **padding** (*int or tuple, optional*): Implicit zero padding. Default: 0
- **dilation** (*int or tuple, optional*): Base dilation multiplier. Default: 1

**Returns:**

- **output** (*torch.Tensor*): Output tensor of shape `(B, C * kH * kW, H_out * W_out)`

**Output Shape Calculation:**

```python
H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
```

#### Module API

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

### 💡 Usage Example: Gradient Flow Analysis

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

### 🧪 Testing

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

**Sample Output:**

```
======================================================================
Running Dynamic Dilation Unfold Tests
======================================================================

=== Test Basic Forward ===
Output shape: torch.Size([2, 27, 64])
✓ Basic forward test passed

=== Test Gradient Input ===
Input gradient shape: torch.Size([2, 2, 6, 6])
Input gradient mean: 0.150234
Input gradient std: 0.489123
✓ Input gradient test passed

=== Test Numerical Gradient ===
Max relative error: 0.000023
Mean relative error: 0.000008
✓ Numerical gradient test passed

======================================================================
✓ All tests passed!
======================================================================
```

### 📊 Performance Benchmark

#### Speed Comparison

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

#### Memory Usage

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

### 🔬 How It Works

#### Algorithm Overview

For each output position `(b, c, h_out, w_out)`:

1. **Read Dilation Rate**: Get `d = dilation_map[b, 0, h_out, w_out]`

2. **Compute Sampling Positions**: For each kernel position `(kh, kw)`:
   ```python
   h_in = h_out * stride_h - pad_h + (kh - (K-1)/2) * base_dilation * d
   w_in = w_out * stride_w - pad_w + (kw - (K-1)/2) * base_dilation * d
   ```

3. **Bilinear Interpolation**: Sample input at `(h_in, w_in)` using bilinear interpolation

4. **Store Result**: Place sampled value in output tensor

#### Backward Pass

**Gradient w.r.t. Input:**
- Distribute output gradients to the 4 neighboring pixels used in bilinear interpolation
- Use atomic operations for thread-safe accumulation

**Gradient w.r.t. Dilation Map:**
- Compute how the sampling position changes with dilation
- Apply chain rule through bilinear interpolation
- Accumulate across all kernel positions and channels

#### Illustration

```
Standard Unfold (dilation=1):          Dynamic Dilation (dilation_map):
                                       
    [·][·][·]                              [·]    [·]    [·]
    [·][X][·]      kernel_size=3           
    [·][·][·]                              [·]    [X]    [·]      dilation=2.0
                                           
                                           [·]    [·]    [·]


    [·][·][·]                              [·]      [·]      [·]
    [·][X][·]      kernel_size=3           
    [·][·][·]                              [·]      [X]      [·]  dilation=3.0
                                           
                                           [·]      [·]      [·]
```

### 🤔 FAQ

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

### 📄 License

This project is licensed under the MIT License.

### 🙏 Acknowledgments

This project is inspired by:
- [Deformable Convolution V2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) by Chengdazhi
- [DCNv2](https://github.com/CharlesShang/DCNv2) by CharlesShang

---

<a name="chinese"></a>
## 中文文档

### 📖 概述

**Dynamic Dilation Unfold（动态扩张展开）** 是一个PyTorch扩展，实现了具有空间可变扩张率的unfold（im2col）操作，并提供高性能的CUDA加速。与标准的`F.unfold`在整个特征图上使用固定扩张率不同，该实现允许每个空间位置拥有自己的扩张率，从而实现更灵活和自适应的特征提取。

该项目受到Deformable Convolution (DCNv2)的启发，但专注于unfold操作，适用于：
- 具有自适应感受野的注意力机制
- 多尺度特征提取
- Transformer中的可变形采样
- 具有可变感受野的自定义池化操作

### ✨ 核心特性

- **🔄 空间可变扩张率**：每个输出位置可以有不同的扩张率
- **⚡ CUDA加速**：优化的CUDA内核，性能卓越
- **🎓 完整的自动微分支持**：对输入和扩张图都可微分
- **🔧 易于集成**：API兼容`torch.nn.Unfold`，可直接替换
- **💪 混合精度**：支持FP16、FP32和FP64
- **🎯 双线性插值**：亚像素位置的平滑采样
- **📦 简单安装**：一行命令完成安装

### 🚀 安装

#### 环境要求

```bash
# 检查环境
python --version  # >= 3.7
python -c "import torch; print(torch.__version__)"  # >= 1.8.0
python -c "import torch; print(torch.version.cuda)"  # 检查CUDA版本
nvcc --version  # 应与PyTorch的CUDA版本匹配
```

#### 从源码安装

```bash
git clone https://github.com/yourusername/dynamic_dilation_unfold.git
cd dynamic_dilation_unfold
pip install -e .
```

如果遇到编译错误：

```bash
# 清理之前的构建
rm -rf build/ *.egg-info
python setup.py clean --all

# 重新安装（显示详细输出）
pip install -e . -v
```

### 📚 快速开始

#### 基础用法

```python
import torch
from dynamic_dilation_unfold import dynamic_dilation_unfold

# 创建输入张量 (B, C, H, W)
x = torch.randn(2, 3, 32, 32).cuda()

# 创建扩张图 (B, 1, H_out, W_out)
# 每个值代表该位置的扩张率
dilation_map = torch.ones(2, 1, 32, 32).cuda()

# 应用动态扩张unfold
output = dynamic_dilation_unfold(
    input=x,
    kernel_size=3,
    dilation_map=dilation_map,
    stride=1,
    padding=1,
    dilation=1  # 基础扩张系数
)

print(f"输入形状:  {x.shape}")           # [2, 3, 32, 32]
print(f"输出形状: {output.shape}")      # [2, 27, 1024]
# 输出: [B, C*kH*kW, H_out*W_out]
```

#### 模块接口

```python
from dynamic_dilation_unfold import DynamicDilationUnfold

# 创建可重用的模块
unfold = DynamicDilationUnfold(
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1
).cuda()

# 在前向传播中使用
output = unfold(x, dilation_map)
```

#### 进阶：空间可变扩张

```python
import torch
import matplotlib.pyplot as plt

# 创建输入
x = torch.randn(1, 3, 64, 64).cuda()

# 创建空间可变的扩张图
dilation_map = torch.ones(1, 1, 64, 64).cuda()

# 左上：小感受野 (扩张率=0.5)
dilation_map[:, :, :32, :32] = 0.5

# 右上：中等感受野 (扩张率=1.0)
dilation_map[:, :, :32, 32:] = 1.0

# 左下：大感受野 (扩张率=2.0)
dilation_map[:, :, 32:, :32] = 2.0

# 右下：超大感受野 (扩张率=3.0)
dilation_map[:, :, 32:, 32:] = 3.0

# 应用unfold
output = dynamic_dilation_unfold(x, kernel_size=3, dilation_map=dilation_map)

# 可视化扩张图
plt.imshow(dilation_map[0, 0].cpu(), cmap='viridis')
plt.colorbar(label='扩张率')
plt.title('空间可变扩张图')
plt.show()
```

### 🔧 API参考

#### 函数API

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

**参数：**

- **input** (*torch.Tensor*): 输入张量，形状为`(B, C, H, W)`
- **kernel_size** (*int 或 tuple*): 滑动窗口的大小，可以是单个整数或元组`(kH, kW)`
- **dilation_map** (*torch.Tensor*): 扩张图，形状为`(B, 1, H_out, W_out)`。每个值>=0，表示该空间位置的扩张率
- **stride** (*int 或 tuple, 可选*): 滑动窗口的步长。默认值: 1
- **padding** (*int 或 tuple, 可选*): 隐式零填充。默认值: 0
- **dilation** (*int 或 tuple, 可选*): 基础扩张系数。默认值: 1

**返回值：**

- **output** (*torch.Tensor*): 输出张量，形状为`(B, C * kH * kW, H_out * W_out)`

**输出形状计算：**

```python
H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
```

#### 模块API

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

### 💡 使用示例：梯度流分析

```python
import torch
from dynamic_dilation_unfold import dynamic_dilation_unfold

# 启用梯度跟踪
x = torch.randn(2, 3, 16, 16, requires_grad=True).cuda()
dilation_map = torch.ones(2, 1, 16, 16, requires_grad=True).cuda() * 1.5

# 前向传播
output = dynamic_dilation_unfold(x, kernel_size=3, dilation_map=dilation_map, padding=1)

# 反向传播
loss = output.mean()
loss.backward()

# 分析梯度
print("输入梯度:")
print(f"  形状: {x.grad.shape}")
print(f"  平均值: {x.grad.mean().item():.6f}")
print(f"  标准差: {x.grad.std().item():.6f}")
print(f"  有NaN: {torch.isnan(x.grad).any()}")

print("\n扩张图梯度:")
print(f"  形状: {dilation_map.grad.shape}")
print(f"  平均值: {dilation_map.grad.mean().item():.6f}")
print(f"  标准差: {dilation_map.grad.std().item():.6f}")
print(f"  有NaN: {torch.isnan(dilation_map.grad).any()}")
```

### 🧪 测试

运行完整的测试套件：

```bash
cd tests
python test_dynamic_unfold.py
```

**测试覆盖：**

- ✅ 基本前向传播
- ✅ 梯度计算（输入和扩张图）
- ✅ 数值梯度验证
- ✅ 不同扩张值
- ✅ 空间可变扩张
- ✅ 模块接口
- ✅ 边界情况（零/大扩张率）
- ✅ 混合精度（FP16/FP32/FP64）

**示例输出：**

```
======================================================================
运行动态扩张Unfold测试
======================================================================

=== 测试基本前向传播 ===
输出形状: torch.Size([2, 27, 64])
✓ 基本前向测试通过

=== 测试输入梯度 ===
输入梯度形状: torch.Size([2, 2, 6, 6])
输入梯度平均值: 0.150234
输入梯度标准差: 0.489123
✓ 输入梯度测试通过

=== 测试数值梯度 ===
最大相对误差: 0.000023
平均相对误差: 0.000008
✓ 数值梯度测试通过

======================================================================
✓ 所有测试通过！
======================================================================
```

### 📊 性能基准测试

#### 速度对比

```python
import torch
import time
from dynamic_dilation_unfold import dynamic_dilation_unfold

def benchmark(func, *args, warmup=10, iterations=100):
    # 预热
    for _ in range(warmup):
        func(*args)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        func(*args)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed / iterations * 1000  # ms

# 设置
B, C, H, W = 4, 64, 128, 128
x = torch.randn(B, C, H, W).cuda()
dilation_map = torch.ones(B, 1, H, W).cuda()

# 基准测试
time_dynamic = benchmark(lambda: dynamic_dilation_unfold(x, 3, dilation_map, padding=1))

print(f"动态扩张Unfold: {time_dynamic:.2f} ms")
print(f"输入形状: {x.shape}")
```

#### 内存使用

```python
import torch
from dynamic_dilation_unfold import dynamic_dilation_unfold

torch.cuda.reset_peak_memory_stats()

x = torch.randn(4, 64, 128, 128).cuda()
dilation_map = torch.ones(4, 1, 128, 128).cuda()

output = dynamic_dilation_unfold(x, kernel_size=3, dilation_map=dilation_map, padding=1)

memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"峰值内存使用: {memory_mb:.2f} MB")
```

### 🔬 工作原理

#### 算法概述

对于每个输出位置`(b, c, h_out, w_out)`：

1. **读取扩张率**：获取`d = dilation_map[b, 0, h_out, w_out]`

2. **计算采样位置**：对于每个kernel位置`(kh, kw)`：
   ```python
   h_in = h_out * stride_h - pad_h + (kh - (K-1)/2) * base_dilation * d
   w_in = w_out * stride_w - pad_w + (kw - (K-1)/2) * base_dilation * d
   ```

3. **双线性插值**：在`(h_in, w_in)`位置使用双线性插值采样输入

4. **存储结果**：将采样值放入输出张量

#### 反向传播

**对输入的梯度：**
- 将输出梯度分配到双线性插值使用的4个邻近像素
- 使用原子操作保证线程安全的累积

**对扩张图的梯度：**
- 计算采样位置如何随扩张率变化
- 通过双线性插值应用链式法则
- 在所有kernel位置和通道上累积

#### 示意图

```
标准Unfold (扩张率=1):              动态扩张 (dilation_map):
                                       
    [·][·][·]                              [·]    [·]    [·]
    [·][X][·]      kernel_size=3           
    [·][·][·]                              [·]    [X]    [·]      扩张率=2.0
                                           
                                           [·]    [·]    [·]


    [·][·][·]                              [·]      [·]      [·]
    [·][X][·]      kernel_size=3           
    [·][·][·]                              [·]      [X]      [·]  扩张率=3.0
                                           
                                           [·]      [·]      [·]
```

### 🤔 常见问题

**Q: 与标准`F.unfold`有什么区别？**

A: 标准`F.unfold`对所有位置使用固定的扩张率。动态扩张Unfold允许每个空间位置具有由`dilation_map`指定的自己的扩张率。

**Q: 可以与卷积层一起使用吗？**

A: 这专门用于unfold操作。对于动态卷积，考虑使用Deformable Convolution (DCNv2)。但是，您可以将此unfold与手动矩阵乘法结合以实现类似效果。

**Q: 如果扩张率为0会发生什么？**

A: 当扩张率为0时，所有kernel位置都从中心点采样。这创建了一个"塌陷"的感受野，在某些应用中可能很有用。

**Q: 这是可微分的吗？**

A: 是的！输入和dilation_map都通过双线性插值具有完整的梯度支持。

**Q: 可以使用小数扩张值吗？**

A: 可以！dilation_map支持任何非负浮点值，实现对感受野大小的平滑、连续控制。

**Q: 计算开销是多少？**

A: 主要开销来自双线性插值（每个kernel位置4次采样）和反向传播中的原子操作。对于类似配置，通常比标准unfold慢2-3倍。

**Q: 支持混合精度训练吗？**

A: 支持！该实现支持FP16（半精度）、FP32（单精度）和FP64（双精度）。

### 📄 许可证

本项目采用MIT许可证。

### 🙏 致谢

本项目受以下项目启发：
- [Deformable Convolution V2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) by Chengdazhi
- [DCNv2](https://github.com/CharlesShang/DCNv2) by CharlesShang

---
