#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N) {
    return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

// Custom atomicAdd for half precision
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
static __inline__ __device__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        half tmpres = __hadd(__half(hsum), __half(val));
        hsum = __half_raw(tmpres);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}
#elif defined(__CUDA_ARCH__)
// For compute capability >= 7.0, use native atomicAdd for half
static __inline__ __device__ void atomicAdd(c10::Half* address, c10::Half val) {
    atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(val));
}
#endif

// Atomic add for c10::Half that works for all CUDA architectures
__device__ __forceinline__ void atomic_add(c10::Half* address, c10::Half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(val));
#else
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        half tmpres = __hadd(reinterpret_cast<const __half&>(hsum), static_cast<__half>(val));
        hsum = __half_raw(tmpres);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
#endif
}

// Overload for float
__device__ __forceinline__ void atomic_add(float* address, float val) {
    atomicAdd(address, val);
}

// Overload for double
__device__ __forceinline__ void atomic_add(double* address, double val) {
    atomicAdd(address, val);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(
    const scalar_t* input,
    const int height,
    const int width,
    scalar_t h,
    scalar_t w) {
    
    if (h <= -1 || height <= h || w <= -1 || width <= w) {
        return 0;
    }
    
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    
    scalar_t lh = h - h_low;
    scalar_t lw = w - w_low;
    scalar_t hh = 1 - lh;
    scalar_t hw = 1 - lw;
    
    scalar_t v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = input[h_low * width + w_low];
    
    scalar_t v2 = 0;
    if (h_low >= 0 && w_high < width)
        v2 = input[h_low * width + w_high];
    
    scalar_t v3 = 0;
    if (h_high < height && w_low >= 0)
        v3 = input[h_high * width + w_low];
    
    scalar_t v4 = 0;
    if (h_high < height && w_high < width)
        v4 = input[h_high * width + w_high];
    
    scalar_t w1 = hh * hw;
    scalar_t w2 = hh * lw;
    scalar_t w3 = lh * hw;
    scalar_t w4 = lh * lw;
    
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename scalar_t>
__global__ void dynamic_dilation_unfold_forward_kernel(
    const int n,
    const scalar_t* input,
    const scalar_t* dilation_map,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int height_out,
    const int width_out,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int channels_per_group,
    scalar_t* output) {
    
    CUDA_KERNEL_LOOP(index, n) {
        // index = b * (channels * kernel_h * kernel_w) * (height_out * width_out) + 
        //         c * (height_out * width_out) + h_out * width_out + w_out
        
        const int w_out = index % width_out;
        const int h_out = (index / width_out) % height_out;
        const int c_out = (index / width_out / height_out) % (channels * kernel_h * kernel_w);
        const int b = index / width_out / height_out / (channels * kernel_h * kernel_w);
        
        const int c = c_out / (kernel_h * kernel_w);
        const int kernel_idx = c_out % (kernel_h * kernel_w);
        const int kh = kernel_idx / kernel_w;
        const int kw = kernel_idx % kernel_w;
        
        // Calculate group index based on channel
        const int group_idx = c / channels_per_group;
        
        // Get dilation value for this output position and group
        const int dilation_map_idx = b * groups * height_out * width_out + 
                                     group_idx * height_out * width_out + 
                                     h_out * width_out + w_out;
        const scalar_t dilation_value = dilation_map[dilation_map_idx];
        
        // Calculate input position
        const scalar_t h_in_base = h_out * stride_h - pad_h;
        const scalar_t w_in_base = w_out * stride_w - pad_w;
        
        // Apply dynamic dilation
        const scalar_t h_offset = (kh - (kernel_h - 1) / 2.0) * dilation_h * dilation_value;
        const scalar_t w_offset = (kw - (kernel_w - 1) / 2.0) * dilation_w * dilation_value;
        
        const scalar_t h_in = h_in_base + h_offset + (kernel_h - 1) / 2.0 * dilation_h;
        const scalar_t w_in = w_in_base + w_offset + (kernel_w - 1) / 2.0 * dilation_w;
        
        // Bilinear interpolation
        const scalar_t* input_ptr = input + (b * channels + c) * height * width;
        output[index] = bilinear_interpolate(input_ptr, height, width, h_in, w_in);
    }
}

template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(
    const scalar_t* input,
    const int height,
    const int width,
    scalar_t h,
    scalar_t w,
    scalar_t& grad_h,
    scalar_t& grad_w,
    const scalar_t grad_output) {
    
    grad_h = 0;
    grad_w = 0;
    
    if (h <= -1 || height <= h || w <= -1 || width <= w) {
        return;
    }
    
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;
    
    scalar_t lh = h - h_low;
    scalar_t lw = w - w_low;
    
    scalar_t v1 = 0, v2 = 0, v3 = 0, v4 = 0;
    bool valid_v1 = false, valid_v2 = false, valid_v3 = false, valid_v4 = false;
    
    if (h_low >= 0 && w_low >= 0) {
        v1 = input[h_low * width + w_low];
        valid_v1 = true;
    }
    if (h_low >= 0 && w_high < width) {
        v2 = input[h_low * width + w_high];
        valid_v2 = true;
    }
    if (h_high < height && w_low >= 0) {
        v3 = input[h_high * width + w_low];
        valid_v3 = true;
    }
    if (h_high < height && w_high < width) {
        v4 = input[h_high * width + w_high];
        valid_v4 = true;
    }
    
    if (valid_v1 || valid_v3) {
        grad_h += (v3 - v1) * (1 - lw) * grad_output;
    }
    if (valid_v2 || valid_v4) {
        grad_h += (v4 - v2) * lw * grad_output;
    }
    if (valid_v1 || valid_v2) {
        grad_w += (v2 - v1) * (1 - lh) * grad_output;
    }
    if (valid_v3 || valid_v4) {
        grad_w += (v4 - v3) * lh * grad_output;
    }
}

template <typename scalar_t>
__global__ void dynamic_dilation_unfold_backward_input_kernel(
    const int n,
    const scalar_t* grad_output,
    const scalar_t* dilation_map,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int height_out,
    const int width_out,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int channels_per_group,
    scalar_t* grad_input) {
    
    CUDA_KERNEL_LOOP(index, n) {
        const int w_out = index % width_out;
        const int h_out = (index / width_out) % height_out;
        const int c_out = (index / width_out / height_out) % (channels * kernel_h * kernel_w);
        const int b = index / width_out / height_out / (channels * kernel_h * kernel_w);
        
        const int c = c_out / (kernel_h * kernel_w);
        const int kernel_idx = c_out % (kernel_h * kernel_w);
        const int kh = kernel_idx / kernel_w;
        const int kw = kernel_idx % kernel_w;
        
        // Calculate group index based on channel
        const int group_idx = c / channels_per_group;
        
        const int dilation_map_idx = b * groups * height_out * width_out + 
                                     group_idx * height_out * width_out + 
                                     h_out * width_out + w_out;
        const scalar_t dilation_value = dilation_map[dilation_map_idx];
        
        const scalar_t h_in_base = h_out * stride_h - pad_h;
        const scalar_t w_in_base = w_out * stride_w - pad_w;
        
        const scalar_t h_offset = (kh - (kernel_h - 1) / 2.0) * dilation_h * dilation_value;
        const scalar_t w_offset = (kw - (kernel_w - 1) / 2.0) * dilation_w * dilation_value;
        
        const scalar_t h_in = h_in_base + h_offset + (kernel_h - 1) / 2.0 * dilation_h;
        const scalar_t w_in = w_in_base + w_offset + (kernel_w - 1) / 2.0 * dilation_w;
        
        if (h_in <= -1 || height <= h_in || w_in <= -1 || width <= w_in) {
            continue;
        }
        
        int h_low = floor(h_in);
        int w_low = floor(w_in);
        int h_high = h_low + 1;
        int w_high = w_low + 1;
        
        scalar_t lh = h_in - h_low;
        scalar_t lw = w_in - w_low;
        scalar_t hh = 1 - lh;
        scalar_t hw = 1 - lw;
        
        const scalar_t grad_out_value = grad_output[index];
        scalar_t* grad_input_ptr = grad_input + (b * channels + c) * height * width;
        
        if (h_low >= 0 && w_low >= 0) {
            atomic_add(&grad_input_ptr[h_low * width + w_low], hh * hw * grad_out_value);
        }
        if (h_low >= 0 && w_high < width) {
            atomic_add(&grad_input_ptr[h_low * width + w_high], hh * lw * grad_out_value);
        }
        if (h_high < height && w_low >= 0) {
            atomic_add(&grad_input_ptr[h_high * width + w_low], lh * hw * grad_out_value);
        }
        if (h_high < height && w_high < width) {
            atomic_add(&grad_input_ptr[h_high * width + w_high], lh * lw * grad_out_value);
        }
    }
}

template <typename scalar_t>
__global__ void dynamic_dilation_unfold_backward_dilation_kernel(
    const int n,
    const scalar_t* grad_output,
    const scalar_t* input,
    const scalar_t* dilation_map,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int height_out,
    const int width_out,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const int groups,
    const int channels_per_group,
    scalar_t* grad_dilation_map) {
    
    CUDA_KERNEL_LOOP(index, n) {
        const int w_out = index % width_out;
        const int h_out = (index / width_out) % height_out;
        const int c_out = (index / width_out / height_out) % (channels * kernel_h * kernel_w);
        const int b = index / width_out / height_out / (channels * kernel_h * kernel_w);
        
        const int c = c_out / (kernel_h * kernel_w);
        const int kernel_idx = c_out % (kernel_h * kernel_w);
        const int kh = kernel_idx / kernel_w;
        const int kw = kernel_idx % kernel_w;
        
        // Calculate group index based on channel
        const int group_idx = c / channels_per_group;
        
        const int dilation_map_idx = b * groups * height_out * width_out + 
                                     group_idx * height_out * width_out + 
                                     h_out * width_out + w_out;
        const scalar_t dilation_value = dilation_map[dilation_map_idx];
        
        const scalar_t h_in_base = h_out * stride_h - pad_h;
        const scalar_t w_in_base = w_out * stride_w - pad_w;
        
        const scalar_t h_offset = (kh - (kernel_h - 1) / 2.0) * dilation_h * dilation_value;
        const scalar_t w_offset = (kw - (kernel_w - 1) / 2.0) * dilation_w * dilation_value;
        
        const scalar_t h_in = h_in_base + h_offset + (kernel_h - 1) / 2.0 * dilation_h;
        const scalar_t w_in = w_in_base + w_offset + (kernel_w - 1) / 2.0 * dilation_w;
        
        const scalar_t* input_ptr = input + (b * channels + c) * height * width;
        
        scalar_t grad_h = 0;
        scalar_t grad_w = 0;
        bilinear_interpolate_gradient(
            input_ptr, height, width, h_in, w_in,
            grad_h, grad_w, grad_output[index]);
        
        const scalar_t grad_dilation = grad_h * (kh - (kernel_h - 1) / 2.0) * dilation_h +
                                       grad_w * (kw - (kernel_w - 1) / 2.0) * dilation_w;
        
        atomic_add(&grad_dilation_map[dilation_map_idx], grad_dilation);
    }
}

torch::Tensor dynamic_dilation_unfold_cuda_forward(
    torch::Tensor input,
    torch::Tensor dilation_map,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups, int channels_per_group) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int height_out = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, channels * kernel_h * kernel_w, height_out * width_out},
                               input.options());
    
    const int num_kernels = batch_size * channels * kernel_h * kernel_w * height_out * width_out;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamic_dilation_unfold_forward_cuda", ([&] {
        dynamic_dilation_unfold_forward_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_kernels,
                input.data_ptr<scalar_t>(),
                dilation_map.data_ptr<scalar_t>(),
                batch_size, channels, height, width,
                height_out, width_out,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dilation_h, dilation_w,
                groups, channels_per_group,
                output.data_ptr<scalar_t>());
    }));
    
    return output;
}

std::vector<torch::Tensor> dynamic_dilation_unfold_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor dilation_map,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups, int channels_per_group) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    
    const int height_out = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto grad_input = torch::zeros_like(input);
    auto grad_dilation_map = torch::zeros_like(dilation_map);
    
    const int num_kernels = batch_size * channels * kernel_h * kernel_w * height_out * width_out;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "dynamic_dilation_unfold_backward_cuda", ([&] {
        dynamic_dilation_unfold_backward_input_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_kernels,
                grad_output.data_ptr<scalar_t>(),
                dilation_map.data_ptr<scalar_t>(),
                batch_size, channels, height, width,
                height_out, width_out,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dilation_h, dilation_w,
                groups, channels_per_group,
                grad_input.data_ptr<scalar_t>());
        
        dynamic_dilation_unfold_backward_dilation_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_kernels,
                grad_output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                dilation_map.data_ptr<scalar_t>(),
                batch_size, channels, height, width,
                height_out, width_out,
                kernel_h, kernel_w,
                stride_h, stride_w,
                pad_h, pad_w,
                dilation_h, dilation_w,
                groups, channels_per_group,
                grad_dilation_map.data_ptr<scalar_t>());
    }));
    
    return {grad_input, grad_dilation_map};
}