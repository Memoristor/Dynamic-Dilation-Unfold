import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dynamic_dilation_unfold import dynamic_dilation_unfold

# ============================================================================
# Part 1: Manual Implementation for Verification
# ============================================================================

def manual_dynamic_dilation_unfold(input, kernel_size, dilation_map, stride=1, padding=0, dilation=1):
    """
    Manual implementation using pure PyTorch for verification
    This is slow but easy to verify correctness
    """
    B, C, H, W = input.shape
    
    if isinstance(kernel_size, int):
        kH = kW = kernel_size
    else:
        kH, kW = kernel_size
    
    if isinstance(stride, int):
        sH = sW = stride
    else:
        sH, sW = stride
    
    if isinstance(padding, int):
        pH = pW = padding
    else:
        pH, pW = padding
    
    if isinstance(dilation, int):
        dH = dW = dilation
    else:
        dH, dW = dilation
    
    # Calculate output dimensions
    H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    
    # Apply padding
    if pH > 0 or pW > 0:
        input_padded = F.pad(input, (pW, pW, pH, pH), mode='constant', value=0)
    else:
        input_padded = input
    
    # Adjust dimensions after padding
    H_padded = H + 2 * pH
    W_padded = W + 2 * pW
    
    # Initialize output
    output = torch.zeros(B, C * kH * kW, H_out * W_out, device=input.device, dtype=input.dtype)
    
    # Ensure dilation_map has correct shape
    if dilation_map.dim() == 4:
        dilation_map = dilation_map.squeeze(1)  # (B, H_out, W_out)
    
    # Manual unfold with dynamic dilation
    for b in range(B):
        for i in range(H_out):
            for j in range(W_out):
                # Get dilation value for this position
                d_val = dilation_map[b, i, j].item()
                
                for c in range(C):
                    for m in range(kH):
                        for n in range(kW):
                            # Calculate sampling position with dynamic dilation
                            # Using centered kernel convention: (m - (kH-1)/2)
                            h_offset = (m - (kH - 1) / 2.0) * dH * d_val
                            w_offset = (n - (kW - 1) / 2.0) * dW * d_val
                            
                            h_in = i * sH + h_offset + (kH - 1) / 2.0 * dH
                            w_in = j * sW + w_offset + (kW - 1) / 2.0 * dW
                            
                            # Bilinear interpolation
                            value = bilinear_sample(input_padded[b, c], h_in, w_in)
                            
                            # Store in output
                            out_c = c * kH * kW + m * kW + n
                            out_idx = i * W_out + j
                            output[b, out_c, out_idx] = value
    
    return output


def bilinear_sample(feature_map, h, w):
    """
    Bilinear sampling at non-integer positions
    feature_map: (H, W)
    h, w: float coordinates
    """
    H, W = feature_map.shape
    
    # Boundary check
    if h < -1 or h > H or w < -1 or w > W:
        return torch.tensor(0.0, device=feature_map.device, dtype=feature_map.dtype)
    
    h_low = int(np.floor(h))
    w_low = int(np.floor(w))
    h_high = h_low + 1
    w_high = w_low + 1
    
    lh = h - h_low
    lw = w - w_low
    hh = 1 - lh
    hw = 1 - lw
    
    v1 = feature_map[h_low, w_low] if (0 <= h_low < H and 0 <= w_low < W) else 0
    v2 = feature_map[h_low, w_high] if (0 <= h_low < H and 0 <= w_high < W) else 0
    v3 = feature_map[h_high, w_low] if (0 <= h_high < H and 0 <= w_low < W) else 0
    v4 = feature_map[h_high, w_high] if (0 <= h_high < H and 0 <= w_high < W) else 0
    
    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


# ============================================================================
# Part 2: Forward Pass Verification
# ============================================================================

def test_forward_pass():
    """Test forward pass correctness"""
    print("=" * 80)
    print("FORWARD PASS VERIFICATION")
    print("=" * 80)
    
    # Test configurations
    test_cases = [
        {
            'name': 'Small Fixed Dilation',
            'B': 1, 'C': 2, 'H': 8, 'W': 8,
            'kernel_size': 3, 'stride': 1, 'padding': 1,
            'dilation_value': 1.0
        },
        {
            'name': 'Large Fixed Dilation',
            'B': 2, 'C': 3, 'H': 16, 'W': 16,
            'kernel_size': 3, 'stride': 1, 'padding': 1,
            'dilation_value': 2.0
        },
        {
            'name': 'Fractional Dilation',
            'B': 1, 'C': 2, 'H': 8, 'W': 8,
            'kernel_size': 3, 'stride': 1, 'padding': 1,
            'dilation_value': 1.5
        },
        {
            'name': 'Small Kernel Size 5',
            'B': 1, 'C': 2, 'H': 12, 'W': 12,
            'kernel_size': 5, 'stride': 1, 'padding': 2,
            'dilation_value': 1.0
        },
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print("-" * 80)
        
        # Create input
        B, C, H, W = test_case['B'], test_case['C'], test_case['H'], test_case['W']
        kernel_size = test_case['kernel_size']
        stride = test_case['stride']
        padding = test_case['padding']
        dilation_value = test_case['dilation_value']
        
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        input_tensor = torch.randn(B, C, H, W).cuda()
        
        # Create uniform dilation map
        H_out = (H + 2 * padding - 1 * (kernel_size - 1) - 1) // stride + 1
        W_out = (W + 2 * padding - 1 * (kernel_size - 1) - 1) // stride + 1
        dilation_map = torch.ones(B, 1, H_out, W_out).cuda() * dilation_value
        
        # CUDA implementation
        output_cuda = dynamic_dilation_unfold(
            input_tensor, 
            kernel_size=kernel_size, 
            dilation_map=dilation_map,
            stride=stride, 
            padding=padding, 
            dilation=1
        )
        
        # Manual implementation
        output_manual = manual_dynamic_dilation_unfold(
            input_tensor.cpu(),
            kernel_size=kernel_size,
            dilation_map=dilation_map.cpu(),
            stride=stride,
            padding=padding,
            dilation=1
        ).cuda()
        
        # Compare results
        abs_diff = torch.abs(output_cuda - output_manual)
        rel_diff = abs_diff / (torch.abs(output_manual) + 1e-8)
        
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        max_rel_diff = rel_diff.max().item()
        mean_rel_diff = rel_diff.mean().item()
        
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {output_cuda.shape}")
        print(f"  Dilation value: {dilation_value}")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print(f"  Mean relative difference: {mean_rel_diff:.2e}")
        
        # Check if passed (tolerance for float operations)
        passed = max_abs_diff < 1e-4 and mean_abs_diff < 1e-5
        
        if passed:
            print(f"  ✓ PASSED")
        else:
            print(f"  ✗ FAILED")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL FORWARD TESTS PASSED")
    else:
        print("✗ SOME FORWARD TESTS FAILED")
    print("=" * 80)
    
    return all_passed


# ============================================================================
# Part 3: Gradient Verification (Numerical Check)
# ============================================================================

def numerical_gradient_check(x, dilation_map, kernel_size, stride, padding, eps=1e-4):
    """
    Compute numerical gradient using finite differences
    """
    # Make copies for numerical gradient computation
    x_copy = x.clone().detach().requires_grad_(True)
    dilation_map_copy = dilation_map.clone().detach().requires_grad_(True)
    
    # Forward pass and compute analytical gradients
    output = dynamic_dilation_unfold(
        x_copy, 
        kernel_size=kernel_size, 
        dilation_map=dilation_map_copy,
        stride=stride, 
        padding=padding, 
        dilation=1
    )
    loss = output.sum()
    loss.backward()
    
    grad_x_analytical = x_copy.grad.clone()
    grad_d_analytical = dilation_map_copy.grad.clone()
    
    # Numerical gradient for input
    grad_x_numerical = torch.zeros_like(x)
    
    print("  Computing numerical gradient for input...")
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    # +eps
                    x_plus = x.clone()
                    x_plus[b, c, h, w] += eps
                    output_plus = dynamic_dilation_unfold(
                        x_plus, kernel_size=kernel_size, dilation_map=dilation_map,
                        stride=stride, padding=padding, dilation=1
                    )
                    loss_plus = output_plus.sum()
                    
                    # -eps
                    x_minus = x.clone()
                    x_minus[b, c, h, w] -= eps
                    output_minus = dynamic_dilation_unfold(
                        x_minus, kernel_size=kernel_size, dilation_map=dilation_map,
                        stride=stride, padding=padding, dilation=1
                    )
                    loss_minus = output_minus.sum()
                    
                    # Numerical gradient
                    grad_x_numerical[b, c, h, w] = (loss_plus - loss_minus) / (2 * eps)
    
    # Numerical gradient for dilation_map
    grad_d_numerical = torch.zeros_like(dilation_map)
    
    print("  Computing numerical gradient for dilation_map...")
    for b in range(dilation_map.shape[0]):
        for h in range(dilation_map.shape[2]):
            for w in range(dilation_map.shape[3]):
                # +eps
                d_plus = dilation_map.clone()
                d_plus[b, 0, h, w] += eps
                output_plus = dynamic_dilation_unfold(
                    x, kernel_size=kernel_size, dilation_map=d_plus,
                    stride=stride, padding=padding, dilation=1
                )
                loss_plus = output_plus.sum()
                
                # -eps
                d_minus = dilation_map.clone()
                d_minus[b, 0, h, w] -= eps
                output_minus = dynamic_dilation_unfold(
                    x, kernel_size=kernel_size, dilation_map=d_minus,
                    stride=stride, padding=padding, dilation=1
                )
                loss_minus = output_minus.sum()
                
                # Numerical gradient
                grad_d_numerical[b, 0, h, w] = (loss_plus - loss_minus) / (2 * eps)
    
    return grad_x_analytical, grad_x_numerical, grad_d_analytical, grad_d_numerical


def test_gradient():
    """Test gradient correctness using numerical gradient check"""
    print("\n" + "=" * 80)
    print("GRADIENT VERIFICATION (Numerical Check)")
    print("=" * 80)
    
    # Use small size for numerical gradient check (it's slow)
    B, C, H, W = 1, 2, 6, 6
    kernel_size = 3
    stride = 1
    padding = 1
    
    print(f"\nTest Configuration:")
    print(f"  Input shape: ({B}, {C}, {H}, {W})")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Stride: {stride}, Padding: {padding}")
    
    # Create input
    torch.manual_seed(123)
    x = torch.randn(B, C, H, W).cuda()
    dilation_map = torch.ones(B, 1, H, W).cuda() * 1.5
    
    # Compute gradients
    print("\nComputing gradients...")
    grad_x_analytical, grad_x_numerical, grad_d_analytical, grad_d_numerical = \
        numerical_gradient_check(x, dilation_map, kernel_size, stride, padding, eps=1e-4)
    
    # Compare input gradients
    print("\n" + "-" * 80)
    print("Input Gradient Comparison:")
    print("-" * 80)
    
    abs_diff_x = torch.abs(grad_x_analytical - grad_x_numerical)
    rel_diff_x = abs_diff_x / (torch.abs(grad_x_numerical) + 1e-8)
    
    print(f"  Analytical gradient - mean: {grad_x_analytical.mean().item():.6f}, "
          f"std: {grad_x_analytical.std().item():.6f}")
    print(f"  Numerical gradient  - mean: {grad_x_numerical.mean().item():.6f}, "
          f"std: {grad_x_numerical.std().item():.6f}")
    print(f"  Max absolute difference: {abs_diff_x.max().item():.2e}")
    print(f"  Mean absolute difference: {abs_diff_x.mean().item():.2e}")
    print(f"  Max relative difference: {rel_diff_x.max().item():.2e}")
    print(f"  Mean relative difference: {rel_diff_x.mean().item():.2e}")
    
    passed_x = rel_diff_x.mean().item() < 0.01  # 1% relative error tolerance
    
    if passed_x:
        print(f"  ✓ Input gradient PASSED")
    else:
        print(f"  ✗ Input gradient FAILED")
    
    # Compare dilation_map gradients
    print("\n" + "-" * 80)
    print("Dilation Map Gradient Comparison:")
    print("-" * 80)
    
    abs_diff_d = torch.abs(grad_d_analytical - grad_d_numerical)
    rel_diff_d = abs_diff_d / (torch.abs(grad_d_numerical) + 1e-8)
    
    print(f"  Analytical gradient - mean: {grad_d_analytical.mean().item():.6f}, "
          f"std: {grad_d_analytical.std().item():.6f}")
    print(f"  Numerical gradient  - mean: {grad_d_numerical.mean().item():.6f}, "
          f"std: {grad_d_numerical.std().item():.6f}")
    print(f"  Max absolute difference: {abs_diff_d.max().item():.2e}")
    print(f"  Mean absolute difference: {abs_diff_d.mean().item():.2e}")
    print(f"  Max relative difference: {rel_diff_d.max().item():.2e}")
    print(f"  Mean relative difference: {rel_diff_d.mean().item():.2e}")
    
    passed_d = rel_diff_d.mean().item() < 0.01  # 1% relative error tolerance
    
    if passed_d:
        print(f"  ✓ Dilation map gradient PASSED")
    else:
        print(f"  ✗ Dilation map gradient FAILED")
    
    # Visualize gradient comparison
    visualize_gradient_comparison(
        grad_x_analytical, grad_x_numerical,
        grad_d_analytical, grad_d_numerical
    )
    
    print("\n" + "=" * 80)
    if passed_x and passed_d:
        print("✓ ALL GRADIENT TESTS PASSED")
    else:
        print("✗ SOME GRADIENT TESTS FAILED")
    print("=" * 80)
    
    return passed_x and passed_d


# ============================================================================
# Part 4: Visualization
# ============================================================================

def visualize_gradient_comparison(grad_x_analytical, grad_x_numerical, 
                                   grad_d_analytical, grad_d_numerical):
    """Visualize gradient comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Input gradients
    grad_x_anal_vis = grad_x_analytical[0, 0].cpu().detach().numpy()
    grad_x_numer_vis = grad_x_numerical[0, 0].cpu().detach().numpy()
    grad_x_diff = np.abs(grad_x_anal_vis - grad_x_numer_vis)
    
    im0 = axes[0, 0].imshow(grad_x_anal_vis, cmap='RdBu_r')
    axes[0, 0].set_title('Input Grad (Analytical)')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(grad_x_numer_vis, cmap='RdBu_r')
    axes[0, 1].set_title('Input Grad (Numerical)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(grad_x_diff, cmap='hot')
    axes[0, 2].set_title('Input Grad (Abs Diff)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Dilation map gradients
    grad_d_anal_vis = grad_d_analytical[0, 0].cpu().detach().numpy()
    grad_d_numer_vis = grad_d_numerical[0, 0].cpu().detach().numpy()
    grad_d_diff = np.abs(grad_d_anal_vis - grad_d_numer_vis)
    
    im3 = axes[1, 0].imshow(grad_d_anal_vis, cmap='RdBu_r')
    axes[1, 0].set_title('Dilation Grad (Analytical)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(grad_d_numer_vis, cmap='RdBu_r')
    axes[1, 1].set_title('Dilation Grad (Numerical)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].imshow(grad_d_diff, cmap='hot')
    axes[1, 2].set_title('Dilation Grad (Abs Diff)')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('gradient_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n  Gradient comparison visualization saved to 'gradient_comparison.png'")
    plt.close()


# ============================================================================
# Part 5: Spatially-Varying Dilation Visualization
# ============================================================================

def test_spatially_varying_dilation():
    """Test with spatially-varying dilation and visualize"""
    print("\n" + "=" * 80)
    print("SPATIALLY-VARYING DILATION TEST")
    print("=" * 80)
    
    B, C, H, W = 1, 3, 32, 32
    kernel_size = 3
    
    # Create input
    torch.manual_seed(42)
    x = torch.randn(B, C, H, W).cuda()
    
    # Create spatially-varying dilation map
    dilation_map = torch.ones(B, 1, H, W).cuda()
    
    # Create quadrants with different dilation rates
    dilation_map[:, :, :H//2, :W//2] = 0.5   # Top-left
    dilation_map[:, :, :H//2, W//2:] = 1.0   # Top-right
    dilation_map[:, :, H//2:, :W//2] = 2.0   # Bottom-left
    dilation_map[:, :, H//2:, W//2:] = 3.0   # Bottom-right
    
    # Apply unfold
    output = dynamic_dilation_unfold(
        x, kernel_size=kernel_size, dilation_map=dilation_map,
        stride=1, padding=1
    )
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Dilation map range: [{dilation_map.min().item():.2f}, {dilation_map.max().item():.2f}]")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image (first channel)
    im0 = axes[0].imshow(x[0, 0].cpu().numpy(), cmap='viridis')
    axes[0].set_title('Input (Channel 0)')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])
    
    # Dilation map
    im1 = axes[1].imshow(dilation_map[0, 0].cpu().numpy(), cmap='plasma')
    axes[1].set_title('Dilation Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='Dilation Rate')
    
    # Add text annotations
    axes[1].text(W//4, H//4, '0.5', ha='center', va='center', 
                color='white', fontsize=20, weight='bold')
    axes[1].text(3*W//4, H//4, '1.0', ha='center', va='center', 
                color='white', fontsize=20, weight='bold')
    axes[1].text(W//4, 3*H//4, '2.0', ha='center', va='center', 
                color='white', fontsize=20, weight='bold')
    axes[1].text(3*W//4, 3*H//4, '3.0', ha='center', va='center', 
                color='white', fontsize=20, weight='bold')
    
    # Output (reshape and show first few channels)
    output_vis = output[0, :9, :].reshape(3, 3, H, W).mean(dim=(0, 1)).cpu().numpy()
    im2 = axes[2].imshow(output_vis, cmap='viridis')
    axes[2].set_title('Output (Averaged Kernel Positions)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('spatially_varying_dilation.png', dpi=150, bbox_inches='tight')
    print(f"  Visualization saved to 'spatially_varying_dilation.png'")
    plt.close()
    
    print("  ✓ Spatially-varying dilation test completed")


# ============================================================================
# Part 6: Main Test Runner
# ============================================================================

def main():
    """Run all validation tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DYNAMIC DILATION UNFOLD VALIDATION" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    
    results = {}
    
    # Test 1: Forward pass verification
    results['forward'] = test_forward_pass()
    
    # Test 2: Gradient verification
    results['gradient'] = test_gradient()
    
    # Test 3: Spatially-varying dilation
    test_spatially_varying_dilation()
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Forward Pass:     {'✓ PASSED' if results['forward'] else '✗ FAILED'}")
    print(f"Gradient Check:   {'✓ PASSED' if results['gradient'] else '✗ FAILED'}")
    print("=" * 80)
    
    if all(results.values()):
        print("\n🎉 ALL VALIDATIONS PASSED! Your implementation is correct! 🎉\n")
        return True
    else:
        print("\n❌ SOME VALIDATIONS FAILED! Please check the implementation. ❌\n")
        return False


if __name__ == '__main__':
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test requires CUDA.")
        exit(1)
    
    # Run all tests
    success = main()
    
    exit(0 if success else 1)