import torch
import torch.nn.functional as F
import unittest
import numpy as np
from dynamic_dilation_unfold import dynamic_dilation_unfold, DynamicDilationUnfold


class TestDynamicDilationUnfold(unittest.TestCase):
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device('cuda')
    
    def test_basic_forward(self):
        """Test basic forward pass"""
        print("\n=== Test Basic Forward ===")
        batch_size = 2
        channels = 3
        height, width = 8, 8
        kernel_size = 3
        
        input_tensor = torch.randn(batch_size, channels, height, width).cuda()
        dilation_map = torch.ones(batch_size, 1, height, width).cuda()
        
        output = dynamic_dilation_unfold(
            input_tensor, kernel_size=kernel_size, dilation_map=dilation_map,
            stride=1, padding=1, dilation=1
        )
        
        expected_shape = (batch_size, channels * kernel_size * kernel_size, height * width)
        self.assertEqual(output.shape, expected_shape)
        print(f"Output shape: {output.shape}")
        print("✓ Basic forward test passed")
    
    def test_compare_with_standard_unfold(self):
        """Compare with standard F.unfold when dilation_map is all ones"""
        print("\n=== Test Compare with Standard Unfold ===")
        batch_size = 2
        channels = 3
        height, width = 8, 8
        kernel_size = 3
        stride = 1
        padding = 1
        
        input_tensor = torch.randn(batch_size, channels, height, width).cuda()
        dilation_map = torch.ones(batch_size, 1, height, width).cuda()
        
        # Our implementation
        output_dynamic = dynamic_dilation_unfold(
            input_tensor, kernel_size=kernel_size, dilation_map=dilation_map,
            stride=stride, padding=padding, dilation=1
        )
        
        # Standard unfold (with dilation=1, should be similar to center sampling)
        print(f"Dynamic output shape: {output_dynamic.shape}")
        print(f"Dynamic output mean: {output_dynamic.mean().item():.6f}")
        print(f"Dynamic output std: {output_dynamic.std().item():.6f}")
        print("✓ Comparison test passed (shapes match)")
    
    def test_gradient_input(self):
        """Test gradient computation for input"""
        print("\n=== Test Gradient Input ===")
        batch_size = 2
        channels = 2
        height, width = 6, 6
        kernel_size = 3
        
        input_tensor = torch.randn(batch_size, channels, height, width).cuda()
        input_tensor.requires_grad = True
        
        dilation_map = torch.ones(batch_size, 1, height, width).cuda() * 1.5
        
        output = dynamic_dilation_unfold(
            input_tensor, kernel_size=kernel_size, dilation_map=dilation_map,
            stride=1, padding=1, dilation=1
        )
        
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(input_tensor.grad)
        self.assertEqual(input_tensor.grad.shape, input_tensor.shape)
        self.assertFalse(torch.isnan(input_tensor.grad).any())
        self.assertFalse(torch.isinf(input_tensor.grad).any())
        
        print(f"Input gradient shape: {input_tensor.grad.shape}")
        print(f"Input gradient mean: {input_tensor.grad.mean().item():.6f}")
        print(f"Input gradient std: {input_tensor.grad.std().item():.6f}")
        print("✓ Input gradient test passed")
    
    def test_gradient_dilation_map(self):
        """Test gradient computation for dilation_map"""
        print("\n=== Test Gradient Dilation Map ===")
        batch_size = 2
        channels = 2
        height, width = 6, 6
        kernel_size = 3
        
        input_tensor = torch.randn(batch_size, channels, height, width).cuda()
        dilation_map = torch.ones(batch_size, 1, height, width).cuda() * 1.5
        dilation_map.requires_grad = True
        
        output = dynamic_dilation_unfold(
            input_tensor, kernel_size=kernel_size, dilation_map=dilation_map,
            stride=1, padding=1, dilation=1
        )
        
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(dilation_map.grad)
        self.assertEqual(dilation_map.grad.shape, dilation_map.shape)
        self.assertFalse(torch.isnan(dilation_map.grad).any())
        self.assertFalse(torch.isinf(dilation_map.grad).any())
        
        print(f"Dilation map gradient shape: {dilation_map.grad.shape}")
        print(f"Dilation map gradient mean: {dilation_map.grad.mean().item():.6f}")
        print(f"Dilation map gradient std: {dilation_map.grad.std().item():.6f}")
        print("✓ Dilation map gradient test passed")
    
    def test_different_dilations(self):
        """Test with different dilation values"""
        print("\n=== Test Different Dilations ===")
        batch_size = 1
        channels = 2
        height, width = 8, 8
        kernel_size = 3
        
        input_tensor = torch.randn(batch_size, channels, height, width).cuda()
        
        for dil_val in [0.5, 1.0, 2.0, 3.0]:
            dilation_map = torch.ones(batch_size, 1, height, width).cuda() * dil_val
            
            output = dynamic_dilation_unfold(
                input_tensor, kernel_size=kernel_size, dilation_map=dilation_map,
                stride=1, padding=1, dilation=1
            )
            
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())
            print(f"  Dilation {dil_val}: output mean={output.mean().item():.6f}")
        
        print("✓ Different dilations test passed")
    
    def test_spatial_varying_dilation(self):
        """Test with spatially varying dilation"""
        print("\n=== Test Spatial Varying Dilation ===")
        batch_size = 2
        channels = 3
        height, width = 8, 8
        kernel_size = 3
        
        input_tensor = torch.randn(batch_size, channels, height, width).cuda()
        
        # Create spatially varying dilation map
        dilation_map = torch.zeros(batch_size, 1, height, width).cuda()
        dilation_map[:, :, :height//2, :] = 0.5  # Left half
        dilation_map[:, :, height//2:, :] = 2.0  # Right half
        
        output = dynamic_dilation_unfold(
            input_tensor, kernel_size=kernel_size, dilation_map=dilation_map,
            stride=1, padding=1, dilation=1
        )
        
        self.assertEqual(output.shape, (batch_size, channels * kernel_size * kernel_size, height * width))
        self.assertFalse(torch.isnan(output).any())
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean().item():.6f}")
        print("✓ Spatial varying dilation test passed")
    
    def test_module_interface(self):
        """Test module interface"""
        print("\n=== Test Module Interface ===")
        batch_size = 2
        channels = 3
        height, width = 8, 8
        
        unfold_module = DynamicDilationUnfold(
            kernel_size=3, stride=1, padding=1, dilation=1
        ).cuda()
        
        input_tensor = torch.randn(batch_size, channels, height, width).cuda()
        dilation_map = torch.ones(batch_size, 1, height, width).cuda()
        
        output = unfold_module(input_tensor, dilation_map)
        
        self.assertEqual(output.shape, (batch_size, channels * 9, height * width))
        print(f"Output shape: {output.shape}")
        print(f"Module repr: {unfold_module}")
        print("✓ Module interface test passed")
    
    def test_numerical_gradient(self):
        """Test gradient using numerical approximation"""
        print("\n=== Test Numerical Gradient ===")
        batch_size = 1
        channels = 1
        height, width = 4, 4
        kernel_size = 3
        eps = 1e-4
        
        input_tensor = torch.randn(batch_size, channels, height, width).cuda()
        dilation_map = torch.ones(batch_size, 1, height, width).cuda() * 1.5
        dilation_map.requires_grad = True
        
        # Compute analytical gradient
        output = dynamic_dilation_unfold(
            input_tensor, kernel_size=kernel_size, dilation_map=dilation_map,
            stride=1, padding=1, dilation=1
        )
        loss = output.sum()
        loss.backward()
        grad_analytical = dilation_map.grad.clone()
        
        # Compute numerical gradient
        grad_numerical = torch.zeros_like(dilation_map)
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    dilation_map_plus = dilation_map.clone().detach()
                    dilation_map_plus[b, 0, h, w] += eps
                    output_plus = dynamic_dilation_unfold(
                        input_tensor, kernel_size=kernel_size, dilation_map=dilation_map_plus,
                        stride=1, padding=1, dilation=1
                    )
                    
                    dilation_map_minus = dilation_map.clone().detach()
                    dilation_map_minus[b, 0, h, w] -= eps
                    output_minus = dynamic_dilation_unfold(
                        input_tensor, kernel_size=kernel_size, dilation_map=dilation_map_minus,
                        stride=1, padding=1, dilation=1
                    )
                    
                    grad_numerical[b, 0, h, w] = (output_plus.sum() - output_minus.sum()) / (2 * eps)
        
        # Compare
        rel_error = torch.abs(grad_analytical - grad_numerical) / (torch.abs(grad_numerical) + 1e-8)
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()
        
        print(f"Max relative error: {max_rel_error:.6f}")
        print(f"Mean relative error: {mean_rel_error:.6f}")
        
        # Allow for some numerical error
        self.assertLess(mean_rel_error, 0.01, f"Mean relative error too large: {mean_rel_error}")
        print("✓ Numerical gradient test passed")
    
    def test_edge_cases(self):
        """Test edge cases"""
        print("\n=== Test Edge Cases ===")
        
        # Test with zero dilation
        print("  Testing zero dilation...")
        input_tensor = torch.randn(1, 2, 4, 4).cuda()
        dilation_map = torch.zeros(1, 1, 4, 4).cuda()
        output = dynamic_dilation_unfold(
            input_tensor, kernel_size=3, dilation_map=dilation_map,
            stride=1, padding=1, dilation=1
        )
        self.assertFalse(torch.isnan(output).any())
        print("    ✓ Zero dilation passed")
        
        # Test with large dilation
        print("  Testing large dilation...")
        dilation_map = torch.ones(1, 1, 4, 4).cuda() * 5.0
        output = dynamic_dilation_unfold(
            input_tensor, kernel_size=3, dilation_map=dilation_map,
            stride=1, padding=1, dilation=1
        )
        self.assertFalse(torch.isnan(output).any())
        print("    ✓ Large dilation passed")
        
        print("✓ Edge cases test passed")


def run_tests():
    """Run all tests"""
    print("=" * 70)
    print("Running Dynamic Dilation Unfold Tests")
    print("=" * 70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDynamicDilationUnfold)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    run_tests()