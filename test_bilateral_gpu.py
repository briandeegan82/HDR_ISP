#!/usr/bin/env python3
"""
Test script for GPU bilateral filter implementation
"""
import numpy as np
import cv2
import time
from util.gpu_utils import gpu_accelerator

def test_bilateral_filter():
    """Test GPU bilateral filter vs CPU bilateral filter"""
    print("Testing GPU Bilateral Filter Implementation")
    print("=" * 50)
    
    # Create a test image
    test_img = np.random.rand(2048, 2048).astype(np.float32)
    test_img = (test_img * 255).astype(np.uint8)
    
    # Test parameters
    d = 15
    sigma_color = 75.0
    sigma_space = 75.0
    
    print(f"Test image shape: {test_img.shape}")
    print(f"Test parameters: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
    print(f"CUDA available: {gpu_accelerator.cuda_available}")
    
    # Test CPU bilateral filter
    print("\nTesting CPU bilateral filter...")
    start_time = time.time()
    cpu_result = cv2.bilateralFilter(test_img, d, sigma_color, sigma_space)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f}s")
    
    # Test GPU bilateral filter
    print("\nTesting GPU bilateral filter...")
    start_time = time.time()
    gpu_result = gpu_accelerator.bilateral_filter_gpu(test_img, d, sigma_color, sigma_space)
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.4f}s")
    
    # Compare results
    if gpu_accelerator.cuda_available:
        diff = np.abs(cpu_result.astype(np.float32) - gpu_result.astype(np.float32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"\nResults comparison:")
        print(f"Max difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")
        
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"Speedup: {speedup:.2f}x")
        
        # Check if results are reasonable
        if max_diff < 1.0:  # Allow for some floating point differences
            print("✓ GPU bilateral filter test passed")
        else:
            print("✗ GPU bilateral filter test failed - results differ significantly")
    else:
        print("\nCUDA not available, both results should be identical")
        if np.array_equal(cpu_result, gpu_result):
            print("✓ GPU fallback test passed")
        else:
            print("✗ GPU fallback test failed")

if __name__ == "__main__":
    test_bilateral_filter() 