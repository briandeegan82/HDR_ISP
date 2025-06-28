#!/usr/bin/env python3
"""
Benchmark script for LDCI module
Tests original Cython, OpenCV CUDA, and OpenCV CPU implementations
"""

import time
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.ldci.ldci import LDCI
from modules.ldci.clahe_opencv_cuda import LDCIOpenCVCUDA
import cv2


def create_test_yuv_image(height, width):
    """Create a test YUV image"""
    # Create a realistic YUV image with some contrast variations
    y_channel = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    u_channel = np.random.randint(100, 150, (height, width), dtype=np.uint8)
    v_channel = np.random.randint(100, 150, (height, width), dtype=np.uint8)
    
    # Add some contrast variations to make CLAHE more meaningful
    y_channel[height//4:3*height//4, width//4:3*width//4] = np.random.randint(20, 80, (height//2, width//2), dtype=np.uint8)
    y_channel[height//8:height//4, width//8:width//4] = np.random.randint(180, 255, (height//8, width//8), dtype=np.uint8)
    
    return np.stack([y_channel, u_channel, v_channel], axis=2)


def benchmark_ldci_implementations():
    """Benchmark different LDCI implementations"""
    
    # Test configurations
    test_sizes = [
        (480, 640),    # Small
        (720, 1280),   # Medium
        (1080, 1920),  # Large
        (2160, 3840),  # 4K
    ]
    
    # Test parameters
    platform = {"in_file": "test_image"}
    sensor_info = {"output_bit_depth": 10}
    parm_ldci = {
        "is_enable": True,
        "is_save": False,
        "wind": 8,  # Window size
        "clip_limit": 2  # Clip limit
    }
    conv_std = 1  # BT.709
    
    implementations = [
        ("Cython", LDCI),
        ("OpenCV CUDA", LDCIOpenCVCUDA),
    ]
    
    print("LDCI Performance Benchmark")
    print("=" * 50)
    print(f"Window size: {parm_ldci['wind']}")
    print(f"Clip limit: {parm_ldci['clip_limit']}")
    print(f"OpenCV CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
    print()
    
    for height, width in test_sizes:
        print(f"Image size: {width}x{height} ({width*height:,} pixels)")
        print("-" * 40)
        
        results = {}
        
        for name, impl_class in implementations:
            try:
                # Create a fresh copy of the test image for each implementation
                fresh_test_img = create_test_yuv_image(height, width)
                
                # Warm up (especially important for JIT compilation)
                if name == "OpenCV CUDA":
                    warmup_img = create_test_yuv_image(100, 100)
                    warmup_ldci = impl_class(warmup_img, platform, sensor_info, parm_ldci, conv_std)
                    warmup_ldci.execute()
                
                # Benchmark
                print(f"[DEBUG] Running {name} implementation...")
                ldci = impl_class(fresh_test_img, platform, sensor_info, parm_ldci, conv_std)
                print(f"[DEBUG] {name} input shape: {fresh_test_img.shape}")
                start_time = time.time()
                result = ldci.execute()
                end_time = time.time()
                print(f"[DEBUG] {name} output shape: {result.shape}")
                
                execution_time = end_time - start_time
                results[name] = execution_time
                
                print(f"{name:12}: {execution_time:.4f}s")
                
            except Exception as e:
                print(f"[DEBUG] Exception in {name}: {e}")
                print(f"{name:12}: ERROR - {str(e)}")
                results[name] = None
                continue
        
        # Calculate speedups
        if results.get("Cython") and results["Cython"] is not None:
            print("\nSpeedup vs Cython:")
            cython_time = results["Cython"]
            for name, time_taken in results.items():
                if name != "Cython" and time_taken is not None:
                    speedup = cython_time / time_taken
                    print(f"{name:12}: {speedup:.2f}x")
                elif name != "Cython":
                    print(f"{name:12}: FAILED")
        print()


def test_opencv_clahe_quality():
    """Test the quality of OpenCV CLAHE vs Cython implementation"""
    print("\n" + "="*50)
    print("Quality Comparison Test")
    print("="*50)
    
    # Create a test image with known contrast issues
    height, width = 480, 640
    test_img = create_test_yuv_image(height, width)
    
    platform = {"in_file": "test_image"}
    sensor_info = {"output_bit_depth": 10}
    parm_ldci = {
        "is_enable": True,
        "is_save": False,
        "wind": 8,
        "clip_limit": 2
    }
    conv_std = 1
    
    try:
        # Test Cython implementation
        cython_ldci = LDCI(test_img.copy(), platform, sensor_info, parm_ldci, conv_std)
        cython_result = cython_ldci.execute()
        
        # Test OpenCV CUDA implementation
        opencv_ldci = LDCIOpenCVCUDA(test_img.copy(), platform, sensor_info, parm_ldci, conv_std)
        opencv_result = opencv_ldci.execute()
        
        # Compare Y channels (where CLAHE is applied)
        cython_y = cython_result[:, :, 0]
        opencv_y = opencv_result[:, :, 0]
        
        # Calculate differences
        diff = np.abs(cython_y.astype(np.int16) - opencv_y.astype(np.int16))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        print(f"Quality comparison results:")
        print(f"  Mean difference: {mean_diff:.2f}")
        print(f"  Max difference: {max_diff}")
        print(f"  Similarity: {100 - (mean_diff/255)*100:.1f}%")
        
        if mean_diff < 5:
            print("  ✓ Quality is very similar")
        elif mean_diff < 15:
            print("  ⚠ Quality is reasonably similar")
        else:
            print("  ✗ Quality differs significantly")
            
    except Exception as e:
        print(f"Quality test failed: {str(e)}")


if __name__ == "__main__":
    benchmark_ldci_implementations()
    test_opencv_clahe_quality() 