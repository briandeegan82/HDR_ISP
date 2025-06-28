#!/usr/bin/env python3
"""
Benchmark script for Color Space Conversion module
Tests original, Numba, CUDA, and GPU implementations
"""

import time
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.color_space_conversion.color_space_conversion import ColorSpaceConversion
from modules.color_space_conversion.color_space_conversion_numba import ColorSpaceConversionNumba
from modules.color_space_conversion.color_space_conversion_cuda import ColorSpaceConversionCUDA
from modules.color_space_conversion.color_space_conversion_gpu import ColorSpaceConversionGPU
from modules.color_space_conversion.color_space_conversion_opencv_cuda import ColorSpaceConversionOpenCVCUDA


def create_test_image(height, width):
    """Create a test RGB image"""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def benchmark_csc_implementations():
    """Benchmark different CSC implementations"""
    
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
    parm_csc = {"is_save": False, "conv_standard": 1}  # BT.709
    parm_cse = {"is_enable": True, "saturation_gain": 1.2}
    
    implementations = [
        ("Original", ColorSpaceConversion),
        ("Numba", ColorSpaceConversionNumba),
        ("CUDA", ColorSpaceConversionCUDA),
        ("OpenCV CUDA", ColorSpaceConversionOpenCVCUDA),
        ("GPU", ColorSpaceConversionGPU),
    ]
    
    print("Color Space Conversion Performance Benchmark")
    print("=" * 60)
    
    for height, width in test_sizes:
        print(f"\nImage size: {width}x{height} ({width*height:,} pixels)")
        print("-" * 40)
        
        # Create test image
        test_img = create_test_image(height, width)
        
        results = {}
        
        for name, impl_class in implementations:
            try:
                # Warm up (especially important for Numba)
                if name == "Numba":
                    warmup_img = create_test_image(100, 100)
                    warmup_csc = impl_class(warmup_img, platform, sensor_info, parm_csc, parm_cse)
                    warmup_csc.execute()
                
                # Benchmark
                csc = impl_class(test_img.copy(), platform, sensor_info, parm_csc, parm_cse)
                
                start_time = time.time()
                result = csc.execute()
                end_time = time.time()
                
                execution_time = end_time - start_time
                results[name] = execution_time
                
                print(f"{name:10}: {execution_time:.4f}s")
                
            except Exception as e:
                print(f"{name:10}: ERROR - {str(e)}")
                results[name] = None
        
        # Calculate speedups
        if results.get("Original") and results["Original"] is not None:
            print("\nSpeedup vs Original:")
            original_time = results["Original"]
            for name, time_taken in results.items():
                if name != "Original" and time_taken is not None:
                    speedup = original_time / time_taken
                    print(f"{name:10}: {speedup:.2f}x")
                elif name != "Original":
                    print(f"{name:10}: FAILED")


if __name__ == "__main__":
    benchmark_csc_implementations() 