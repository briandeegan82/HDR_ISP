#!/usr/bin/env python3
"""
Performance test script for Auto Exposure module optimizations
This script compares original NumPy, Numba, and CUDA implementations.
"""

import time
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.auto_exposure.auto_exposure import AutoExposure
from modules.auto_exposure.auto_exposure_numba import AutoExposureNumba

# Try to import CUDA implementation
try:
    from modules.auto_exposure.auto_exposure_cuda import AutoExposureCUDA
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA implementation not available (CuPy not installed)")


def create_test_image(width, height, bit_depth=12):
    """Create a test RGB image with realistic values"""
    max_value = (1 << bit_depth) - 1
    img = np.random.randint(0, max_value + 1, (height, width, 3), dtype=np.uint16)
    return img


def get_test_parameters():
    """Get realistic auto exposure parameters"""
    return {
        "is_enable": True,
        "is_debug": False,
        "center_illuminance": 90,
        "histogram_skewness": 0.9
    }


def test_implementation(impl_class, img, sensor_info, parm_ae, name):
    """Test a specific implementation"""
    try:
        instance = impl_class(img.copy(), sensor_info, parm_ae)
        # Warm up run
        _ = instance.execute()
        # Performance test
        start_time = time.time()
        result = instance.execute()
        execution_time = time.time() - start_time
        return result, execution_time, True
    except Exception as e:
        print(f"  {name} failed: {e}")
        return None, 0, False


def run_performance_comparison():
    print("Auto Exposure Performance Analysis")
    print("=" * 50)
    image_sizes = [
        (640, 480),      # Small
        (1920, 1080),    # HD
        (2592, 1536),    # Medium
        (3840, 2160),    # 4K
        (4096, 3072),    # Large
    ]
    sensor_info = {
        "width": 2592,
        "height": 1536,
        "bit_depth": 12,
        "output_bit_depth": 12,
        "bayer_pattern": "rggb"
    }
    parm_ae = get_test_parameters()
    results = {
        'sizes': [],
        'numpy': [],
        'numba': [],
        'cuda': [],
        'pixels': []
    }
    for width, height in image_sizes:
        print(f"\nTesting image size: {width}x{height} ({width*height:,} pixels)")
        sensor_info["width"] = width
        sensor_info["height"] = height
        img = create_test_image(width, height)
        num_pixels = width * height
        
        # Original NumPy
        numpy_result, numpy_time, numpy_success = test_implementation(
            AutoExposure, img, sensor_info, parm_ae, "NumPy"
        )
        
        # Numba
        numba_result, numba_time, numba_success = test_implementation(
            AutoExposureNumba, img, sensor_info, parm_ae, "Numba"
        )
        
        # CUDA
        cuda_result, cuda_time, cuda_success = test_implementation(
            AutoExposureCUDA, img, sensor_info, parm_ae, "CUDA"
        ) if CUDA_AVAILABLE else (None, 0, False)
        
        results['sizes'].append(f"{width}x{height}")
        results['pixels'].append(num_pixels)
        results['numpy'].append(numpy_time if numpy_success else None)
        results['numba'].append(numba_time if numba_success else None)
        results['cuda'].append(cuda_time if cuda_success else None)
        
        print(f"  NumPy:       {numpy_time:.4f}s" if numpy_success else "  NumPy:       Failed")
        print(f"  Numba:       {numba_time:.4f}s" if numba_success else "  Numba:       Failed")
        print(f"  CUDA:        {cuda_time:.4f}s" if cuda_success else "  CUDA:        Failed")
        
        # Speedups
        if numpy_success and numba_success:
            print(f"  Numba speedup: {numpy_time / numba_time:.2f}x")
        if numpy_success and cuda_success:
            print(f"  CUDA speedup: {numpy_time / cuda_time:.2f}x")
        
        # Accuracy (check if results are the same)
        if numpy_success and numba_success and numpy_result is not None and numba_result is not None:
            accuracy = "✓" if numpy_result == numba_result else "✗"
            print(f"  Numba accuracy: {accuracy} (result: {numba_result})")
        if numpy_success and cuda_success and numpy_result is not None and cuda_result is not None:
            accuracy = "✓" if numpy_result == cuda_result else "✗"
            print(f"  CUDA accuracy: {accuracy} (result: {cuda_result})")
    
    return results


if __name__ == "__main__":
    print("Auto Exposure Performance Analysis")
    print("=" * 40)
    results = run_performance_comparison()
    print("\nAnalysis completed!") 