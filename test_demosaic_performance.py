#!/usr/bin/env python3
"""
Performance test script for Demosaic module optimizations
This script compares Cython, Numba, custom GPU, and OpenCV CUDA implementations.
"""

import time
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.demosaic.demosaic import Demosaic
from modules.demosaic.demosaic_numba import DemosaicNumba
from modules.demosaic.demosaic_gpu import DemosaicGPU
from modules.demosaic.demosaic_opencv_cuda import DemosaicOpenCVCUDA


def create_test_image(width, height, bit_depth=12):
    """Create a test image with realistic values"""
    max_value = (1 << bit_depth) - 1
    img = np.random.randint(0, max_value + 1, (height, width), dtype=np.uint16)
    return img


def get_test_parameters():
    """Get realistic demosaic parameters"""
    return {
        "is_save": False,
        "is_debug": False
    }


def test_implementation(impl_class, img, platform, sensor_info, parm_dga, name):
    """Test a specific implementation"""
    try:
        instance = impl_class(img.copy(), platform, sensor_info, parm_dga)
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
    print("Demosaic Performance Analysis")
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
    platform = {
        "disable_progress_bar": True,
        "leave_pbar_string": False,
        "in_file": "test_image"
    }
    parm_dga = get_test_parameters()
    results = {
        'sizes': [],
        'cython': [],
        'numba': [],
        'gpu': [],
        'opencv_cuda': [],
        'pixels': []
    }
    for width, height in image_sizes:
        print(f"\nTesting image size: {width}x{height} ({width*height:,} pixels)")
        sensor_info["width"] = width
        sensor_info["height"] = height
        img = create_test_image(width, height)
        num_pixels = width * height
        # Cython
        cython_result, cython_time, cython_success = test_implementation(
            Demosaic, img, platform, sensor_info, parm_dga, "Cython"
        )
        # Numba
        numba_result, numba_time, numba_success = test_implementation(
            DemosaicNumba, img, platform, sensor_info, parm_dga, "Numba"
        )
        # Custom GPU
        gpu_result, gpu_time, gpu_success = test_implementation(
            DemosaicGPU, img, platform, sensor_info, parm_dga, "Custom GPU"
        )
        # OpenCV CUDA
        opencv_cuda_result, opencv_cuda_time, opencv_cuda_success = test_implementation(
            DemosaicOpenCVCUDA, img, platform, sensor_info, parm_dga, "OpenCV CUDA"
        )
        results['sizes'].append(f"{width}x{height}")
        results['pixels'].append(num_pixels)
        results['cython'].append(cython_time if cython_success else None)
        results['numba'].append(numba_time if numba_success else None)
        results['gpu'].append(gpu_time if gpu_success else None)
        results['opencv_cuda'].append(opencv_cuda_time if opencv_cuda_success else None)
        print(f"  Cython:      {cython_time:.4f}s" if cython_success else "  Cython:      Failed")
        print(f"  Numba:       {numba_time:.4f}s" if numba_success else "  Numba:       Failed")
        print(f"  Custom GPU:  {gpu_time:.4f}s" if gpu_success else "  Custom GPU:  Failed")
        print(f"  OpenCV CUDA: {opencv_cuda_time:.4f}s" if opencv_cuda_success else "  OpenCV CUDA: Failed")
        # Speedups
        if cython_success and numba_success:
            print(f"  Numba speedup: {cython_time / numba_time:.2f}x")
        if cython_success and gpu_success:
            print(f"  Custom GPU speedup: {cython_time / gpu_time:.2f}x")
        if cython_success and opencv_cuda_success:
            print(f"  OpenCV CUDA speedup: {cython_time / opencv_cuda_time:.2f}x")
        # Accuracy
        if cython_success and numba_success:
            diff = np.abs(cython_result.astype(np.float32) - numba_result.astype(np.float32))
            print(f"  Numba accuracy: max_diff={np.max(diff):.2f}, mean_diff={np.mean(diff):.2f}")
        if cython_success and gpu_success:
            diff = np.abs(cython_result.astype(np.float32) - gpu_result.astype(np.float32))
            print(f"  Custom GPU accuracy: max_diff={np.max(diff):.2f}, mean_diff={np.mean(diff):.2f}")
        if cython_success and opencv_cuda_success:
            diff = np.abs(cython_result.astype(np.float32) - opencv_cuda_result.astype(np.float32))
            print(f"  OpenCV CUDA accuracy: max_diff={np.max(diff):.2f}, mean_diff={np.mean(diff):.2f}")
    return results

if __name__ == "__main__":
    print("Demosaic Performance Analysis")
    print("=" * 40)
    results = run_performance_comparison()
    print("\nAnalysis completed!") 