#!/usr/bin/env python3
"""
Performance test script for Gamma Correction module optimizations
This script compares original NumPy, Numba, and CUDA implementations.
"""

import time
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.gamma_correction.gamma_correction import GammaCorrection
from modules.gamma_correction.gamma_correction_numba import GammaCorrectionNumba

# Try to import CUDA implementation
try:
    from modules.gamma_correction.gamma_correction_cuda import GammaCorrectionCUDA
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
    """Get realistic gamma correction parameters"""
    return {
        "is_enable": True,
        "is_save": False
    }


def test_implementation(impl_class, img, platform, sensor_info, parm_gmm, name):
    """Test a specific implementation"""
    try:
        instance = impl_class(img.copy(), platform, sensor_info, parm_gmm)
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
    print("Gamma Correction Performance Analysis")
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
    parm_gmm = get_test_parameters()
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
            GammaCorrection, img, platform, sensor_info, parm_gmm, "NumPy"
        )
        
        # Numba
        numba_result, numba_time, numba_success = test_implementation(
            GammaCorrectionNumba, img, platform, sensor_info, parm_gmm, "Numba"
        )
        
        # CUDA
        cuda_result, cuda_time, cuda_success = test_implementation(
            GammaCorrectionCUDA, img, platform, sensor_info, parm_gmm, "CUDA"
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
        
        # Accuracy
        if numpy_success and numba_success and numpy_result is not None and numba_result is not None:
            diff = np.abs(numpy_result.astype(np.float32) - numba_result.astype(np.float32))
            print(f"  Numba accuracy: max_diff={np.max(diff):.2f}, mean_diff={np.mean(diff):.2f}")
        if numpy_success and cuda_success and numpy_result is not None and cuda_result is not None:
            diff = np.abs(numpy_result.astype(np.float32) - cuda_result.astype(np.float32))
            print(f"  CUDA accuracy: max_diff={np.max(diff):.2f}, mean_diff={np.mean(diff):.2f}")
    
    return results


if __name__ == "__main__":
    print("Gamma Correction Performance Analysis")
    print("=" * 40)
    results = run_performance_comparison()
    print("\nAnalysis completed!") 