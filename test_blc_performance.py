#!/usr/bin/env python3
"""
Test script for Black Level Correction Performance Comparison
This script compares the performance of different implementations:
- Original NumPy implementation
- Numba-optimized implementation
- Halide-optimized implementation
"""

import numpy as np
import time
import cv2
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.black_level_correction.black_level_correction import BlackLevelCorrection
from modules.black_level_correction.black_level_correction_numba import BlackLevelCorrectionNumba
from modules.black_level_correction.black_level_correction_halide import BlackLevelCorrectionHalideFallback


def create_test_image(width=2592, height=1536):
    """Create a test image with realistic sensor data"""
    
    # Create a base image with some texture and noise
    base_img = np.random.randint(0, 4095, (height, width), dtype=np.uint16)
    
    # Add some structured content (gradients, patterns)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Add Bayer pattern variations
    bayer_pattern = np.zeros((height, width), dtype=np.uint16)
    bayer_pattern[0::2, 0::2] = 100   # Red channel
    bayer_pattern[0::2, 1::2] = 150   # Green Red channel
    bayer_pattern[1::2, 0::2] = 120   # Green Blue channel
    bayer_pattern[1::2, 1::2] = 80    # Blue channel
    
    # Combine with base image
    test_img = np.clip(base_img + bayer_pattern, 0, 4095).astype(np.uint16)
    
    return test_img


def test_blc_implementations():
    """Test all BLC implementations"""
    
    print("Black Level Correction Performance Comparison")
    print("=" * 60)
    
    # Create test image
    print("Creating test image...")
    test_img = create_test_image(2592, 1536)
    
    # Define test parameters
    sensor_info = {
        "width": 2592,
        "height": 1536,
        "bit_depth": 12,
        "bayer_pattern": "rggb"
    }
    
    parm_blc = {
        "is_enable": True,
        "r_offset": 200,
        "gr_offset": 200,
        "gb_offset": 200,
        "b_offset": 200,
        "is_linear": True,
        "r_sat": 4095,
        "gr_sat": 4095,
        "gb_sat": 4095,
        "b_sat": 4095,
        "is_debug": True,
        "is_save": False
    }
    
    platform = {
        "disable_progress_bar": True,
        "leave_pbar_string": False,
        "in_file": "test_image"
    }
    
    results = {}
    
    # Test original implementation
    print("\nTesting Original Implementation...")
    original_blc = BlackLevelCorrection(test_img.copy(), platform, sensor_info, parm_blc)
    
    start_time = time.time()
    original_result = original_blc.execute()
    original_time = time.time() - start_time
    
    results['original'] = {
        'time': original_time,
        'result': original_result
    }
    print(f"Original implementation time: {original_time:.3f}s")
    
    # Test Numba implementation
    print("\nTesting Numba Implementation...")
    try:
        numba_blc = BlackLevelCorrectionNumba(test_img.copy(), platform, sensor_info, parm_blc)
        
        start_time = time.time()
        numba_result = numba_blc.execute()
        numba_time = time.time() - start_time
        
        results['numba'] = {
            'time': numba_time,
            'result': numba_result
        }
        print(f"Numba implementation time: {numba_time:.3f}s")
    except Exception as e:
        print(f"Numba implementation failed: {e}")
        results['numba'] = None
    
    # Test Halide implementation
    print("\nTesting Halide Implementation...")
    try:
        halide_blc = BlackLevelCorrectionHalideFallback(test_img.copy(), platform, sensor_info, parm_blc)
        
        start_time = time.time()
        halide_result = halide_blc.execute()
        halide_time = time.time() - start_time
        
        results['halide'] = {
            'time': halide_time,
            'result': halide_result
        }
        print(f"Halide implementation time: {halide_time:.3f}s")
    except Exception as e:
        print(f"Halide implementation failed: {e}")
        results['halide'] = None
    
    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    original_time = results['original']['time']
    print(f"Original (NumPy):     {original_time:.3f}s (baseline)")
    
    if results['numba']:
        numba_time = results['numba']['time']
        numba_speedup = original_time / numba_time
        print(f"Numba:               {numba_time:.3f}s ({numba_speedup:.2f}x speedup)")
        
        # Check correctness
        numba_diff = np.abs(original_result.astype(np.float32) - results['numba']['result'].astype(np.float32))
        numba_max_diff = np.max(numba_diff)
        numba_mean_diff = np.mean(numba_diff)
        print(f"  Max difference: {numba_max_diff:.2f}, Mean difference: {numba_mean_diff:.2f}")
    
    if results['halide']:
        halide_time = results['halide']['time']
        halide_speedup = original_time / halide_time
        print(f"Halide:              {halide_time:.3f}s ({halide_speedup:.2f}x speedup)")
        
        # Check correctness
        halide_diff = np.abs(original_result.astype(np.float32) - results['halide']['result'].astype(np.float32))
        halide_max_diff = np.max(halide_diff)
        halide_mean_diff = np.mean(halide_diff)
        print(f"  Max difference: {halide_max_diff:.2f}, Mean difference: {halide_mean_diff:.2f}")
    
    # Performance analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    print("Black Level Correction Characteristics:")
    print("- Memory-bound operation (simple arithmetic per pixel)")
    print("- No data dependencies between pixels")
    print("- Regular access pattern (Bayer pattern)")
    print("- Low computational intensity")
    
    print("\nExpected Performance:")
    print("- Numba: Should provide 2-5x speedup due to:")
    print("  * Compilation to machine code")
    print("  * SIMD vectorization")
    print("  * Parallel execution")
    print("  * Reduced Python overhead")
    
    print("- Halide: Should provide 3-10x speedup due to:")
    print("  * Advanced loop optimization")
    print("  * Cache-friendly memory access")
    print("  * SIMD vectorization")
    print("  * Parallel scheduling")
    
    print("- CUDA: May not provide significant speedup due to:")
    print("  * Memory transfer overhead")
    print("  * Low computational intensity")
    print("  * Simple arithmetic operations")
    
    # Save results for visual inspection
    print("\nSaving results for visual inspection...")
    
    def normalize_for_display(img):
        return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    cv2.imwrite("test_blc_original.png", normalize_for_display(test_img))
    cv2.imwrite("test_blc_original_corrected.png", normalize_for_display(original_result))
    
    if results['numba']:
        cv2.imwrite("test_blc_numba_corrected.png", normalize_for_display(results['numba']['result']))
        cv2.imwrite("test_blc_numba_diff.png", normalize_for_display(numba_diff))
    
    if results['halide']:
        cv2.imwrite("test_blc_halide_corrected.png", normalize_for_display(results['halide']['result']))
        cv2.imwrite("test_blc_halide_diff.png", normalize_for_display(halide_diff))
    
    print("Saved images:")
    print("- test_blc_original.png: Original test image")
    print("- test_blc_original_corrected.png: Corrected by original implementation")
    if results['numba']:
        print("- test_blc_numba_corrected.png: Corrected by Numba implementation")
        print("- test_blc_numba_diff.png: Difference between original and Numba")
    if results['halide']:
        print("- test_blc_halide_corrected.png: Corrected by Halide implementation")
        print("- test_blc_halide_diff.png: Difference between original and Halide")
    
    return results


def test_with_different_sizes():
    """Test performance with different image sizes"""
    
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    
    sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 1536),
        (2592, 1536),
        (4096, 3072)
    ]
    
    sensor_info = {
        "bit_depth": 12,
        "bayer_pattern": "rggb"
    }
    
    parm_blc = {
        "is_enable": True,
        "r_offset": 200,
        "gr_offset": 200,
        "gb_offset": 200,
        "b_offset": 200,
        "is_linear": True,
        "r_sat": 4095,
        "gr_sat": 4095,
        "gb_sat": 4095,
        "b_sat": 4095,
        "is_debug": False,
        "is_save": False
    }
    
    platform = {
        "disable_progress_bar": True,
        "leave_pbar_string": False,
        "in_file": "test_image"
    }
    
    print(f"{'Size':<12} {'Pixels':<10} {'Original':<10} {'Numba':<10} {'Halide':<10}")
    print("-" * 60)
    
    for width, height in sizes:
        sensor_info["width"] = width
        sensor_info["height"] = height
        
        test_img = create_test_image(width, height)
        pixels = width * height
        
        # Test original
        original_blc = BlackLevelCorrection(test_img.copy(), platform, sensor_info, parm_blc)
        start_time = time.time()
        original_blc.execute()
        original_time = time.time() - start_time
        
        # Test Numba
        try:
            numba_blc = BlackLevelCorrectionNumba(test_img.copy(), platform, sensor_info, parm_blc)
            start_time = time.time()
            numba_blc.execute()
            numba_time = time.time() - start_time
        except:
            numba_time = float('inf')
        
        # Test Halide
        try:
            halide_blc = BlackLevelCorrectionHalideFallback(test_img.copy(), platform, sensor_info, parm_blc)
            start_time = time.time()
            halide_blc.execute()
            halide_time = time.time() - start_time
        except:
            halide_time = float('inf')
        
        print(f"{width}x{height:<6} {pixels:<10} {original_time:<10.3f} {numba_time:<10.3f} {halide_time:<10.3f}")


if __name__ == "__main__":
    print("Black Level Correction Performance Test")
    print("=" * 40)
    
    # Test all implementations
    results = test_blc_implementations()
    
    # Test scaling
    test_with_different_sizes()
    
    print("\nTest completed!") 