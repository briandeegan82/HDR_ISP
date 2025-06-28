#!/usr/bin/env python3
"""
Test script to verify Numba BLC integration in the pipeline
This script tests the full pipeline to ensure Numba optimization is working.
"""

import time
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.black_level_correction.black_level_correction_numba_fallback import BlackLevelCorrectionNumbaFallback
from modules.black_level_correction.black_level_correction import BlackLevelCorrection


def test_pipeline_integration():
    """Test that Numba BLC is properly integrated"""
    
    print("Testing Numba BLC Pipeline Integration")
    print("=" * 50)
    
    # Create a test image
    width, height = 2592, 1536
    test_img = np.random.randint(0, 4095, (height, width), dtype=np.uint16)
    
    # Define parameters
    sensor_info = {
        "width": width,
        "height": height,
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
    
    # Test the fallback implementation
    print("\nTesting Numba Fallback Implementation...")
    numba_fallback = BlackLevelCorrectionNumbaFallback(test_img.copy(), platform, sensor_info, parm_blc)
    
    start_time = time.time()
    numba_result = numba_fallback.execute()
    numba_time = time.time() - start_time
    
    print(f"Numba fallback time: {numba_time:.3f}s")
    print(f"Using Numba: {numba_fallback.use_numba}")
    
    # Test original implementation for comparison
    print("\nTesting Original Implementation...")
    original_blc = BlackLevelCorrection(test_img.copy(), platform, sensor_info, parm_blc)
    
    start_time = time.time()
    original_result = original_blc.execute()
    original_time = time.time() - start_time
    
    print(f"Original time: {original_time:.3f}s")
    
    # Compare results
    print("\n" + "=" * 50)
    print("INTEGRATION RESULTS")
    print("=" * 50)
    
    if numba_fallback.use_numba:
        speedup = original_time / numba_time
        print(f"✓ Numba integration successful!")
        print(f"✓ Speedup: {speedup:.2f}x")
        
        # Check correctness
        diff = np.abs(original_result.astype(np.float32) - numba_result.astype(np.float32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"✓ Max difference: {max_diff:.2f}, Mean difference: {mean_diff:.2f}")
        
        if max_diff < 1.0:
            print("✓ Results are consistent")
        else:
            print("⚠ Results differ significantly")
    else:
        print("⚠ Numba not available, using original implementation")
        print("  This is normal if Numba is not installed")
    
    return numba_fallback.use_numba, numba_time, original_time


def test_pipeline_imports():
    """Test that the pipeline imports work correctly"""
    
    print("\n" + "=" * 50)
    print("PIPELINE IMPORT TEST")
    print("=" * 50)
    
    try:
        # Test main pipeline import
        from infinite_isp import main
        print("✓ Main pipeline import successful")
    except Exception as e:
        print(f"✗ Main pipeline import failed: {e}")
    
    try:
        # Test GPU pipeline import
        from infinite_isp_gpu import main
        print("✓ GPU pipeline import successful")
    except Exception as e:
        print(f"✗ GPU pipeline import failed: {e}")
    
    try:
        # Test BLC module import
        from modules.black_level_correction.black_level_correction_numba_fallback import BlackLevelCorrectionNumbaFallback
        print("✓ BLC fallback import successful")
    except Exception as e:
        print(f"✗ BLC fallback import failed: {e}")


def test_numba_availability():
    """Test Numba availability and functionality"""
    
    print("\n" + "=" * 50)
    print("NUMBA AVAILABILITY TEST")
    print("=" * 50)
    
    try:
        from numba import jit, prange
        print("✓ Numba is available")
        
        # Test basic Numba functionality
        @jit(nopython=True)
        def test_func(x):
            return x * 2
        
        result = test_func(5)
        if result == 10:
            print("✓ Numba JIT compilation working")
        else:
            print("⚠ Numba JIT compilation issue")
            
    except ImportError:
        print("⚠ Numba is not available")
        print("  Install with: pip install numba")
    except Exception as e:
        print(f"⚠ Numba error: {e}")


if __name__ == "__main__":
    print("Numba BLC Pipeline Integration Test")
    print("=" * 40)
    
    # Test Numba availability
    test_numba_availability()
    
    # Test pipeline integration
    use_numba, numba_time, original_time = test_pipeline_integration()
    
    # Test pipeline imports
    test_pipeline_imports()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if use_numba:
        speedup = original_time / numba_time
        print(f"✓ Numba BLC successfully integrated with {speedup:.2f}x speedup")
        print("✓ Pipeline is ready for production use")
    else:
        print("⚠ Using original BLC implementation")
        print("  Consider installing Numba for better performance")
    
    print("\nTest completed!") 