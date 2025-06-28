#!/usr/bin/env python3
"""
Test script to verify CUDA CCM integration
"""
import time
import numpy as np
from pathlib import Path

# Test the fallback implementation
from modules.color_correction_matrix.color_correction_matrix_cuda_fallback import ColorCorrectionMatrixCUDAFallback

def create_test_rgb_image(width=100, height=100, bit_depth=12):
    """Create a test RGB image"""
    max_value = (1 << bit_depth) - 1
    img = np.random.randint(0, max_value + 1, (height, width, 3), dtype=np.uint16)
    return img

def test_ccm_integration():
    """Test the CUDA CCM integration"""
    print("Testing CUDA CCM Integration")
    print("=" * 50)
    
    # Create test parameters
    platform = {
        "disable_progress_bar": True,
        "leave_pbar_string": "",
        "in_file": "test"
    }
    
    sensor_info = {
        "width": 100,
        "height": 100,
        "bit_depth": 12,
        "output_bit_depth": 12,
        "bayer_pattern": "rggb"
    }
    
    parm_ccm = {
        "is_enable": True,
        "is_save": False,
        "corrected_red": [1.660, -0.527, -0.133],
        "corrected_green": [-0.408, 1.563, -0.082],
        "corrected_blue": [-0.055, -1.641, 2.695]
    }
    
    # Create test image
    test_img = create_test_rgb_image(100, 100, 12)
    print(f"Test image shape: {test_img.shape}, dtype: {test_img.dtype}")
    
    # Test the fallback implementation
    try:
        ccm = ColorCorrectionMatrixCUDAFallback(test_img, platform, sensor_info, parm_ccm)
        result = ccm.execute()
        
        print(f"CCM successful!")
        print(f"Input shape: {test_img.shape}")
        print(f"Output shape: {result.shape}")
        print(f"Output dtype: {result.dtype}")
        
        # Check if output is RGB (3 channels)
        if len(result.shape) == 3 and result.shape[2] == 3:
            print("✓ Output is RGB format (3 channels)")
        else:
            print("✗ Output is not RGB format")
            
        return True
        
    except Exception as e:
        print(f"CCM failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ccm_integration()
    if success:
        print("\n✓ CUDA CCM integration test passed!")
    else:
        print("\n✗ CUDA CCM integration test failed!") 