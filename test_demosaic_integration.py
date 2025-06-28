#!/usr/bin/env python3
"""
Test script to verify OpenCV CUDA demosaic integration
"""
import time
import numpy as np
import cv2
from pathlib import Path

# Test the fallback implementation
from modules.demosaic.demosaic_opencv_cuda_fallback import DemosaicOpenCVCUDAFallback

def create_test_bayer_image(width=100, height=100, bayer_pattern="rggb"):
    """Create a test Bayer pattern image"""
    # Create a simple test pattern
    img = np.zeros((height, width), dtype=np.uint16)
    
    # Fill with a gradient pattern
    for i in range(height):
        for j in range(width):
            img[i, j] = (i * 256 + j) % 65536
    
    return img

def test_demosaic_integration():
    """Test the OpenCV CUDA demosaic integration"""
    print("Testing OpenCV CUDA Demosaic Integration")
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
        "bit_depth": 16,
        "output_bit_depth": 16,
        "bayer_pattern": "rggb"
    }
    
    parm_dem = {
        "is_enable": True,
        "is_debug": True,
        "is_save": False
    }
    
    # Create test image
    test_img = create_test_bayer_image(100, 100, "rggb")
    print(f"Test image shape: {test_img.shape}, dtype: {test_img.dtype}")
    
    # Test the fallback implementation
    try:
        demosaic = DemosaicOpenCVCUDAFallback(test_img, platform, sensor_info, parm_dem)
        result = demosaic.execute()
        
        print(f"Demosaic successful!")
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
        print(f"Demosaic failed: {e}")
        return False

if __name__ == "__main__":
    success = test_demosaic_integration()
    if success:
        print("\n✓ OpenCV CUDA demosaic integration test passed!")
    else:
        print("\n✗ OpenCV CUDA demosaic integration test failed!") 