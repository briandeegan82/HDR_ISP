#!/usr/bin/env python3
"""
Test script for Halide-based Dead Pixel Correction
This script demonstrates the usage of the Halide implementation
and compares it with the original implementation.
"""

import numpy as np
import time
import cv2
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.dead_pixel_correction.dead_pixel_correction_halide import DeadPixelCorrectionHalideFallback
from modules.dead_pixel_correction.dead_pixel_correction import DeadPixelCorrection


def create_test_image_with_dead_pixels(width=512, height=512, num_dead_pixels=50):
    """Create a test image with synthetic dead pixels"""
    
    # Create a base image with some texture
    base_img = np.random.randint(0, 255, (height, width), dtype=np.uint16)
    
    # Add some structured content (gradients, patterns)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    pattern = ((x_coords + y_coords) % 256).astype(np.uint16)
    base_img = np.clip(base_img.astype(np.int32) + pattern, 0, 65535).astype(np.uint16)
    
    # Add some dead pixels (very bright or very dark)
    for _ in range(num_dead_pixels):
        x = np.random.randint(2, width-2)
        y = np.random.randint(2, height-2)
        
        # Make it either very bright (hot pixel) or very dark (dead pixel)
        if np.random.random() > 0.5:
            base_img[y, x] = 65535  # Very bright
        else:
            base_img[y, x] = 0      # Very dark
    
    return base_img


def test_dpc_implementations():
    """Test both original and Halide implementations"""
    
    print("Testing Dead Pixel Correction Implementations")
    print("=" * 50)
    
    # Create test image
    print("Creating test image with synthetic dead pixels...")
    test_img = create_test_image_with_dead_pixels(512, 512, 100)
    
    # Define test parameters
    sensor_info = {
        "width": 512,
        "height": 512,
        "bit_depth": 16,
        "bayer_pattern": "RGGB"
    }
    
    parm_dpc = {
        "is_enable": True,
        "dp_threshold": 50.0,
        "is_debug": True,
        "is_save": False
    }
    
    platform = {
        "disable_progress_bar": True,
        "leave_pbar_string": False,
        "in_file": "test_image"
    }
    
    # Test original implementation
    print("\nTesting Original Implementation...")
    original_dpc = DeadPixelCorrection(test_img.copy(), sensor_info, parm_dpc, platform)
    
    start_time = time.time()
    original_result = original_dpc.execute()
    original_time = time.time() - start_time
    
    print(f"Original implementation time: {original_time:.3f}s")
    
    # Test Halide implementation (with fallback)
    print("\nTesting Halide Implementation...")
    halide_dpc = DeadPixelCorrectionHalideFallback(test_img.copy(), sensor_info, parm_dpc, platform)
    
    start_time = time.time()
    halide_result = halide_dpc.execute()
    halide_time = time.time() - start_time
    
    print(f"Halide implementation time: {halide_time:.3f}s")
    
    # Compare results
    print("\nComparing Results...")
    
    # Calculate difference
    diff = np.abs(original_result.astype(np.float32) - halide_result.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Maximum difference: {max_diff:.2f}")
    print(f"Mean difference: {mean_diff:.2f}")
    
    # Check if results are similar (allowing for small numerical differences)
    if max_diff < 1.0:
        print("✓ Results are consistent between implementations")
    else:
        print("⚠ Results differ significantly - this may indicate an implementation issue")
    
    # Performance comparison
    if halide_time > 0 and original_time > 0:
        speedup = original_time / halide_time
        print(f"\nPerformance Comparison:")
        print(f"Speedup: {speedup:.2f}x")
        if speedup > 1.0:
            print(f"✓ Halide implementation is {speedup:.2f}x faster")
        else:
            print(f"⚠ Original implementation is {1/speedup:.2f}x faster")
    
    # Save results for visual inspection
    print("\nSaving results for visual inspection...")
    
    # Normalize for display
    def normalize_for_display(img):
        return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    # Save original, corrected, and difference images
    cv2.imwrite("test_original.png", normalize_for_display(test_img))
    cv2.imwrite("test_original_corrected.png", normalize_for_display(original_result))
    cv2.imwrite("test_halide_corrected.png", normalize_for_display(halide_result))
    cv2.imwrite("test_difference.png", normalize_for_display(diff))
    
    print("Saved images:")
    print("- test_original.png: Original test image")
    print("- test_original_corrected.png: Corrected by original implementation")
    print("- test_halide_corrected.png: Corrected by Halide implementation")
    print("- test_difference.png: Difference between implementations")
    
    return original_result, halide_result, original_time, halide_time


def test_with_real_image(image_path):
    """Test with a real image if provided"""
    
    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found")
        return
    
    print(f"\nTesting with real image: {image_path}")
    print("=" * 50)
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to load image")
        return
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ensure 16-bit
    if img.dtype != np.uint16:
        img = img.astype(np.uint16) * 256
    
    # Define parameters
    sensor_info = {
        "width": img.shape[1],
        "height": img.shape[0],
        "bit_depth": 16,
        "bayer_pattern": "RGGB"
    }
    
    parm_dpc = {
        "is_enable": True,
        "dp_threshold": 50.0,
        "is_debug": True,
        "is_save": False
    }
    
    platform = {
        "disable_progress_bar": True,
        "leave_pbar_string": False,
        "in_file": os.path.basename(image_path)
    }
    
    # Test both implementations
    print("Testing Original Implementation...")
    original_dpc = DeadPixelCorrection(img.copy(), sensor_info, parm_dpc, platform)
    start_time = time.time()
    original_result = original_dpc.execute()
    original_time = time.time() - start_time
    
    print("Testing Halide Implementation...")
    halide_dpc = DeadPixelCorrectionHalideFallback(img.copy(), sensor_info, parm_dpc, platform)
    start_time = time.time()
    halide_result = halide_dpc.execute()
    halide_time = time.time() - start_time
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f"{base_name}_original_corrected.png", normalize_for_display(original_result))
    cv2.imwrite(f"{base_name}_halide_corrected.png", normalize_for_display(halide_result))
    
    print(f"Original time: {original_time:.3f}s")
    print(f"Halide time: {halide_time:.3f}s")
    if halide_time > 0 and original_time > 0:
        print(f"Speedup: {original_time / halide_time:.2f}x")


def normalize_for_display(img):
    """Normalize image for display"""
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


if __name__ == "__main__":
    print("Halide Dead Pixel Correction Test")
    print("=" * 40)
    
    # Test with synthetic image
    test_dpc_implementations()
    
    # Test with real image if provided
    if len(sys.argv) > 1:
        test_with_real_image(sys.argv[1])
    
    print("\nTest completed!") 