#!/usr/bin/env python3
"""
Performance test script for PWC (Piecewise Curve) module optimizations
This script compares the original implementation with Numba, CUDA, and Halide versions.
"""

import time
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.pwc_generation.pwc_generation import PiecewiseCurve
from modules.pwc_generation.pwc_generation_numba import PiecewiseCurveNumba
from modules.pwc_generation.pwc_generation_cuda import PiecewiseCurveCuda
from modules.pwc_generation.pwc_generation_halide import PiecewiseCurveHalide


def create_test_image(width, height, bit_depth=12):
    """Create a test image with realistic values"""
    max_value = (1 << bit_depth) - 1
    # Create image with some companded values
    img = np.random.randint(0, max_value + 1, (height, width), dtype=np.uint16)
    return img


def get_test_parameters():
    """Get realistic PWC parameters"""
    return {
        "is_enable": True,
        "companded_pin": [0, 512, 1024, 2048, 4095],  # Input knee points
        "companded_pout": [0, 256, 768, 1792, 4095],  # Output knee points
        "pedestal": 64,  # Pedestal value to subtract
        "is_save": False,
        "is_debug": False
    }


def test_implementation(impl_class, img, platform, sensor_info, parm_cmpd, name):
    """Test a specific implementation"""
    try:
        # Create instance
        instance = impl_class(img.copy(), platform, sensor_info, parm_cmpd)
        
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
    """Run comprehensive performance comparison"""
    
    print("PWC Performance Analysis")
    print("=" * 50)
    
    # Test image sizes
    image_sizes = [
        (640, 480),      # Small
        (1920, 1080),    # HD
        (2592, 1536),    # Medium
        (3840, 2160),    # 4K
        (4096, 3072),    # Large
    ]
    
    # Sensor info
    sensor_info = {
        "width": 2592,
        "height": 1536,
        "bit_depth": 12,
        "bayer_pattern": "rggb"
    }
    
    # Platform info
    platform = {
        "disable_progress_bar": True,
        "leave_pbar_string": False,
        "in_file": "test_image"
    }
    
    # PWC parameters
    parm_cmpd = get_test_parameters()
    
    # Results storage
    results = {
        'sizes': [],
        'original': [],
        'numba': [],
        'cuda': [],
        'halide': [],
        'pixels': []
    }
    
    # Test each image size
    for width, height in image_sizes:
        print(f"\nTesting image size: {width}x{height} ({width*height:,} pixels)")
        
        # Update sensor info for this size
        sensor_info["width"] = width
        sensor_info["height"] = height
        
        # Create test image
        img = create_test_image(width, height)
        num_pixels = width * height
        
        # Test original implementation
        original_result, original_time, original_success = test_implementation(
            PiecewiseCurve, img, platform, sensor_info, parm_cmpd, "Original"
        )
        
        # Test Numba implementation
        numba_result, numba_time, numba_success = test_implementation(
            PiecewiseCurveNumba, img, platform, sensor_info, parm_cmpd, "Numba"
        )
        
        # Test CUDA implementation
        cuda_result, cuda_time, cuda_success = test_implementation(
            PiecewiseCurveCuda, img, platform, sensor_info, parm_cmpd, "CUDA"
        )
        
        # Test Halide implementation
        halide_result, halide_time, halide_success = test_implementation(
            PiecewiseCurveHalide, img, platform, sensor_info, parm_cmpd, "Halide"
        )
        
        # Store results
        results['sizes'].append(f"{width}x{height}")
        results['pixels'].append(num_pixels)
        results['original'].append(original_time if original_success else None)
        results['numba'].append(numba_time if numba_success else None)
        results['cuda'].append(cuda_time if cuda_success else None)
        results['halide'].append(halide_time if halide_success else None)
        
        # Print results for this size
        print(f"  Original: {original_time:.4f}s" if original_success else "  Original: Failed")
        print(f"  Numba:    {numba_time:.4f}s" if numba_success else "  Numba:    Failed")
        print(f"  CUDA:     {cuda_time:.4f}s" if cuda_success else "  CUDA:     Failed")
        print(f"  Halide:   {halide_time:.4f}s" if halide_success else "  Halide:   Failed")
        
        # Calculate speedups
        if original_success and numba_success:
            speedup = original_time / numba_time
            print(f"  Numba speedup: {speedup:.2f}x")
        
        if original_success and cuda_success:
            speedup = original_time / cuda_time
            print(f"  CUDA speedup:  {speedup:.2f}x")
        
        if original_success and halide_success:
            speedup = original_time / halide_time
            print(f"  Halide speedup: {speedup:.2f}x")
        
        # Verify correctness
        if original_success and numba_success:
            diff = np.abs(original_result.astype(np.float32) - numba_result.astype(np.float32))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"  Numba accuracy: max_diff={max_diff:.2f}, mean_diff={mean_diff:.2f}")
        
        if original_success and cuda_success:
            diff = np.abs(original_result.astype(np.float32) - cuda_result.astype(np.float32))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"  CUDA accuracy:  max_diff={max_diff:.2f}, mean_diff={mean_diff:.2f}")
        
        if original_success and halide_success:
            diff = np.abs(original_result.astype(np.float32) - halide_result.astype(np.float32))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"  Halide accuracy: max_diff={max_diff:.2f}, mean_diff={mean_diff:.2f}")
    
    return results


def plot_results(results):
    """Plot performance results"""
    try:
        plt.figure(figsize=(15, 10))
        
        # Performance comparison
        plt.subplot(2, 2, 1)
        x = range(len(results['sizes']))
        width = 0.2
        
        implementations = ['original', 'numba', 'cuda', 'halide']
        colors = ['blue', 'green', 'red', 'orange']
        labels = ['Original', 'Numba', 'CUDA', 'Halide']
        
        for i, (impl, color, label) in enumerate(zip(implementations, colors, labels)):
            times = [t for t in results[impl] if t is not None]
            if times:
                plt.bar([xi + i*width for xi in x[:len(times)]], times, width, 
                       label=label, color=color, alpha=0.7)
        
        plt.xlabel('Image Size')
        plt.ylabel('Execution Time (s)')
        plt.title('PWC Performance Comparison')
        plt.xticks([xi + width*1.5 for xi in x], results['sizes'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Speedup comparison
        plt.subplot(2, 2, 2)
        if results['original'] and any(t is not None for t in results['original']):
            original_times = [t for t in results['original'] if t is not None]
            
            for impl, color, label in zip(['numba', 'cuda', 'halide'], ['green', 'red', 'orange'], ['Numba', 'CUDA', 'Halide']):
                times = [t for t in results[impl] if t is not None]
                if times and len(times) == len(original_times):
                    speedups = [orig / opt for orig, opt in zip(original_times, times)]
                    plt.plot(range(len(speedups)), speedups, 'o-', color=color, label=label, linewidth=2, markersize=6)
        
        plt.xlabel('Image Size Index')
        plt.ylabel('Speedup Factor')
        plt.title('PWC Speedup Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # Performance vs image size
        plt.subplot(2, 2, 3)
        pixels = results['pixels']
        
        for impl, color, label in zip(['original', 'numba', 'cuda', 'halide'], ['blue', 'green', 'red', 'orange'], ['Original', 'Numba', 'CUDA', 'Halide']):
            times = [t for t in results[impl] if t is not None]
            if times and len(times) == len(pixels):
                plt.loglog(pixels, times, 'o-', color=color, label=label, linewidth=2, markersize=6)
        
        plt.xlabel('Number of Pixels')
        plt.ylabel('Execution Time (s)')
        plt.title('PWC Performance vs Image Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Efficiency comparison (time per pixel)
        plt.subplot(2, 2, 4)
        for impl, color, label in zip(['original', 'numba', 'cuda', 'halide'], ['blue', 'green', 'red', 'orange'], ['Original', 'Numba', 'CUDA', 'Halide']):
            times = [t for t in results[impl] if t is not None]
            if times and len(times) == len(pixels):
                efficiency = [t / p * 1e6 for t, p in zip(times, pixels)]  # microseconds per pixel
                plt.plot(range(len(efficiency)), efficiency, 'o-', color=color, label=label, linewidth=2, markersize=6)
        
        plt.xlabel('Image Size Index')
        plt.ylabel('Time per Pixel (μs)')
        plt.title('PWC Efficiency Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pwc_performance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nPerformance plot saved as: pwc_performance_analysis.png")
        
    except Exception as e:
        print(f"Plotting failed: {e}")


def print_summary(results):
    """Print performance summary"""
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Calculate average speedups
    if results['original'] and any(t is not None for t in results['original']):
        original_times = [t for t in results['original'] if t is not None]
        
        for impl, name in [('numba', 'Numba'), ('cuda', 'CUDA'), ('halide', 'Halide')]:
            times = [t for t in results[impl] if t is not None]
            if times and len(times) == len(original_times):
                speedups = [orig / opt for orig, opt in zip(original_times, times)]
                avg_speedup = np.mean(speedups)
                max_speedup = np.max(speedups)
                min_speedup = np.min(speedups)
                print(f"{name:8s}: avg={avg_speedup:.2f}x, min={min_speedup:.2f}x, max={max_speedup:.2f}x")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 20)
    
    if results['numba'] and any(t is not None for t in results['numba']):
        print("✓ Numba: Good for CPU optimization, easy integration")
    
    if results['cuda'] and any(t is not None for t in results['cuda']):
        print("✓ CUDA: Best for large images with GPU available")
    
    if results['halide'] and any(t is not None for t in results['halide']):
        print("✓ Halide: Good for complex image processing pipelines")
    
    print("\nBest choice depends on:")
    print("- Image size (larger = better GPU performance)")
    print("- Hardware availability (CPU vs GPU)")
    print("- Integration complexity requirements")


if __name__ == "__main__":
    print("PWC Performance Analysis")
    print("=" * 40)
    
    # Run performance comparison
    results = run_performance_comparison()
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print_summary(results)
    
    print("\nAnalysis completed!") 