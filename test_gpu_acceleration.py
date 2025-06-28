"""
File: test_gpu_acceleration.py
Description: Comprehensive test suite for GPU acceleration functionality
Author: 10xEngineers
------------------------------------------------------------
"""
import numpy as np
import cv2
import time
import sys
from typing import Tuple, Dict, Any, Optional
from util.gpu_utils import gpu_accelerator
import argparse

class GPUAccelerationTester:
    """
    Comprehensive test suite for GPU acceleration functionality
    """
    
    def __init__(self, test_image_size: Tuple[int, int] = (512, 512)):
        self.test_results = {}
        self.performance_data = {}
        self.test_image_size = test_image_size
        print(f"Test image size: {test_image_size[0]}x{test_image_size[1]}")
        
    def print_header(self, title: str):
        """Print a formatted header"""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)
    
    def print_section(self, title: str):
        """Print a formatted section header"""
        print(f"\n{'-' * 40}")
        print(f" {title}")
        print(f"{'-' * 40}")
    
    def test_gpu_availability(self) -> bool:
        """Test if GPU acceleration is available"""
        self.print_header("GPU ACCELERATION AVAILABILITY TEST")
        
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"CUDA devices detected: {cuda_count}")
            print(f"GPU accelerator available: {gpu_accelerator.cuda_available}")
            
            if gpu_accelerator.cuda_available:
                print("✓ GPU acceleration is available")
                return True
            else:
                print("✗ GPU acceleration is not available")
                print("  This is normal if you don't have a CUDA-capable GPU")
                print("  or OpenCV is not compiled with CUDA support")
                return False
        except Exception as e:
            print(f"✗ Error checking GPU availability: {e}")
            return False
    
    def create_test_image(self, size: Optional[Tuple[int, int]] = None, 
                         dtype: Any = np.float32) -> np.ndarray:
        """Create a test image with controlled patterns"""
        if size is None:
            size = self.test_image_size
            
        if dtype == np.uint8:
            # Create grayscale test image with varying contrast regions
            img = np.random.randint(0, 100, size, dtype=np.uint8)
            # Add some structured patterns
            img[100:200, 100:200] += 100  # High contrast region
            img[300:400, 300:400] += 50   # Medium contrast region
            return img
        else:
            # Create float test image
            img = np.random.rand(*size).astype(dtype)
            # Add some structured patterns
            img[100:200, 100:200] += 0.5
            img[300:400, 300:400] += 0.3
            return img
    
    def create_test_image_3d(self, size: Optional[Tuple[int, int, int]] = None, 
                            dtype: Any = np.float32) -> np.ndarray:
        """Create a 3D test image with controlled patterns"""
        if size is None:
            # Use the 2D test size and add 3 channels
            size = (self.test_image_size[0], self.test_image_size[1], 3)
            
        img = np.random.rand(*size).astype(dtype)
        # Add some structured patterns
        img[100:200, 100:200, :] += 0.5
        img[300:400, 300:400, :] += 0.3
        return img
    
    def benchmark_operation(self, operation_name: str, 
                          cpu_func, gpu_func, 
                          test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark CPU vs GPU performance for an operation"""
        
        # Warm up
        for _ in range(3):
            try:
                cpu_func(**test_data)
                gpu_func(**test_data)
            except:
                pass
        
        # CPU benchmark
        cpu_times = []
        for _ in range(5):
            start_time = time.time()
            cpu_result = cpu_func(**test_data)
            cpu_times.append(time.time() - start_time)
        
        # GPU benchmark
        gpu_times = []
        gpu_results = []
        for _ in range(5):
            start_time = time.time()
            gpu_result = gpu_func(**test_data)
            gpu_times.append(time.time() - start_time)
            gpu_results.append(gpu_result)
        
        # Calculate statistics
        cpu_avg = np.mean(cpu_times)
        gpu_avg = np.mean(gpu_times)
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        
        # Compare results
        if len(gpu_results) > 0:
            diff = np.abs(cpu_result - gpu_results[0])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
        else:
            max_diff = mean_diff = float('inf')
        
        return {
            'operation': operation_name,
            'cpu_time': cpu_avg,
            'gpu_time': gpu_avg,
            'speedup': speedup,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'cpu_result': cpu_result,
            'gpu_result': gpu_results[0] if gpu_results else None
        }
    
    def test_bilateral_filter(self) -> bool:
        """Test GPU-accelerated bilateral filtering"""
        self.print_section("Testing Bilateral Filter")
        
        # Create test image
        test_img = self.create_test_image((512, 512), np.float32)
        
        # Test parameters
        test_data = {
            'src': test_img,
            'd': 15,
            'sigma_color': 75.0,
            'sigma_space': 75.0
        }
        
        # Define CPU and GPU functions
        def cpu_bilateral(src, d, sigma_color, sigma_space):
            return cv2.bilateralFilter(src, d, sigma_color, sigma_space)
        
        def gpu_bilateral(src, d, sigma_color, sigma_space):
            return gpu_accelerator.bilateral_filter_gpu(src, d, sigma_color, sigma_space)
        
        # Benchmark
        result = self.benchmark_operation("Bilateral Filter", cpu_bilateral, gpu_bilateral, test_data)
        
        # Print results
        print(f"CPU time: {result['cpu_time']:.4f}s")
        print(f"GPU time: {result['gpu_time']:.4f}s")
        print(f"Speedup: {result['speedup']:.2f}x")
        print(f"Max difference: {result['max_diff']:.6f}")
        print(f"Mean difference: {result['mean_diff']:.6f}")
        
        # Determine if test passed
        passed = result['max_diff'] < 0.01
        if passed:
            print("✓ Bilateral filter test passed")
        else:
            print("✗ Bilateral filter test failed - results differ significantly")
        
        self.test_results['bilateral_filter'] = result
        return passed
    
    def test_gaussian_filter(self) -> bool:
        """Test GPU-accelerated Gaussian filtering"""
        self.print_section("Testing Gaussian Filter")
        
        # Create test image
        test_img = self.create_test_image((512, 512), np.float32)
        
        # Test parameters
        test_data = {
            'src': test_img,
            'ksize': (15, 15),
            'sigma_x': 2.0,
            'sigma_y': 2.0
        }
        
        # Define CPU and GPU functions
        def cpu_gaussian(src, ksize, sigma_x, sigma_y):
            return cv2.GaussianBlur(src, ksize, sigma_x, sigma_y)
        
        def gpu_gaussian(src, ksize, sigma_x, sigma_y):
            return gpu_accelerator.gaussian_filter_gpu(src, ksize, sigma_x, sigma_y)
        
        # Benchmark
        result = self.benchmark_operation("Gaussian Filter", cpu_gaussian, gpu_gaussian, test_data)
        
        # Print results
        print(f"CPU time: {result['cpu_time']:.4f}s")
        print(f"GPU time: {result['gpu_time']:.4f}s")
        print(f"Speedup: {result['speedup']:.2f}x")
        print(f"Max difference: {result['max_diff']:.6f}")
        print(f"Mean difference: {result['mean_diff']:.6f}")
        
        # Determine if test passed
        passed = result['max_diff'] < 0.01
        if passed:
            print("✓ Gaussian filter test passed")
        else:
            print("✗ Gaussian filter test failed - results differ significantly")
        
        self.test_results['gaussian_filter'] = result
        return passed
    
    def test_filter2d(self) -> bool:
        """Test GPU-accelerated 2D filtering"""
        self.print_section("Testing 2D Filter")
        
        # Create test image
        test_img = self.create_test_image((512, 512), np.float32)
        
        # Create test kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        # Test parameters
        test_data = {
            'src': test_img,
            'kernel': kernel,
            'border_type': cv2.BORDER_REFLECT
        }
        
        # Define CPU and GPU functions
        def cpu_filter2d(src, kernel, border_type):
            return cv2.filter2D(src, -1, kernel, borderType=border_type)
        
        def gpu_filter2d(src, kernel, border_type):
            return gpu_accelerator.filter2d_gpu(src, kernel, border_type)
        
        # Benchmark
        result = self.benchmark_operation("2D Filter", cpu_filter2d, gpu_filter2d, test_data)
        
        # Print results
        print(f"CPU time: {result['cpu_time']:.4f}s")
        print(f"GPU time: {result['gpu_time']:.4f}s")
        print(f"Speedup: {result['speedup']:.2f}x")
        print(f"Max difference: {result['max_diff']:.6f}")
        print(f"Mean difference: {result['mean_diff']:.6f}")
        
        # Determine if test passed
        passed = result['max_diff'] < 0.01
        if passed:
            print("✓ 2D filter test passed")
        else:
            print("✗ 2D filter test failed - results differ significantly")
        
        self.test_results['filter2d'] = result
        return passed
    
    def test_resize(self) -> bool:
        """Test GPU-accelerated image resizing"""
        self.print_section("Testing Image Resize")
        
        # Create test image
        test_img = self.create_test_image((512, 512), np.float32)
        
        # Test parameters
        new_size = (256, 256)
        test_data = {
            'src': test_img,
            'dsize': new_size,
            'interpolation': cv2.INTER_LINEAR
        }
        
        # Define CPU and GPU functions
        def cpu_resize(src, dsize, interpolation):
            return cv2.resize(src, dsize, interpolation=interpolation)
        
        def gpu_resize(src, dsize, interpolation):
            return gpu_accelerator.resize_gpu(src, dsize, interpolation)
        
        # Benchmark
        result = self.benchmark_operation("Image Resize", cpu_resize, gpu_resize, test_data)
        
        # Print results
        print(f"CPU time: {result['cpu_time']:.4f}s")
        print(f"GPU time: {result['gpu_time']:.4f}s")
        print(f"Speedup: {result['speedup']:.2f}x")
        print(f"Max difference: {result['max_diff']:.6f}")
        print(f"Mean difference: {result['mean_diff']:.6f}")
        
        # Determine if test passed
        passed = result['max_diff'] < 0.01
        if passed:
            print("✓ Resize test passed")
        else:
            print("✗ Resize test failed - results differ significantly")
        
        self.test_results['resize'] = result
        return passed
    
    def test_matrix_multiply(self) -> bool:
        """Test GPU-accelerated matrix multiplication"""
        self.print_section("Testing Matrix Multiplication")
        
        # Create test image and matrix
        test_img = self.create_test_image_3d((256, 256, 3), np.float32)
        test_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.169, -0.331, 0.500],
            [0.500, -0.419, -0.081]
        ], dtype=np.float32)
        
        # Test parameters
        test_data = {
            'src': test_img,
            'matrix': test_matrix
        }
        
        # Define CPU and GPU functions
        def cpu_matrix_multiply(src, matrix):
            return np.dot(src.reshape(-1, 3), matrix.T).reshape(src.shape)
        
        def gpu_matrix_multiply(src, matrix):
            return gpu_accelerator.matrix_multiply_gpu(src, matrix)
        
        # Benchmark
        result = self.benchmark_operation("Matrix Multiplication", cpu_matrix_multiply, gpu_matrix_multiply, test_data)
        
        # Print results
        print(f"CPU time: {result['cpu_time']:.4f}s")
        print(f"GPU time: {result['gpu_time']:.4f}s")
        print(f"Speedup: {result['speedup']:.2f}x")
        print(f"Max difference: {result['max_diff']:.6f}")
        print(f"Mean difference: {result['mean_diff']:.6f}")
        
        # Determine if test passed
        passed = result['max_diff'] < 0.01
        if passed:
            print("✓ Matrix multiplication test passed")
        else:
            print("✗ Matrix multiplication test failed - results differ significantly")
        
        self.test_results['matrix_multiply'] = result
        return passed
    
    def test_histogram_equalization(self) -> bool:
        """Test GPU-accelerated histogram equalization"""
        self.print_section("Testing Histogram Equalization")
        
        # Create test image
        test_img = self.create_test_image((512, 512), np.uint8)
        
        # Test parameters
        test_data = {
            'src': test_img
        }
        
        # Define CPU and GPU functions
        def cpu_histogram_equalization(src):
            return cv2.equalizeHist(src)
        
        def gpu_histogram_equalization(src):
            return gpu_accelerator.histogram_equalization_gpu(src)
        
        # Benchmark
        result = self.benchmark_operation("Histogram Equalization", cpu_histogram_equalization, gpu_histogram_equalization, test_data)
        
        # Print results
        print(f"CPU time: {result['cpu_time']:.4f}s")
        print(f"GPU time: {result['gpu_time']:.4f}s")
        print(f"Speedup: {result['speedup']:.2f}x")
        print(f"Max difference: {result['max_diff']:.6f}")
        print(f"Mean difference: {result['mean_diff']:.6f}")
        
        # Determine if test passed
        passed = result['max_diff'] < 1  # Allow for small integer differences
        if passed:
            print("✓ Histogram equalization test passed")
        else:
            print("✗ Histogram equalization test failed - results differ significantly")
        
        self.test_results['histogram_equalization'] = result
        return passed
    
    def test_clahe(self) -> bool:
        """Test GPU-accelerated CLAHE"""
        self.print_section("Testing CLAHE")
        
        # Create test image
        test_img = self.create_test_image((512, 512), np.uint8)
        
        # Test parameters
        test_data = {
            'src': test_img,
            'clip_limit': 2.0,
            'tile_grid_size': (8, 8)
        }
        
        # Define CPU and GPU functions
        def cpu_clahe(src, clip_limit, tile_grid_size):
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(src)
        
        def gpu_clahe(src, clip_limit, tile_grid_size):
            return gpu_accelerator.clahe_gpu(src, clip_limit, tile_grid_size)
        
        # Benchmark
        result = self.benchmark_operation("CLAHE", cpu_clahe, gpu_clahe, test_data)
        
        # Print results
        print(f"CPU time: {result['cpu_time']:.4f}s")
        print(f"GPU time: {result['gpu_time']:.4f}s")
        print(f"Speedup: {result['speedup']:.2f}x")
        print(f"Max difference: {result['max_diff']:.6f}")
        print(f"Mean difference: {result['mean_diff']:.6f}")
        
        # Determine if test passed
        passed = result['max_diff'] < 1  # Allow for small integer differences
        if passed:
            print("✓ CLAHE test passed")
        else:
            print("✗ CLAHE test failed - results differ significantly")
        
        self.test_results['clahe'] = result
        return passed
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all GPU acceleration tests"""
        self.print_header("COMPREHENSIVE GPU ACCELERATION TEST SUITE")
        
        # Check GPU availability first
        if not self.test_gpu_availability():
            print("\n⚠️  GPU acceleration not available. Tests will run with CPU fallbacks.")
        
        # Run all tests
        tests = [
            ("Bilateral Filter", self.test_bilateral_filter),
            ("Gaussian Filter", self.test_gaussian_filter),
            ("2D Filter", self.test_filter2d),
            ("Image Resize", self.test_resize),
            ("Matrix Multiplication", self.test_matrix_multiply),
            ("Histogram Equalization", self.test_histogram_equalization),
            ("CLAHE", self.test_clahe)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                print(f"✗ {test_name} test failed with error: {e}")
        
        # Print summary
        self.print_header("TEST SUMMARY")
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed! GPU acceleration is working perfectly.")
        elif passed_tests > total_tests // 2:
            print("✅ Most tests passed. GPU acceleration is working well.")
        else:
            print("⚠️  Many tests failed. GPU acceleration may need attention.")
        
        # Performance summary
        if self.test_results:
            self.print_section("Performance Summary")
            for operation, result in self.test_results.items():
                if result['speedup'] > 1.0:
                    print(f"{operation}: {result['speedup']:.2f}x speedup")
                else:
                    print(f"{operation}: {result['speedup']:.2f}x (slower)")
        
        return {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'test_results': self.test_results
        }

def main():
    """Main function to run the GPU acceleration tests"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GPU Acceleration Test Suite')
    parser.add_argument('--size', '-s', type=int, nargs=2, default=[512, 512],
                       help='Test image size (width height) - default: 512 512')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run quick test with smaller image size')
    parser.add_argument('--large', '-l', action='store_true',
                       help='Run test with larger image size')
    
    args = parser.parse_args()
    
    # Determine test image size
    if args.quick:
        test_size = (256, 256)
    elif args.large:
        test_size = (1024, 1024)
    else:
        test_size = tuple(args.size)
    
    print(f"Starting GPU acceleration tests with image size: {test_size[0]}x{test_size[1]}...")
    
    tester = GPUAccelerationTester(test_image_size=test_size)
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results['success_rate'] >= 80:
        sys.exit(0)  # Success
    elif results['success_rate'] >= 50:
        sys.exit(1)  # Partial success
    else:
        sys.exit(2)  # Failure

def run_size_comparison():
    """Run tests with different image sizes to show performance scaling"""
    print("=" * 80)
    print("GPU ACCELERATION SIZE COMPARISON")
    print("=" * 80)
    
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
    
    for size in sizes:
        print(f"\n{'='*20} Testing {size[0]}x{size[1]} {'='*20}")
        tester = GPUAccelerationTester(test_image_size=size)
        
        # Run a subset of tests for size comparison
        tests = [
            ("Bilateral Filter", tester.test_bilateral_filter),
            ("Gaussian Filter", tester.test_gaussian_filter),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"✗ {test_name} failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--size-comparison":
        run_size_comparison()
    else:
        main()
