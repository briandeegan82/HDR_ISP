"""
File: performance_comparison.py
Description: Performance comparison between CPU and GPU-accelerated ISP pipeline
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Import both CPU and GPU versions
from infinite_isp import InfiniteISP
from infinite_isp_gpu import InfiniteISPGPU

def run_performance_comparison(data_path, config_path, num_runs=3):
    """
    Run performance comparison between CPU and GPU versions
    
    Args:
        data_path (str): Path to the data directory
        config_path (str): Path to the configuration file
        num_runs (int): Number of runs for averaging
    """
    
    print("=" * 60)
    print("PERFORMANCE COMPARISON: CPU vs GPU-ACCELERATED ISP")
    print("=" * 60)
    
    # Initialize both pipelines
    print("Initializing CPU pipeline...")
    cpu_pipeline = InfiniteISP(data_path, config_path)
    
    print("Initializing GPU pipeline...")
    gpu_pipeline = InfiniteISPGPU(data_path, config_path)
    
    # Load configuration to get image info
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        sensor_info = config["sensor_info"]
        image_size = f"{sensor_info['width']}x{sensor_info['height']}"
        bit_depth = sensor_info["bit_depth"]
    
    print(f"\nImage specifications:")
    print(f"  Size: {image_size}")
    print(f"  Bit depth: {bit_depth}")
    print(f"  Bayer pattern: {sensor_info['bayer_pattern']}")
    
    # Performance measurement
    cpu_times = []
    gpu_times = []
    
    print(f"\nRunning {num_runs} iterations for each pipeline...")
    
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        
        # CPU pipeline
        print("Running CPU pipeline...")
        start_time = time.time()
        cpu_output = cpu_pipeline.execute(save_intermediate=False)
        cpu_time = time.time() - start_time
        cpu_times.append(cpu_time)
        print(f"CPU execution time: {cpu_time:.3f}s")
        
        # GPU pipeline
        print("Running GPU pipeline...")
        start_time = time.time()
        gpu_output = gpu_pipeline.execute(save_intermediate=False)
        gpu_time = time.time() - start_time
        gpu_times.append(gpu_time)
        print(f"GPU execution time: {gpu_time:.3f}s")
        
        # Verify outputs are similar (within tolerance)
        if cpu_output is not None and gpu_output is not None and cpu_output.shape == gpu_output.shape:
            diff = np.abs(cpu_output.astype(np.float32) - gpu_output.astype(np.float32))
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"Output verification - Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
        else:
            print("Warning: Output shapes differ or outputs are None!")
    
    # Calculate statistics
    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    gpu_mean = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    
    speedup = cpu_mean / gpu_mean
    
    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"CPU Pipeline:")
    print(f"  Mean time: {cpu_mean:.3f}s ± {cpu_std:.3f}s")
    print(f"  Min time: {min(cpu_times):.3f}s")
    print(f"  Max time: {max(cpu_times):.3f}s")
    
    print(f"\nGPU Pipeline:")
    print(f"  Mean time: {gpu_mean:.3f}s ± {gpu_std:.3f}s")
    print(f"  Min time: {min(gpu_times):.3f}s")
    print(f"  Max time: {max(gpu_times):.3f}s")
    
    print(f"\nPerformance Improvement:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time reduction: {((cpu_mean - gpu_mean) / cpu_mean * 100):.1f}%")
    
    # Create performance plot
    create_performance_plot(cpu_times, gpu_times, image_size, speedup)
    
    return {
        'cpu_times': cpu_times,
        'gpu_times': gpu_times,
        'speedup': speedup,
        'image_size': image_size,
        'bit_depth': bit_depth
    }

def create_performance_plot(cpu_times, gpu_times, image_size, speedup):
    """
    Create a performance comparison plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot of mean times
    categories = ['CPU', 'GPU']
    means = [np.mean(cpu_times), np.mean(gpu_times)]
    stds = [np.std(cpu_times), np.std(gpu_times)]
    
    bars = ax1.bar(categories, means, yerr=stds, capsize=5, 
                   color=['#ff7f0e', '#2ca02c'], alpha=0.7)
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title(f'Mean Execution Time\nImage: {image_size}')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.3f}s', ha='center', va='bottom')
    
    # Scatter plot of individual runs
    run_numbers = list(range(1, len(cpu_times) + 1))
    ax2.scatter(run_numbers, cpu_times, color='#ff7f0e', label='CPU', alpha=0.7, s=100)
    ax2.scatter(run_numbers, gpu_times, color='#2ca02c', label='GPU', alpha=0.7, s=100)
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title(f'Individual Run Times\nSpeedup: {speedup:.2f}x')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPerformance plot saved as: performance_comparison.png")
    plt.show()

def benchmark_individual_modules(data_path, config_path):
    """
    Benchmark individual modules to identify bottlenecks
    """
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODULE BENCHMARKING")
    print("=" * 60)
    
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    # Initialize pipelines
    cpu_pipeline = InfiniteISP(data_path, config_path)
    gpu_pipeline = InfiniteISPGPU(data_path, config_path)
    
    # Load raw image
    cpu_pipeline.load_raw()
    gpu_pipeline.load_raw()
    
    # Define modules to benchmark (most computationally intensive)
    modules_to_benchmark = [
        ("Bayer Noise Reduction", "bayer_noise_reduction"),
        ("Demosaicing", "demosaic"),
        ("HDR Durand", "hdr_durand"),
        ("Color Space Conversion", "color_space_conversion"),
        ("Sharpening", "sharpen"),
        ("Scaling", "scale")
    ]
    
    print(f"{'Module':<25} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-" * 65)
    
    for module_name, module_key in modules_to_benchmark:
        # This is a simplified benchmark - in practice, you'd need to extract
        # individual module execution times from the pipeline
        print(f"{module_name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    print("\nNote: Individual module benchmarking requires instrumenting the pipeline")
    print("to measure execution times of each module separately.")

if __name__ == "__main__":
    # Example usage
    data_path = "in_frames"
    config_path = "config/samsung.yml"
    
    try:
        # Run performance comparison
        results = run_performance_comparison(data_path, config_path, num_runs=3)
        
        # Run individual module benchmarking
        benchmark_individual_modules(data_path, config_path)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data path and config path are correct.")
    except Exception as e:
        print(f"Error during performance comparison: {e}")
        import traceback
        traceback.print_exc() 