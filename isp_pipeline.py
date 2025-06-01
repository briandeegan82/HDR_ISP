"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from infinite_isp import InfiniteISP
import argparse
import time
import cProfile
import pstats
import io
import numpy as np
import os

def get_file_size_mb(file_path):
    """Get file size in megabytes"""
    return os.path.getsize(file_path) / (1024 * 1024)

def profile_color_space_conversion(isp_instance, img_path, save_intermediate):
    """Profile color space conversion operations"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute the pipeline
    isp_instance.execute(img_path=img_path, save_intermediate=save_intermediate)
    
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    # Print profiling results
    print("\nColor Space Conversion Profiling Results:")
    print(s.getvalue())

def main():
    parser = argparse.ArgumentParser(description='Run the ISP pipeline')
    parser.add_argument('--config', default="./config/svs_cam.yml", help='Path to config file')
    parser.add_argument('--data', default="./in_frames/hdr_mode/", help='Path to input data directory')
    parser.add_argument('--filename', default='RV.raw', help='Input filename')
    parser.add_argument('--save-intermediate', action='store_true', help='Save intermediate images at each stage')
    parser.add_argument('--profile', action='store_true', help='Profile color space conversion operations')
    parser.add_argument('--memory-map', action='store_true', help='Use memory mapping for large files')
    parser.add_argument('--memory-threshold', type=float, default=100.0, 
                       help='File size threshold in MB to trigger memory mapping')
    
    args = parser.parse_args()

    # Check if we should use memory mapping
    full_path = os.path.join(args.data, args.filename)
    use_memory_map = args.memory_map and get_file_size_mb(full_path) > args.memory_threshold
    
    if use_memory_map:
        print(f"Using memory mapping for large file: {full_path}")
        # Create a memory-mapped array
        img_data = np.memmap(full_path, dtype=np.uint8, mode='r+')
    else:
        img_data = None

    infinite_isp = InfiniteISP(args.data, args.config, memory_mapped_data=img_data if use_memory_map else None)
    
    if args.profile:
        profile_color_space_conversion(infinite_isp, args.filename, args.save_intermediate)
    else:
        infinite_isp.execute(img_path=args.filename, save_intermediate=args.save_intermediate)

if __name__ == "__main__":
    main()
