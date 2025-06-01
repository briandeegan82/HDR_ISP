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
    
    args = parser.parse_args()

    infinite_isp = InfiniteISP(args.data, args.config)
    
    if args.profile:
        profile_color_space_conversion(infinite_isp, args.filename, args.save_intermediate)
    else:
        infinite_isp.execute(img_path=args.filename, save_intermediate=args.save_intermediate)

if __name__ == "__main__":
    main()
