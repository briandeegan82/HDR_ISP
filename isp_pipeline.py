"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from infinite_isp import InfiniteISP
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the ISP pipeline')
    parser.add_argument('--config', default="./config/svs_cam.yml", help='Path to config file')
    parser.add_argument('--data', default="./in_frames/hdr_mode/", help='Path to input data directory')
    parser.add_argument('--filename', default='RV.raw', help='Input filename')
    parser.add_argument('--save-intermediate', action='store_true', help='Save intermediate images at each stage')
    
    args = parser.parse_args()

    infinite_isp = InfiniteISP(args.data, args.config)
    infinite_isp.execute(img_path=args.filename, save_intermediate=args.save_intermediate)

if __name__ == "__main__":
    main()
