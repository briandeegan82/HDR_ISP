"""
This script is used to run isp_pipeline.py on a dataset placed in ./inframes/normal/data
It also fetches if a separate config of a raw image is present othewise uses the default config
"""

import os
from pathlib import Path
from tqdm import tqdm
from infinite_isp import InfiniteISP
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Optional, Dict, Any
import logging
import traceback
import numpy as np
import sys

from util.config_utils import parse_file_name, extract_raw_metadata
from util.utils import OUTPUT_DIR, OUTPUT_ARRAY_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = "./in_frames/hdr_mode/svs"
CONFIG_PATH = "./config/svs_cam.yml"
OUTPUT_PATH = "./out_frames/hdr_mode/svs"  # Added output path
VIDEO_MODE = False
EXTRACT_SENSOR_INFO = True
UPDATE_BLC_WB = True
MAX_WORKERS = 4  # Adjust based on your system's capabilities

# Override the hardcoded output paths in util.utils
OUTPUT_DIR = OUTPUT_PATH
OUTPUT_ARRAY_DIR = OUTPUT_PATH

def setup_output_directory():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    logger.info(f"Output directory set to: {OUTPUT_PATH}")

def process_single_image(raw: str, default_config: str) -> None:
    """
    Process a single image with a fresh InfiniteISP instance
    """
    try:
        logger.info(f"Starting to process image: {raw}")
        # Create a fresh ISP instance for each image
        infinite_isp = InfiniteISP(DATASET_PATH, default_config)
        
        # Configure output settings
        infinite_isp.c_yaml["platform"]["generate_tv"] = False
        infinite_isp.c_yaml["platform"]["output_dir"] = OUTPUT_PATH
        infinite_isp.c_yaml["platform"]["module_output_dir"] = OUTPUT_PATH
        infinite_isp.c_yaml["platform"]["save_output"] = True
        
        raw_path_object = Path(raw)
        config_file = raw_path_object.stem + "-configs.yml"
        
        # check if the config file exists in the DATASET_PATH
        if find_files(config_file, DATASET_PATH):
            logger.info(f"Found config file: {config_file}")
            infinite_isp.load_config(DATASET_PATH + config_file)
            # Ensure output settings are maintained after loading config
            infinite_isp.c_yaml["platform"]["output_dir"] = OUTPUT_PATH
            infinite_isp.c_yaml["platform"]["module_output_dir"] = OUTPUT_PATH
            infinite_isp.c_yaml["platform"]["save_output"] = True
            infinite_isp.execute()
        else:
            logger.info(f"No specific config found for {raw}, using default config")

            if EXTRACT_SENSOR_INFO:
                if raw_path_object.suffix.lower() == ".raw":
                    logger.info(f"Extracting sensor info from filename for {raw}")
                    sensor_info = parse_file_name(raw)
                    if sensor_info:
                        infinite_isp.update_sensor_info(sensor_info)
                        logger.info(f"Updated sensor_info into config: {sensor_info}")
                    else:
                        logger.warning("No information in filename - sensor_info not updated")
                else:
                    logger.info(f"Extracting sensor info from metadata for {raw}")
                    sensor_info = extract_raw_metadata(DATASET_PATH + raw)
                    if sensor_info:
                        infinite_isp.update_sensor_info(sensor_info, UPDATE_BLC_WB)
                        logger.info(f"Updated sensor_info into config: {sensor_info}")
                    else:
                        logger.warning("Not compatible file for metadata - sensor_info not updated")

            infinite_isp.execute(raw)
            
            # Verify output after execution
            output_files = os.listdir(OUTPUT_PATH)
            logger.info(f"Output directory contents after {raw} processing: {output_files}")

        logger.info(f"Successfully processed image: {raw}")
    except Exception as e:
        logger.error(f"Error processing {raw}: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

def video_processing():
    """
    Processed Images in a folder [DATASET_PATH] like frames of an Image.
    - All images are processed with same config file located at CONFIG_PATH
    - 3A Stats calculated on a frame are applied on the next frame
    """
    raw_files = [f_name for f_name in os.listdir(DATASET_PATH) if ".raw" in f_name]
    raw_files.sort()

    infinite_isp = InfiniteISP(DATASET_PATH, CONFIG_PATH)
    infinite_isp.c_yaml["platform"]["generate_tv"] = False
    infinite_isp.c_yaml["platform"]["render_3a"] = False
    infinite_isp.c_yaml["platform"]["output_dir"] = OUTPUT_PATH
    infinite_isp.c_yaml["platform"]["module_output_dir"] = OUTPUT_PATH
    infinite_isp.c_yaml["platform"]["save_output"] = True

    for file in tqdm(raw_files, disable=False, leave=True):
        infinite_isp.execute(file)
        infinite_isp.load_3a_statistics()

def dataset_processing():
    """
    Processed each image as a single entity that may or may not have its config
    - If config file in the dataset folder has format filename-configs.yml it will
    be use to proocess the image otherwise default config is used.
    - For 3a-rendered output - set 3a_render flag in config file to true.
    """
    # Ensure output directory exists
    setup_output_directory()
    
    default_config = CONFIG_PATH
    directory_content = os.listdir(DATASET_PATH)
    
    # Get all raw images, case-insensitive
    raw_images = [
        x for x in directory_content
        if Path(DATASET_PATH, x).suffix.lower() in [".raw", ".nef", ".dng"]
    ]
    
    logger.info(f"Found {len(raw_images)} raw images: {raw_images}")
    logger.info(f"Input directory contents: {directory_content}")

    print(f"Processing {len(raw_images)} images using {MAX_WORKERS} workers...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list of futures
        futures = [
            executor.submit(process_single_image, raw, default_config)
            for raw in raw_images
        ]
        
        # Use tqdm to show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), 
                     total=len(futures),
                     desc="Processing images",
                     ncols=100):
            pass

def find_files(filename, search_path):
    """
    This function is used to find the files in the search_path
    """
    for _, _, files in os.walk(search_path):
        if filename in files:
            return True
    return False

if __name__ == "__main__":
    if VIDEO_MODE:
        print("PROCESSING VIDEO FRAMES ONE BY ONE IN SEQUENCE")
        video_processing()
    else:
        print("PROCESSING DATASET IMAGES IN PARALLEL")
        dataset_processing()
