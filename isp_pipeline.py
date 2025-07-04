"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

from infinite_isp import InfiniteISP

CONFIG_PATH = "./config/samsung.yml"
RAW_DATA = "./in_frames/hdr_mode/"
FILENAME = 'Samsung1H1RAWexample.tiff'

if __name__ == "__main__":

    infinite_isp = InfiniteISP(RAW_DATA, CONFIG_PATH)

    # set generate_tv flag to false
    infinite_isp.execute(img_path=FILENAME)
