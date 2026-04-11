"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""

import os
from pathlib import Path

from brilliant_isp import BrilliantISP

CONFIG_PATH = "./config/AD_cam.yml"
# Override without editing:  ISP_RAW_DATA=/path/to/raw/dir python isp_pipeline.py
_DEFAULT_RAW = (
    "/media/brian/62B0FCB91A04680F1/drive_test/69B44B5E/"
    "NUIG_20260313_174146_005/output/AD_FL/raw/"
)
RAW_DATA = os.environ.get("ISP_RAW_DATA", _DEFAULT_RAW)
FILENAME = "frame_000000.raw"

if __name__ == "__main__":
    raw_file = Path(RAW_DATA).expanduser().resolve() / FILENAME
    if not raw_file.is_file():
        raise SystemExit(
            f"Raw file not found: {raw_file}\n"
            "Mount the storage device or set ISP_RAW_DATA to the directory that contains "
            f"{FILENAME!r} (example: ISP_RAW_DATA=/path/to/raw python isp_pipeline.py)."
        )
    brilliant_isp = BrilliantISP(RAW_DATA, CONFIG_PATH, outFileName="")
    brilliant_isp.execute(img_path=FILENAME)
