"""
File: demosaic_opencv_cuda.py
Description: OpenCV CUDA demosaicing wrapper for benchmarking
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
import cv2
from util.utils import save_output_array

# Map Bayer pattern string to OpenCV code
BAYER_TO_OPENCV = {
    "RGGB": cv2.COLOR_BayerRG2BGR,
    "BGGR": cv2.COLOR_BayerBG2BGR,
    "GRBG": cv2.COLOR_BayerGR2BGR,
    "GBRG": cv2.COLOR_BayerGB2BGR,
}
BAYER_TO_OPENCV_CUDA = {
    "RGGB": cv2.COLOR_BayerRG2BGR,
    "BGGR": cv2.COLOR_BayerBG2BGR,
    "GRBG": cv2.COLOR_BayerGR2BGR,
    "GBRG": cv2.COLOR_BayerGB2BGR,
}

class DemosaicOpenCVCUDA:
    """
    OpenCV CUDA demosaicing wrapper
    """
    def __init__(self, img, platform, sensor_info, parm_dga):
        self.img = img
        self.bayer = sensor_info["bayer_pattern"].upper()
        self.bit_depth = sensor_info["output_bit_depth"]
        self.is_save = parm_dga["is_save"]
        self.sensor_info = sensor_info
        self.platform = platform
        self.is_debug = parm_dga.get("is_debug", False)

    def save(self):
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_demosaic_opencv_cuda_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        start = time.time()
        # OpenCV expects 8/16-bit single channel input
        img_in = self.img
        if img_in.ndim == 3:
            img_in = img_in[..., 0]  # Take only one channel if accidentally 3D
        if img_in.dtype != np.uint16 and img_in.dtype != np.uint8:
            img_in = img_in.astype(np.uint16)
        # Upload to GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img_in)
        # Demosaic
        code = BAYER_TO_OPENCV_CUDA.get(self.bayer, cv2.COLOR_BayerRG2BGR)
        gpu_demosaic = cv2.cuda.demosaicing(gpu_img, code)
        # Download result
        demos_out = gpu_demosaic.download()
        # Clip and convert to uint16
        demos_out = np.clip(demos_out, 0, 2**self.bit_depth - 1)
        demos_out = np.uint16(demos_out)
        if self.is_debug:
            print(f"  OpenCV CUDA Demosaic execution time: {time.time() - start:.3f}s")
        self.img = demos_out
        self.save()
        return self.img 