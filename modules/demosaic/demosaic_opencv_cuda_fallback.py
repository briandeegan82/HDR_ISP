"""
File: demosaic_opencv_cuda_fallback.py
Description: OpenCV CUDA demosaicing with fallback to original Cython implementation
Code / Paper  Reference: https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
import cv2

from util.utils import save_output_array

# Try to import OpenCV CUDA, fallback to original if not available
try:
    import cv2.cuda
    OPENCV_CUDA_AVAILABLE = True
    print("OpenCV CUDA demosaic: Using OpenCV CUDA acceleration")
except ImportError:
    OPENCV_CUDA_AVAILABLE = False
    print("OpenCV CUDA demosaic: OpenCV CUDA not available, using Cython fallback")

# Import original implementation as fallback
from modules.demosaic.demosaic import Demosaic as DemosaicOriginal


class DemosaicOpenCVCUDAFallback:
    """
    OpenCV CUDA demosaicing with fallback to original Cython implementation
    """

    def __init__(self, img, platform, sensor_info, parm_dem_or_dga):
        self.img = img
        self.sensor_info = sensor_info
        self.platform = platform
        
        # Handle both parm_dem and parm_dga parameter structures
        if "is_enable" in parm_dem_or_dga:
            # This is parm_dem from pipeline
            self.enable = parm_dem_or_dga["is_enable"]
            self.is_debug = parm_dem_or_dga.get("is_debug", False)
            self.is_save = parm_dem_or_dga.get("is_save", False)
            # Create a parm_dga structure for the fallback
            parm_dga = {"is_save": self.is_save}
        else:
            # This is parm_dga from test scripts
            self.enable = True  # Assume enabled for test scripts
            self.is_debug = parm_dem_or_dga.get("is_debug", False)
            self.is_save = parm_dem_or_dga.get("is_save", False)
            parm_dga = parm_dem_or_dga
        
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.bpp = self.sensor_info["bit_depth"]
        
        # Create fallback instance
        self.fallback_demosaic = DemosaicOriginal(img, platform, sensor_info, parm_dga)

    def _get_opencv_demosaic_code(self, bayer_pattern):
        """Convert bayer pattern to OpenCV demosaic code"""
        pattern_map = {
            "rggb": cv2.COLOR_BayerRG2RGB,
            "bggr": cv2.COLOR_BayerBG2RGB,
            "grbg": cv2.COLOR_BayerGR2RGB,
            "gbrg": cv2.COLOR_BayerGB2RGB,
        }
        return pattern_map.get(bayer_pattern.lower(), cv2.COLOR_BayerRG2RGB)

    def _opencv_cuda_demosaic(self):
        """Apply OpenCV CUDA demosaicing"""
        start = time.time()
        
        # Ensure image is in correct format
        if self.img.dtype != np.uint8 and self.img.dtype != np.uint16:
            # Convert to uint16 if needed
            img_uint16 = np.clip(self.img, 0, 65535).astype(np.uint16)
        else:
            img_uint16 = self.img.astype(np.uint16)
        
        # Get bayer pattern and convert to OpenCV code
        bayer_pattern = self.sensor_info["bayer_pattern"]
        demosaic_code = self._get_opencv_demosaic_code(bayer_pattern)
        
        # Upload to GPU
        gpu_img = cv2.cuda.GpuMat()
        gpu_img.upload(img_uint16)
        
        # Apply demosaicing
        gpu_demosaic = cv2.cuda.demosaicing(gpu_img, demosaic_code)
        
        # Download result
        demos_out = gpu_demosaic.download()
        
        # Convert back to float32 if original was float
        if self.img.dtype == np.float32:
            demos_out = demos_out.astype(np.float32)
        
        if self.is_debug:
            print(f"  OpenCV CUDA Demosaic execution time: {time.time() - start:.3f}s")
        
        return demos_out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_demosaic_opencv_cuda_",
                self.platform,
                self.bpp,
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute demosaicing with OpenCV CUDA or fallback"""
        if not self.enable:
            return self.img

        if OPENCV_CUDA_AVAILABLE:
            try:
                self.img = self._opencv_cuda_demosaic()
                self.save()
                return self.img
            except Exception as e:
                print(f"OpenCV CUDA demosaic failed: {e}")
                print("Falling back to Cython implementation...")
                # Fall through to original implementation
        
        # Use original implementation
        return self.fallback_demosaic.execute() 