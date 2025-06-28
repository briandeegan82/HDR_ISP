"""
File: auto_exposure_cuda_fallback.py
Description: CUDA-optimized Auto Exposure with fallback to Numba/NumPy
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

# Try to import CuPy, fallback to Numba if not available
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Try to import Numba, fallback to original if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Import original implementation as fallback
from modules.auto_exposure.auto_exposure import AutoExposure as AutoExposureOriginal
from modules.auto_exposure.auto_exposure_numba import (
    convert_to_8bit_numba, rgb_to_greyscale_numba, calculate_skewness_numba
)

# CUDA helpers
if CUDA_AVAILABLE:
    from modules.auto_exposure.auto_exposure_cuda import (
        convert_to_8bit_cuda, rgb_to_greyscale_cuda, calculate_skewness_cuda
    )

class AutoExposureCUDAFallback:
    """
    CUDA-optimized Auto Exposure with fallback to Numba/NumPy
    """
    def __init__(self, img, sensor_info, parm_ae):
        self.img = img
        self.enable = parm_ae["is_enable"]
        self.is_debug = parm_ae["is_debug"]
        self.center_illuminance = parm_ae["center_illuminance"]
        self.histogram_skewness_range = parm_ae["histogram_skewness"]
        self.sensor_info = sensor_info
        self.param_ae = parm_ae
        self.bit_depth = sensor_info["bit_depth"]
        # Create fallback instance
        self.fallback_ae = AutoExposureOriginal(img, sensor_info, parm_ae)

    def get_exposure_feedback(self):
        if CUDA_AVAILABLE:
            self.img = convert_to_8bit_cuda(self.img, self.bit_depth)
            self.bit_depth = 8
            return self.determine_exposure_cuda()
        elif NUMBA_AVAILABLE:
            self.img = convert_to_8bit_numba(self.img, self.bit_depth)
            self.bit_depth = 8
            return self.determine_exposure_numba()
        else:
            return self.fallback_ae.get_exposure_feedback()

    def determine_exposure_cuda(self):
        grey_img = rgb_to_greyscale_cuda(self.img, self.bit_depth)
        avg_lum = np.average(grey_img)
        if self.is_debug:
            print("Average luminance is = ", avg_lum)
        skewness = calculate_skewness_cuda(grey_img, self.center_illuminance)
        upper_limit = self.histogram_skewness_range
        lower_limit = -1 * upper_limit
        if self.is_debug:
            print("   - AE - Histogram Skewness Range = ", upper_limit)
        if skewness < lower_limit:
            return -1
        elif skewness > upper_limit:
            return 1
        else:
            return 0

    def determine_exposure_numba(self):
        grey_img = rgb_to_greyscale_numba(self.img, self.bit_depth)
        avg_lum = np.average(grey_img)
        if self.is_debug:
            print("Average luminance is = ", avg_lum)
        skewness = calculate_skewness_numba(grey_img, self.center_illuminance)
        upper_limit = self.histogram_skewness_range
        lower_limit = -1 * upper_limit
        if self.is_debug:
            print("   - AE - Histogram Skewness Range = ", upper_limit)
        if skewness < lower_limit:
            return -1
        elif skewness > upper_limit:
            return 1
        else:
            return 0

    def execute(self):
        if not self.enable:
            return None
        if CUDA_AVAILABLE:
            start = time.time()
            ae_feedback = self.get_exposure_feedback()
            print(f"  CUDA AE execution time: {time.time()-start:.3f}s")
            return ae_feedback
        elif NUMBA_AVAILABLE:
            start = time.time()
            ae_feedback = self.get_exposure_feedback()
            print(f"  Numba AE execution time: {time.time()-start:.3f}s")
            return ae_feedback
        else:
            return self.fallback_ae.execute() 