"""
File: gamma_correction_numba_fallback.py
Description: Numba-optimized Gamma Correction with fallback to original NumPy
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array

# Try to import Numba, fallback to original if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Import original implementation as fallback
from modules.gamma_correction.gamma_correction import GammaCorrection as GammaCorrectionOriginal

@jit(nopython=True)
def generate_gamma_lut_numba(bit_depth):
    max_val = 2**bit_depth - 1
    lut = np.arange(0, max_val + 1, dtype=np.uint32)
    lut = (np.round(max_val * ((lut / max_val) ** (1 / 2.2)))).astype(np.uint32)
    return lut

@jit(nopython=True, parallel=True)
def apply_gamma_numba(img, lut):
    height, width, channels = img.shape
    result = np.empty_like(img)
    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                result[i, j, c] = lut[img[i, j, c]]
    return result

class GammaCorrectionNumbaFallback:
    """
    Numba-optimized Gamma Correction with fallback to original NumPy implementation
    """
    def __init__(self, img, platform, sensor_info, parm_gmm):
        self.img = img
        self.enable = parm_gmm["is_enable"]
        self.sensor_info = sensor_info
        self.output_bit_depth = sensor_info["output_bit_depth"]
        self.parm_gmm = parm_gmm
        self.is_save = parm_gmm["is_save"]
        self.platform = platform
        # Create fallback instance
        self.fallback_gc = GammaCorrectionOriginal(img, platform, sensor_info, parm_gmm)

    def generate_gamma_lut(self, bit_depth):
        return generate_gamma_lut_numba(bit_depth)

    def apply_gamma(self):
        lut = self.generate_gamma_lut(self.output_bit_depth)
        return apply_gamma_numba(self.img, lut)

    def save(self):
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_gamma_correction_numba_",
                self.platform,
                self.sensor_info["output_bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        if not self.enable:
            return self.img
        if NUMBA_AVAILABLE:
            try:
                start = time.time()
                self.img = self.apply_gamma()
                print(f"  Numba Gamma execution time: {time.time() - start:.3f}s")
                self.save()
                return self.img
            except Exception as e:
                print(f"Numba Gamma failed: {e}")
                print("Falling back to NumPy implementation...")
        # Use original implementation
        return self.fallback_gc.execute() 