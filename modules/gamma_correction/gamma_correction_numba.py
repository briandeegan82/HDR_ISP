"""
File: gamma_correction_numba.py
Description: Numba-optimized Gamma Correction implementation
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from numba import jit, prange

from util.utils import save_output_array


@jit(nopython=True)
def generate_gamma_lut_numba(bit_depth):
    """
    Numba-optimized Gamma LUT generation
    """
    max_val = 2**bit_depth - 1
    lut = np.arange(0, max_val + 1, dtype=np.uint32)
    lut = (np.round(max_val * ((lut / max_val) ** (1 / 2.2)))).astype(np.uint32)
    return lut


@jit(nopython=True, parallel=True)
def apply_gamma_numba(img, lut):
    """
    Numba-optimized Gamma LUT application
    """
    height, width, channels = img.shape
    result = np.empty_like(img)
    
    for i in prange(height):
        for j in range(width):
            for c in range(channels):
                result[i, j, c] = lut[img[i, j, c]]
    
    return result


class GammaCorrectionNumba:
    """
    Numba-optimized Gamma Correction
    """

    def __init__(self, img, platform, sensor_info, parm_gmm):
        self.img = img
        self.enable = parm_gmm["is_enable"]
        self.sensor_info = sensor_info
        self.output_bit_depth = sensor_info["output_bit_depth"]
        self.parm_gmm = parm_gmm
        self.is_save = parm_gmm["is_save"]
        self.platform = platform

    def generate_gamma_lut(self, bit_depth):
        """
        Generates Gamma LUT with Numba optimization
        """
        return generate_gamma_lut_numba(bit_depth)

    def apply_gamma(self):
        """
        Apply Gamma LUT with Numba optimization
        """
        # generate LUT
        lut = self.generate_gamma_lut(self.output_bit_depth)

        # apply LUT with Numba optimization
        return apply_gamma_numba(self.img, lut)

    def save(self):
        """
        Function to save module output
        """
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
        """
        Execute Gamma Correction with Numba optimization
        """
        if self.enable is True:
            start = time.time()
            gc_out = self.apply_gamma()
            print(f"  Numba Gamma execution time: {time.time() - start:.3f}s")
            self.img = gc_out

        self.save()
        return self.img 