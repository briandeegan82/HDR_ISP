"""
File: color_correction_matrix_numba_fallback.py
Description: Numba-optimized Color Correction Matrix with fallback to original NumPy
Code / Paper  Reference: https://www.imatest.com/docs/colormatrix/
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
    print("CCM Numba: Using Numba acceleration")
except ImportError:
    NUMBA_AVAILABLE = False
    print("CCM Numba: Numba not available, using NumPy fallback")

# Import original implementation as fallback
from modules.color_correction_matrix.color_correction_matrix import ColorCorrectionMatrix as ColorCorrectionMatrixOriginal


@jit(nopython=True, parallel=True)
def apply_ccm_numba(img, ccm_mat, output_bit_depth):
    """
    Numba-optimized Color Correction Matrix application
    """
    height, width, channels = img.shape
    max_value = (2**output_bit_depth - 1)
    
    # Normalize to 0-1
    img_norm = img.astype(np.float32) / max_value
    
    # Apply CCM
    result = np.empty_like(img_norm)
    
    for i in prange(height):
        for j in range(width):
            pixel = img_norm[i, j]
            # Matrix multiplication: pixel * ccm_mat.T
            result[i, j, 0] = pixel[0] * ccm_mat[0, 0] + pixel[1] * ccm_mat[0, 1] + pixel[2] * ccm_mat[0, 2]
            result[i, j, 1] = pixel[0] * ccm_mat[1, 0] + pixel[1] * ccm_mat[1, 1] + pixel[2] * ccm_mat[1, 2]
            result[i, j, 2] = pixel[0] * ccm_mat[2, 0] + pixel[1] * ccm_mat[2, 1] + pixel[2] * ccm_mat[2, 2]
    
    # Clip and convert back
    result = np.clip(result, 0, 1)
    result = (result * max_value).astype(np.uint16)
    
    return result


class ColorCorrectionMatrixNumbaFallback:
    """
    Numba-optimized Color Correction Matrix with fallback to original NumPy implementation
    """

    def __init__(self, img, platform, sensor_info, parm_ccm):
        self.img = img
        self.enable = parm_ccm["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ccm = parm_ccm
        self.output_bit_depth = sensor_info["output_bit_depth"]
        self.ccm_mat = None
        self.is_save = parm_ccm["is_save"]
        self.platform = platform
        
        # Create fallback instance
        self.fallback_ccm = ColorCorrectionMatrixOriginal(img, platform, sensor_info, parm_ccm)

    def _numba_apply_ccm(self):
        """
        Apply CCM with Numba optimization
        """
        r_1 = np.array(self.parm_ccm["corrected_red"])
        r_2 = np.array(self.parm_ccm["corrected_green"])
        r_3 = np.array(self.parm_ccm["corrected_blue"])

        self.ccm_mat = np.array([r_1, r_2, r_3], dtype=np.float32)

        # Apply Numba-optimized CCM
        return apply_ccm_numba(self.img, self.ccm_mat, self.output_bit_depth)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_color_correction_matrix_numba_",
                self.platform,
                self.sensor_info["output_bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute CCM with Numba optimization or fallback"""
        if not self.enable:
            return self.img

        if NUMBA_AVAILABLE:
            try:
                start = time.time()
                self.img = self._numba_apply_ccm()
                print(f"  Numba CCM execution time: {time.time() - start:.3f}s")
                self.save()
                return self.img
            except Exception as e:
                print(f"Numba CCM failed: {e}")
                print("Falling back to NumPy implementation...")
                # Fall through to original implementation
        
        # Use original implementation
        return self.fallback_ccm.execute() 