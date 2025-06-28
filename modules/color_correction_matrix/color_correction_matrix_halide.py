"""
File: color_correction_matrix_halide.py
Description: Halide-optimized Color Correction Matrix implementation
Code / Paper  Reference: https://www.imatest.com/docs/colormatrix/
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

# Try to import Halide, fallback to original if not available
try:
    import halide as hl
    HALIDE_AVAILABLE = True
except ImportError:
    HALIDE_AVAILABLE = False

from util.utils import save_output_array


def apply_ccm_halide(img, ccm_mat, output_bit_depth):
    """
    Halide-optimized Color Correction Matrix application
    """
    if not HALIDE_AVAILABLE:
        raise ImportError("Halide not available")
    
    max_value = (2**output_bit_depth - 1)
    
    # Define Halide function for CCM
    def ccm_halide():
        # Input image
        input_img = hl.Func("input_img")
        x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")
        
        # CCM matrix
        ccm = hl.Func("ccm")
        i, j = hl.Var("i"), hl.Var("j")
        ccm[i, j] = hl.cast(hl.Float(32), ccm_mat[i, j])
        
        # Normalize input
        normalized = hl.Func("normalized")
        normalized[x, y, c] = hl.cast(hl.Float(32), input_img[x, y, c]) / max_value
        
        # Apply CCM
        result = hl.Func("result")
        result[x, y, c] = (normalized[x, y, 0] * ccm[c, 0] + 
                          normalized[x, y, 1] * ccm[c, 1] + 
                          normalized[x, y, 2] * ccm[c, 2])
        
        # Clip and convert back
        clipped = hl.Func("clipped")
        clipped[x, y, c] = hl.cast(hl.UInt(16), 
                                 hl.clamp(result[x, y, c], 0, 1) * max_value)
        
        return clipped
    
    # Compile and run
    height, width, channels = img.shape
    
    # Create input buffer
    input_buffer = hl.Buffer(hl.UInt(16), [width, height, channels])
    input_buffer.set_min(0, 0, 0)
    for i in range(width):
        for j in range(height):
            for k in range(channels):
                input_buffer[i, j, k] = img[j, i, k]
    
    # Compile function
    ccm_func = ccm_halide()
    ccm_func.compile_jit()
    
    # Run
    output_buffer = ccm_func.realize([width, height, channels])
    
    # Convert back to numpy
    result = np.empty((height, width, channels), dtype=np.uint16)
    for i in range(width):
        for j in range(height):
            for k in range(channels):
                result[j, i, k] = output_buffer[i, j, k]
    
    return result


class ColorCorrectionMatrixHalide:
    "Halide-optimized Color Correction Matrix"

    def __init__(self, img, platform, sensor_info, parm_ccm):
        self.img = img
        self.enable = parm_ccm["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ccm = parm_ccm
        self.output_bit_depth = sensor_info["output_bit_depth"]
        self.ccm_mat = None
        self.is_save = parm_ccm["is_save"]
        self.platform = platform

    def apply_ccm(self):
        """
        Apply CCM with Halide optimization
        """
        r_1 = np.array(self.parm_ccm["corrected_red"])
        r_2 = np.array(self.parm_ccm["corrected_green"])
        r_3 = np.array(self.parm_ccm["corrected_blue"])

        self.ccm_mat = np.array([r_1, r_2, r_3], dtype=np.float32)

        # Apply Halide-optimized CCM
        return apply_ccm_halide(self.img, self.ccm_mat, self.output_bit_depth)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_color_correction_matrix_halide_",
                self.platform,
                self.sensor_info["output_bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute CCM with Halide optimization if enabled."""
        if self.enable:
            start = time.time()
            ccm_out = self.apply_ccm()
            print(f"  Halide CCM execution time: {time.time() - start:.3f}s")
            self.img = ccm_out

        self.save()
        return self.img 