"""
File: gamma_correction_cuda.py
Description: CUDA-optimized Gamma Correction implementation
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
import cupy as cp

from util.utils import save_output_array


def generate_gamma_lut_cuda(bit_depth):
    """
    CUDA-optimized Gamma LUT generation
    """
    max_val = 2**bit_depth - 1
    lut = cp.arange(0, max_val + 1, dtype=cp.uint32)
    lut = (cp.round(max_val * ((lut / max_val) ** (1 / 2.2)))).astype(cp.uint32)
    return lut.get()  # Convert to NumPy array


def apply_gamma_cuda(img, lut):
    """
    CUDA-optimized Gamma LUT application
    """
    # Upload to GPU
    gpu_img = cp.asarray(img)
    gpu_lut = cp.asarray(lut)
    
    # Apply LUT using advanced indexing
    result = gpu_lut[gpu_img]
    
    # Download result
    return cp.asnumpy(result)


class GammaCorrectionCUDA:
    """
    CUDA-optimized Gamma Correction
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
        Generates Gamma LUT with CUDA optimization
        """
        return generate_gamma_lut_cuda(bit_depth)

    def apply_gamma(self):
        """
        Apply Gamma LUT with CUDA optimization
        """
        # generate LUT
        lut = self.generate_gamma_lut(self.output_bit_depth)

        # apply LUT with CUDA optimization
        return apply_gamma_cuda(self.img, lut)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_gamma_correction_cuda_",
                self.platform,
                self.sensor_info["output_bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute Gamma Correction with CUDA optimization
        """
        if self.enable is True:
            start = time.time()
            gc_out = self.apply_gamma()
            print(f"  CUDA Gamma execution time: {time.time() - start:.3f}s")
            self.img = gc_out

        self.save()
        return self.img 