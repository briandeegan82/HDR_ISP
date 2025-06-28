"""
File: color_correction_matrix_cuda.py
Description: CUDA-optimized Color Correction Matrix implementation
Code / Paper  Reference: https://www.imatest.com/docs/colormatrix/
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
import cupy as cp

from util.utils import save_output_array


def apply_ccm_cuda(img, ccm_mat, output_bit_depth):
    """
    CUDA-optimized Color Correction Matrix application
    """
    max_value = (2**output_bit_depth - 1)
    
    # Upload to GPU
    gpu_img = cp.asarray(img)
    gpu_ccm = cp.asarray(ccm_mat)
    
    # Normalize to 0-1
    gpu_img_norm = gpu_img.astype(cp.float32) / max_value
    
    # Reshape to (N, 3) for matrix multiplication
    height, width, channels = gpu_img_norm.shape
    img_reshaped = gpu_img_norm.reshape(-1, 3)
    
    # Apply CCM: img_reshaped * ccm_mat.T
    result = cp.matmul(img_reshaped, gpu_ccm.T)
    
    # Clip and reshape back
    result = cp.clip(result, 0, 1)
    result = result.reshape(height, width, 3)
    
    # Convert back to original range
    result = (result * max_value).astype(cp.uint16)
    
    # Download result
    return cp.asnumpy(result)


class ColorCorrectionMatrixCUDA:
    "CUDA-optimized Color Correction Matrix"

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
        Apply CCM with CUDA optimization
        """
        r_1 = np.array(self.parm_ccm["corrected_red"])
        r_2 = np.array(self.parm_ccm["corrected_green"])
        r_3 = np.array(self.parm_ccm["corrected_blue"])

        self.ccm_mat = np.array([r_1, r_2, r_3], dtype=np.float32)

        # Apply CUDA-optimized CCM
        return apply_ccm_cuda(self.img, self.ccm_mat, self.output_bit_depth)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_color_correction_matrix_cuda_",
                self.platform,
                self.sensor_info["output_bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute CCM with CUDA optimization if enabled."""
        if self.enable:
            start = time.time()
            ccm_out = self.apply_ccm()
            print(f"  CUDA CCM execution time: {time.time() - start:.3f}s")
            self.img = ccm_out

        self.save()
        return self.img 