"""
File: color_correction_matrix_cuda_fallback.py
Description: CUDA-optimized Color Correction Matrix with fallback to Numba/NumPy
Code / Paper  Reference: https://www.imatest.com/docs/colormatrix/
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array

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


class ColorCorrectionMatrixCUDAFallback:
    """
    CUDA-optimized Color Correction Matrix with fallback to Numba/NumPy
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

    def _prepare_ccm_matrix(self):
        """Prepare CCM matrix from parameters"""
        r_1 = np.array(self.parm_ccm["corrected_red"])
        r_2 = np.array(self.parm_ccm["corrected_green"])
        r_3 = np.array(self.parm_ccm["corrected_blue"])
        return np.array([r_1, r_2, r_3], dtype=np.float32)

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
        """Execute CCM with CUDA, Numba, or NumPy fallback"""
        if not self.enable:
            return self.img

        # Prepare CCM matrix
        self.ccm_mat = self._prepare_ccm_matrix()

        # Try CUDA first
        if CUDA_AVAILABLE:
            try:
                print("CCM: Using CUDA acceleration")
                start = time.time()
                self.img = apply_ccm_cuda(self.img, self.ccm_mat, self.output_bit_depth)
                print(f"  CUDA CCM execution time: {time.time() - start:.3f}s")
                self.save()
                return self.img
            except Exception as e:
                print(f"CUDA CCM failed: {e}")
                print("Falling back to Numba/NumPy implementation...")
                # Fall through to next option

        # Try Numba next
        if NUMBA_AVAILABLE:
            try:
                print("CCM: Using Numba acceleration")
                start = time.time()
                self.img = apply_ccm_numba(self.img, self.ccm_mat, self.output_bit_depth)
                print(f"  Numba CCM execution time: {time.time() - start:.3f}s")
                self.save()
                return self.img
            except Exception as e:
                print(f"Numba CCM failed: {e}")
                print("Falling back to NumPy implementation...")
                # Fall through to original implementation

        # Use original NumPy implementation
        print("CCM: Using NumPy implementation")
        return self.fallback_ccm.execute() 