"""
File: color_space_conversion_gpu.py
Description: GPU-accelerated color space conversion
Author: 10xEngineers
------------------------------------------------------------
"""
import numpy as np
import time
from util.gpu_utils import gpu_accelerator
from util.utils import save_output_array_yuv

# Define the conversion matrices as constants
BT709_MATRIX = np.array([[47, 157, 16], [-26, -86, 112], [112, -102, -10]], dtype=np.float32)
BT601_MATRIX = np.array([[77, 150, 29], [131, -110, -21], [-44, -87, 138]], dtype=np.float32)

class ColorSpaceConversionGPU:
    """
    GPU-accelerated Color Space Conversion
    """

    def __init__(self, img, platform, sensor_info, parm_csc, parm_cse):
        self.img = img
        self.enable = parm_csc["is_enable"]
        self.sensor_info = sensor_info
        self.parm_csc = parm_csc
        self.parm_cse = parm_cse
        self.is_save = parm_csc["is_save"]
        self.platform = platform

    def rgb_to_yuv_gpu(self, img, conv_std, saturation_gain, bit_depth):
        """
        GPU-accelerated RGB-to-YUV Colorspace conversion
        """
        height, width, channels = img.shape
        
        # Select conversion matrix based on standard
        rgb2yuv_mat = BT709_MATRIX if conv_std == 1 else BT601_MATRIX
        
        # Apply saturation gain to U and V components
        rgb2yuv_mat[1, :] *= saturation_gain
        rgb2yuv_mat[2, :] *= saturation_gain
        
        # Scale matrix for bit depth
        bit_scale = 2 ** (bit_depth - 8)
        rgb2yuv_mat = rgb2yuv_mat / bit_scale
        
        # Use GPU-accelerated matrix multiplication
        yuv_img = gpu_accelerator.matrix_multiply_gpu(img.astype(np.float32), rgb2yuv_mat)
        
        # Apply offsets
        y_offset = 2 ** (bit_depth / 2)
        uv_offset = 2 ** (bit_depth - 1)
        
        yuv_img[:, :, 0] += y_offset
        yuv_img[:, :, 1] += uv_offset
        yuv_img[:, :, 2] += uv_offset
        
        # Clip to valid range
        max_val = (2 ** bit_depth) - 1
        yuv_img = np.clip(yuv_img, 0, max_val)
        
        return yuv_img.astype(np.uint8)

    def apply_csc(self):
        """
        Apply GPU-accelerated color space conversion
        """
        if self.parm_csc["conv_standard"] == 1:
            conv_std = 1  # BT709
        else:
            conv_std = 0  # BT601
            
        saturation_gain = self.parm_cse["saturation_gain"]
        bit_depth = self.sensor_info["output_bit_depth"]
        
        # Apply GPU-accelerated color space conversion
        yuv_img = self.rgb_to_yuv_gpu(self.img, conv_std, saturation_gain, bit_depth)
        
        return yuv_img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_color_space_conversion_gpu_",
                self.platform,
                self.parm_csc["conv_standard"],
            )

    def execute(self):
        """
        Applying GPU-accelerated color space conversion to input image
        """
        if self.enable is True:
            start = time.time()
            csc_out = self.apply_csc()
            print(f"GPU Color Space Conversion execution time: {time.time() - start:.3f}s")
            self.img = csc_out

        self.save()
        return self.img 