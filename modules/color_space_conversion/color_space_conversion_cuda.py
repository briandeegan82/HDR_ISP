"""
File: color_space_conversion_cuda.py
Description: CUDA-optimized color space conversion
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
import cupy as cp
from util.utils import save_output_array_yuv


def rgb_to_yuv_cuda(img, rgb2yuv_mat, saturation_gain, bit_depth):
    """
    CUDA-optimized RGB-to-YUV conversion
    """
    # Transfer data to GPU
    img_gpu = cp.asarray(img, dtype=cp.float32)
    rgb2yuv_mat_gpu = cp.asarray(rgb2yuv_mat, dtype=cp.float32)
    
    height, width, channels = img_gpu.shape
    
    # Pre-compute constants
    bit_scale = 2 ** 8
    y_offset = 2 ** (bit_depth / 2)
    uv_offset = 2 ** (bit_depth - 1)
    max_val = (2 ** bit_depth) - 1
    final_scale = 2 ** (bit_depth - 8)
    
    # Reshape image for matrix multiplication
    img_reshaped = img_gpu.reshape(-1, 3).T
    
    # Matrix multiplication
    yuv_2d = cp.matmul(rgb2yuv_mat_gpu, img_reshaped)
    yuv_2d = cp.asarray(yuv_2d, dtype=cp.float64) / bit_scale
    
    # Apply saturation gain to U and V channels
    yuv_2d[1, :] *= saturation_gain
    yuv_2d[2, :] *= saturation_gain
    
    # Apply offsets
    yuv_2d[0, :] += y_offset
    yuv_2d[1, :] += uv_offset
    yuv_2d[2, :] += uv_offset
    
    # Clip to valid range
    yuv_2d = cp.clip(yuv_2d, 0, max_val)
    
    # Transpose back and reshape
    yuv2d_t = yuv_2d.T
    yuv2d_t = cp.round(yuv2d_t / final_scale)
    yuv2d_t = cp.clip(yuv2d_t, 0, 255)
    
    # Transfer back to CPU and reshape
    result = cp.asnumpy(yuv2d_t).reshape(height, width, channels).astype(np.uint8)
    
    return result


class ColorSpaceConversionCUDA:
    """
    CUDA-optimized Color Space Conversion
    """

    def __init__(self, img, platform, sensor_info, parm_csc, parm_cse):
        self.img = img.copy()
        self.is_save = parm_csc["is_save"]
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_csc = parm_csc
        self.bit_depth = sensor_info["output_bit_depth"]
        self.conv_std = self.parm_csc["conv_standard"]
        self.rgb2yuv_mat = None
        self.yuv_img = None
        self.parm_cse = parm_cse

    def rgb_to_yuv_8bit(self):
        """
        CUDA-optimized RGB-to-YUV Colorspace conversion 8bit
        """
        total_start = time.time()

        if self.conv_std == 1:
            # for BT. 709
            self.rgb2yuv_mat = np.array(
                [[47, 157, 16], [-26, -86, 112], [112, -102, -10]], dtype=np.float32
            )
        else:
            # for BT.601/407
            # conversion metrix with 8bit integer co-efficients - m=8
            self.rgb2yuv_mat = np.array(
                [[77, 150, 29], [131, -110, -21], [-44, -87, 138]], dtype=np.float32
            )

        start = time.time()
        saturation_gain = self.parm_cse['saturation_gain'] if self.parm_cse['is_enable'] else 1.0
        
        # Use CUDA-optimized conversion
        self.img = rgb_to_yuv_cuda(self.img, self.rgb2yuv_mat, saturation_gain, self.bit_depth)
        
        #print(f"  CUDA RGB to YUV conversion time: {time.time() - start:.3f}s")
        #print(f"  Total RGB to YUV conversion time: {time.time() - total_start:.3f}s")
        return self.img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_color_space_conversion_cuda_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Execute CUDA-optimized Color Space Conversion
        """
        #print("Color Space Conversion (CUDA) = True")

        start = time.time()
        csc_out = self.rgb_to_yuv_8bit()
        #print(f"  Total execution time: {time.time() - start:.3f}s")
        self.img = csc_out
        self.save()
        return self.img 