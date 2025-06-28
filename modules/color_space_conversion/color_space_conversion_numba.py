"""
File: color_space_conversion_numba.py
Description: Numba-optimized color space conversion
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
from numba import jit, prange
from util.utils import save_output_array_yuv


@jit(nopython=True, parallel=True)
def rgb_to_yuv_numba(img, rgb2yuv_mat, saturation_gain, bit_depth):
    """
    Numba-optimized RGB-to-YUV conversion
    """
    height, width, channels = img.shape
    yuv_img = np.empty((height, width, channels), dtype=np.uint8)
    
    # Pre-compute constants (using integer arithmetic for better performance)
    bit_scale = 256  # 2^8
    y_offset = int(2 ** (bit_depth / 2))
    uv_offset = int(2 ** (bit_depth - 1))
    max_val = int((2 ** bit_depth) - 1)
    final_scale = int(2 ** (bit_depth - 8))
    
    # Pre-compute matrix coefficients for better cache locality
    m00, m01, m02 = rgb2yuv_mat[0, 0], rgb2yuv_mat[0, 1], rgb2yuv_mat[0, 2]
    m10, m11, m12 = rgb2yuv_mat[1, 0], rgb2yuv_mat[1, 1], rgb2yuv_mat[1, 2]
    m20, m21, m22 = rgb2yuv_mat[2, 0], rgb2yuv_mat[2, 1], rgb2yuv_mat[2, 2]
    
    for i in prange(height):
        for j in range(width):
            r, g, b = img[i, j, 0], img[i, j, 1], img[i, j, 2]
            
            # Matrix multiplication (using pre-computed coefficients)
            y = (m00 * r + m01 * g + m02 * b) // bit_scale
            u = (m10 * r + m11 * g + m12 * b) // bit_scale
            v = (m20 * r + m21 * g + m22 * b) // bit_scale
            
            # Apply saturation gain to U and V
            u = int(u * saturation_gain)
            v = int(v * saturation_gain)
            
            # Apply offsets
            y += y_offset
            u += uv_offset
            v += uv_offset
            
            # Clip to valid range
            y = max(0, min(y, max_val))
            u = max(0, min(u, max_val))
            v = max(0, min(v, max_val))
            
            # Final scaling and clipping
            y = y // final_scale
            u = u // final_scale
            v = v // final_scale
            
            y = max(0, min(y, 255))
            u = max(0, min(u, 255))
            v = max(0, min(v, 255))
            
            yuv_img[i, j, 0] = y
            yuv_img[i, j, 1] = u
            yuv_img[i, j, 2] = v
    
    return yuv_img


@jit(nopython=True, parallel=True)
def rgb_to_yuv_numba_ultra(img, rgb2yuv_mat, saturation_gain, bit_depth):
    """
    Ultra-optimized Numba RGB-to-YUV conversion with better memory layout
    """
    height, width, channels = img.shape
    yuv_img = np.empty((height, width, channels), dtype=np.uint8)
    
    # Pre-compute all constants
    bit_scale = 256
    y_offset = int(2 ** (bit_depth / 2))
    uv_offset = int(2 ** (bit_depth - 1))
    max_val = int((2 ** bit_depth) - 1)
    final_scale = int(2 ** (bit_depth - 8))
    
    # Pre-compute matrix coefficients
    m00, m01, m02 = rgb2yuv_mat[0, 0], rgb2yuv_mat[0, 1], rgb2yuv_mat[0, 2]
    m10, m11, m12 = rgb2yuv_mat[1, 0], rgb2yuv_mat[1, 1], rgb2yuv_mat[1, 2]
    m20, m21, m22 = rgb2yuv_mat[2, 0], rgb2yuv_mat[2, 1], rgb2yuv_mat[2, 2]
    
    # Pre-compute saturation gain as integer for better performance
    sat_gain_int = int(saturation_gain * 256)  # Scale by 256 for integer arithmetic
    
    for i in prange(height):
        for j in range(width):
            r, g, b = img[i, j, 0], img[i, j, 1], img[i, j, 2]
            
            # Matrix multiplication with integer arithmetic
            y = (m00 * r + m01 * g + m02 * b) // bit_scale
            u = (m10 * r + m11 * g + m12 * b) // bit_scale
            v = (m20 * r + m21 * g + m22 * b) // bit_scale
            
            # Apply saturation gain using integer arithmetic
            u = (u * sat_gain_int) // 256
            v = (v * sat_gain_int) // 256
            
            # Apply offsets
            y += y_offset
            u += uv_offset
            v += uv_offset
            
            # Clip and scale in one operation
            y = max(0, min(y, max_val)) // final_scale
            u = max(0, min(u, max_val)) // final_scale
            v = max(0, min(v, max_val)) // final_scale
            
            # Final clipping
            yuv_img[i, j, 0] = max(0, min(y, 255))
            yuv_img[i, j, 1] = max(0, min(u, 255))
            yuv_img[i, j, 2] = max(0, min(v, 255))
    
    return yuv_img


@jit(nopython=True, parallel=True)
def rgb_to_yuv_numba_uint16(img, rgb2yuv_mat, saturation_gain, bit_depth):
    """
    Ultra-optimized Numba RGB-to-YUV conversion for uint16 input
    """
    height, width, channels = img.shape
    yuv_img = np.empty((height, width, channels), dtype=np.uint8)
    
    # Pre-compute all constants for uint16 processing
    bit_scale = 256
    y_offset = int(2 ** (bit_depth / 2))
    uv_offset = int(2 ** (bit_depth - 1))
    max_val = int((2 ** bit_depth) - 1)
    final_scale = int(2 ** (bit_depth - 8))
    
    # Pre-compute matrix coefficients
    m00, m01, m02 = rgb2yuv_mat[0, 0], rgb2yuv_mat[0, 1], rgb2yuv_mat[0, 2]
    m10, m11, m12 = rgb2yuv_mat[1, 0], rgb2yuv_mat[1, 1], rgb2yuv_mat[1, 2]
    m20, m21, m22 = rgb2yuv_mat[2, 0], rgb2yuv_mat[2, 1], rgb2yuv_mat[2, 2]
    
    # Pre-compute saturation gain as integer for better performance
    sat_gain_int = int(saturation_gain * 256)  # Scale by 256 for integer arithmetic
    
    # For uint16 input, we need to scale down to 8-bit range first
    input_scale = 256  # Scale uint16 (0-65535) to uint8 range (0-255)
    
    for i in prange(height):
        for j in range(width):
            # Scale uint16 input to uint8 range for processing
            r = img[i, j, 0] // input_scale
            g = img[i, j, 1] // input_scale
            b = img[i, j, 2] // input_scale
            
            # Matrix multiplication with integer arithmetic
            y = (m00 * r + m01 * g + m02 * b) // bit_scale
            u = (m10 * r + m11 * g + m12 * b) // bit_scale
            v = (m20 * r + m21 * g + m22 * b) // bit_scale
            
            # Apply saturation gain using integer arithmetic
            u = (u * sat_gain_int) // 256
            v = (v * sat_gain_int) // 256
            
            # Apply offsets
            y += y_offset
            u += uv_offset
            v += uv_offset
            
            # Clip and scale in one operation
            y = max(0, min(y, max_val)) // final_scale
            u = max(0, min(u, max_val)) // final_scale
            v = max(0, min(v, max_val)) // final_scale
            
            # Final clipping
            yuv_img[i, j, 0] = max(0, min(y, 255))
            yuv_img[i, j, 1] = max(0, min(u, 255))
            yuv_img[i, j, 2] = max(0, min(v, 255))
    
    return yuv_img


@jit(nopython=True, parallel=True)
def rgb_to_yuv_numba_uint16_fast(img, rgb2yuv_mat, saturation_gain, bit_depth):
    """
    Ultra-fast Numba RGB-to-YUV conversion for uint16 input with optimized memory access
    """
    height, width, channels = img.shape
    yuv_img = np.empty((height, width, channels), dtype=np.uint8)
    
    # Pre-compute all constants
    y_offset = int(2 ** (bit_depth / 2))
    uv_offset = int(2 ** (bit_depth - 1))
    max_val = int((2 ** bit_depth) - 1)
    final_scale = int(2 ** (bit_depth - 8))
    
    # Pre-compute matrix coefficients
    m00, m01, m02 = rgb2yuv_mat[0, 0], rgb2yuv_mat[0, 1], rgb2yuv_mat[0, 2]
    m10, m11, m12 = rgb2yuv_mat[1, 0], rgb2yuv_mat[1, 1], rgb2yuv_mat[1, 2]
    m20, m21, m22 = rgb2yuv_mat[2, 0], rgb2yuv_mat[2, 1], rgb2yuv_mat[2, 2]
    
    # Pre-compute saturation gain as integer
    sat_gain_int = int(saturation_gain * 256)
    
    # Pre-compute scaling factors for uint16 to uint8 conversion
    # Use bit shifting for faster division: 256 = 2^8
    input_scale = 8  # log2(256)
    
    for i in prange(height):
        for j in range(width):
            # Fast uint16 to uint8 conversion using bit shifting
            r = img[i, j, 0] >> input_scale
            g = img[i, j, 1] >> input_scale
            b = img[i, j, 2] >> input_scale
            
            # Matrix multiplication (optimized for integer arithmetic)
            y = (m00 * r + m01 * g + m02 * b) >> 8  # Divide by 256 using bit shift
            u = (m10 * r + m11 * g + m12 * b) >> 8
            v = (m20 * r + m21 * g + m22 * b) >> 8
            
            # Apply saturation gain using bit shifting
            u = (u * sat_gain_int) >> 8
            v = (v * sat_gain_int) >> 8
            
            # Apply offsets and clipping in one step
            y = y + y_offset
            u = u + uv_offset
            v = v + uv_offset
            
            # Clip to valid range
            if y < 0: y = 0
            elif y > max_val: y = max_val
            if u < 0: u = 0
            elif u > max_val: u = max_val
            if v < 0: v = 0
            elif v > max_val: v = max_val
            
            # Final scaling and clipping
            y = y >> (bit_depth - 8)  # Divide by final_scale using bit shift
            u = u >> (bit_depth - 8)
            v = v >> (bit_depth - 8)
            
            # Final 8-bit clipping
            if y > 255: y = 255
            if u > 255: u = 255
            if v > 255: v = 255
            
            yuv_img[i, j, 0] = y
            yuv_img[i, j, 1] = u
            yuv_img[i, j, 2] = v
    
    return yuv_img


class ColorSpaceConversionNumba:
    """
    Numba-optimized Color Space Conversion
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
        Numba-optimized RGB-to-YUV Colorspace conversion 8bit
        """
        total_start = time.time()

        if self.conv_std == 1:
            # for BT. 709
            self.rgb2yuv_mat = np.array(
                [[47, 157, 16], [-26, -86, 112], [112, -102, -10]], dtype=np.int32
            )
        else:
            # for BT.601/407
            # conversion metrix with 8bit integer co-efficients - m=8
            self.rgb2yuv_mat = np.array(
                [[77, 150, 29], [131, -110, -21], [-44, -87, 138]], dtype=np.int32
            )

        start = time.time()
        saturation_gain = self.parm_cse['saturation_gain'] if self.parm_cse['is_enable'] else 1.0
        
        # Select appropriate kernel based on input dtype
        if self.img.dtype == np.uint16:
            self.img = rgb_to_yuv_numba_uint16_fast(self.img, self.rgb2yuv_mat, saturation_gain, self.bit_depth)
        else:
            self.img = rgb_to_yuv_numba_ultra(self.img, self.rgb2yuv_mat, saturation_gain, self.bit_depth)
        
        return self.img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_color_space_conversion_numba_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Execute Numba-optimized Color Space Conversion
        """
        #print("Color Space Conversion (Numba) = True")

        start = time.time()
        csc_out = self.rgb_to_yuv_8bit()
        #print(f"  Total execution time: {time.time() - start:.3f}s")
        self.img = csc_out
        self.save()
        return self.img 