# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport round
from cython.parallel import prange

# Define the conversion matrices as constants
BT709_MATRIX = np.array([[47, 157, 16], [-26, -86, 112], [112, -102, -10]], dtype=np.float32)
BT601_MATRIX = np.array([[77, 150, 29], [131, -110, -21], [-44, -87, 138]], dtype=np.float32)

def rgb_to_yuv_8bit_cython(np.ndarray img, int conv_std, float saturation_gain, int bit_depth):
    """
    Optimized RGB-to-YUV Colorspace conversion using Cython
    """
    cdef:
        int height, width, channels
        float[:, :] rgb2yuv_mat
        float bit_scale, max_val, y_offset, uv_offset
        unsigned char[:, :, :] yuv_img
        int i, j
        float y, u, v
        float r, g, b
        np.uint32_t[:, :, :] img_view = img
    
    # Get dimensions from input array
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    
    rgb2yuv_mat = BT709_MATRIX if conv_std == 1 else BT601_MATRIX
    bit_scale = 2 ** (bit_depth - 8)
    max_val = (2 ** bit_depth) - 1
    y_offset = 2 ** (bit_depth / 2)
    uv_offset = 2 ** (bit_depth - 1)
    
    yuv_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Parallel processing of pixels
    for i in prange(height, nogil=True):
        for j in range(width):
            # Get RGB values
            r = <float>img_view[i, j, 0]
            g = <float>img_view[i, j, 1]
            b = <float>img_view[i, j, 2]
            
            # Direct matrix multiplication
            y = (rgb2yuv_mat[0, 0] * r + 
                 rgb2yuv_mat[0, 1] * g + 
                 rgb2yuv_mat[0, 2] * b) / 256.0
            u = (rgb2yuv_mat[1, 0] * r + 
                 rgb2yuv_mat[1, 1] * g + 
                 rgb2yuv_mat[1, 2] * b) / 256.0
            v = (rgb2yuv_mat[2, 0] * r + 
                 rgb2yuv_mat[2, 1] * g + 
                 rgb2yuv_mat[2, 2] * b) / 256.0
            
            # Apply saturation gain
            u *= saturation_gain
            v *= saturation_gain
            
            # Apply offsets and clipping in one step
            y = round(y + y_offset)
            u = round(u + uv_offset)
            v = round(v + uv_offset)
            
            # Clipping and scaling
            y = max(0, min(max_val, y)) / bit_scale
            u = max(0, min(max_val, u)) / bit_scale
            v = max(0, min(max_val, v)) / bit_scale
            
            # Final clipping to 8-bit range
            yuv_img[i, j, 0] = <unsigned char>max(0, min(255, y))
            yuv_img[i, j, 1] = <unsigned char>max(0, min(255, u))
            yuv_img[i, j, 2] = <unsigned char>max(0, min(255, v))
    
    return np.asarray(yuv_img) 