"""
File: malvar_he_cutler_cy.pyx
Description: Cython-optimized implementation of the Malvar-He-Cutler algorithm for CFA interpolation
Code / Paper  Reference: https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf
Author: 10xEngineers
------------------------------------------------------------
"""
import numpy as np
cimport numpy as np
import cv2
from libc.math cimport fabs
import time
import cython
import traceback

# Define numpy data types
np.import_array()
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void apply_kernel_5x5(DTYPE_t[:, :] image, DTYPE_t[:, :] output,
                                DTYPE_t[:, :] kernel, int height, int width) nogil noexcept:
    """
    Optimized 5x5 convolution using memory views
    """
    cdef:
        int i, j, ki, kj
        DTYPE_t sum_val
    
    for i in range(2, height-2):
        for j in range(2, width-2):
            sum_val = 0.0
            for ki in range(5):
                for kj in range(5):
                    sum_val += image[i-2+ki, j-2+kj] * kernel[ki, kj]
            output[i-2, j-2] = sum_val

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_filter_cy(np.ndarray[DTYPE_t, ndim=2] image, 
                   np.ndarray[DTYPE_t, ndim=2] kernel):
    """
    Cython-optimized 2D convolution
    """
    cdef:
        int height = image.shape[0]
        int width = image.shape[1]
        np.ndarray[DTYPE_t, ndim=2] output = np.zeros((height, width), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] padded
        DTYPE_t[:, :] padded_view
        DTYPE_t[:, :] output_view
        DTYPE_t[:, :] kernel_view = kernel
    
    try:
        # Handle borders with reflection
        padded = np.pad(image, 2, mode='reflect')
        padded_view = padded
        output_view = output
        
        # Apply kernel
        with nogil:
            apply_kernel_5x5(padded_view, output_view, kernel_view, height, width)
        
        return output
    except Exception as e:
        print(f"Error in apply_filter_cy: {str(e)}")
        print(traceback.format_exc())
        raise

@cython.boundscheck(False)
@cython.wraparound(False)
def update_channels_cy(DTYPE_t[:, :] r_channel,
                      DTYPE_t[:, :] b_channel,
                      DTYPE_t[:, :] rb_at_g_rbbr,
                      DTYPE_t[:, :] rb_at_g_brrb,
                      DTYPE_t[:, :] rb_at_gr_bbrr,
                      np.uint8_t[:, :] r_rows,
                      np.uint8_t[:, :] r_cols,
                      np.uint8_t[:, :] b_rows,
                      np.uint8_t[:, :] b_cols):
    """
    Cython-optimized channel updates using memory views
    """
    cdef:
        int height = r_channel.shape[0]
        int width = r_channel.shape[1]
        int i, j
        np.ndarray[DTYPE_t, ndim=2] r_out = np.empty((height, width), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] b_out = np.empty((height, width), dtype=DTYPE)
        DTYPE_t[:, :] r_out_view = r_out
        DTYPE_t[:, :] b_out_view = b_out
    
    try:
        # Copy initial values
        for i in range(height):
            for j in range(width):
                r_out_view[i,j] = r_channel[i,j]
                b_out_view[i,j] = b_channel[i,j]
        
        # Update channels using vectorized operations
        for i in range(height):
            for j in range(width):
                if r_rows[i,0] and b_cols[0,j]:
                    r_out_view[i,j] = rb_at_g_rbbr[i,j]
                elif b_rows[i,0] and r_cols[0,j]:
                    r_out_view[i,j] = rb_at_g_brrb[i,j]
                elif b_rows[i,0] and b_cols[0,j]:
                    r_out_view[i,j] = rb_at_gr_bbrr[i,j]
                
                if b_rows[i,0] and r_cols[0,j]:
                    b_out_view[i,j] = rb_at_g_rbbr[i,j]
                elif r_rows[i,0] and b_cols[0,j]:
                    b_out_view[i,j] = rb_at_g_brrb[i,j]
                elif r_rows[i,0] and r_cols[0,j]:
                    b_out_view[i,j] = rb_at_gr_bbrr[i,j]
        
        return r_out, b_out
    except Exception as e:
        print(f"Error in update_channels_cy: {str(e)}")
        print(traceback.format_exc())
        raise

@cython.boundscheck(False)
@cython.wraparound(False)
def apply_malvar_cy(np.ndarray[DTYPE_t, ndim=2] raw_in,
                   np.ndarray[DTYPE_t, ndim=2] mask_r,
                   np.ndarray[DTYPE_t, ndim=2] mask_g,
                   np.ndarray[DTYPE_t, ndim=2] mask_b):
    """
    Cython-optimized Malvar-He-Cutler demosaicing implementation
    """
    cdef:
        int height = raw_in.shape[0]
        int width = raw_in.shape[1]
        np.ndarray[DTYPE_t, ndim=3] demos_out = np.empty((height, width, 3), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] r_channel = np.empty((height, width), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] g_channel = np.empty((height, width), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] b_channel = np.empty((height, width), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] filtered_g
        np.ndarray[DTYPE_t, ndim=2] rb_at_g_rbbr, rb_at_g_brrb, rb_at_gr_bbrr
        np.ndarray[np.uint8_t, ndim=2] r_rows, r_cols, b_rows, b_cols
        int i, j
        DTYPE_t[:, :] raw_view = raw_in
        DTYPE_t[:, :] mask_r_view = mask_r
        DTYPE_t[:, :] mask_g_view = mask_g
        DTYPE_t[:, :] mask_b_view = mask_b
        DTYPE_t[:, :] r_channel_view = r_channel
        DTYPE_t[:, :] g_channel_view = g_channel
        DTYPE_t[:, :] b_channel_view = b_channel
        DTYPE_t[:, :] filtered_g_view
    
    try:
        # Define filter kernels
        g_at_r_and_b = np.float32([
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0]
        ]) * 0.125

        r_at_gr_and_b_at_gb = np.float32([
            [0, 0, 0.5, 0, 0],
            [0, -1, 0, -1, 0],
            [-1, 4, 5, 4, -1],
            [0, -1, 0, -1, 0],
            [0, 0, 0.5, 0, 0]
        ]) * 0.125

        r_at_gb_and_b_at_gr = np.transpose(r_at_gr_and_b_at_gb)

        r_at_b_and_b_at_r = np.float32([
            [0, 0, -1.5, 0, 0],
            [0, 2, 0, 2, 0],
            [-1.5, 0, 6, 0, -1.5],
            [0, 2, 0, 2, 0],
            [0, 0, -1.5, 0, 0]
        ]) * 0.125

        # Create initial channels using memory views
        for i in range(height):
            for j in range(width):
                r_channel_view[i,j] = raw_view[i,j] * mask_r_view[i,j]
                g_channel_view[i,j] = raw_view[i,j] * mask_g_view[i,j]
                b_channel_view[i,j] = raw_view[i,j] * mask_b_view[i,j]

        # Interpolate green channel
        filtered_g = apply_filter_cy(raw_in, g_at_r_and_b)
        filtered_g_view = filtered_g
        for i in range(height):
            for j in range(width):
                g_channel_view[i,j] += filtered_g_view[i,j] * (1 - mask_g_view[i,j])

        # Apply filters for red and blue channels
        rb_at_g_rbbr = apply_filter_cy(raw_in, r_at_gr_and_b_at_gb)
        rb_at_g_brrb = apply_filter_cy(raw_in, r_at_gb_and_b_at_gr)
        rb_at_gr_bbrr = apply_filter_cy(raw_in, r_at_b_and_b_at_r)

        # Create boolean masks for rows and columns
        r_rows = np.any(mask_r == 1, axis=1).astype(np.uint8)[:, np.newaxis]
        r_cols = np.any(mask_r == 1, axis=0).astype(np.uint8)[np.newaxis, :]
        b_rows = np.any(mask_b == 1, axis=1).astype(np.uint8)[:, np.newaxis]
        b_cols = np.any(mask_b == 1, axis=0).astype(np.uint8)[np.newaxis, :]

        # Update red and blue channels
        r_channel, b_channel = update_channels_cy(
            r_channel, b_channel,
            rb_at_g_rbbr, rb_at_g_brrb, rb_at_gr_bbrr,
            r_rows, r_cols, b_rows, b_cols
        )

        # Combine channels
        demos_out[:, :, 0] = r_channel
        demos_out[:, :, 1] = g_channel
        demos_out[:, :, 2] = b_channel

        return demos_out
    except Exception as e:
        print(f"Error in apply_malvar_cy: {str(e)}")
        print(traceback.format_exc())
        raise

class Malvar:
    """
    CFA interpolation or Demosaicing with Cython optimizations
    """

    def __init__(self, raw_in, platform, sensor_info, parm_dem):
        try:
            self.img = raw_in
            self.platform = platform
            self.sensor_info = sensor_info
            self.parm_dem = parm_dem
            self.masks = self._create_masks()
        except Exception as e:
            print(f"Error in Malvar.__init__: {str(e)}")
            print(traceback.format_exc())
            raise

    def _create_masks(self):
        """
        Create masks for R, G, and B channels based on Bayer pattern
        """
        try:
            height, width = self.img.shape
            bayer_pattern = self.sensor_info.get('bayer_pattern', 'RGGB').upper()
            
            # Create empty masks
            mask_r = np.zeros((height, width), dtype=np.float32)
            mask_g = np.zeros((height, width), dtype=np.float32)
            mask_b = np.zeros((height, width), dtype=np.float32)
            
            # Set mask values based on Bayer pattern
            if bayer_pattern == 'RGGB':
                mask_r[::2, ::2] = 1
                mask_g[::2, 1::2] = 1
                mask_g[1::2, ::2] = 1
                mask_b[1::2, 1::2] = 1
            elif bayer_pattern == 'GRBG':
                mask_g[::2, ::2] = 1
                mask_r[::2, 1::2] = 1
                mask_b[1::2, ::2] = 1
                mask_g[1::2, 1::2] = 1
            elif bayer_pattern == 'GBRG':
                mask_g[::2, ::2] = 1
                mask_b[::2, 1::2] = 1
                mask_r[1::2, ::2] = 1
                mask_g[1::2, 1::2] = 1
            elif bayer_pattern == 'BGGR':
                mask_b[::2, ::2] = 1
                mask_g[::2, 1::2] = 1
                mask_g[1::2, ::2] = 1
                mask_r[1::2, 1::2] = 1
            else:
                raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
            
            return mask_r, mask_g, mask_b
        except Exception as e:
            print(f"Error in _create_masks: {str(e)}")
            print(traceback.format_exc())
            raise

    def execute(self):
        """
        Execute the demosaicing process
        """
        try:
            result = self.apply_malvar()
            return result
        except Exception as e:
            print(f"Error in execute: {str(e)}")
            print(traceback.format_exc())
            raise

    def apply_malvar(self):
        """
        Demosaicing the given raw image using Malvar-He-Cutler
        """
        try:
            # Get masks and convert input to float32
            mask_r, mask_g, mask_b = self.masks
            raw_in = np.float32(self.img)
            
            # Apply the Cython-optimized demosaicing
            return apply_malvar_cy(raw_in, mask_r, mask_g, mask_b)
        except Exception as e:
            print(f"Error in apply_malvar: {str(e)}")
            print(traceback.format_exc())
            raise 