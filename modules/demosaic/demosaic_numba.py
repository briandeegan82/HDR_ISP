"""
File: demosaic_numba.py
Description: Numba-optimized CFA interpolation algorithms
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
from numba import jit, prange
from util.utils import save_output_array


@jit(nopython=True, cache=True)
def create_masks_numba(height, width, bayer_pattern):
    """
    Create Bayer pattern masks using Numba optimization
    """
    mask_r = np.zeros((height, width), dtype=np.float32)
    mask_g = np.zeros((height, width), dtype=np.float32)
    mask_b = np.zeros((height, width), dtype=np.float32)
    
    if bayer_pattern == 0:  # RGGB
        mask_r[::2, ::2] = 1
        mask_g[::2, 1::2] = 1
        mask_g[1::2, ::2] = 1
        mask_b[1::2, 1::2] = 1
    elif bayer_pattern == 1:  # GRBG
        mask_g[::2, ::2] = 1
        mask_r[::2, 1::2] = 1
        mask_b[1::2, ::2] = 1
        mask_g[1::2, 1::2] = 1
    elif bayer_pattern == 2:  # GBRG
        mask_g[::2, ::2] = 1
        mask_b[::2, 1::2] = 1
        mask_r[1::2, ::2] = 1
        mask_g[1::2, 1::2] = 1
    elif bayer_pattern == 3:  # BGGR
        mask_b[::2, ::2] = 1
        mask_g[::2, 1::2] = 1
        mask_g[1::2, ::2] = 1
        mask_r[1::2, 1::2] = 1
    
    return mask_r, mask_g, mask_b


@jit(nopython=True, cache=True)
def apply_convolution_5x5_numba(img, kernel):
    """
    Apply 5x5 convolution using Numba optimization
    """
    height, width = img.shape
    output = np.zeros((height, width), dtype=np.float32)
    
    # Pad the image with reflection
    padded = np.zeros((height + 4, width + 4), dtype=np.float32)
    padded[2:-2, 2:-2] = img
    
    # Reflect borders
    padded[0:2, 2:-2] = img[1::-1, :]  # Top
    padded[-2:, 2:-2] = img[-1:-3:-1, :]  # Bottom
    padded[2:-2, 0:2] = img[:, 1::-1]  # Left
    padded[2:-2, -2:] = img[:, -1:-3:-1]  # Right
    
    # Apply convolution
    for i in range(height):
        for j in range(width):
            sum_val = 0.0
            for ki in range(5):
                for kj in range(5):
                    sum_val += padded[i + ki, j + kj] * kernel[ki, kj]
            output[i, j] = sum_val
    
    return output


@jit(nopython=True, parallel=True, cache=True)
def update_channels_numba(r_channel, b_channel, rb_at_g_rbbr, rb_at_g_brrb, rb_at_gr_bbrr,
                         r_rows, r_cols, b_rows, b_cols):
    """
    Update red and blue channels using Numba optimization
    """
    height, width = r_channel.shape
    r_out = np.empty_like(r_channel)
    b_out = np.empty_like(b_channel)
    
    for i in prange(height):
        for j in range(width):
            # Copy initial values
            r_out[i, j] = r_channel[i, j]
            b_out[i, j] = b_channel[i, j]
            
            # Update red channel
            if r_rows[i] and b_cols[j]:
                r_out[i, j] = rb_at_g_rbbr[i, j]
            elif b_rows[i] and r_cols[j]:
                r_out[i, j] = rb_at_g_brrb[i, j]
            elif b_rows[i] and b_cols[j]:
                r_out[i, j] = rb_at_gr_bbrr[i, j]
            
            # Update blue channel
            if b_rows[i] and r_cols[j]:
                b_out[i, j] = rb_at_g_rbbr[i, j]
            elif r_rows[i] and b_cols[j]:
                b_out[i, j] = rb_at_g_brrb[i, j]
            elif r_rows[i] and r_cols[j]:
                b_out[i, j] = rb_at_gr_bbrr[i, j]
    
    return r_out, b_out


@jit(nopython=True, cache=True)
def apply_malvar_numba(raw_in, mask_r, mask_g, mask_b):
    """
    Numba-optimized Malvar-He-Cutler demosaicing implementation
    """
    height, width = raw_in.shape
    
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

    r_at_gb_and_b_at_gr = np.float32([
        [0, 0, 0.5, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 0.5, 0, 0]
    ]).T * 0.125

    r_at_b_and_b_at_r = np.float32([
        [0, 0, -1.5, 0, 0],
        [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5],
        [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0]
    ]) * 0.125

    # Create initial channels
    r_channel = raw_in * mask_r
    g_channel = raw_in * mask_g
    b_channel = raw_in * mask_b

    # Interpolate green channel
    filtered_g = apply_convolution_5x5_numba(raw_in, g_at_r_and_b)
    g_channel = g_channel + filtered_g * (1 - mask_g)

    # Apply filters for red and blue channels
    rb_at_g_rbbr = apply_convolution_5x5_numba(raw_in, r_at_gr_and_b_at_gb)
    rb_at_g_brrb = apply_convolution_5x5_numba(raw_in, r_at_gb_and_b_at_gr)
    rb_at_gr_bbrr = apply_convolution_5x5_numba(raw_in, r_at_b_and_b_at_r)

    # Create boolean masks for rows and columns
    r_rows = np.zeros(height, dtype=np.uint8)
    r_cols = np.zeros(width, dtype=np.uint8)
    b_rows = np.zeros(height, dtype=np.uint8)
    b_cols = np.zeros(width, dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            if mask_r[i, j] > 0:
                r_rows[i] = 1
                r_cols[j] = 1
            if mask_b[i, j] > 0:
                b_rows[i] = 1
                b_cols[j] = 1

    # Update red and blue channels
    r_channel, b_channel = update_channels_numba(
        r_channel, b_channel,
        rb_at_g_rbbr, rb_at_g_brrb, rb_at_gr_bbrr,
        r_rows, r_cols, b_rows, b_cols
    )

    # Combine channels
    demos_out = np.empty((height, width, 3), dtype=np.float32)
    demos_out[:, :, 0] = r_channel
    demos_out[:, :, 1] = g_channel
    demos_out[:, :, 2] = b_channel

    return demos_out


class DemosaicNumba:
    """
    Numba-optimized CFA interpolation
    """

    def __init__(self, img, platform, sensor_info, parm_dga):
        self.img = img
        self.bayer = sensor_info["bayer_pattern"]
        self.bit_depth = sensor_info["output_bit_depth"]
        self.is_save = parm_dga["is_save"]
        self.sensor_info = sensor_info
        self.platform = platform
        self.is_debug = parm_dga.get("is_debug", False)
        
        # Convert bayer pattern to integer for Numba
        bayer_patterns = {"RGGB": 0, "GRBG": 1, "GBRG": 2, "BGGR": 3}
        self.bayer_int = bayer_patterns.get(self.bayer.upper(), 0)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_demosaic_numba_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Applying Numba-optimized demosaicing to bayer image
        """
        start = time.time()
        
        # Convert input to float32
        raw_in = np.float32(self.img)
        
        # Create masks using Numba
        mask_r, mask_g, mask_b = create_masks_numba(
            raw_in.shape[0], raw_in.shape[1], self.bayer_int
        )
        
        # Apply Numba-optimized demosaicing
        demos_out = apply_malvar_numba(raw_in, mask_r, mask_g, mask_b)
        
        # Clip and convert to uint16
        demos_out = np.clip(demos_out, 0, 2**self.bit_depth - 1)
        demos_out = np.uint16(demos_out)
        
        if self.is_debug:
            print(f"  Numba Demosaic execution time: {time.time() - start:.3f}s")
        
        self.img = demos_out
        self.save()
        return self.img 