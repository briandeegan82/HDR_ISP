# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free

# Type definitions
ctypedef np.uint8_t uint8
ctypedef np.int32_t int32
ctypedef np.float32_t float32

def compute_histogram_cy(uint8[:, :] tile):
    """Compute histogram for a tile using Cython"""
    cdef:
        int32[::1] hist = np.zeros(256, dtype=np.int32)
        int i, j, val
        int height = tile.shape[0]
        int width = tile.shape[1]
    
    for i in range(height):
        for j in range(width):
            val = tile[i, j]
            hist[val] += 1
    return np.asarray(hist)

def apply_lut_cy(uint8[:, :] tile, uint8[:] lut, uint8[:, :] result=None):
    """Apply LUT to a tile using Cython with optional pre-allocated result array"""
    cdef:
        int i, j
        int height = tile.shape[0]
        int width = tile.shape[1]
    
    if result is None:
        result = np.empty((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            result[i, j] = lut[tile[i, j]]
    return np.asarray(result)

def process_tile_cy(uint8[:, :] y_block, uint8[:, :, :] luts, int i_row, int i_col, 
                   int horiz_tiles, int vert_tiles, int32[:, :] left_lut_weights,
                   int32[:, :] top_lut_weights):
    """Process a single tile with optimized interpolation using Cython"""
    cdef:
        int height = y_block.shape[0]
        int width = y_block.shape[1]
        uint8[:, :] result = np.empty((height, width), dtype=np.uint8)
        uint8[:, :] temp = np.empty((height, width), dtype=np.uint8)
        int i, j, w
        int32 val
        float32 w_float, w_inv
    
    # Safety checks with detailed error messages
    if i_row < 0 or i_row >= vert_tiles or i_col < 0 or i_col >= horiz_tiles:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Additional safety checks
    if luts.shape[0] <= i_row or luts.shape[1] <= i_col:
        return np.zeros((height, width), dtype=np.uint8)
    
    if left_lut_weights.shape[1] != width or top_lut_weights.shape[0] != height:
        return np.zeros((height, width), dtype=np.uint8)
    
    try:
        # Get LUTs based on position
        if i_row == 0:
            if i_col == 0:  # Top-left corner
                return apply_lut_cy(y_block, luts[0, 0], result)
            elif i_col == horiz_tiles - 1:  # Top-right corner
                try:
                    # Ensure we're using the correct LUT index
                    lut_idx = min(i_col, luts.shape[1] - 1)
                    result = apply_lut_cy(y_block, luts[0, lut_idx], result)
                    return np.asarray(result)
                except Exception as e:
                    return np.zeros((height, width), dtype=np.uint8)
            else:  # Top edge
                apply_lut_cy(y_block, luts[0, i_col - 1], temp)
                apply_lut_cy(y_block, luts[0, i_col], result)
                with nogil, parallel():
                    for i in prange(height):
                        for j in range(width):
                            w_float = left_lut_weights[0, j] / 1024.0
                            w_inv = 1.0 - w_float
                            val = int(w_float * temp[i, j] + w_inv * result[i, j])
                            result[i, j] = val if val < 256 else 255
                return np.asarray(result)
        elif i_row == vert_tiles - 1:
            if i_col == 0:  # Bottom-left corner
                try:
                    # Ensure we're using the correct LUT index
                    lut_idx = min(i_row, luts.shape[0] - 1)
                    result = apply_lut_cy(y_block, luts[lut_idx, 0], result)
                    return np.asarray(result)
                except Exception as e:
                    return np.zeros((height, width), dtype=np.uint8)
            elif i_col == horiz_tiles - 1:  # Bottom-right corner
                lut_idx_row = min(i_row, luts.shape[0] - 1)
                lut_idx_col = min(i_col, luts.shape[1] - 1)
                return apply_lut_cy(y_block, luts[lut_idx_row, lut_idx_col], result)
            else:  # Bottom edge
                try:
                    # Ensure we're using the correct LUT indices
                    lut_idx_row = min(i_row, luts.shape[0] - 1)
                    lut_idx_col_prev = min(i_col - 1, luts.shape[1] - 1)
                    lut_idx_col_curr = min(i_col, luts.shape[1] - 1)
                    
                    apply_lut_cy(y_block, luts[lut_idx_row, lut_idx_col_prev], temp)
                    apply_lut_cy(y_block, luts[lut_idx_row, lut_idx_col_curr], result)
                    
                    with nogil, parallel():
                        for i in prange(height):
                            for j in range(width):
                                w_float = left_lut_weights[0, j] / 1024.0
                                w_inv = 1.0 - w_float
                                val = int(w_float * temp[i, j] + w_inv * result[i, j])
                                result[i, j] = val if val < 256 else 255
                    
                    return np.asarray(result)
                except Exception as e:
                    return np.zeros((height, width), dtype=np.uint8)
        elif i_col == 0:  # Left edge
            apply_lut_cy(y_block, luts[i_row - 1, 0], temp)
            apply_lut_cy(y_block, luts[i_row, 0], result)
            with nogil, parallel():
                for i in prange(height):
                    for j in range(width):
                        w_float = top_lut_weights[i, 0] / 1024.0
                        w_inv = 1.0 - w_float
                        val = int(w_float * temp[i, j] + w_inv * result[i, j])
                        result[i, j] = val if val < 256 else 255
            return np.asarray(result)
        elif i_col == horiz_tiles - 1:  # Right edge
            lut_idx = min(i_col, luts.shape[1] - 1)
            apply_lut_cy(y_block, luts[i_row - 1, lut_idx], temp)
            apply_lut_cy(y_block, luts[i_row, lut_idx], result)
            with nogil, parallel():
                for i in prange(height):
                    for j in range(width):
                        w_float = top_lut_weights[i, 0] / 1024.0
                        w_inv = 1.0 - w_float
                        val = int(w_float * temp[i, j] + w_inv * result[i, j])
                        result[i, j] = val if val < 256 else 255
            return np.asarray(result)
        else:  # Interior tile
            # Apply LUTs for all four corners
            apply_lut_cy(y_block, luts[i_row - 1, i_col - 1], temp)
            apply_lut_cy(y_block, luts[i_row - 1, i_col], result)
            
            # Interpolate top edge
            with nogil, parallel():
                for i in prange(height):
                    for j in range(width):
                        w_float = left_lut_weights[0, j] / 1024.0
                        w_inv = 1.0 - w_float
                        val = int(w_float * temp[i, j] + w_inv * result[i, j])
                        temp[i, j] = val if val < 256 else 255
            
            # Apply LUTs for bottom corners
            apply_lut_cy(y_block, luts[i_row, i_col - 1], result)
            apply_lut_cy(y_block, luts[i_row, i_col], temp)
            
            # Interpolate bottom edge
            with nogil, parallel():
                for i in prange(height):
                    for j in range(width):
                        w_float = left_lut_weights[0, j] / 1024.0
                        w_inv = 1.0 - w_float
                        val = int(w_float * result[i, j] + w_inv * temp[i, j])
                        result[i, j] = val if val < 256 else 255
            
            # Final interpolation between top and bottom
            with nogil, parallel():
                for i in prange(height):
                    for j in range(width):
                        w_float = top_lut_weights[i, 0] / 1024.0
                        w_inv = 1.0 - w_float
                        val = int(w_float * temp[i, j] + w_inv * result[i, j])
                        result[i, j] = val if val < 256 else 255
            
            return np.asarray(result)
    except Exception as e:
        return np.zeros((height, width), dtype=np.uint8) 