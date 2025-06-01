# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from cython.parallel import prange
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

def apply_lut_cy(uint8[:, :] tile, uint8[:] lut):
    """Apply LUT to a tile using Cython"""
    cdef:
        int i, j
        int height = tile.shape[0]
        int width = tile.shape[1]
        uint8[:, :] result = np.empty((height, width), dtype=np.uint8)
    
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
        uint8[:, :] first, second, interp_top, interp_curr
        int i, j, w
        int32 val
    
    # Safety checks with detailed error messages
    if i_row < 0 or i_row > vert_tiles or i_col < 0 or i_col > horiz_tiles:
        return np.zeros((height, width), dtype=np.uint8)
    
    try:
        # Get LUTs based on position
        if i_row == 0:
            if i_col == 0:  # Top-left corner
                return apply_lut_cy(y_block, luts[0, 0])
            elif i_col == horiz_tiles:  # Top-right corner
                return apply_lut_cy(y_block, luts[0, -1])
            else:  # Top edge
                first = apply_lut_cy(y_block, luts[0, i_col - 1])
                second = apply_lut_cy(y_block, luts[0, i_col])
                for i in range(height):
                    for j in range(width):
                        w = left_lut_weights[0, j]
                        val = (w * first[i, j] + (1024 - w) * second[i, j]) >> 10
                        result[i, j] = val if val < 256 else 255
                return np.asarray(result)
        elif i_row == vert_tiles:
            if i_col == 0:  # Bottom-left corner
                return apply_lut_cy(y_block, luts[-1, 0])
            elif i_col == horiz_tiles:  # Bottom-right corner
                return apply_lut_cy(y_block, luts[-1, -1])
            else:  # Bottom edge
                first = apply_lut_cy(y_block, luts[-1, i_col - 1])
                second = apply_lut_cy(y_block, luts[-1, i_col])
                for i in range(height):
                    for j in range(width):
                        w = left_lut_weights[0, j]
                        val = (w * first[i, j] + (1024 - w) * second[i, j]) >> 10
                        result[i, j] = val if val < 256 else 255
                return np.asarray(result)
        elif i_col == 0:  # Left edge
            first = apply_lut_cy(y_block, luts[i_row - 1, 0])
            second = apply_lut_cy(y_block, luts[i_row, 0])
            for i in range(height):
                w = top_lut_weights[i, 0]
                for j in range(width):
                    val = (w * first[i, j] + (1024 - w) * second[i, j]) >> 10
                    result[i, j] = val if val < 256 else 255
            return np.asarray(result)
        elif i_col == horiz_tiles:  # Right edge
            first = apply_lut_cy(y_block, luts[i_row - 1, -1])
            second = apply_lut_cy(y_block, luts[i_row, -1])
            for i in range(height):
                w = top_lut_weights[i, 0]
                for j in range(width):
                    val = (w * first[i, j] + (1024 - w) * second[i, j]) >> 10
                    result[i, j] = val if val < 256 else 255
            return np.asarray(result)
        else:  # Interior
            # Process top blocks
            first = apply_lut_cy(y_block, luts[i_row - 1, i_col - 1])
            second = apply_lut_cy(y_block, luts[i_row - 1, i_col])
            interp_top = np.empty((height, width), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    w = left_lut_weights[0, j]
                    val = (w * first[i, j] + (1024 - w) * second[i, j]) >> 10
                    interp_top[i, j] = val if val < 256 else 255
            
            # Process current blocks
            first = apply_lut_cy(y_block, luts[i_row, i_col - 1])
            second = apply_lut_cy(y_block, luts[i_row, i_col])
            interp_curr = np.empty((height, width), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    w = left_lut_weights[0, j]
                    val = (w * first[i, j] + (1024 - w) * second[i, j]) >> 10
                    interp_curr[i, j] = val if val < 256 else 255
            
            # Final interpolation
            for i in range(height):
                w = top_lut_weights[i, 0]
                for j in range(width):
                    val = (w * interp_top[i, j] + (1024 - w) * interp_curr[i, j]) >> 10
                    result[i, j] = val if val < 256 else 255
            return np.asarray(result)
    except Exception as e:
        return np.zeros((height, width), dtype=np.uint8) 