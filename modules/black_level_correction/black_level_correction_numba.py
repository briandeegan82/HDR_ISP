"""
File: black_level_correction_numba.py
Description: Numba-optimized black level correction implementation
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from numba import jit, prange

from util.utils import save_output_array


@jit(nopython=True, parallel=True, cache=True)
def apply_blc_numba_rggb(raw, r_offset, gr_offset, gb_offset, b_offset, 
                        r_sat, gr_sat, gb_sat, b_sat, bpp, is_linearize):
    """Numba-optimized BLC for RGGB pattern"""
    height, width = raw.shape
    
    for y in prange(height):
        for x in range(width):
            if y % 2 == 0:  # Even rows
                if x % 2 == 0:  # Even columns - Red
                    raw[y, x] = raw[y, x] - r_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (r_sat - r_offset) * ((2**bpp) - 1)
                else:  # Odd columns - Green Red
                    raw[y, x] = raw[y, x] - gr_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (gr_sat - gr_offset) * ((2**bpp) - 1)
            else:  # Odd rows
                if x % 2 == 0:  # Even columns - Green Blue
                    raw[y, x] = raw[y, x] - gb_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                else:  # Odd columns - Blue
                    raw[y, x] = raw[y, x] - b_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (b_sat - b_offset) * ((2**bpp) - 1)
    
    return raw


@jit(nopython=True, parallel=True, cache=True)
def apply_blc_numba_bggr(raw, r_offset, gr_offset, gb_offset, b_offset, 
                        r_sat, gr_sat, gb_sat, b_sat, bpp, is_linearize):
    """Numba-optimized BLC for BGGR pattern"""
    height, width = raw.shape
    
    for y in prange(height):
        for x in range(width):
            if y % 2 == 0:  # Even rows
                if x % 2 == 0:  # Even columns - Blue
                    raw[y, x] = raw[y, x] - b_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (b_sat - b_offset) * ((2**bpp) - 1)
                else:  # Odd columns - Green Blue
                    raw[y, x] = raw[y, x] - gb_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (gb_sat - gb_offset) * ((2**bpp) - 1)
            else:  # Odd rows
                if x % 2 == 0:  # Even columns - Green Red
                    raw[y, x] = raw[y, x] - gr_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                else:  # Odd columns - Red
                    raw[y, x] = raw[y, x] - r_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (r_sat - r_offset) * ((2**bpp) - 1)
    
    return raw


@jit(nopython=True, parallel=True, cache=True)
def apply_blc_numba_grbg(raw, r_offset, gr_offset, gb_offset, b_offset, 
                        r_sat, gr_sat, gb_sat, b_sat, bpp, is_linearize):
    """Numba-optimized BLC for GRBG pattern"""
    height, width = raw.shape
    
    for y in prange(height):
        for x in range(width):
            if y % 2 == 0:  # Even rows
                if x % 2 == 0:  # Even columns - Green Red
                    raw[y, x] = raw[y, x] - gr_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                else:  # Odd columns - Red
                    raw[y, x] = raw[y, x] - r_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (r_sat - r_offset) * ((2**bpp) - 1)
            else:  # Odd rows
                if x % 2 == 0:  # Even columns - Blue
                    raw[y, x] = raw[y, x] - b_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (b_sat - b_offset) * ((2**bpp) - 1)
                else:  # Odd columns - Green Blue
                    raw[y, x] = raw[y, x] - gb_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (gb_sat - gb_offset) * ((2**bpp) - 1)
    
    return raw


@jit(nopython=True, parallel=True, cache=True)
def apply_blc_numba_gbrg(raw, r_offset, gr_offset, gb_offset, b_offset, 
                        r_sat, gr_sat, gb_sat, b_sat, bpp, is_linearize):
    """Numba-optimized BLC for GBRG pattern"""
    height, width = raw.shape
    
    for y in prange(height):
        for x in range(width):
            if y % 2 == 0:  # Even rows
                if x % 2 == 0:  # Even columns - Green Blue
                    raw[y, x] = raw[y, x] - gb_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                else:  # Odd columns - Blue
                    raw[y, x] = raw[y, x] - b_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (b_sat - b_offset) * ((2**bpp) - 1)
            else:  # Odd rows
                if x % 2 == 0:  # Even columns - Red
                    raw[y, x] = raw[y, x] - r_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (r_sat - r_offset) * ((2**bpp) - 1)
                else:  # Odd columns - Green Red
                    raw[y, x] = raw[y, x] - gr_offset
                    if is_linearize:
                        raw[y, x] = raw[y, x] / (gr_sat - gr_offset) * ((2**bpp) - 1)
    
    return raw


class BlackLevelCorrectionNumba:
    """
    Numba-optimized Black Level Correction
    """

    def __init__(self, img, platform, sensor_info, parm_blc):
        self.img = img
        self.enable = parm_blc["is_enable"]
        self.sensor_info = sensor_info
        self.param_blc = parm_blc
        self.is_linearize = self.param_blc["is_linear"]
        self.is_save = parm_blc["is_save"]
        self.platform = platform
        self.is_debug = parm_blc.get("is_debug", False)

    def apply_blc_parameters(self):
        """
        Apply BLC parameters using Numba optimization
        """
        # Get config parameters
        bayer = self.sensor_info["bayer_pattern"]
        bpp = self.sensor_info["bit_depth"]
        r_offset = self.param_blc["r_offset"]
        gb_offset = self.param_blc["gb_offset"]
        gr_offset = self.param_blc["gr_offset"]
        b_offset = self.param_blc["b_offset"]

        r_sat = self.param_blc["r_sat"]
        gr_sat = self.param_blc["gr_sat"]
        gb_sat = self.param_blc["gb_sat"]
        b_sat = self.param_blc["b_sat"]

        raw = np.float32(self.img)

        # Apply Numba-optimized BLC based on Bayer pattern
        if bayer == "rggb":
            raw = apply_blc_numba_rggb(raw, r_offset, gr_offset, gb_offset, b_offset,
                                     r_sat, gr_sat, gb_sat, b_sat, bpp, self.is_linearize)
        elif bayer == "bggr":
            raw = apply_blc_numba_bggr(raw, r_offset, gr_offset, gb_offset, b_offset,
                                     r_sat, gr_sat, gb_sat, b_sat, bpp, self.is_linearize)
        elif bayer == "grbg":
            raw = apply_blc_numba_grbg(raw, r_offset, gr_offset, gb_offset, b_offset,
                                     r_sat, gr_sat, gb_sat, b_sat, bpp, self.is_linearize)
        elif bayer == "gbrg":
            raw = apply_blc_numba_gbrg(raw, r_offset, gr_offset, gb_offset, b_offset,
                                     r_sat, gr_sat, gb_sat, b_sat, bpp, self.is_linearize)

        raw_blc = np.uint32(np.clip(raw, 0, (2**bpp) - 1))
        return raw_blc

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_black_level_correction_numba_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute Numba-optimized Black Level Correction
        """
        if self.enable:
            start = time.time()
            blc_out = self.apply_blc_parameters()
            if hasattr(self, 'is_debug') and self.is_debug:
                print(f"  Numba BLC execution time: {time.time() - start:.3f}s")
            self.img = blc_out
        self.save()
        return self.img 