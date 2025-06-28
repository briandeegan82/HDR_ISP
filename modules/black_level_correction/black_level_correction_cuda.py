"""
File: black_level_correction_cuda.py
Description: CUDA-optimized black level correction implementation
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from numba import cuda

from util.utils import save_output_array


@cuda.jit
def apply_blc_cuda_rggb(raw, r_offset, gr_offset, gb_offset, b_offset, 
                       r_sat, gr_sat, gb_sat, b_sat, bpp, is_linearize):
    """CUDA-optimized BLC for RGGB pattern"""
    y, x = cuda.grid(2)
    height, width = raw.shape
    
    if y < height and x < width:
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


@cuda.jit
def apply_blc_cuda_bggr(raw, r_offset, gr_offset, gb_offset, b_offset, 
                       r_sat, gr_sat, gb_sat, b_sat, bpp, is_linearize):
    """CUDA-optimized BLC for BGGR pattern"""
    y, x = cuda.grid(2)
    height, width = raw.shape
    
    if y < height and x < width:
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


@cuda.jit
def apply_blc_cuda_grbg(raw, r_offset, gr_offset, gb_offset, b_offset, 
                       r_sat, gr_sat, gb_sat, b_sat, bpp, is_linearize):
    """CUDA-optimized BLC for GRBG pattern"""
    y, x = cuda.grid(2)
    height, width = raw.shape
    
    if y < height and x < width:
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


@cuda.jit
def apply_blc_cuda_gbrg(raw, r_offset, gr_offset, gb_offset, b_offset, 
                       r_sat, gr_sat, gb_sat, b_sat, bpp, is_linearize):
    """CUDA-optimized BLC for GBRG pattern"""
    y, x = cuda.grid(2)
    height, width = raw.shape
    
    if y < height and x < width:
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


class BlackLevelCorrectionCUDA:
    """
    CUDA-optimized Black Level Correction
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
        
        # Check CUDA availability
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available")

    def apply_blc_parameters(self):
        """
        Apply BLC parameters using CUDA optimization
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
        height, width = raw.shape

        # Configure CUDA grid and block dimensions
        threadsperblock = (16, 16)
        blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_y, blockspergrid_x)

        # Copy data to GPU
        raw_gpu = cuda.to_device(raw)

        # Apply CUDA-optimized BLC based on Bayer pattern
        if bayer == "rggb":
            apply_blc_cuda_rggb[blockspergrid, threadsperblock](
                raw_gpu, r_offset, gr_offset, gb_offset, b_offset,
                r_sat, gr_sat, gb_sat, b_sat, bpp, self.is_linearize)
        elif bayer == "bggr":
            apply_blc_cuda_bggr[blockspergrid, threadsperblock](
                raw_gpu, r_offset, gr_offset, gb_offset, b_offset,
                r_sat, gr_sat, gb_sat, b_sat, bpp, self.is_linearize)
        elif bayer == "grbg":
            apply_blc_cuda_grbg[blockspergrid, threadsperblock](
                raw_gpu, r_offset, gr_offset, gb_offset, b_offset,
                r_sat, gr_sat, gb_sat, b_sat, bpp, self.is_linearize)
        elif bayer == "gbrg":
            apply_blc_cuda_gbrg[blockspergrid, threadsperblock](
                raw_gpu, r_offset, gr_offset, gb_offset, b_offset,
                r_sat, gr_sat, gb_sat, b_sat, bpp, self.is_linearize)

        # Copy result back to CPU
        raw = raw_gpu.copy_to_host()
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
                "Out_black_level_correction_cuda_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute CUDA-optimized Black Level Correction
        """
        if self.enable:
            start = time.time()
            blc_out = self.apply_blc_parameters()
            if self.is_debug:
                print(f"  CUDA BLC execution time: {time.time() - start:.3f}s")
            self.img = blc_out
        self.save()
        return self.img 