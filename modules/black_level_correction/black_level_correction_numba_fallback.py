"""
File: black_level_correction_numba_fallback.py
Description: Numba-optimized black level correction with automatic fallback
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array

# Try to import Numba, but provide fallback if not available
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = None
    prange = None


class BlackLevelCorrectionNumbaFallback:
    """
    Numba-optimized Black Level Correction with automatic fallback
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
        
        # Try to use Numba, fallback to original if not available
        if NUMBA_AVAILABLE:
            try:
                from modules.black_level_correction.black_level_correction_numba import BlackLevelCorrectionNumba
                self.numba_blc = BlackLevelCorrectionNumba(img, platform, sensor_info, parm_blc)
                self.use_numba = True
                if self.is_debug:
                    print("  Using Numba implementation for BLC")
            except Exception as e:
                if self.is_debug:
                    print(f"  Numba initialization failed, falling back to original implementation: {e}")
                self.use_numba = False
                from modules.black_level_correction.black_level_correction import BlackLevelCorrection
                self.original_blc = BlackLevelCorrection(img, platform, sensor_info, parm_blc)
        else:
            if self.is_debug:
                print("  Numba not available, using original implementation")
            self.use_numba = False
            from modules.black_level_correction.black_level_correction import BlackLevelCorrection
            self.original_blc = BlackLevelCorrection(img, platform, sensor_info, parm_blc)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_black_level_correction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute Black Level Correction with automatic fallback
        """
        if self.enable:
            start = time.time()
            
            if self.use_numba:
                blc_out = self.numba_blc.apply_blc_parameters()
                if self.is_debug:
                    print(f"  Numba BLC execution time: {time.time() - start:.3f}s")
            else:
                blc_out = self.original_blc.apply_blc_parameters()
                if self.is_debug:
                    print(f"  Original BLC execution time: {time.time() - start:.3f}s")
            
            self.img = blc_out
        self.save()
        return self.img 