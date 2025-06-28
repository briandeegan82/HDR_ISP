"""
File: pwc_generation_numba_fallback.py
Description: Numba-optimized Piecewise Curve decompanding with automatic fallback
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


class PiecewiseCurveNumbaFallback:
    """
    Numba-optimized Piecewise Curve decompanding with automatic fallback
    """

    def __init__(self, img, platform, sensor_info, parm_cmpd):
        self.img = img
        self.enable = parm_cmpd["is_enable"]
        self.sensor_info = sensor_info
        self.bit_depth = sensor_info["bit_depth"]
        self.parm_cmpd = parm_cmpd
        self.companded_pin = parm_cmpd["companded_pin"]
        self.companded_pout = parm_cmpd["companded_pout"]
        self.is_save = parm_cmpd["is_save"]
        self.platform = platform
        self.is_debug = parm_cmpd.get("is_debug", False)
        
        # Try to use Numba, fallback to original if not available
        if NUMBA_AVAILABLE:
            try:
                from modules.pwc_generation.pwc_generation_numba import PiecewiseCurveNumba
                self.numba_pwc = PiecewiseCurveNumba(img, platform, sensor_info, parm_cmpd)
                self.use_numba = True
                if self.is_debug:
                    print("  Using Numba implementation for PWC")
            except Exception as e:
                if self.is_debug:
                    print(f"  Numba initialization failed, falling back to original implementation: {e}")
                self.use_numba = False
                from modules.pwc_generation.pwc_generation import PiecewiseCurve
                self.original_pwc = PiecewiseCurve(img, platform, sensor_info, parm_cmpd)
        else:
            if self.is_debug:
                print("  Numba not available, using original implementation")
            self.use_numba = False
            from modules.pwc_generation.pwc_generation import PiecewiseCurve
            self.original_pwc = PiecewiseCurve(img, platform, sensor_info, parm_cmpd)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_decompanding",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute PWC decompanding with automatic fallback
        """
        if self.enable:
            start = time.time()
            
            if self.use_numba:
                pwc_out = self.numba_pwc.execute()
                if self.is_debug:
                    print(f"  Numba PWC execution time: {time.time() - start:.3f}s")
            else:
                pwc_out = self.original_pwc.execute()
                if self.is_debug:
                    print(f"  Original PWC execution time: {time.time() - start:.3f}s")
            
            self.img = pwc_out
        self.save()
        return self.img 