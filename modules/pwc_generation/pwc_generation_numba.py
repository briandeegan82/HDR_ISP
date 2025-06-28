"""
File: pwc_generation_numba.py
Description: Numba-optimized Piecewise Curve decompanding
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from numba import jit, prange
from util.utils import save_output_array


@jit(nopython=True, cache=True)
def generate_decompanding_lut_numba(companded_pin, companded_pout, max_input_value):
    """
    Generate a decompanding lookup table (LUT) using Numba optimization.
    
    Parameters:
    - companded_pin: Array of input knee points
    - companded_pout: Array of corresponding output knee points
    - max_input_value: Maximum input value for the LUT
    
    Returns:
    - lut: A numpy array representing the decompanding lookup table
    """
    # Initialize the LUT with zeros
    lut = np.zeros(max_input_value + 1, dtype=np.float64)
    
    # Generate the LUT by interpolating between the knee points
    for i in range(len(companded_pin) - 1):
        start_in = companded_pin[i]
        end_in = companded_pin[i + 1]
        start_out = companded_pout[i]
        end_out = companded_pout[i + 1]
        
        # Linear interpolation between the knee points
        for x in range(start_in, end_in + 1):
            t = (x - start_in) / (end_in - start_in)
            lut[x] = start_out + t * (end_out - start_out)
    
    # Handle values beyond the last knee point (extend the last segment)
    last_in = companded_pin[-1]
    last_out = companded_pout[-1]
    for x in range(last_in, max_input_value + 1):
        lut[x] = last_out
    
    return lut


@jit(nopython=True, parallel=True, cache=True)
def apply_lut_and_pedestal_numba(img, lut, pedestal):
    """
    Apply LUT and subtract pedestal using Numba optimization.
    
    Parameters:
    - img: Input image array
    - lut: Lookup table array
    - pedestal: Pedestal value to subtract
    
    Returns:
    - Processed image array
    """
    height, width = img.shape
    result = np.empty_like(img, dtype=np.float64)
    
    for i in prange(height):
        for j in range(width):
            # Apply LUT lookup
            lut_value = lut[img[i, j]]
            # Subtract pedestal and clip to non-negative
            result[i, j] = max(0.0, lut_value - pedestal)
    
    return result


class PiecewiseCurveNumba:
    """
    Numba-optimized Piecewise Curve decompanding
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
        PWC decompanding with Numba optimization
        """
        if self.enable:
            start = time.time()
            
            # Generate LUT using Numba
            lut = generate_decompanding_lut_numba(
                np.array(self.parm_cmpd["companded_pin"], dtype=np.int32),
                np.array(self.parm_cmpd["companded_pout"], dtype=np.float64),
                self.parm_cmpd["companded_pin"][-1]
            )
            
            # Apply LUT and pedestal using Numba
            self.img = apply_lut_and_pedestal_numba(
                self.img.astype(np.int32),
                lut,
                self.parm_cmpd["pedestal"]
            )
            
            if self.is_debug:
                print(f"  Numba PWC execution time: {time.time() - start:.3f}s")
        
        self.save()
        return self.img.astype(np.uint32) 