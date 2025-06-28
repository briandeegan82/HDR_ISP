"""
File: pwc_generation_halide.py
Description: Halide-optimized Piecewise Curve decompanding
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from util.utils import save_output_array

# Try to import Halide, but provide fallback if not available
try:
    import halide as hl
    HALIDE_AVAILABLE = True
except ImportError:
    HALIDE_AVAILABLE = False
    hl = None


def generate_decompanding_lut_halide(companded_pin, companded_pout, max_input_value):
    """
    Generate a decompanding lookup table (LUT) using Halide optimization.
    
    Parameters:
    - companded_pin: Array of input knee points
    - companded_pout: Array of corresponding output knee points
    - max_input_value: Maximum input value for the LUT
    
    Returns:
    - lut: A numpy array representing the decompanding lookup table
    """
    if not HALIDE_AVAILABLE:
        # Fallback to numpy implementation
        lut = np.zeros(max_input_value + 1, dtype=np.float64)
        
        for i in range(len(companded_pin) - 1):
            start_in = companded_pin[i]
            end_in = companded_pin[i + 1]
            start_out = companded_pout[i]
            end_out = companded_pout[i + 1]
            
            for x in range(start_in, end_in + 1):
                t = (x - start_in) / (end_in - start_in)
                lut[x] = start_out + t * (end_out - start_out)
        
        last_in = companded_pin[-1]
        last_out = companded_pout[-1]
        lut[last_in:] = last_out
        
        return lut
    
    # Halide implementation
    lut = np.zeros(max_input_value + 1, dtype=np.float64)
    
    # Use Halide for interpolation
    for i in range(len(companded_pin) - 1):
        start_in = companded_pin[i]
        end_in = companded_pin[i + 1]
        start_out = companded_pout[i]
        end_out = companded_pout[i + 1]
        
        # Create Halide function for this segment
        x = hl.Var('x')
        t = (x - start_in) / (end_in - start_in)
        segment_func = hl.Func('segment')
        segment_func[x] = start_out + t * (end_out - start_out)
        
        # Apply to the range
        segment_range = (start_in, end_in + 1)
        segment_result = segment_func.realize([end_in - start_in + 1])
        lut[start_in:end_in + 1] = np.array(segment_result)
    
    # Handle values beyond the last knee point
    last_in = companded_pin[-1]
    last_out = companded_pout[-1]
    lut[last_in:] = last_out
    
    return lut


def apply_lut_and_pedestal_halide(img, lut, pedestal):
    """
    Apply LUT and subtract pedestal using Halide optimization.
    
    Parameters:
    - img: Input image array
    - lut: Lookup table array
    - pedestal: Pedestal value to subtract
    
    Returns:
    - Processed image array
    """
    if not HALIDE_AVAILABLE:
        # Fallback to numpy implementation
        # Ensure image is integer type for LUT indexing
        img_int = img.astype(np.int32)
        lut_values = lut[img_int]
        result = np.clip(lut_values - pedestal, 0, None)
        return result
    
    # Halide implementation
    x, y = hl.Var('x'), hl.Var('y')
    
    # Create Halide functions
    input_func = hl.Func('input')
    lut_func = hl.Func('lut')
    output_func = hl.Func('output')
    
    # Define the input
    input_func[x, y] = img[y, x]  # Note: Halide uses (x, y) order
    
    # Define the LUT lookup
    lut_func[x] = lut[x]
    
    # Define the output with pedestal subtraction and clipping
    output_func[x, y] = hl.max(0.0, lut_func[input_func[x, y]] - pedestal)
    
    # Realize the result
    result = output_func.realize([img.shape[1], img.shape[0]])  # (width, height)
    
    # Convert back to numpy with correct shape
    return np.array(result).transpose(1, 0)  # Back to (height, width)


class PiecewiseCurveHalide:
    """
    Halide-optimized Piecewise Curve decompanding
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
        
        if self.is_debug:
            if HALIDE_AVAILABLE:
                print("  Halide available for PWC optimization")
            else:
                print("  Halide not available, using fallback implementation")
    
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
        PWC decompanding with Halide optimization
        """
        if self.enable:
            start = time.time()
            
            # Generate LUT using Halide
            lut = generate_decompanding_lut_halide(
                self.parm_cmpd["companded_pin"],
                self.parm_cmpd["companded_pout"],
                self.parm_cmpd["companded_pin"][-1]
            )
            
            # Apply LUT and pedestal using Halide
            self.img = apply_lut_and_pedestal_halide(
                self.img.astype(np.int32),
                lut,
                self.parm_cmpd["pedestal"]
            )
            
            if self.is_debug:
                print(f"  Halide PWC execution time: {time.time() - start:.3f}s")
        
        self.save()
        return self.img.astype(np.uint32) 