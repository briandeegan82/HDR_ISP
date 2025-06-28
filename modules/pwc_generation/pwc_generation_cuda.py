"""
File: pwc_generation_cuda.py
Description: CUDA-optimized Piecewise Curve decompanding using PyTorch
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from util.utils import save_output_array

# Try to import PyTorch, but provide fallback if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def generate_decompanding_lut_cuda(companded_pin, companded_pout, max_input_value):
    """
    Generate a decompanding lookup table (LUT) using CUDA optimization.
    
    Parameters:
    - companded_pin: Array of input knee points
    - companded_pout: Array of corresponding output knee points
    - max_input_value: Maximum input value for the LUT
    
    Returns:
    - lut: A PyTorch tensor representing the decompanding lookup table
    """
    if not TORCH_AVAILABLE:
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
    
    # Convert to PyTorch tensors
    companded_pin = torch.tensor(companded_pin, dtype=torch.int32)
    companded_pout = torch.tensor(companded_pout, dtype=torch.float32)
    
    # Initialize the LUT with zeros
    lut = torch.zeros(max_input_value + 1, dtype=torch.float32)
    
    # Generate the LUT by interpolating between the knee points
    for i in range(len(companded_pin) - 1):
        start_in = companded_pin[i]
        end_in = companded_pin[i + 1]
        start_out = companded_pout[i]
        end_out = companded_pout[i + 1]
        
        # Create range for this segment
        x_range = torch.arange(start_in, end_in + 1, dtype=torch.float32)
        
        # Linear interpolation between the knee points
        t = (x_range - start_in) / (end_in - start_in)
        lut[start_in:end_in + 1] = start_out + t * (end_out - start_out)
    
    # Handle values beyond the last knee point (extend the last segment)
    last_in = companded_pin[-1]
    last_out = companded_pout[-1]
    lut[last_in:] = last_out
    
    return lut


def apply_lut_and_pedestal_cuda(img, lut, pedestal):
    """
    Apply LUT and subtract pedestal using CUDA optimization.
    
    Parameters:
    - img: Input image tensor
    - lut: Lookup table tensor
    - pedestal: Pedestal value to subtract
    
    Returns:
    - Processed image tensor
    """
    if not TORCH_AVAILABLE:
        # Fallback to numpy implementation
        lut_values = lut[img]
        result = np.clip(lut_values - pedestal, 0, None)
        return result
    
    # Move tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img.to(device)
    lut = lut.to(device)
    pedestal = torch.tensor(pedestal, dtype=torch.float32, device=device)
    
    # Apply LUT lookup using advanced indexing
    lut_values = lut[img]
    
    # Subtract pedestal and clip to non-negative
    result = torch.clamp(lut_values - pedestal, min=0.0)
    
    return result


class PiecewiseCurveCuda:
    """
    CUDA-optimized Piecewise Curve decompanding
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
        
        # Check CUDA availability
        if TORCH_AVAILABLE:
            self.cuda_available = torch.cuda.is_available()
            if self.is_debug and self.cuda_available:
                print(f"  CUDA available: {torch.cuda.get_device_name()}")
            elif self.is_debug:
                print("  CUDA not available, using CPU")
        else:
            self.cuda_available = False
            if self.is_debug:
                print("  PyTorch not available, using fallback implementation")
    
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
        PWC decompanding with CUDA optimization
        """
        if self.enable:
            start = time.time()
            
            if TORCH_AVAILABLE:
                # Convert input to PyTorch tensor
                img_tensor = torch.tensor(self.img, dtype=torch.int32)
                
                # Generate LUT using CUDA
                lut = generate_decompanding_lut_cuda(
                    self.parm_cmpd["companded_pin"],
                    self.parm_cmpd["companded_pout"],
                    self.parm_cmpd["companded_pin"][-1]
                )
                
                # Apply LUT and pedestal using CUDA
                result_tensor = apply_lut_and_pedestal_cuda(
                    img_tensor,
                    lut,
                    self.parm_cmpd["pedestal"]
                )
                
                # Convert back to numpy
                self.img = result_tensor.cpu().numpy()
            else:
                # Fallback to original implementation
                from modules.pwc_generation.pwc_generation import PiecewiseCurve
                original_pwc = PiecewiseCurve(self.img, self.platform, self.sensor_info, self.parm_cmpd)
                self.img = original_pwc.execute()
            
            if self.is_debug:
                print(f"  CUDA PWC execution time: {time.time() - start:.3f}s")
        
        self.save()
        return self.img.astype(np.uint32) 