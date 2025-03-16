import numpy as np
import matplotlib.pyplot as plt
import time
from util.utils import save_output_array


class PiecewiseCurve:
    
    def __init__(self, img, platform, sensor_info, parm_cmpd):
        self.img = img
        self.enable = parm_cmpd["is_enable"]
        self.sensor_info = sensor_info
        self.bit_depth = sensor_info["bit_depth"]
        self.parm_cmpd = parm_cmpd
        self.num_kneepoints = parm_cmpd["num_kneepoints"]
        self.A = parm_cmpd["A"]
        self.input_range = parm_cmpd["input_range"]
        self.output_range = parm_cmpd["output_range"]
        self.is_save = parm_cmpd["is_save"]
        self.platform = platform
    
    def generate_companding_curve(self):
        """
        Generates a piecewise A-Law companding curve with non-uniformly distributed kneepoints.
        """
        min_input = 0
        max_input = self.output_range
        min_output = 0
        max_output = self.input_range
        
        # Generate non-uniform kneepoints (e.g., 1, 1/2, 1/4, 1/8, etc.)
        x_knees = np.array([1 / (2 ** i) for i in range(self.num_kneepoints)])
        x_knees = np.append(x_knees, 0)
        x_knees = np.flip(x_knees)  # Reverse to get [1, 1/2, 1/4, ...]
        
        # Apply A-Law companding formula for the positive half only
        y_knees = np.where(
            x_knees == 0, 1e-6,  # If x_knees is 0, return 1e-6
            np.where(
                x_knees < (1 / self.A),  # If x_knees < (1 / self.A), return (self.A * x_knees) / (1 + np.log(self.A))
                (self.A * x_knees) / (1 + np.log(self.A)),
                (1 + np.log(self.A * x_knees)) / (1 + np.log(self.A))  # Else, return this value
            )
        )
        
        # Scale to the output range
        x_knees = x_knees * (max_input - min_input) + min_input
        y_knees = y_knees * (max_output - min_output) + min_output
        
        # Generate fine-grained x values for interpolation
        x_values = np.arange(min_input, max_input, 1)
        y_values = np.interp(x_values, x_knees, y_knees)
        
        return x_values, y_values, x_knees, y_knees, "Piecewise A-Law Companding Curve (Non-Uniform Kneepoints)", 'g'
    
    import numpy as np

    def generate_decompanding_curve(self):
        """
        Generates a piecewise A-Law decompanding curve to reverse the companding process.
        This function reverses the non-uniformly distributed kneepoints used in companding.
        """
        min_input = 0
        max_input = self.input_range  # Note: input to decompanding is output from companding
        min_output = 0
        max_output = self.output_range  # Note: output from decompanding is input to companding
        
        # Generate non-uniform kneepoints as in the companding function
        x_knees_orig = np.array([1 / (2 ** i) for i in range(self.num_kneepoints)])
        x_knees_orig = np.append(x_knees_orig, 0)
        x_knees_orig = np.flip(x_knees_orig)  # Reverse to get [0, ..., 1/4, 1/2, 1]
        
        # Apply A-Law companding formula for the positive half only (same as in companding)
        y_knees_orig = np.where(
            x_knees_orig == 0, 1e-6,
            np.where(
                x_knees_orig < (1 / self.A),
                (self.A * x_knees_orig) / (1 + np.log(self.A)),
                (1 + np.log(self.A * x_knees_orig)) / (1 + np.log(self.A))
            )
        )
        
        # Scale to the original ranges
        x_knees_orig = x_knees_orig * (max_output - min_output) + min_output
        y_knees_orig = y_knees_orig * (max_input - min_input) + min_input
        
        # For decompanding, we swap x and y to get the inverse function
        x_knees = y_knees_orig
        y_knees = x_knees_orig
        
        # Generate fine-grained x values for interpolation
        x_values = np.arange(min_input, max_input, 1)
        y_values = np.interp(x_values, x_knees, y_knees)
        
        # For the analytical inverse (necessary for precise decompanding)
        def decompand_value(y):
            """Analytically compute the inverse of A-Law companding for a single value"""
            # Scale y to [0, 1] range
            y_scaled = (y - min_input) / (max_input - min_input)
            
            # Apply inverse A-Law formula
            if y_scaled < (1 / self.A) / (1 + np.log(self.A)):
                # Inverse of: y = (A * x) / (1 + log(A))
                x_scaled = y_scaled * (1 + np.log(self.A)) / self.A
            else:
                # Inverse of: y = (1 + log(A * x)) / (1 + log(A))
                x_scaled = np.exp(y_scaled * (1 + np.log(self.A)) - 1) / self.A
            
            # Scale x back to original range
            x = x_scaled * (max_output - min_output) + min_output
            return x
        
        # Store the analytical inverse function for precise computations
        self.decompand_value = decompand_value
        
        return x_values, y_values, x_knees, y_knees, "Piecewise A-Law Decompanding Curve", 'r'  
    
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
        PWC decompanding
        """
        print("Decompanding = " + str(self.enable))

        if self.enable:
            start = time.time()
            x_values, y_values, x_knees, y_knees,_,_ = self.generate_decompanding_curve()
            lut = y_values.astype(np.uint32)
            self.img = lut[self.img]
            print(f"  Execution time: {time.time() - start:.3f}s")
        self.save()
        return self.img