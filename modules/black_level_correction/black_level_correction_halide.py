"""
File: black_level_correction_halide.py
Description: Halide-optimized black level correction implementation
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


class BlackLevelCorrectionHalide:
    """
    Halide-optimized Black Level Correction
    """

    def __init__(self, img, platform, sensor_info, parm_blc):
        if not HALIDE_AVAILABLE:
            raise ImportError("Halide is not available")
            
        self.img = img
        self.enable = parm_blc["is_enable"]
        self.sensor_info = sensor_info
        self.param_blc = parm_blc
        self.is_linearize = self.param_blc["is_linear"]
        self.is_save = parm_blc["is_save"]
        self.platform = platform
        self.is_debug = parm_blc.get("is_debug", False)
        
        # Initialize Halide pipeline
        self._init_halide_pipeline()

    def _init_halide_pipeline(self):
        """Initialize the Halide pipeline for black level correction"""
        
        def halide_blc_func(input_data, bayer_pattern, r_offset, gr_offset, gb_offset, b_offset,
                           r_sat, gr_sat, gb_sat, b_sat, bpp, is_linearize):
            import halide as hl
            x, y = hl.Var("x"), hl.Var("y")
            input_buf = hl.Buffer(input_data)
            
            # Create Bayer pattern masks
            is_even_row = y % 2 == 0
            is_even_col = x % 2 == 0
            
            # Define offsets and saturation values based on Bayer pattern
            if bayer_pattern == "rggb":
                # RGGB pattern: R at (0,0), G at (0,1), G at (1,0), B at (1,1)
                offset = hl.select(
                    is_even_row & is_even_col, r_offset,      # Red
                    is_even_row & ~is_even_col, gr_offset,    # Green Red
                    ~is_even_row & is_even_col, gb_offset,    # Green Blue
                    b_offset                                  # Blue
                )
                sat = hl.select(
                    is_even_row & is_even_col, r_sat,         # Red
                    is_even_row & ~is_even_col, gr_sat,       # Green Red
                    ~is_even_row & is_even_col, gb_sat,       # Green Blue
                    b_sat                                     # Blue
                )
            elif bayer_pattern == "bggr":
                # BGGR pattern: B at (0,0), G at (0,1), G at (1,0), R at (1,1)
                offset = hl.select(
                    is_even_row & is_even_col, b_offset,      # Blue
                    is_even_row & ~is_even_col, gb_offset,    # Green Blue
                    ~is_even_row & is_even_col, gr_offset,    # Green Red
                    r_offset                                  # Red
                )
                sat = hl.select(
                    is_even_row & is_even_col, b_sat,         # Blue
                    is_even_row & ~is_even_col, gb_sat,       # Green Blue
                    ~is_even_row & is_even_col, gr_sat,       # Green Red
                    r_sat                                     # Red
                )
            elif bayer_pattern == "grbg":
                # GRBG pattern: G at (0,0), R at (0,1), B at (1,0), G at (1,1)
                offset = hl.select(
                    is_even_row & is_even_col, gr_offset,     # Green Red
                    is_even_row & ~is_even_col, r_offset,     # Red
                    ~is_even_row & is_even_col, b_offset,     # Blue
                    gb_offset                                 # Green Blue
                )
                sat = hl.select(
                    is_even_row & is_even_col, gr_sat,        # Green Red
                    is_even_row & ~is_even_col, r_sat,        # Red
                    ~is_even_row & is_even_col, b_sat,        # Blue
                    gb_sat                                    # Green Blue
                )
            elif bayer_pattern == "gbrg":
                # GBRG pattern: G at (0,0), B at (0,1), R at (1,0), G at (1,1)
                offset = hl.select(
                    is_even_row & is_even_col, gb_offset,     # Green Blue
                    is_even_row & ~is_even_col, b_offset,     # Blue
                    ~is_even_row & is_even_col, r_offset,     # Red
                    gr_offset                                 # Green Red
                )
                sat = hl.select(
                    is_even_row & is_even_col, gb_sat,        # Green Blue
                    is_even_row & ~is_even_col, b_sat,        # Blue
                    ~is_even_row & is_even_col, r_sat,        # Red
                    gr_sat                                    # Green Red
                )
            else:
                # Default to RGGB
                offset = hl.select(
                    is_even_row & is_even_col, r_offset,
                    is_even_row & ~is_even_col, gr_offset,
                    ~is_even_row & is_even_col, gb_offset,
                    b_offset
                )
                sat = hl.select(
                    is_even_row & is_even_col, r_sat,
                    is_even_row & ~is_even_col, gr_sat,
                    ~is_even_row & is_even_col, gb_sat,
                    b_sat
                )
            
            # Apply black level correction
            pixel = input_buf[x, y] - offset
            
            # Apply linearization if enabled
            if is_linearize:
                pixel = pixel / (sat - offset) * ((2**bpp) - 1)
            
            # Define the pipeline
            func = hl.Func("black_level_correction")
            func[x, y] = pixel
            
            # Schedule for performance
            func.parallel(y).vectorize(x, 8)
            
            return func
        
        self.halide_blc_func = halide_blc_func

    def apply_blc_parameters(self):
        """
        Apply BLC parameters using Halide optimization
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

        # Create Halide function
        func = self.halide_blc_func(raw, bayer, r_offset, gr_offset, gb_offset, b_offset,
                                  r_sat, gr_sat, gb_sat, b_sat, bpp, self.is_linearize)
        
        # Create output buffer
        output_data = np.empty((height, width), dtype=np.float32)
        
        # Execute Halide pipeline
        func.realize(hl.Buffer(output_data))
        
        raw_blc = np.uint32(np.clip(output_data, 0, (2**bpp) - 1))
        return raw_blc

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_black_level_correction_halide_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute Halide-optimized Black Level Correction
        """
        if self.enable:
            start = time.time()
            blc_out = self.apply_blc_parameters()
            if self.is_debug:
                print(f"  Halide BLC execution time: {time.time() - start:.3f}s")
            self.img = blc_out
        self.save()
        return self.img


# Fallback to original implementation if Halide is not available
class BlackLevelCorrectionHalideFallback:
    """
    Black Level Correction with Halide fallback
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
        
        # Try to use Halide, fallback to original if not available
        if HALIDE_AVAILABLE:
            try:
                self.halide_blc = BlackLevelCorrectionHalide(img, platform, sensor_info, parm_blc)
                self.use_halide = True
                if self.is_debug:
                    print("  Using Halide implementation for BLC")
            except Exception as e:
                if self.is_debug:
                    print(f"  Halide initialization failed, falling back to original implementation: {e}")
                self.use_halide = False
                from modules.black_level_correction.black_level_correction import BlackLevelCorrection
                self.original_blc = BlackLevelCorrection(img, platform, sensor_info, parm_blc)
        else:
            if self.is_debug:
                print("  Halide not available, using original implementation")
            self.use_halide = False
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
            
            if self.use_halide:
                blc_out = self.halide_blc.apply_blc_parameters()
                if self.is_debug:
                    print(f"  Halide BLC execution time: {time.time() - start:.3f}s")
            else:
                blc_out = self.original_blc.apply_blc_parameters()
                if self.is_debug:
                    print(f"  Original BLC execution time: {time.time() - start:.3f}s")
            
            self.img = blc_out
        self.save()
        return self.img 