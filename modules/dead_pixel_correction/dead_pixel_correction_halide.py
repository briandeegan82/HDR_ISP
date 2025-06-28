"""
File: dead_pixel_correction_halide.py
Description: Corrects the hot or dead pixels using Halide
Code / Paper  Reference: https://ieeexplore.ieee.org/document/9194921
Implementation inspired from: (OpenISP) https://github.com/cruxopen/openISP
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array
from modules.dead_pixel_correction.dynamic_dpc import DynamicDPC as DynDPC

# Try to import Halide, but provide fallback if not available
try:
    import halide as hl
    HALIDE_AVAILABLE = True
except ImportError:
    HALIDE_AVAILABLE = False
    hl = None


class DeadPixelCorrectionHalide:
    "Dead Pixel Correction using Halide"

    def __init__(self, img, sensor_info, parm_dpc, platform):
        if not HALIDE_AVAILABLE:
            raise ImportError("Halide is not available")
            
        self.img = img
        self.enable = parm_dpc["is_enable"]
        self.sensor_info = sensor_info
        self.parm_dpc = parm_dpc
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.bpp = self.sensor_info["bit_depth"]
        self.threshold = self.parm_dpc["dp_threshold"]
        self.is_debug = self.parm_dpc["is_debug"]
        self.is_save = parm_dpc["is_save"]
        self.platform = platform
        
        # Initialize Halide pipeline
        self._init_halide_pipeline()

    def _init_halide_pipeline(self):
        """Initialize the Halide pipeline for dead pixel correction"""
        def halide_dpc_func(input_data, threshold):
            import halide as hl
            x, y = hl.Var("x"), hl.Var("y")
            input_buf = hl.Buffer(input_data)
            # Mirror boundary condition for safe access
            bounded = hl.BoundaryConditions.mirror_interior(input_buf)
            center = bounded[x, y]
            n0 = bounded[x-2, y-2]
            n1 = bounded[x, y-2]
            n2 = bounded[x+2, y-2]
            n3 = bounded[x-2, y]
            n4 = bounded[x+2, y]
            n5 = bounded[x-2, y+2]
            n6 = bounded[x, y+2]
            n7 = bounded[x+2, y+2]
            min1 = hl.min(n0, n1)
            min2 = hl.min(n2, n3)
            min3 = hl.min(n4, n5)
            min4 = hl.min(n6, n7)
            min5 = hl.min(min1, min2)
            min6 = hl.min(min3, min4)
            min_neighbor = hl.min(min5, min6)
            max1 = hl.max(n0, n1)
            max2 = hl.max(n2, n3)
            max3 = hl.max(n4, n5)
            max4 = hl.max(n6, n7)
            max5 = hl.max(max1, max2)
            max6 = hl.max(max3, max4)
            max_neighbor = hl.max(max5, max6)
            cond1 = (center < min_neighbor) | (center > max_neighbor)
            diff0 = hl.abs(center - n0)
            diff1 = hl.abs(center - n1)
            diff2 = hl.abs(center - n2)
            diff3 = hl.abs(center - n3)
            diff4 = hl.abs(center - n4)
            diff5 = hl.abs(center - n5)
            diff6 = hl.abs(center - n6)
            diff7 = hl.abs(center - n7)
            cond2 = (diff0 > threshold) & (diff1 > threshold) & (diff2 > threshold) & \
                    (diff3 > threshold) & (diff4 > threshold) & (diff5 > threshold) & \
                    (diff6 > threshold) & (diff7 > threshold)
            is_dead_pixel = cond1 & cond2
            grad_v = hl.abs(2 * center - bounded[x, y-2] - bounded[x, y+2])
            grad_h = hl.abs(2 * center - bounded[x-2, y] - bounded[x+2, y])
            grad_ld = hl.abs(2 * center - bounded[x-2, y-2] - bounded[x+2, y+2])
            grad_rd = hl.abs(2 * center - bounded[x+2, y-2] - bounded[x-2, y+2])
            min_grad1 = hl.min(grad_v, grad_h)
            min_grad2 = hl.min(grad_ld, grad_rd)
            min_grad = hl.min(min_grad1, min_grad2)
            corr_v = (bounded[x, y-2] + bounded[x, y+2]) / 2.0
            corr_h = (bounded[x-2, y] + bounded[x+2, y]) / 2.0
            corr_ld = (bounded[x-2, y-2] + bounded[x+2, y+2]) / 2.0
            corr_rd = (bounded[x+2, y-2] + bounded[x-2, y+2]) / 2.0
            corrected_value = hl.select(
                min_grad == grad_v, corr_v,
                min_grad == grad_h, corr_h,
                min_grad == grad_ld, corr_ld,
                corr_rd
            )
            output = hl.select(is_dead_pixel, corrected_value, center)
            func = hl.Func("dead_pixel_correction")
            func[x, y] = output
            func.parallel(y).vectorize(x, 8)
            return func
        self.halide_dpc_func = halide_dpc_func

    def apply_halide_dpc(self):
        """Apply DPC using Halide"""
        input_data = np.asarray(self.img, dtype=np.float32)
        height, width = input_data.shape
        output_data = np.empty((height, width), dtype=np.float32)
        func = self.halide_dpc_func(input_data, float(self.threshold))
        func.realize(hl.Buffer(output_data))
        return output_data

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_dead_pixel_correction_halide_",
                self.platform,
                self.bpp,
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute DPC Module using Halide"""

        if self.enable:
            start = time.time()
            self.img = np.float32(self.img)
            dpc_out = self.apply_halide_dpc()
            if self.is_debug:
                print(f"  Halide DPC execution time: {time.time() - start:.3f}s")
            self.img = dpc_out

        self.save()
        return self.img


# Fallback to original implementation if Halide is not available
class DeadPixelCorrectionHalideFallback:
    "Dead Pixel Correction with Halide fallback"

    def __init__(self, img, sensor_info, parm_dpc, platform):
        self.img = img
        self.enable = parm_dpc["is_enable"]
        self.sensor_info = sensor_info
        self.parm_dpc = parm_dpc
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.bpp = self.sensor_info["bit_depth"]
        self.threshold = self.parm_dpc["dp_threshold"]
        self.is_debug = self.parm_dpc["is_debug"]
        self.is_save = parm_dpc["is_save"]
        self.platform = platform
        
        # Try to use Halide, fallback to original if not available
        if HALIDE_AVAILABLE:
            try:
                self.halide_dpc = DeadPixelCorrectionHalide(img, sensor_info, parm_dpc, platform)
                self.use_halide = True
                if self.is_debug:
                    print("  Using Halide implementation for DPC")
            except Exception as e:
                if self.is_debug:
                    print(f"  Halide initialization failed, falling back to original implementation: {e}")
                self.use_halide = False
                self.original_dpc = DynDPC(img, sensor_info, parm_dpc)
        else:
            if self.is_debug:
                print("  Halide not available, using original implementation")
            self.use_halide = False
            self.original_dpc = DynDPC(img, sensor_info, parm_dpc)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_dead_pixel_correction_",
                self.platform,
                self.bpp,
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute DPC Module with automatic fallback"""

        if self.enable:
            start = time.time()
            self.img = np.float32(self.img)
            
            if self.use_halide:
                dpc_out = self.halide_dpc.apply_halide_dpc()
                if self.is_debug:
                    print(f"  Halide DPC execution time: {time.time() - start:.3f}s")
            else:
                dpc_out = self.original_dpc.dynamic_dpc()
                if self.is_debug:
                    print(f"  Original DPC execution time: {time.time() - start:.3f}s")
            
            self.img = dpc_out

        self.save()
        return self.img 