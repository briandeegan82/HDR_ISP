"""
File: color_space_conversion_fallback.py
Description: Fallback implementation for color space conversion with automatic acceleration selection
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
from util.utils import save_output_array_yuv

# Import implementations with error handling
try:
    from .color_space_conversion_numba import ColorSpaceConversionNumba
    NUMBA_AVAILABLE = True
    print("Color Space Conversion: Numba acceleration available")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Color Space Conversion: Numba acceleration not available")

try:
    from .color_space_conversion_cuda import ColorSpaceConversionCUDA
    CUDA_AVAILABLE = True
    print("Color Space Conversion: CUDA acceleration available")
except ImportError:
    CUDA_AVAILABLE = False
    print("Color Space Conversion: CUDA acceleration not available")

try:
    from .color_space_conversion_gpu import ColorSpaceConversionGPU
    GPU_AVAILABLE = True
    print("Color Space Conversion: GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    print("Color Space Conversion: GPU acceleration not available")

try:
    from .color_space_conversion_opencv_cuda import ColorSpaceConversionOpenCVCUDA
    OPENCV_CUDA_AVAILABLE = True
    print("Color Space Conversion: OpenCV CUDA acceleration available")
except ImportError:
    OPENCV_CUDA_AVAILABLE = False
    print("Color Space Conversion: OpenCV CUDA acceleration not available")

try:
    from .color_space_conversion_gpu import ColorSpaceConversionCuPy
    CUPY_AVAILABLE = True
    print("Color Space Conversion: CuPy acceleration available")
except ImportError:
    CUPY_AVAILABLE = False
    print("Color Space Conversion: CuPy acceleration not available")

from .color_space_conversion import ColorSpaceConversion


class ColorSpaceConversionFallback:
    """
    Fallback implementation for Color Space Conversion with automatic acceleration selection
    Priority: CuPy > OpenCV CUDA > Numba > CUDA > GPU > Original
    """

    def __init__(self, img, platform, sensor_info, parm_csc, parm_cse):
        self.img = img.copy()
        self.is_save = parm_csc["is_save"]
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_csc = parm_csc
        self.bit_depth = sensor_info["output_bit_depth"]
        self.conv_std = self.parm_csc["conv_standard"]
        self.rgb2yuv_mat = None
        self.yuv_img = None
        self.parm_cse = parm_cse
        
        # Select the best available implementation
        self.selected_impl = self._select_implementation()
        self.impl_instance = self.selected_impl(self.img, platform, sensor_info, parm_csc, parm_cse)

    def _select_implementation(self):
        """
        Select the best available implementation based on availability and performance
        Priority: CuPy > OpenCV CUDA > Numba > CUDA > GPU > Original
        """
        if CUPY_AVAILABLE:
            return ColorSpaceConversionCuPy
        elif OPENCV_CUDA_AVAILABLE:
            return ColorSpaceConversionOpenCVCUDA
        elif NUMBA_AVAILABLE:
            return ColorSpaceConversionNumba
        elif CUDA_AVAILABLE:
            return ColorSpaceConversionCUDA
        elif GPU_AVAILABLE:
            return ColorSpaceConversionGPU
        else:
            return ColorSpaceConversion

    def rgb_to_yuv_8bit(self):
        """
        Execute RGB-to-YUV conversion using the selected implementation
        """
        # Handle different method names across implementations
        if self.selected_impl.__name__ == 'ColorSpaceConversionGPU':
            method = getattr(self.impl_instance, 'apply_csc')
        else:
            method = getattr(self.impl_instance, 'rgb_to_yuv_8bit')
        return method()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                f"Out_color_space_conversion_{self.selected_impl.__name__.lower()}_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Execute Color Space Conversion with automatic acceleration
        """
        impl_name = self.selected_impl.__name__.replace('ColorSpaceConversion', '')
        if impl_name == '':
            impl_name = 'Original'
        print(f"Color Space Conversion ({impl_name}) = True")

        start = time.time()
        csc_out = self.rgb_to_yuv_8bit()
        print(f"  Total execution time: {time.time() - start:.3f}s")
        self.img = csc_out
        self.save()
        return self.img 