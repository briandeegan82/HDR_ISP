"""
File: color_space_conversion_opencv_cuda.py
Description: OpenCV CUDA-optimized color space conversion
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
import cv2
from util.utils import save_output_array_yuv

# Import OpenCV CUDA modules
try:
    import cv2.cuda
    OPENCV_CUDA_AVAILABLE = True
except ImportError:
    OPENCV_CUDA_AVAILABLE = False


def rgb_to_yuv_opencv_cuda(img, conv_std, saturation_gain, bit_depth):
    """
    OpenCV CUDA-optimized RGB-to-YCrCb conversion using built-in cvtColor
    """
    # Convert to float32 for processing
    img_float = img.astype(np.float32)
    
    if OPENCV_CUDA_AVAILABLE and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # Upload to GPU
        gpu_img = cv2.cuda.GpuMat()
        gpu_img.upload(img_float)
        
        # Convert RGB to YCrCb using OpenCV CUDA's built-in function
        gpu_yuv = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2YCrCb)
        
        # Download from GPU
        yuv_img = gpu_yuv.download()
        
        # Apply saturation gain to Cr and Cb channels
        if saturation_gain != 1.0:
            yuv_img[:, :, 1] *= saturation_gain  # Cr channel
            yuv_img[:, :, 2] *= saturation_gain  # Cb channel
        
        # Apply bit depth scaling and offsets (same as original)
        y_offset = 2 ** (bit_depth / 2)
        uv_offset = 2 ** (bit_depth - 1)
        max_val = (2 ** bit_depth) - 1
        final_scale = 2 ** (bit_depth - 8)
        
        # Scale and apply offsets
        yuv_img[:, :, 0] = yuv_img[:, :, 0] * 255 + y_offset
        yuv_img[:, :, 1] = yuv_img[:, :, 1] * 255 + uv_offset
        yuv_img[:, :, 2] = yuv_img[:, :, 2] * 255 + uv_offset
        
        # Clip to valid range
        yuv_img = np.clip(yuv_img, 0, max_val)
        
        # Final scaling
        yuv_img = np.round(yuv_img / final_scale)
        yuv_img = np.clip(yuv_img, 0, 255)
        
        return yuv_img.astype(np.uint8)
    else:
        # Fallback to CPU if CUDA not available
        yuv_img = cv2.cvtColor(img_float, cv2.COLOR_RGB2YCrCb)
        
        # Apply saturation gain to Cr and Cb channels
        if saturation_gain != 1.0:
            yuv_img[:, :, 1] *= saturation_gain  # Cr channel
            yuv_img[:, :, 2] *= saturation_gain  # Cb channel
        
        # Apply the same processing as above
        y_offset = 2 ** (bit_depth / 2)
        uv_offset = 2 ** (bit_depth - 1)
        max_val = (2 ** bit_depth) - 1
        final_scale = 2 ** (bit_depth - 8)
        
        yuv_img[:, :, 0] = yuv_img[:, :, 0] * 255 + y_offset
        yuv_img[:, :, 1] = yuv_img[:, :, 1] * 255 + uv_offset
        yuv_img[:, :, 2] = yuv_img[:, :, 2] * 255 + uv_offset
        
        yuv_img = np.clip(yuv_img, 0, max_val)
        yuv_img = np.round(yuv_img / final_scale)
        yuv_img = np.clip(yuv_img, 0, 255)
        
        return yuv_img.astype(np.uint8)


class ColorSpaceConversionOpenCVCUDA:
    """
    OpenCV CUDA-optimized Color Space Conversion
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

    def rgb_to_yuv_8bit(self):
        """
        OpenCV CUDA-optimized RGB-to-YUV Colorspace conversion 8bit
        """
        total_start = time.time()

        start = time.time()
        saturation_gain = self.parm_cse['saturation_gain'] if self.parm_cse['is_enable'] else 1.0
        
        # Use OpenCV CUDA-optimized conversion
        self.img = rgb_to_yuv_opencv_cuda(self.img, self.conv_std, saturation_gain, self.bit_depth)
        
        #print(f"  OpenCV CUDA RGB to YUV conversion time: {time.time() - start:.3f}s")
        #print(f"  Total RGB to YUV conversion time: {time.time() - total_start:.3f}s")
        return self.img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_color_space_conversion_opencv_cuda_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Execute OpenCV CUDA-optimized Color Space Conversion
        """
        #print("Color Space Conversion (OpenCV CUDA) = True")

        start = time.time()
        csc_out = self.rgb_to_yuv_8bit()
        #print(f"  Total execution time: {time.time() - start:.3f}s")
        self.img = csc_out
        self.save()
        return self.img 