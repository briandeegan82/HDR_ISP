"""
File: scale_gpu.py
Description: GPU-accelerated scaling implementation
Code / Paper  Reference:
https://patentimages.storage.googleapis.com/f9/11/65/a2b66f52c6dbd4/US8538199.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import re
import numpy as np
import cv2
from util.utils import crop
from util.utils import save_output_array_yuv, save_output_array
from util.gpu_utils import gpu_accelerator

class BilinearInterpolationGPU:
    """GPU-accelerated Bilinear Interpolation for scaling."""

    def __init__(self, img, scale_factor):
        self.img = img
        self.scale_factor = scale_factor

    def apply_bilinear_interpolation(self):
        """
        Apply GPU-accelerated bilinear interpolation
        """
        height, width = self.img.shape[:2]
        new_height = int(height * self.scale_factor)
        new_width = int(width * self.scale_factor)
        
        # Use GPU-accelerated resizing
        scaled_img = gpu_accelerator.resize_gpu(
            self.img.astype(np.float32), 
            (new_width, new_height), 
            cv2.INTER_LINEAR
        )
        
        return scaled_img

class NearestNeighborGPU:
    """GPU-accelerated Nearest Neighbor Interpolation for scaling."""

    def __init__(self, img, scale_factor):
        self.img = img
        self.scale_factor = scale_factor

    def apply_nearest_neighbor(self):
        """
        Apply GPU-accelerated nearest neighbor interpolation
        """
        height, width = self.img.shape[:2]
        new_height = int(height * self.scale_factor)
        new_width = int(width * self.scale_factor)
        
        # Use GPU-accelerated resizing
        scaled_img = gpu_accelerator.resize_gpu(
            self.img.astype(np.float32), 
            (new_width, new_height), 
            cv2.INTER_NEAREST
        )
        
        return scaled_img

class ScaleGPU:
    """GPU-accelerated Scale color image to given size."""

    def __init__(self, img, platform, sensor_info, parm_sca, conv_std):
        self.img = img
        self.enable = parm_sca["is_enable"]
        self.sensor_info = sensor_info
        self.parm_sca = parm_sca
        self.is_save = parm_sca["is_save"]
        self.platform = platform
        self.conv_std = conv_std
        self.get_scaling_params()

    def get_scaling_params(self):
        """
        Extract scaling parameters from configuration
        """
        scale_factor_str = self.parm_sca["scale_factor"]
        
        # Parse scale factor (e.g., "1/2", "2/1", "1.5")
        if "/" in scale_factor_str:
            num, den = map(float, scale_factor_str.split("/"))
            self.scale_factor = num / den
        else:
            self.scale_factor = float(scale_factor_str)
        
        self.interpolation_method = self.parm_sca.get("interpolation_method", "bilinear")

    def apply_scaling(self):
        """
        Apply GPU-accelerated scaling based on interpolation method
        """
        if self.interpolation_method.lower() == "nearest":
            scaler = NearestNeighborGPU(self.img, self.scale_factor)
            return scaler.apply_nearest_neighbor()
        else:
            # Default to bilinear interpolation
            scaler = BilinearInterpolationGPU(self.img, self.scale_factor)
            return scaler.apply_bilinear_interpolation()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            if self.conv_std == "yuv":
                save_output_array_yuv(
                    self.platform["in_file"],
                    self.img,
                    "Out_scale_gpu_",
                    self.platform,
                    self.conv_std,
                )
            else:
                save_output_array(
                    self.platform["in_file"],
                    self.img,
                    "Out_scale_gpu_",
                    self.platform,
                    self.sensor_info["bit_depth"],
                    self.sensor_info["bayer_pattern"],
                )

    def execute(self):
        """
        Applying GPU-accelerated scaling to input image
        """
        if self.enable is True:
            start = time.time()
            scaled_img = self.apply_scaling()
            print(f"GPU Scaling execution time: {time.time() - start:.3f}s")
            self.img = scaled_img

        self.save()
        return self.img 