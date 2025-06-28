"""
File: sharpen_gpu.py
Description: GPU-accelerated sharpening for Infinite-ISP.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
"""

import time
import numpy as np
from util.gpu_utils import gpu_accelerator
from util.utils import save_output_array_yuv

class UnsharpMaskingGPU:
    """
    GPU-accelerated Unsharp Masking Algorithm
    """

    def __init__(self, img, sharpen_sigma, sharpen_strength):
        self.img = img
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_strength = sharpen_strength

    def apply_sharpen(self):
        """
        Applying GPU-accelerated sharpening to the input image
        """
        luma = np.float32(self.img[:, :, 0])

        # Filter the luma component of the image with a GPU-accelerated Gaussian LPF
        # Smoothing magnitude can be controlled with the sharpen_sigma parameter
        ksize = (int(self.sharpen_sigma * 6 + 1), int(self.sharpen_sigma * 6 + 1))
        if ksize[0] % 2 == 0:
            ksize = (ksize[0] + 1, ksize[1] + 1)
        
        smoothened = gpu_accelerator.gaussian_filter_gpu(luma.astype(np.float32), ksize, self.sharpen_sigma)
        
        # Sharpen the image with unsharp mask
        # Strength is tuneable with the sharpen_strength parameter
        sharpened = luma + ((luma - smoothened) * self.sharpen_strength)

        if self.img.dtype == "float32":
            self.img[:, :, 0] = np.clip(sharpened, 0, 1)
        else:
            self.img[:, :, 0] = np.uint8(np.clip(sharpened, 0, 255))
        return self.img

class SharpeningGPU:
    """
    GPU-accelerated Sharpening
    """

    def __init__(self, img, platform, sensor_info, parm_sha, conv_std):
        self.img = img
        self.enable = parm_sha["is_enable"]
        self.sensor_info = sensor_info
        self.parm_sha = parm_sha
        self.is_save = parm_sha["is_save"]
        self.platform = platform
        self.conv_std = conv_std

    def apply_unsharp_masking(self):
        """
        Apply function for GPU-accelerated Sharpening Algorithm - Unsharp Masking
        """
        usm = UnsharpMaskingGPU(
            self.img, self.parm_sha["sharpen_sigma"], self.parm_sha["sharpen_strength"]
        )
        return usm.apply_sharpen()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_Sharpening_gpu_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Applying GPU-accelerated sharpening to input image
        """
        if self.enable is True:
            start = time.time()
            s_out = self.apply_unsharp_masking()
            print(f"GPU Sharpening execution time: {time.time() - start:.3f}s")
            self.img = s_out

        self.save()
        return self.img 