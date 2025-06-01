"""
File: bayer_noise_reduction.py
Description: Noise reduction in bayer domain
Author: 10xEngineers
------------------------------------------------------------
"""

import time
import numpy as np
from numba import jit, prange
from modules.bayer_noise_reduction.joint_bf import JointBF as JBF, interpolate_green_channel_numba, gauss_kern_raw_numpy, fast_joint_bilateral_filter_opencv
from util.utils import save_output_array
from scipy.ndimage import grey_dilation, gaussian_filter
import cv2

def extract_channels_numpy(img, bayer_pattern, height, width):
    """Pure NumPy channel extraction (no Numba)"""
    img = img.astype(np.float32)
    if bayer_pattern == "rggb":
        in_img_r = img[0:height:2, 0:width:2].copy()
        in_img_b = img[1:height:2, 1:width:2].copy()
    elif bayer_pattern == "bggr":
        in_img_r = img[1:height:2, 1:width:2].copy()
        in_img_b = img[0:height:2, 0:width:2].copy()
    elif bayer_pattern == "grbg":
        in_img_r = img[0:height:2, 1:width:2].copy()
        in_img_b = img[1:height:2, 0:width:2].copy()
    elif bayer_pattern == "gbrg":
        in_img_r = img[1:height:2, 0:width:2].copy()
        in_img_b = img[0:height:2, 1:width:2].copy()
    else:
        raise ValueError(f"Unknown bayer pattern: {bayer_pattern}")
    return in_img_r, in_img_b

def combine_channels_numpy(out_img_r, out_img_g, out_img_b, bayer_pattern, height, width):
    """Pure NumPy channel combination (no Numba)"""
    out_img_r = out_img_r.astype(np.float32)
    out_img_g = out_img_g.astype(np.float32)
    out_img_b = out_img_b.astype(np.float32)
    bnr_out_img = out_img_g.copy()
    if bayer_pattern == "rggb":
        bnr_out_img[0:height:2, 0:width:2] = out_img_r
        bnr_out_img[1:height:2, 1:width:2] = out_img_b
    elif bayer_pattern == "bggr":
        bnr_out_img[1:height:2, 1:width:2] = out_img_r
        bnr_out_img[0:height:2, 0:width:2] = out_img_b
    elif bayer_pattern == "grbg":
        bnr_out_img[0:height:2, 1:width:2] = out_img_r
        bnr_out_img[1:height:2, 0:width:2] = out_img_b
    elif bayer_pattern == "gbrg":
        bnr_out_img[1:height:2, 0:width:2] = out_img_r
        bnr_out_img[0:height:2, 1:width:2] = out_img_b
    else:
        raise ValueError(f"Unknown bayer pattern: {bayer_pattern}")
    return bnr_out_img

class BayerNoiseReduction:
    """
    Noise Reduction in Bayer domain
    """

    def __init__(self, img, sensor_info, parm_bnr, platform):
        self.img = img
        self.enable = parm_bnr["is_enable"]
        self.sensor_info = sensor_info
        self.parm_bnr = parm_bnr
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.is_save = parm_bnr["is_save"]
        self.platform = platform

    def apply_bnr(self):
        """
        Apply bnr to the input image and return the output image
        """
        # Extract parameters
        bayer_pattern = self.sensor_info["bayer_pattern"]
        width, height = self.sensor_info["width"], self.sensor_info["height"]
        bit_depth = self.sensor_info["hdr_bit_depth"]
        
        # Print raw R, G, B pixel stats before normalization
        raw_img = np.float32(self.img)
        g_mask = np.zeros_like(raw_img)
        r_mask = np.zeros_like(raw_img)
        b_mask = np.zeros_like(raw_img)
        if bayer_pattern == "rggb":
            r_mask[0:height:2, 0:width:2] = 1
            g_mask[0:height:2, 1:width:2] = 1
            g_mask[1:height:2, 0:width:2] = 1
            b_mask[1:height:2, 1:width:2] = 1
        elif bayer_pattern == "bggr":
            b_mask[0:height:2, 0:width:2] = 1
            g_mask[0:height:2, 1:width:2] = 1
            g_mask[1:height:2, 0:width:2] = 1
            r_mask[1:height:2, 1:width:2] = 1
        elif bayer_pattern == "grbg":
            g_mask[0:height:2, 0:width:2] = 1
            r_mask[0:height:2, 1:width:2] = 1
            b_mask[1:height:2, 0:width:2] = 1
            g_mask[1:height:2, 1:width:2] = 1
        elif bayer_pattern == "gbrg":
            g_mask[0:height:2, 0:width:2] = 1
            b_mask[0:height:2, 1:width:2] = 1
            r_mask[1:height:2, 0:width:2] = 1
            g_mask[1:height:2, 1:width:2] = 1
        else:
            raise ValueError(f"Unknown bayer pattern: {bayer_pattern}")
        print(f"Raw R pixels: mean={raw_img[r_mask==1].mean():.4f}, min={raw_img[r_mask==1].min():.4f}, max={raw_img[r_mask==1].max():.4f}")
        print(f"Raw G pixels: mean={raw_img[g_mask==1].mean():.4f}, min={raw_img[g_mask==1].min():.4f}, max={raw_img[g_mask==1].max():.4f}")
        print(f"Raw B pixels: mean={raw_img[b_mask==1].mean():.4f}, min={raw_img[b_mask==1].min():.4f}, max={raw_img[b_mask==1].max():.4f}")
        
        # Convert to float32 and normalize only if needed
        in_img = np.float32(self.img)
        if in_img.max() > 1.0:
            in_img = in_img / (2**bit_depth - 1)
            print("Input image normalized to [0, 1] range.")
        else:
            print("Input image already in [0, 1] range. No normalization applied.")
        
        # Extract channels using Numba-accelerated function
        start = time.time()
        in_img_r, in_img_b = extract_channels_numpy(in_img, bayer_pattern, height, width)
        
        # Define the G interpolation kernel (5x5)
        interp_kern_g = np.array([
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0]
        ], dtype=np.float32) / 8.0
        
        # Interpolate green channel using the kernel
        interp_g = cv2.filter2D(in_img, -1, interp_kern_g)
        interp_g = np.clip(interp_g, 0, 1)
        
        # Extract guide image at R and B positions to match source image sizes
        if bayer_pattern == "rggb":
            guide_r = interp_g[0:height:2, 0:width:2]
            guide_b = interp_g[1:height:2, 1:width:2]
        elif bayer_pattern == "bggr":
            guide_r = interp_g[1:height:2, 1:width:2]
            guide_b = interp_g[0:height:2, 0:width:2]
        elif bayer_pattern == "grbg":
            guide_r = interp_g[0:height:2, 1:width:2]
            guide_b = interp_g[1:height:2, 0:width:2]
        elif bayer_pattern == "gbrg":
            guide_r = interp_g[1:height:2, 0:width:2]
            guide_b = interp_g[0:height:2, 1:width:2]
        
        # Use different filter sizes for different channels
        filt_size_g = self.parm_bnr["filter_window"]
        filt_size_r = int((self.parm_bnr["filter_window"] + 1) / 2)
        filt_size_b = int((self.parm_bnr["filter_window"] + 1) / 2)
        
        # Apply joint bilateral filter with original parameters
        out_img_r = fast_joint_bilateral_filter_opencv(
            in_img_r, guide_r, filt_size_r, self.parm_bnr["r_std_dev_r"], self.parm_bnr["r_std_dev_s"]
        )
        out_img_g = fast_joint_bilateral_filter_opencv(
            interp_g, interp_g, filt_size_g, self.parm_bnr["g_std_dev_r"], self.parm_bnr["g_std_dev_s"]
        )
        out_img_b = fast_joint_bilateral_filter_opencv(
            in_img_b, guide_b, filt_size_b, self.parm_bnr["b_std_dev_r"], self.parm_bnr["b_std_dev_s"]
        )
        
        # Combine channels
        bnr_out_img = combine_channels_numpy(out_img_r, out_img_g, out_img_b, bayer_pattern, height, width)
        
        # Convert back to original bit depth
        bnr_out_img = np.uint32(np.clip(bnr_out_img, 0, 1) * ((2**bit_depth) - 1))
        return bnr_out_img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_bayer_noise_reduction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Appling BNR to input RAW image and returns the output image
        """
        print("Bayer Noise Reduction = " + str(self.enable))

        if self.enable is True:
            start = time.time()
            bnr_out = self.apply_bnr()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = bnr_out

        self.save()
        return self.img
