"""
File: joint_bf.py
Description: Noise reduction in bayer domain uding joint bilateral filter
Code / Paper  Reference:
https://www.researchgate.net/publication/261753644_Green_Channel_Guiding_Denoising_on_Bayer_Image
Author: 10xEngineers
------------------------------------------------------------
"""

import warnings
import numpy as np
from scipy import ndimage
from tqdm import tqdm
from scipy.ndimage import convolve
import cv2
import cv2.ximgproc

def pad_reflect_numba(img, pad_len):
    """Numba-compatible reflect padding implementation"""
    h, w = img.shape
    padded = np.zeros((h + 2*pad_len, w + 2*pad_len), dtype=img.dtype)
    
    # Copy original image
    padded[pad_len:pad_len+h, pad_len:pad_len+w] = img
    
    # Pad top and bottom
    for i in range(pad_len):
        padded[i, pad_len:pad_len+w] = img[pad_len-i-1, :]  # Top
        padded[pad_len+h+i, pad_len:pad_len+w] = img[h-i-1, :]  # Bottom
    
    # Pad left and right
    for j in range(pad_len):
        padded[:, j] = padded[:, 2*pad_len-j-1]  # Left
        padded[:, pad_len+w+j] = padded[:, pad_len+w-j-1]  # Right
    
    return padded

def gauss_kern_raw_numba(kern, std_dev, stride):
    """Numba-accelerated Gaussian kernel generation"""
    if kern % 2 == 0:
        kern = kern + 1
    if kern <= 0:
        kern = 3

    out_kern = np.zeros((kern, kern), dtype=np.float32)
    center = (kern - 1) / 2
    
    for i in range(kern):
        for j in range(kern):
            x = stride * (i - center)
            y = stride * (j - center)
            out_kern[i, j] = np.exp(-1 * (x*x + y*y) / (2 * (std_dev**2)))
    
    out_kern /= np.sum(out_kern)
    return out_kern

def gauss_kern_raw_numpy(kern, std_dev, stride):
    """Pure NumPy Gaussian kernel generation (no Numba)"""
    if kern % 2 == 0:
        kern = kern + 1
    if kern <= 0:
        kern = 3
    center = (kern - 1) / 2
    x = np.arange(kern) - center
    y = np.arange(kern) - center
    xx, yy = np.meshgrid(x, y)
    xx = stride * xx
    yy = stride * yy
    out_kern = np.exp(-1 * (xx**2 + yy**2) / (2 * (std_dev**2)))
    out_kern /= np.sum(out_kern)
    return out_kern

def fast_joint_bilateral_filter_numba(in_img, guide_img, s_kern, stddev_r, spatial_kern):
    """Numba-accelerated joint bilateral filtering with improved performance"""
    height, width = in_img.shape
    pad_len = int((spatial_kern - 1) / 2)
    
    # Pre-compute spatial weights
    spatial_weights = s_kern.astype(np.float32)
    
    # Pre-compute range weight denominator
    range_weight_denom = -0.5 / (stddev_r * stddev_r)
    
    # Pre-allocate output array
    filt_out = np.zeros_like(in_img)
    
    # Pad images once
    in_img_ext = pad_reflect_numba(in_img, pad_len)
    guide_img_ext = pad_reflect_numba(guide_img, pad_len)
    
    # Process in blocks for better cache locality
    block_size = 256  # Increased block size for better cache utilization
    
    for block_i in range(0, height, block_size):
        for block_j in range(0, width, block_size):
            # Calculate block boundaries
            end_i = min(block_i + block_size, height)
            end_j = min(block_j + block_size, width)
            
            # Get block slices
            block_in = in_img_ext[block_i:end_i+spatial_kern-1, block_j:end_j+spatial_kern-1]
            block_guide = guide_img_ext[block_i:end_i+spatial_kern-1, block_j:end_j+spatial_kern-1]
            block_center = guide_img[block_i:end_i, block_j:end_j]
            
            # Process each pixel in the block
            for i in range(end_i - block_i):
                for j in range(end_j - block_j):
                    # Get window slices
                    window_in = block_in[i:i+spatial_kern, j:j+spatial_kern]
                    window_guide = block_guide[i:i+spatial_kern, j:j+spatial_kern]
                    
                    # Calculate range weights (vectorized)
                    diff = block_center[i, j] - window_guide
                    range_weights = np.exp(range_weight_denom * diff * diff)
                    
                    # Calculate weights (vectorized)
                    weights = spatial_weights * range_weights
                    
                    # Calculate weighted sum (vectorized)
                    sum_weight = np.sum(weights)
                    sum_val = np.sum(weights * window_in)
                    
                    # Store result
                    filt_out[block_i + i, block_j + j] = sum_val / (sum_weight + 1e-10)
    
    return filt_out

def interpolate_green_channel_numba(img, bayer_pattern, height, width):
    """Fully vectorized green channel interpolation using 2D convolution"""
    # Convert input to float32 if not already
    img = img.astype(np.float32)
    
    # Pre-compute kernel values and their products for faster computation
    k = np.array([-1, 2, 4, 2, -1], dtype=np.float32) * 0.125
    k_prod = np.outer(k, k)  # Pre-compute outer product for faster convolution

    # Perform 2D convolution (fully vectorized)
    interp_g = convolve(img, k_prod, mode='reflect')
    interp_g = np.clip(interp_g, 0, 1)

    # Initialize output arrays
    interp_g_at_r = np.zeros((height//2, width//2), dtype=np.float32)
    interp_g_at_b = np.zeros((height//2, width//2), dtype=np.float32)

    # Extract interpolated values based on bayer pattern (vectorized)
    if bayer_pattern == "rggb":
        interp_g_at_r = interp_g[0:height:2, 0:width:2]
        interp_g_at_b = interp_g[1:height:2, 1:width:2]
    elif bayer_pattern == "bggr":
        interp_g_at_r = interp_g[1:height:2, 1:width:2]
        interp_g_at_b = interp_g[0:height:2, 0:width:2]
    elif bayer_pattern == "grbg":
        interp_g_at_r = interp_g[0:height:2, 1:width:2]
        interp_g_at_b = interp_g[1:height:2, 0:width:2]
    elif bayer_pattern == "gbrg":
        interp_g_at_r = interp_g[1:height:2, 0:width:2]
        interp_g_at_b = interp_g[0:height:2, 1:width:2]

    return interp_g, interp_g_at_r, interp_g_at_b

def fast_joint_bilateral_filter_opencv(src, guide, d, sigmaColor, sigmaSpace):
    """Fast joint bilateral filter using OpenCV's ximgproc.jointBilateralFilter"""
    src = src.astype(np.float32)
    guide = guide.astype(np.float32)
    # OpenCV expects single-channel images in HxW float32 or uint8
    return cv2.ximgproc.jointBilateralFilter(guide, src, d, sigmaColor, sigmaSpace)

class JointBF:
    """
    Bayer noise reduction using joint bilateral filer technique
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

    def apply_jbf(self):
        """
        Apply bnr to the input image and return the output image
        """
        in_img = self.img
        bayer_pattern = self.sensor_info["bayer_pattern"]
        width, height = self.sensor_info["width"], self.sensor_info["height"]
        bit_depth = self.sensor_info["hdr_bit_depth"]

        # Extract BNR parameters
        filt_size = self.parm_bnr["filter_window"]
        stddev_s_red, stddev_r_red = (
            self.parm_bnr["r_std_dev_s"],
            self.parm_bnr["r_std_dev_r"],
        )
        stddev_s_green, stddev_r_green = (
            self.parm_bnr["g_std_dev_s"],
            self.parm_bnr["g_std_dev_r"],
        )
        stddev_s_blue, stddev_r_blue = (
            self.parm_bnr["b_std_dev_s"],
            self.parm_bnr["b_std_dev_r"],
        )

        # Convert to float32 and normalize
        in_img = np.float32(in_img) / (2**bit_depth - 1)

        # Extract color channels based on bayer pattern
        if bayer_pattern == "rggb":
            in_img_r = in_img[0:height:2, 0:width:2]
            in_img_b = in_img[1:height:2, 1:width:2]
        elif bayer_pattern == "bggr":
            in_img_r = in_img[1:height:2, 1:width:2]
            in_img_b = in_img[0:height:2, 0:width:2]
        elif bayer_pattern == "grbg":
            in_img_r = in_img[0:height:2, 1:width:2]
            in_img_b = in_img[1:height:2, 0:width:2]
        elif bayer_pattern == "gbrg":
            in_img_r = in_img[1:height:2, 0:width:2]
            in_img_b = in_img[0:height:2, 1:width:2]

        # Interpolate green channel using Numba-accelerated function
        interp_g, interp_g_at_r, interp_g_at_b = interpolate_green_channel_numba(
            in_img, bayer_pattern, height, width
        )

        # Generate Gaussian kernels using Numba
        s_kern_r = gauss_kern_raw_numba(filt_size, stddev_s_red, 2)
        s_kern_g = gauss_kern_raw_numba(filt_size, stddev_s_green, 1)
        s_kern_b = gauss_kern_raw_numba(filt_size, stddev_s_blue, 2)

        # Apply fast joint bilateral filter using Numba
        out_img_r = fast_joint_bilateral_filter_numba(in_img_r, interp_g_at_r, s_kern_r, stddev_r_red, filt_size)
        out_img_g = fast_joint_bilateral_filter_numba(interp_g, interp_g, s_kern_g, stddev_r_green, filt_size)
        out_img_b = fast_joint_bilateral_filter_numba(in_img_b, interp_g_at_b, s_kern_b, stddev_r_blue, filt_size)

        # Combine channels back into bayer pattern
        bnr_out_img = np.zeros_like(in_img)
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

        # Convert back to original bit depth
        bnr_out_img = np.uint32(np.clip(bnr_out_img, 0, 1) * ((2**bit_depth) - 1))
        return bnr_out_img

    def gauss_kern_raw(self, kern, std_dev, stride):
        """
        Applying Gaussian Filter
        """
        if kern % 2 == 0:
            warnings.warn("kernel size (kern) cannot be even, setting it as odd value")
            kern = kern + 1

        if kern <= 0:
            warnings.warn("kernel size (kern) cannot be <= zero, setting it as 3")
            kern = 3

        out_kern = np.zeros((kern, kern), dtype=np.float32)

        for i in range(0, kern):
            for j in range(0, kern):
                # stride is used to adjust the gaussian weights for neighbourhood
                # pixel that are 'stride' spaces apart in a bayer image
                out_kern[i, j] = np.exp(
                    -1
                    * (
                        (stride * (i - ((kern - 1) / 2))) ** 2
                        + (stride * (j - ((kern - 1) / 2))) ** 2
                    )
                    / (2 * (std_dev**2))
                )

        sum_kern = np.sum(out_kern)
        out_kern[0:kern:1, 0:kern:1] = out_kern[0:kern:1, 0:kern:1] / sum_kern

        return out_kern

    def joint_bilateral_filter(
        self, in_img, guide_img, spatial_kern, stddev_s, range_kern, stddev_r, stride
    ):
        """
        Applying Joint Bilateral Filter
        """

        # check if filter window sizes spatial_kern and range_kern greater
        # than zero and are odd
        if spatial_kern <= 0:
            spatial_kern = 3
            warnings.warn(
                "spatial kernel size (spatial_kern) cannot be <= zero, setting it as 3"
            )
        elif spatial_kern % 2 == 0:
            warnings.warn(
                "range kernel size (spatial_kern) cannot be even, "
                "assigning it an odd value"
            )
            spatial_kern = spatial_kern + 1

        # check if range_kern > spatial_kern
        if range_kern > spatial_kern:
            warnings.warn(
                "range kernel size (range_kern) cannot be more "
                "than spatial kernel size (spatial_kern)"
            )
            range_kern = spatial_kern

        # spawn a NxN gaussian kernel
        s_kern = self.gauss_kern_raw(spatial_kern, stddev_s, stride)

        # pad the image with half arm length of the kernel;
        # padType='constant' => pad value = 0; 'reflect' is more suitable
        pad_len = int((spatial_kern - 1) / 2)
        kern_arm = pad_len
        in_img_ext = np.pad(in_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect")
        guide_img_ext = np.pad(
            guide_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect"
        )

        filt_out = np.zeros(in_img.shape, dtype=np.float32)

        for i in tqdm(
            range(kern_arm, np.size(in_img, 0) + kern_arm),
            disable=self.is_progress,
            leave=self.is_leave,
        ):
            for j in range(kern_arm, np.size(in_img, 1) + kern_arm):
                guide_img_ext_center_pix = guide_img_ext[i, j]
                guide_img_ext_filt_window = guide_img_ext[
                    i - kern_arm : i + kern_arm + 1, j - kern_arm : j + kern_arm + 1
                ]
                in_img_ext_filt_window = in_img_ext[
                    i - kern_arm : i + kern_arm + 1, j - kern_arm : j + kern_arm + 1
                ]

                # normalization fact of a filter window = sum(matrix multiplication of
                # spatial kernel and range kernel weights) = sum of filter weights
                norm_fact = np.sum(
                    s_kern[0:spatial_kern, 0:spatial_kern]
                    * np.exp(
                        -1
                        * (guide_img_ext_center_pix - guide_img_ext_filt_window) ** 2
                        / (2 * stddev_r**2)
                    )
                )

                # filter output for a window = sum(spatial kernel weights x range kernel weights x
                # windowed input image) / normalization factor
                filt_out[i - kern_arm, j - kern_arm] = np.sum(
                    s_kern[0:spatial_kern, 0:spatial_kern]
                    * np.exp(
                        -1
                        * (guide_img_ext_center_pix - guide_img_ext_filt_window) ** 2
                        / (2 * stddev_r**2)
                    )
                    * in_img_ext_filt_window
                )
                filt_out[i - kern_arm, j - kern_arm] = (
                    filt_out[i - kern_arm, j - kern_arm] / norm_fact
                )

        return filt_out

    def fast_joint_bilateral_filter(
        self, in_img, guide_img, spatial_kern, stddev_s, range_kern, stddev_r, stride
    ):
        """
        Applying Joint Bilateral Filter
        """

        # check if filter window sizes spatial_kern and range_kern greater than zero and are odd
        if spatial_kern <= 0:
            spatial_kern = 3
            warnings.warn(
                "spatial kernel size (spatial_kern) cannot be <= zero, setting it as 3"
            )
        elif spatial_kern % 2 == 0:
            warnings.warn(
                "range kernel size (spatial_kern) cannot be even, assigning it an odd value"
            )
            spatial_kern = spatial_kern + 1

        if range_kern <= 0:
            range_kern = 3
            warnings.warn(
                "range kernel size (range_kern) cannot be <= zero, setting it as 3"
            )
        elif range_kern % 2 == 0:
            warnings.warn(
                "range kernel size (range_kern) cannot be even, assigning it an odd value"
            )
            range_kern = range_kern + 1

        # check if range_kern > spatial_kern
        if range_kern > spatial_kern:
            warnings.warn(
                "range kernel size (range_kern) cannot be more than..."
                "spatial kernel size (spatial_kern)"
            )
            range_kern = spatial_kern

        # spawn a NxN gaussian kernel
        s_kern = self.gauss_kern_raw(spatial_kern, stddev_s, stride)

        # pad the image with half arm length of the kernel;
        # padType='constant' => pad value = 0; 'reflect' is more suitable
        pad_len = int((spatial_kern - 1) / 2)
        in_img_ext = np.pad(in_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect")
        guide_img_ext = np.pad(
            guide_img, ((pad_len, pad_len), (pad_len, pad_len)), "reflect"
        )

        filt_out = np.zeros(in_img.shape, dtype=np.float32)
        norm_fact = np.zeros(in_img.shape)
        sum_filt_out = np.zeros(in_img.shape)

        for i in range(spatial_kern):
            for j in range(spatial_kern):
                # Creating shifted arrays for processing each pixel in the window
                in_img_ext_array = in_img_ext[
                    i : i + in_img.shape[0], j : j + in_img.shape[1], ...
                ]
                guide_img_ext_array = guide_img_ext[
                    i : i + in_img.shape[0], j : j + in_img.shape[1], ...
                ]

                # Adding normalization factor for each pixel needed to average out the
                # final result
                norm_fact += s_kern[i, j] * np.exp(
                    -1 * (guide_img - guide_img_ext_array) ** 2 / (2 * stddev_r**2)
                )

                # Summing up the final result
                sum_filt_out += (
                    s_kern[i, j]
                    * np.exp(
                        -1
                        * (guide_img - guide_img_ext_array) ** 2
                        / (2 * stddev_r**2)
                    )
                    * in_img_ext_array
                )

        filt_out = sum_filt_out / norm_fact

        return filt_out
