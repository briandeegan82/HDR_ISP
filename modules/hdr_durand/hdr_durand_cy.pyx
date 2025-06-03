"""
File: hdr_durand_cy.pyx
Description: Cython-optimized implementation of HDR Durand tone mapping
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import numpy as np
cimport numpy as np
from libc.math cimport exp, log10, pow
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import zoom
from util.utils import save_output_array
import time

# Define numpy data types
np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def bilateral_filter_cy(np.ndarray[DTYPE_t, ndim=2] image, 
                       DTYPE_t sigma_color, 
                       DTYPE_t sigma_space):
    """
    Cython-optimized bilateral filter implementation.
    """
    cdef:
        int height = image.shape[0]
        int width = image.shape[1]
        np.ndarray[DTYPE_t, ndim=2] output = np.zeros((height, width), dtype=DTYPE)
        np.ndarray[DTYPE_t, ndim=2] spatial_filtered = gaussian_filter(image, sigma=sigma_space, mode='reflect', truncate=4.0)
        np.ndarray[DTYPE_t, ndim=2] intensity_diff = image - spatial_filtered
        np.ndarray[DTYPE_t, ndim=2] range_kernel = np.exp(-0.5 * (intensity_diff / sigma_color) ** 2)
    
    output = spatial_filtered + range_kernel * intensity_diff
    return output

def apply_tone_mapping_cy(np.ndarray[DTYPE_t, ndim=2] img,
                         DTYPE_t sigma_space,
                         DTYPE_t sigma_color,
                         DTYPE_t contrast_factor,
                         int downsample_factor):
    """
    Cython-optimized tone mapping implementation.
    """
    cdef:
        DTYPE_t epsilon = 1e-6
        np.ndarray[DTYPE_t, ndim=2] log_luminance
        np.ndarray[DTYPE_t, ndim=2] log_base
        np.ndarray[DTYPE_t, ndim=2] log_detail
        np.ndarray[DTYPE_t, ndim=2] compressed_log_base
        np.ndarray[DTYPE_t, ndim=2] log_output
        np.ndarray[DTYPE_t, ndim=2] output_luminance
    
    # Convert to log domain
    log_luminance = np.log10(img + epsilon)
    print("log_luminance:", log_luminance.min(), log_luminance.max(), log_luminance.mean(), log_luminance.std())
    
    # Apply bilateral filter directly (no downsampling/upsampling)
    log_base = bilateral_filter_cy(log_luminance.astype(np.float64), sigma_color, sigma_space)
    print("log_base:", log_base.min(), log_base.max(), log_base.mean(), log_base.std())
    
    # Extract detail layer
    log_detail = log_luminance - log_base
    print("log_detail:", log_detail.min(), log_detail.max(), log_detail.mean(), log_detail.std())
    
    # Compress base layer
    compressed_log_base = log_base / contrast_factor
    print("compressed_log_base:", compressed_log_base.min(), compressed_log_base.max(), compressed_log_base.mean(), compressed_log_base.std())
    
    # Recombine layers
    log_output = compressed_log_base + log_detail
    print("log_output:", log_output.min(), log_output.max(), log_output.mean(), log_output.std())
    
    # Convert back from log domain
    output_luminance = np.power(10, log_output)
    print("output_luminance:", output_luminance.min(), output_luminance.max(), output_luminance.mean(), output_luminance.std())
    
    # Normalize
    output_luminance = (output_luminance - np.min(output_luminance)) / (np.max(output_luminance) - np.min(output_luminance))
    print("normalized output_luminance:", output_luminance.min(), output_luminance.max(), output_luminance.mean(), output_luminance.std())
    
    return output_luminance

class HDRDurandToneMapping:
    """
    HDR Durand Tone Mapping Algorithm Implementation with Cython optimizations
    """
    
    def __init__(self, img, platform, sensor_info, params):
        self.img = img.copy()
        self.is_enable = params.get("is_enable", True)
        self.is_save = params.get("is_save", False)
        self.is_debug = params.get("is_debug", False)
        self.sigma_space = params.get("sigma_space", 2.0)
        self.sigma_color = params.get("sigma_color", 0.4)
        self.contrast_factor = params.get("contrast_factor", 2.0)
        self.downsample_factor = params.get("downsample_factor", 4)
        self.output_bit_depth = sensor_info.get("output_bit_depth", 8)
        self.sensor_info = sensor_info
        self.platform = platform
    
    def apply_tone_mapping(self):
        """ Durand's tone mapping implementation using Cython optimizations. """
        output_luminance = apply_tone_mapping_cy(
            self.img.astype(np.float64),
            self.sigma_space,
            self.sigma_color,
            self.contrast_factor,
            self.downsample_factor
        )
        
        if self.output_bit_depth == 8:
            return (output_luminance * 255).astype(np.uint8)
        elif self.output_bit_depth == 16:
            return (output_luminance * 65535).astype(np.uint16)
        elif self.output_bit_depth == 32:
            return output_luminance.astype(np.float32)
        else:
            raise ValueError("Unsupported output bit depth. Use 8, 16, or 32.")
    
    def save(self):
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_hdr_durand_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"]
            )
    
    def execute(self):
        if self.is_enable is True:
            print("Executing HDR Durand Tone Mapping (Cython)...")
            start = time.time()
            self.img = self.apply_tone_mapping()
            print(f"Execution time: {time.time() - start:.3f}s")
            
        self.save()
        return self.img 