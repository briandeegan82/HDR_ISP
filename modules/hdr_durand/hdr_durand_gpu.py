"""
File: hdr_durand_gpu.py
Description: GPU-accelerated HDR Durand Tone Mapping Algorithm Implementation
Author: 10xEngineers
------------------------------------------------------------
"""
import numpy as np
from scipy.ndimage import zoom
from util.utils import save_output_array
from util.gpu_utils import gpu_accelerator
import time

class HDRDurandToneMappingGPU:
    """
    GPU-accelerated HDR Durand Tone Mapping Algorithm Implementation
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
    
    def normalize(self, image):
        """ Normalize image to [0,1] range."""
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    
    def fast_bilateral_filter_gpu(self, image):
        """
        GPU-accelerated approximate bilateral filtering using a downsampled approach.
        """
        small_img = zoom(image, 1 / self.downsample_factor, order=1)
        small_filtered = self.bilateral_filter_gpu(small_img, self.sigma_color, self.sigma_space)
        return zoom(small_filtered, self.downsample_factor, order=1)
    
    def bilateral_filter_gpu(self, image, sigma_color, sigma_space):
        """
        GPU-accelerated bilateral filter using OpenCV CUDA.
        """
        # Use GPU-accelerated bilateral filtering
        return gpu_accelerator.bilateral_filter_gpu(
            image.astype(np.float32), 
            int(sigma_space * 2 + 1), 
            sigma_color, 
            sigma_space
        )
    
    def apply_tone_mapping(self):
        """ GPU-accelerated Durand's tone mapping implementation. """
        # Convert to log domain
        epsilon = 1e-6  # Small value to avoid log(0)
        log_luminance = np.log10(self.img + epsilon)
    
        # Apply GPU-accelerated bilateral filter to get the base layer
        log_base = self.bilateral_filter_gpu(log_luminance.astype(np.float32), 
                                           self.sigma_color, 
                                           self.sigma_space)
    
        # Extract the detail layer
        log_detail = log_luminance - log_base
    
        # Compress the base layer (reduce contrast)
        compressed_log_base = log_base / self.contrast_factor
    
        # Recombine base and detail layers
        log_output = compressed_log_base + log_detail
    
        # Convert back from log domain
        output_luminance = np.power(10, log_output)
    
        # Normalize to [0, 1] range
        output_luminance = (output_luminance - np.min(output_luminance)) / (np.max(output_luminance) - np.min(output_luminance))
    
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
            save_output_array(self.platform["in_file"], self.img, "Out_hdr_durand_gpu_", 
                              self.platform, self.sensor_info["bit_depth"], self.sensor_info["bayer_pattern"])
    
    def execute(self):
        if self.is_enable is True:
            start = time.time()
            self.img = self.apply_tone_mapping()
            print(f"GPU HDR Durand execution time: {time.time() - start:.3f}s")
            
        self.save()
        return self.img 