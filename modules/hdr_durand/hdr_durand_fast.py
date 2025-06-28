import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from util.utils import save_output_array
from util.gpu_utils import gpu_accelerator
import time

# Dummy profile decorator for normal runs (no-op if not using kernprof)
try:
    profile
except NameError:
    def profile(func):
        return func

class HDRDurandToneMapping:
    """
    HDR Durand Tone Mapping Algorithm Implementation
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
    
    def fast_bilateral_filter(self, image):
        """
        Approximate bilateral filtering using a downsampled approach.
        """
        small_img = zoom(image, 1 / self.downsample_factor, order=1)
        small_filtered = self.bilateral_filter(small_img, self.sigma_color, self.sigma_space)
        return zoom(small_filtered, self.downsample_factor, order=1)
    
    def bilateral_filter(self, image, sigma_color, sigma_space):
        """
        GPU-accelerated bilateral filter using OpenCV CUDA.
        """
        # Convert parameters for OpenCV bilateral filter
        d = int(sigma_space * 2 + 1)  # Diameter based on sigma_space
        sigma_color_opencv = sigma_color * 255  # Scale for OpenCV
        
        return gpu_accelerator.bilateral_filter_gpu(
            image.astype(np.float32), d, sigma_color_opencv, sigma_space
        )
    
    @profile
    def apply_tone_mapping(self):
        """ Durand's tone mapping implementation with detailed timing. """
        import time
        t0 = time.time()
        epsilon = 1e-6  # Small value to avoid log(0)
        log_luminance = np.log10(self.img + epsilon)
        print(f"[Durand] Log conversion: {time.time() - t0:.4f}s")

        t1 = time.time()
        log_base = self.bilateral_filter(log_luminance.astype(np.float32), 
                                   self.sigma_color, 
                                   self.sigma_space)
        # Explicit GPU sync if using CuPy
        try:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
            print("[Durand] CuPy GPU sync after bilateral filter")
        except ImportError:
            pass
        print(f"[Durand] Bilateral filter: {time.time() - t1:.4f}s")

        t2 = time.time()
        log_detail = log_luminance - log_base
        compressed_log_base = log_base / self.contrast_factor
        log_output = compressed_log_base + log_detail
        print(f"[Durand] Detail/recombine: {time.time() - t2:.4f}s")

        t3 = time.time()
        output_luminance = np.power(10, log_output)
        output_luminance = (output_luminance - np.min(output_luminance)) / (np.max(output_luminance) - np.min(output_luminance))
        print(f"[Durand] Normalize/scale: {time.time() - t3:.4f}s")

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
            save_output_array(self.platform["in_file"], self.img, "Out_hdr_durand_", 
                              self.platform, self.sensor_info["bit_depth"], self.sensor_info["bayer_pattern"])
    
    @profile
    def execute(self):
        if self.is_enable is True:
            start = time.time()
            print("[Durand] Execute start")
            self.img = self.apply_tone_mapping()
            print(f"[Durand] Execute end: {time.time() - start:.4f}s")
        self.save()
        return self.img
        
