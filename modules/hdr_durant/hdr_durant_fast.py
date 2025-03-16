import cv2
import numpy as np
from util.utils import save_output_array
import time

class HDRDurandFast:
    """
    HDR Durand Tone Mapping with Fast Bilateral Filter
    """

    def __init__(self, img, platform, sensor_info, param_durant):
        self.img = img.copy()
        self.is_save = param_durant["is_save"]
        self.is_debug = param_durant["is_debug"]
        self.sigma_space = param_durant["sigma_space"]
        self.sigma_color = param_durant["sigma_color"]
        self.contrast_factor = param_durant["contrast_factor"]
        self.downsample_factor = param_durant["downsample_factor"]
        self.output_bit_depth = sensor_info["bit_depth"]
        self.sensor_info = sensor_info
        self.platform = platform
        self.param_hdr = param_durant

    def normalize_image(self, image, lower_percentile=1, upper_percentile=99):
        """
        Normalize the image using percentiles to avoid outliers.

        Parameters:
            image (numpy.ndarray): Input image.
            lower_percentile (float): Lower percentile for normalization.
            upper_percentile (float): Upper percentile for normalization.

        Returns:
            numpy.ndarray: Normalized image.
        """
        # Compute the percentiles
        lower = np.percentile(image, lower_percentile)
        upper = np.percentile(image, upper_percentile)

        # Clip the image to the percentile range
        image_clipped = np.clip(image, lower, upper)

        # Normalize to [0, 1]
        image_normalized = (image_clipped - lower) / (upper - lower)

        return image_normalized
    
    def adaptive_normalization(self, image):
        """
        Normalize the image adaptively using mean and standard deviation.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Normalized image.
        """
        mean = np.mean(image)
        std = np.std(image)

        # Scale the image based on mean and standard deviation
        image_normalized = (image - mean) / (std + 1e-6)  # Add small epsilon to avoid division by zero

        # Clip to [0, 1] and scale to the desired range
        image_normalized = np.clip(image_normalized, 0, 1)

        return image_normalized
    
    def log_normalization(self, image):
        """
        Normalize the image using logarithmic scaling.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Normalized image.
        """
        # Add a small constant to avoid log(0)
        image_log = np.log1p(image)

        # Normalize to [0, 1]
        image_normalized = (image_log - np.min(image_log)) / (np.max(image_log) - np.min(image_log))

        return image_normalized

    def fast_bilateral_filter(self, image):
        """
        Apply a fast approximation of the bilateral filter.

        Parameters:
            image (numpy.ndarray): Input image (grayscale or color).

        Returns:
            numpy.ndarray: Filtered image.
        """
        # Downsample the image
        small_image = cv2.resize(image, None, fx=1/self.downsample_factor, fy=1/self.downsample_factor, interpolation=cv2.INTER_LINEAR)

        # Apply bilateral filter on the downsampled image
        small_filtered = cv2.bilateralFilter(small_image, d=-1, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)

        # Upsample the filtered image to the original size
        filtered_image = cv2.resize(small_filtered, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        return filtered_image

    def durand_tone_mapping_fast(self):
        """
        Apply Durand's tone mapping algorithm with a fast bilateral filter approximation.

        Returns:
            numpy.ndarray: Tone-mapped LDR image scaled to the specified bit depth.
        """
        # Step 1: Convert to logarithmic domain
        log_hdr = np.log1p(self.img)

        # Step 2: Apply fast bilateral filtering to separate base and detail layers
        base_layer = self.fast_bilateral_filter(log_hdr)

        # Step 3: Compress the base layer
        compressed_base = base_layer / self.contrast_factor

        # Step 4: Recombine the layers
        detail_layer = log_hdr - base_layer
        log_ldr = compressed_base + detail_layer

        # Step 5: Convert back to linear domain
        ldr_image = np.expm1(log_ldr)


        # Normalize the result to the range [0, 1]
        ldr_image = (ldr_image - np.min(ldr_image)) / (np.max(ldr_image) - np.min(ldr_image))

        # Scale the output to the desired bit depth
        if self.output_bit_depth == 8:
            ldr_image_scaled = (ldr_image * 255).astype(np.uint8)
        elif self.output_bit_depth == 16:
            ldr_image_scaled = (ldr_image * 65535).astype(np.uint16)
        else:
            raise ValueError("Unsupported output bit depth. Supported values are 8 or 16.")

        return ldr_image_scaled

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_hdr_durant_fast_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute HDR Durand Tone Mapping Module
        """
        print("HDR Durand Tone Mapping (fast) = True")

        start = time.time()
        ldr_image = self.durand_tone_mapping_fast()
        print(f"  Execution time: {time.time() - start:.3f}s")

        self.img = ldr_image
        self.save()
        return self.img