"""
File: auto_exposure_cuda.py
Description: CUDA-optimized Auto Exposure implementation
Code / Paper  Reference: https://www.atlantis-press.com/article/25875811.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
import cupy as cp

def convert_to_8bit_cuda(img, bit_depth):
    """Convert image to 8-bit using bit shifting with CUDA"""
    gpu_img = cp.asarray(img)
    return cp.asnumpy(gpu_img >> (bit_depth - 8))

def rgb_to_greyscale_cuda(img, bit_depth):
    """Convert RGB to grayscale using luminance weights with CUDA"""
    gpu_img = cp.asarray(img)
    
    # Luminance conversion using matrix multiplication
    luminance_weights = cp.array([0.299, 0.587, 0.144])
    grey_img = cp.clip(
        cp.dot(gpu_img[..., :3], luminance_weights),
        0, (2**bit_depth) - 1
    ).astype(cp.uint16)
    
    return cp.asnumpy(grey_img)

def calculate_skewness_cuda(img, center_illuminance):
    """Calculate histogram skewness using CUDA optimization"""
    gpu_img = cp.asarray(img)
    
    # Subtract central luminance
    img_centered = gpu_img.astype(cp.float64) - center_illuminance
    
    # Calculate moments
    img_size = img_centered.size
    m_2 = cp.sum(img_centered * img_centered) / img_size
    m_3 = cp.sum(img_centered * img_centered * img_centered) / img_size
    
    # Calculate skewness
    g_1 = cp.sqrt(img_size * (img_size - 1)) / (img_size - 2)
    
    # Avoid division by zero
    if abs(m_2) < 1e-10:
        return 0.0
    
    skewness = (m_3 / (abs(m_2) ** 1.5)) * g_1
    
    # Handle NaN values
    if cp.isnan(skewness):
        return 0.0
    
    return float(skewness)

class AutoExposureCUDA:
    """
    CUDA-optimized Auto Exposure Module
    """

    def __init__(self, img, sensor_info, parm_ae):
        self.img = img
        self.enable = parm_ae["is_enable"]
        self.is_debug = parm_ae["is_debug"]
        self.center_illuminance = parm_ae["center_illuminance"]
        self.histogram_skewness_range = parm_ae["histogram_skewness"]
        self.sensor_info = sensor_info
        self.param_ae = parm_ae
        self.bit_depth = sensor_info["bit_depth"]

    def get_exposure_feedback(self):
        """
        Get Correct Exposure by Adjusting Digital Gain with CUDA optimization
        """
        # Convert Image into 8-bit for AE Calculation
        self.img = convert_to_8bit_cuda(self.img, self.bit_depth)
        self.bit_depth = 8

        # calculate the exposure metric
        return self.determine_exposure()

    def determine_exposure(self):
        """
        Image Exposure Estimation using Skewness Luminance of Histograms
        """
        # For Luminance Histograms, Image is first converted into greyscale image
        grey_img = self.get_greyscale_image(self.img)
        avg_lum = np.average(grey_img)
        
        if self.is_debug:
            print("Average luminance is = ", avg_lum)

        # Histogram skewness Calculation for AE Stats
        skewness = self.get_luminance_histogram_skewness(grey_img)

        # get the ranges
        upper_limit = self.histogram_skewness_range
        lower_limit = -1 * upper_limit

        if self.is_debug:
            print("   - AE - Histogram Skewness Range = ", upper_limit)

        # see if skewness is within range
        if skewness < lower_limit:
            return -1
        elif skewness > upper_limit:
            return 1
        else:
            return 0

    def get_greyscale_image(self, img):
        """
        Conversion of an Image into Greyscale Image with CUDA optimization
        """
        return rgb_to_greyscale_cuda(img, self.bit_depth)

    def get_luminance_histogram_skewness(self, img):
        """
        Skewness Calculation with CUDA optimization
        """
        return calculate_skewness_cuda(img, self.center_illuminance)

    def execute(self):
        """
        Execute Auto Exposure with CUDA optimization
        """
        if self.enable is False:
            return None
        else:
            start = time.time()
            ae_feedback = self.get_exposure_feedback()
            print(f"  CUDA AE execution time: {time.time()-start:.3f}s")
            return ae_feedback 