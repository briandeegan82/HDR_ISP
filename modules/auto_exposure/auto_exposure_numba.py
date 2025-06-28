"""
File: auto_exposure_numba.py
Description: Numba-optimized Auto Exposure implementation
Code / Paper  Reference: https://www.atlantis-press.com/article/25875811.pdf
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np
from numba import jit, prange

@jit(nopython=True)
def convert_to_8bit_numba(img, bit_depth):
    """Convert image to 8-bit using bit shifting"""
    return img >> (bit_depth - 8)

@jit(nopython=True, parallel=True)
def rgb_to_greyscale_numba(img, bit_depth):
    """Convert RGB to grayscale using luminance weights"""
    height, width, channels = img.shape
    grey_img = np.empty((height, width), dtype=np.uint16)
    max_val = (2**bit_depth) - 1
    
    for i in prange(height):
        for j in range(width):
            # Luminance conversion: 0.299*R + 0.587*G + 0.144*B
            luminance = img[i, j, 0] * 0.299 + img[i, j, 1] * 0.587 + img[i, j, 2] * 0.144
            
            # Manual clipping
            if luminance < 0:
                luminance = 0
            elif luminance > max_val:
                luminance = max_val
            
            grey_img[i, j] = int(luminance)
    
    return grey_img

@jit(nopython=True)
def calculate_skewness_numba(img, center_illuminance):
    """Calculate histogram skewness using Numba optimization"""
    # Subtract central luminance
    img_centered = img.astype(np.float64) - center_illuminance
    
    # Calculate moments
    img_size = img_centered.size
    m_2 = np.sum(img_centered * img_centered) / img_size
    m_3 = np.sum(img_centered * img_centered * img_centered) / img_size
    
    # Calculate skewness
    g_1 = np.sqrt(img_size * (img_size - 1)) / (img_size - 2)
    
    # Avoid division by zero
    if abs(m_2) < 1e-10:
        return 0.0
    
    skewness = (m_3 / (abs(m_2) ** 1.5)) * g_1
    
    # Handle NaN values
    if np.isnan(skewness):
        return 0.0
    
    return skewness

class AutoExposureNumba:
    """
    Numba-optimized Auto Exposure Module
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
        Get Correct Exposure by Adjusting Digital Gain with Numba optimization
        """
        # Convert Image into 8-bit for AE Calculation
        self.img = convert_to_8bit_numba(self.img, self.bit_depth)
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
        Conversion of an Image into Greyscale Image with Numba optimization
        """
        return rgb_to_greyscale_numba(img, self.bit_depth)

    def get_luminance_histogram_skewness(self, img):
        """
        Skewness Calculation with Numba optimization
        """
        return calculate_skewness_numba(img, self.center_illuminance)

    def execute(self):
        """
        Execute Auto Exposure with Numba optimization
        """
        if self.enable is False:
            return None
        else:
            start = time.time()
            ae_feedback = self.get_exposure_feedback()
            print(f"  Numba AE execution time: {time.time()-start:.3f}s")
            return ae_feedback 