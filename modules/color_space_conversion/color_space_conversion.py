"""
File: color_space_conversion.py
Description: Converts RGB to YUV or YCbCr
Code / Paper  Reference: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
                         https://www.itu.int/rec/R-REC-BT.601/
                         https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en
                         https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-
                         8-bit-yuv-formats-for-video-rendering
                         https://web.archive.org/web/20180423091842/http://www.equasys.de/
                         colorconversion.html
                         Author: 10xEngineers Pvt Ltd
------------------------------------------------------------------------------
"""

import time
import numpy as np

from util.utils import save_output_array_yuv


class ColorSpaceConversion:
    """
    Color Space Conversion
    """

    def __init__(self, img, platform, sensor_info, parm_csc, parm_cse):
        self.img = img.copy()
        self.is_save = parm_csc["is_save"]
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_csc = parm_csc
        self.bit_depth = sensor_info["output_bit_depth"]
        self.conv_std = self.parm_csc["conv_standard"]
        self.rgb2yuv_mat = None
        self.yuv_img = None
        self.parm_cse = parm_cse

    def rgb_to_yuv_8bit(self):
        """
        RGB-to-YUV Colorspace conversion 8bit
        """
        total_start = time.time()

        if self.conv_std == 1:
            # for BT. 709
            self.rgb2yuv_mat = np.array(
                [[47, 157, 16], [-26, -86, 112], [112, -102, -10]]
            )
        else:
            # for BT.601/407
            # conversion metrix with 8bit integer co-efficients - m=8
            self.rgb2yuv_mat = np.array(
                [[77, 150, 29], [131, -110, -21], [-44, -87, 138]]
            )

        start = time.time()
        # make nx3 2d matrix of image and convert to 3xn for matrix multiplication
        mat2d_t = self.img.reshape(-1, 3).T
        #print(f"  Matrix reshape and transpose time: {time.time() - start:.3f}s")

        start = time.time()
        # convert to YUV and combine bit depth conversion
        yuv_2d = np.matmul(self.rgb2yuv_mat, mat2d_t)
        yuv_2d = np.float64(yuv_2d) / (2**8)
        yuv_2d = np.round(yuv_2d)  # More efficient than where/floor/ceil
        #print(f"  Matrix multiplication and bit depth conversion time: {time.time() - start:.3f}s")

        # color saturation enhancment block:
        if self.parm_cse['is_enable']:
            start = time.time()
            gain = self.parm_cse['saturation_gain']
            yuv_2d[1:, :] *= gain  # Apply gain to both U and V channels at once
            #print(f"  Color saturation enhancement time: {time.time() - start:.3f}s")

        start = time.time()
        # Combine black-level/DC offset and final normalization
        yuv_2d[0, :] += 2 ** (self.bit_depth / 2)
        yuv_2d[1:, :] += 2 ** (self.bit_depth - 1)
        
        # Combine transpose, clip, and final normalization
        yuv2d_t = yuv_2d.T
        yuv2d_t = np.clip(yuv2d_t, 0, (2**self.bit_depth) - 1)
        yuv2d_t = np.round(yuv2d_t / (2 ** (self.bit_depth - 8)))
        yuv2d_t = np.clip(yuv2d_t, 0, 255)
        self.img = yuv2d_t.reshape(self.img.shape).astype(np.uint8)
        #print(f"  Final processing time: {time.time() - start:.3f}s")

        #print(f"  Total RGB to YUV conversion time: {time.time() - total_start:.3f}s")
        return self.img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_color_space_conversion_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Execute Color Space Conversion
        """
        print("Color Space Conversion (default) = True")

        start = time.time()
        csc_out = self.rgb_to_yuv_8bit()
        print(f"  Total execution time: {time.time() - start:.3f}s")
        self.img = csc_out
        self.save()
        return self.img
