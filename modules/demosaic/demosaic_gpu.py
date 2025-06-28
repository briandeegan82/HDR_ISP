"""
File: demosaic_gpu.py
Description: GPU-accelerated CFA interpolation algorithms
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
import cv2
from util.utils import save_output_array
from util.gpu_utils import gpu_accelerator

class DemosaicGPU:
    "GPU-accelerated CFA Interpolation"

    def __init__(self, img, platform, sensor_info, parm_dga):
        self.img = img
        self.bayer = sensor_info["bayer_pattern"]
        self.bit_depth = sensor_info["output_bit_depth"]
        self.is_save = parm_dga["is_save"]
        self.sensor_info = sensor_info
        self.platform = platform

    def masks_cfa_bayer(self):
        """
        Generating masks for the given bayer pattern
        """
        pattern = self.bayer
        # dict will be creating 3 channel boolean type array of given shape with the name
        # tag like 'r_channel': [False False ....] , 'g_channel': [False False ....] ,
        # 'b_channel': [False False ....]
        channels = dict(
            (channel, np.zeros(self.img.shape, dtype=bool)) for channel in "rgb"
        )

        # Following comment will create boolean masks for each channel r_channel,
        # g_channel and b_channel
        for channel, (y_channel, x_channel) in zip(
            pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]
        ):
            channels[channel][y_channel::2, x_channel::2] = True

        # tuple will return 3 channel boolean pattern for r_channel,
        # g_channel and b_channel with True at corresponding value
        # For example in rggb pattern, the r_channel mask would then be
        # [ [ True, False, True, False], [ False, False, False, False]]
        return tuple(channels[c] for c in "rgb")

    def apply_cfa_gpu(self):
        """
        GPU-accelerated demosaicing using Malvar-He-Cutler algorithm
        """
        # 3D masks according to the given bayer
        mask_r, mask_g, mask_b = self.masks_cfa_bayer()
        raw_in = np.float32(self.img)

        # Declaring 3D Demosaiced image
        demos_out = np.empty((raw_in.shape[0], raw_in.shape[1], 3))

        # 5x5 2D Filter coefficients for linear interpolation of
        # r_channel,g_channel and b_channel channels
        # These filters helps to retain corresponding pixels information using
        # laplacian while interpolation

        # g_channel at r_channel & b_channel location,
        g_at_r_and_b = (
            np.float32(
                [
                    [0, 0, -1, 0, 0],
                    [0, 0, 2, 0, 0],
                    [-1, 2, 4, 2, -1],
                    [0, 0, 2, 0, 0],
                    [0, 0, -1, 0, 0],
                ]
            )
            * 0.125
        )

        # r_channel at green in r_channel row & b_channel column --
        # b_channel at green in b_channel row & r_channel column
        r_at_gr_and_b_at_gb = (
            np.float32(
                [
                    [0, 0, 0.5, 0, 0],
                    [0, -1, 0, -1, 0],
                    [-1, 4, 5, 4, -1],
                    [0, -1, 0, -1, 0],
                    [0, 0, 0.5, 0, 0],
                ]
            )
            * 0.125
        )

        # r_channel at green in b_channel row & r_channel column --
        # b_channel at green in r_channel row & b_channel column
        r_at_gb_and_b_at_gr = np.transpose(r_at_gr_and_b_at_gb)

        # r_channel at blue in b_channel row & b_channel column --
        # b_channel at red in r_channel row & r_channel column
        r_at_b_and_b_at_r = (
            np.float32(
                [
                    [0, 0, -1.5, 0, 0],
                    [0, 2, 0, 2, 0],
                    [-1.5, 0, 6, 0, -1.5],
                    [0, 2, 0, 2, 0],
                    [0, 0, -1.5, 0, 0],
                ]
            )
            * 0.125
        )

        # Creating r_channel, g_channel & b_channel channels from raw_in
        start = time.time()
        r_channel = raw_in * mask_r
        g_channel = raw_in * mask_g
        b_channel = raw_in * mask_b

        # Creating g_channel channel first after applying g_at_r_and_b filter
        start = time.time()
        # Pre-compute masks
        non_g_mask = np.logical_or(mask_r == 1, mask_b == 1)
        g_mask = mask_g == 1
        
        # Apply GPU-accelerated filter to the entire image
        filtered_g = gpu_accelerator.filter2d_gpu(raw_in.astype(np.float32), g_at_r_and_b)
        
        # Combine original and filtered values
        g_channel = g_channel * g_mask + filtered_g * non_g_mask

        # Applying other linear filters - optimized using GPU-accelerated filtering
        start = time.time()
        # Apply all filters using GPU acceleration
        rb_at_g_rbbr = gpu_accelerator.filter2d_gpu(raw_in.astype(np.float32), r_at_gr_and_b_at_gb)
        rb_at_g_brrb = gpu_accelerator.filter2d_gpu(raw_in.astype(np.float32), r_at_gb_and_b_at_gr)
        rb_at_gr_bbrr = gpu_accelerator.filter2d_gpu(raw_in.astype(np.float32), r_at_b_and_b_at_r)

        # Extracting Red rows and columns - optimized
        start = time.time()
        # Pre-compute boolean masks for rows and columns
        r_rows_mask = np.any(mask_r == 1, axis=1)
        r_cols_mask = np.any(mask_r == 1, axis=0)
        b_rows_mask = np.any(mask_b == 1, axis=1)
        b_cols_mask = np.any(mask_b == 1, axis=0)
        
        # Create row and column masks using broadcasting
        r_rows = r_rows_mask[:, np.newaxis]
        r_cols = r_cols_mask[np.newaxis, :]
        b_rows = b_rows_mask[:, np.newaxis]
        b_cols = b_cols_mask[np.newaxis, :]

        # For R channel we have to update pixels at [r_channel rows
        # and b_channel cols] & at [b_channel rows and r_channel cols]
        # 3 pixels need to be updated near one given r_channel
        start = time.time()
        # Update R channel
        r_channel = np.where(
            np.logical_and(r_rows, b_cols),
            rb_at_g_rbbr,
            np.where(
                np.logical_and(b_rows, r_cols),
                rb_at_g_brrb,
                np.where(
                    np.logical_and(b_rows, b_cols),
                    rb_at_gr_bbrr,
                    r_channel
                )
            )
        )

        # For B channel we have to update pixels at [r_channel rows
        # and b_channel cols] & at [b_channel rows and r_channel cols]
        # 3 pixels need to be updated near one given b_channel
        b_channel = np.where(
            np.logical_and(r_rows, b_cols),
            rb_at_g_brrb,
            np.where(
                np.logical_and(b_rows, r_cols),
                rb_at_g_rbbr,
                np.where(
                    np.logical_and(r_rows, r_cols),
                    rb_at_gr_bbrr,
                    b_channel
                )
            )
        )

        # Stack the channels to create the final demosaiced image
        demos_out[:, :, 0] = r_channel
        demos_out[:, :, 1] = g_channel
        demos_out[:, :, 2] = b_channel

        # Clipping the pixels values within the bit range
        demos_out = np.clip(demos_out, 0, 2**self.bit_depth - 1)
        demos_out = np.uint16(demos_out)
        return demos_out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_demosaic_gpu_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Applying GPU-accelerated demosaicing to bayer image
        """
        start = time.time()
        cfa_out = self.apply_cfa_gpu()
        print(f"GPU Demosaic execution time: {time.time() - start:.3f}s")
        self.img = cfa_out
        self.save()
        return self.img 