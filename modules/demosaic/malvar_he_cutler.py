"""
File: malvar_he_cutler.py
Description: Implements the Malvar-He-Cutler algorithm for cfa interpolation
Code / Paper  Reference: https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
from scipy.signal import correlate2d
import cv2


class Malvar:
    """
    CFA interpolation or Demosaicing
    """

    def __init__(self, raw_in, masks):
        self.img = raw_in
        self.masks = masks

    def apply_malvar(self):
        """
        Demosaicing the given raw image using Malvar-He-Cutler
        """
        total_start = time.time()
        
        # 3D masks accoridng to the given bayer
        mask_r, mask_g, mask_b = self.masks
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
        print(f"  Channel separation time: {time.time() - start:.3f}s")

        # Creating g_channel channel first after applying g_at_r_and_b filter
        start = time.time()
        # Pre-compute masks
        non_g_mask = np.logical_or(mask_r == 1, mask_b == 1)
        g_mask = mask_g == 1
        
        # Apply filter to the entire image
        filtered_g = cv2.filter2D(raw_in, -1, g_at_r_and_b, borderType=cv2.BORDER_REFLECT)
        
        # Combine original and filtered values
        g_channel = g_channel * g_mask + filtered_g * non_g_mask
        print(f"  Green channel interpolation time: {time.time() - start:.3f}s")

        # Applying other linear filters - optimized using cv2.filter2D
        start = time.time()
        # Stack the three filters into a single 3D array
        filters = np.stack([r_at_gr_and_b_at_gb, r_at_gb_and_b_at_gr, r_at_b_and_b_at_r])
        
        # Apply all filters at once using cv2.filter2D
        rb_at_g_rbbr = cv2.filter2D(raw_in, -1, r_at_gr_and_b_at_gb, borderType=cv2.BORDER_REFLECT)
        rb_at_g_brrb = cv2.filter2D(raw_in, -1, r_at_gb_and_b_at_gr, borderType=cv2.BORDER_REFLECT)
        rb_at_gr_bbrr = cv2.filter2D(raw_in, -1, r_at_b_and_b_at_r, borderType=cv2.BORDER_REFLECT)
        print(f"  Red/Blue channel interpolation time: {time.time() - start:.3f}s")

        # After convolving the input raw image with rest of the filters,
        # now we have the respective interpolated data, now we just have
        # to extract the updated pixels values according to the
        # position they are meant to be updated

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
        print(f"  Row/Column mask creation time: {time.time() - start:.3f}s")

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
        
        # Update B channel
        b_channel = np.where(
            np.logical_and(b_rows, r_cols),
            rb_at_g_rbbr,
            np.where(
                np.logical_and(r_rows, b_cols),
                rb_at_g_brrb,
                np.where(
                    np.logical_and(r_rows, r_cols),
                    rb_at_gr_bbrr,
                    b_channel
                )
            )
        )
        print(f"  Channel combination time: {time.time() - start:.3f}s")

        demos_out[:, :, 0] = r_channel
        demos_out[:, :, 1] = g_channel
        demos_out[:, :, 2] = b_channel

        print(f"  Total CFA interpolation time: {time.time() - total_start:.3f}s")
        return demos_out
