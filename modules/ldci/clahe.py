"""
File: clahe.py
Description: Implements the contrast adjustment using contrast limited adaptive
histogram equalization (CLAHE) approach.
Code / Paper  Reference:
https://arxiv.org/ftp/arxiv/papers/2108/2108.12818.pdf#:~:text
=The%20technique%20to%20equalize%20the,a%20linear%20trend%20(CDF).
Implementation inspired from: MATLAB &
Fast Open ISP Author: Qiu Jueqin (qiujueqin@gmail.com)
Author: x10xEngineers
------------------------------------------------------------
"""
import math
import numpy as np
import time
from . import clahe_cy  # Import the Cython module


class CLAHE:
    """
    Contrast Limited Adaptive Histogram Equalization
    """

    def __init__(self, yuv, platform, sensor_info, parm_ldci):
        self.yuv = yuv
        self.img = yuv
        self.enable = parm_ldci["is_enable"]
        self.sensor_info = sensor_info
        self.wind = parm_ldci["wind"]
        self.clip_limit = parm_ldci["clip_limit"]
        self.is_save = parm_ldci["is_save"]
        self.platform = platform

    def pad_array(self, array, pads, mode="reflect"):
        """
        Pad an array with the given margins on left, right, top and bottom
        """
        if isinstance(pads, (list, tuple, np.ndarray)):
            if len(pads) == 2:
                pads = ((pads[0], pads[0]), (pads[1], pads[1])) + ((0, 0),) * (
                    array.ndim - 2
                )
            elif len(pads) == 4:
                pads = ((pads[0], pads[1]), (pads[2], pads[3])) + ((0, 0),) * (
                    array.ndim - 2
                )
            else:
                raise NotImplementedError

        return np.pad(array, pads, mode)

    def crop(self, array, crops):
        """
        Crop an array within the given margins
        """
        if isinstance(crops, (list, tuple, np.ndarray)):
            if len(crops) == 2:
                top_crop = bottom_crop = crops[0]
                left_crop = right_crop = crops[1]
            elif len(crops) == 4:
                top_crop, bottom_crop, left_crop, right_crop = crops
            else:
                raise NotImplementedError
        else:
            top_crop = bottom_crop = left_crop = right_crop = crops

        height, width = array.shape[:2]
        return array[
            top_crop : height - bottom_crop, left_crop : width - right_crop, ...
        ]

    def get_tile_lut(self, tiled_array):
        """
        Generating LUT using histogram equalization
        """
        # Computing histograms--hist will have bincounts
        hist_start = time.time()
        hist = clahe_cy.compute_histogram_cy(tiled_array)
        hist_time = time.time() - hist_start
        print(f"      Histogram computation: {hist_time:.3f}s")

        clip_start = time.time()
        clip_limit = self.clip_limit

        # Clipping each bin counts within the range of window size to avoid artifacts
        # Applying a check to keep the clipping limit within the appropriate range
        if clip_limit >= self.wind:
            clip_limit = 0.08 * self.wind

        clipped_hist = np.clip(hist, 0, clip_limit)
        num_clipped_pixels = (hist - clipped_hist).sum()
        clip_time = time.time() - clip_start
        print(f"      Histogram clipping: {clip_time:.3f}s")

        # Adding clipped pixels to each bin and getting its sum for normalization
        norm_start = time.time()
        hist = clipped_hist + num_clipped_pixels / 256 + 1
        pdf = hist / hist.sum()
        cdf = np.cumsum(pdf)
        norm_time = time.time() - norm_start
        print(f"      PDF/CDF computation: {norm_time:.3f}s")

        # Computing cdf and getting the LUT for the array
        lut_start = time.time()
        look_up_table = (cdf * 255).astype(np.uint8)
        lut_time = time.time() - lut_start
        print(f"      LUT finalization: {lut_time:.3f}s")

        total_time = time.time() - hist_start
        print(f"    Total LUT generation time: {total_time:.3f}s")

        return look_up_table

    def interp_blocks(self, weights, block, first_block_lut, second_block_lut):
        """
        Interpolating blocks using alpha blending weights
        """
        # alpha blending = weights
        first = weights * first_block_lut[block].astype(np.int32)
        second = (1024 - weights) * second_block_lut[block].astype(np.int32)

        # Interpolating both the LUTs
        return np.right_shift(first + second, 10).astype(np.uint8)

    def interp_top_bottom_block(self, left_lut_weights, block, left_lut, current_lut):
        """
        Interpolating blocks present at top and bottom of the arrays
        """
        return self.interp_blocks(left_lut_weights, block, left_lut, current_lut)

    def interp_left_right_block(self, top_lut_weights, block, top_lut, current_lut):
        """
        Interpolating blocks present at left and right of the arrays
        """
        return self.interp_blocks(top_lut_weights, block, top_lut, current_lut)

    def interp_neighbor_block(
        self,
        left_lut_weights,
        top_lut_weights,
        block,
        tl_lut,
        top_lut,
        left_lut,
        current_lut,
    ):
        """
        Interpolating blocks present in the middle of the arrays
        """
        interp_top_blocks = self.interp_blocks(left_lut_weights, block, tl_lut, top_lut)
        interp_current_blocks = self.interp_blocks(
            left_lut_weights, block, left_lut, current_lut
        )

        interp_final = np.right_shift(
            top_lut_weights * interp_top_blocks
            + (1024 - top_lut_weights) * interp_current_blocks,
            10,
        ).astype(np.uint8)
        return interp_final

    def is_corner_block(self, x_tiles, y_tiles, i_col, i_row):
        """
        Checking if the current image block is locating in a corner region
        """
        return (
            (i_row == 0 and i_col == 0)
            or (i_row == 0 and i_col == x_tiles)
            or (i_row == y_tiles and i_col == 0)
            or (i_row == y_tiles and i_col == x_tiles)
        )

    def is_top_or_bottom_block(self, x_tiles, y_tiles, i_col, i_row):
        """
        Checking if the current image block is locating in teh top or bottom region
        """
        return (i_row == 0 or i_row == y_tiles) and not self.is_corner_block(
            x_tiles, y_tiles, i_col, i_row
        )

    def is_left_or_right_block(self, x_tiles, y_tiles, i_col, i_row):
        """
        Checking if the current image block is locating in the left or right region
        """
        return (i_col == 0 or i_col == x_tiles) and not self.is_corner_block(
            x_tiles, y_tiles, i_col, i_row
        )

    def get_tile_lut_vectorized(self, tiled_arrays):
        """
        Generating LUTs using histogram equalization for multiple tiles at once
        """
        # Computing histograms for all tiles at once
        hist_start = time.time()
        # Reshape to 2D array where each row is a tile
        tiles_flat = tiled_arrays.reshape(-1, self.wind * self.wind)
        # Compute histograms for all tiles at once
        hists = np.apply_along_axis(
            lambda x: np.histogram(x, bins=256, range=(0, 255))[0],
            1, tiles_flat
        )
        hist_time = time.time() - hist_start
        print(f"      Vectorized histogram computation: {hist_time:.3f}s")

        clip_start = time.time()
        clip_limit = self.clip_limit
        if clip_limit >= self.wind:
            clip_limit = 0.08 * self.wind

        # Clip all histograms at once
        clipped_hists = np.clip(hists, 0, clip_limit)
        num_clipped_pixels = (hists - clipped_hists).sum(axis=1, keepdims=True)

        # Add clipped pixels to each bin and normalize
        hists = clipped_hists + num_clipped_pixels / 256 + 1
        pdfs = hists / hists.sum(axis=1, keepdims=True)
        cdfs = np.cumsum(pdfs, axis=1)

        # Generate LUTs for all tiles at once
        lut_start = time.time()
        luts = (cdfs * 255).astype(np.uint8)
        lut_time = time.time() - lut_start
        print(f"      Vectorized LUT generation: {lut_time:.3f}s")
        
        # Reshape LUTs to match tile grid
        vert_tiles = math.ceil(self.yuv.shape[0] / self.wind)
        horiz_tiles = math.ceil(self.yuv.shape[1] / self.wind)
        luts = luts.reshape(vert_tiles, horiz_tiles, 256)
        print(f"      LUT shape after reshaping: {luts.shape}")
        
        # Verify LUT dimensions
        if luts.shape[0] != vert_tiles or luts.shape[1] != horiz_tiles:
            raise ValueError(f"LUT shape mismatch: got {luts.shape}, expected ({vert_tiles}, {horiz_tiles}, 256)")
        
        return luts

    def apply_clahe(self):
        """
        Applying clahe algorithm for contrast enhancement
        """
        print("\nCLAHE Processing:")
        total_start = time.time()
        
        try:
            wind = self.wind
            in_yuv = self.yuv

            # Extracting Luminance channel from yuv as LDCI will be applied to Y channel only
            prep_start = time.time()
            yuv = in_yuv[:, :, 0]
            img_height, img_width = yuv.shape

            # pipeline specific: if input is in analog yuv
            if in_yuv.dtype == "float32":
                yuv = np.round(255 * yuv).astype(np.uint8)

            # output clipped equalized histogram
            out_ceh = np.empty(shape=(img_height, img_width, 3), dtype=np.uint8)

            # computing number of tiles (tiles = block = window).
            vert_tiles = math.ceil(img_height / wind)
            horiz_tiles = math.ceil(img_width / wind)
            tile_height = wind
            tile_width = wind

            print(f"Processing image: {img_height}x{img_width} with {vert_tiles}x{horiz_tiles} tiles")

            # Computing number of columns and rows to be padded in the image
            # for getting proper block/tile
            row_pads = tile_height * vert_tiles - img_height
            col_pads = tile_width * horiz_tiles - img_width
            pads = (
                row_pads // 2,
                row_pads - row_pads // 2,
                col_pads // 2,
                col_pads - col_pads // 2,
            )

            # Assigning linearized LUT weights to top and left blocks
            left_lut_weights = np.linspace(1024, 0, tile_width, dtype=np.int32).reshape(
                (1, -1)
            )
            top_lut_weights = np.linspace(1024, 0, tile_height, dtype=np.int32).reshape(
                (-1, 1)
            )

            # Creating a copy of yuv image
            y_padded = yuv
            y_padded = self.pad_array(y_padded, pads=pads)

            # Extract all tiles at once
            tiles = np.zeros((vert_tiles * horiz_tiles, wind, wind), dtype=np.uint8)
            for i in range(vert_tiles):
                for j in range(horiz_tiles):
                    start_row = i * tile_height
                    end_row = (i + 1) * tile_height
                    start_col = j * tile_width
                    end_col = (j + 1) * tile_width
                    tiles[i * horiz_tiles + j] = y_padded[start_row:end_row, start_col:end_col]

            # Generate LUTs for all tiles at once
            lut_start = time.time()
            luts = self.get_tile_lut_vectorized(tiles)
            lut_time = time.time() - lut_start
            print(f"Generated LUTs in {lut_time:.3f}s")

            # Declaring an empty array for output array after padding is done
            y_ceh = np.empty_like(y_padded)

            # For loops for processing image array tile by tile
            interp_start = time.time()
            print(f"Processing tiles...")
            
            for i_row in range(vert_tiles):
                for i_col in range(horiz_tiles):
                    try:
                        # Extracting tile/block
                        start_row_index = i_row * tile_height
                        end_row_index = min(start_row_index + tile_height, y_padded.shape[0])
                        start_col_index = i_col * tile_width
                        end_col_index = min(start_col_index + tile_width, y_padded.shape[1])

                        # Extracting the tile for processing
                        y_block = y_padded[start_row_index:end_row_index, start_col_index:end_col_index].astype(np.uint8)

                        # Process tile using Cython implementation
                        processed_tile = clahe_cy.process_tile_cy(
                            y_block, luts, i_row, i_col, horiz_tiles, vert_tiles,
                            left_lut_weights, top_lut_weights
                        )
                        
                        # Verify processed tile is valid before assignment
                        if processed_tile is None or processed_tile.size == 0:
                            raise ValueError(f"Processed tile is empty or None at position ({i_row}, {i_col})")
                        
                        if processed_tile.shape != y_block.shape:
                            raise ValueError(f"Shape mismatch: processed tile {processed_tile.shape} != input tile {y_block.shape}")
                        
                        y_ceh[start_row_index:end_row_index, start_col_index:end_col_index] = processed_tile
                        
                    except Exception as e:
                        print(f"Error processing tile ({i_row}, {i_col}): {str(e)}")
                        raise

            print("All tiles processed. Cropping output...")
            
            # Crop the output to original size
            y_ceh = self.crop(y_ceh, pads)
            
            # Copy the processed Y channel to output
            out_ceh[:, :, 0] = y_ceh
            out_ceh[:, :, 1] = in_yuv[:, :, 1]
            out_ceh[:, :, 2] = in_yuv[:, :, 2]

            total_time = time.time() - total_start
            print(f"CLAHE processing completed in {total_time:.3f}s")

            return out_ceh

        except Exception as e:
            print(f"\nError in CLAHE processing: {str(e)}")
            print(f"Image shape: {in_yuv.shape}")
            print(f"Window size: {wind}")
            print(f"Clip limit: {self.clip_limit}")
            raise
