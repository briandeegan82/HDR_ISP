import numpy as np
from typing import Tuple, Optional, List

class ImageTiler:
    """
    Utility class for handling image tiling operations.
    """
    
    def __init__(self, tile_size: Tuple[int, int], overlap: Optional[Tuple[int, int]] = None):
        """
        Initialize the tiler with tile size and optional overlap.
        
        Args:
            tile_size: Tuple of (height, width) for each tile
            overlap: Tuple of (vertical_overlap, horizontal_overlap) in pixels
        """
        self.tile_size = tile_size
        self.overlap = overlap or (0, 0)
        
    def get_tile_boundaries(self, image_shape: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Calculate tile boundaries for an image.
        
        Args:
            image_shape: Tuple of (height, width) of the input image
            
        Returns:
            List of tuples containing (start, end) coordinates for each tile
        """
        height, width = image_shape
        tile_h, tile_w = self.tile_size
        overlap_h, overlap_w = self.overlap
        
        # Calculate number of tiles
        n_tiles_h = (height + tile_h - 1) // tile_h
        n_tiles_w = (width + tile_w - 1) // tile_w
        
        boundaries = []
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile boundaries
                start_h = max(0, i * tile_h - overlap_h)
                end_h = min(height, (i + 1) * tile_h + overlap_h)
                start_w = max(0, j * tile_w - overlap_w)
                end_w = min(width, (j + 1) * tile_w + overlap_w)
                
                # Skip empty tiles
                if end_h <= start_h or end_w <= start_w:
                    continue
                    
                boundaries.append(((start_h, start_w), (end_h, end_w)))
                
        return boundaries
    
    def extract_tile(self, image: np.ndarray, boundary: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
        """
        Extract a tile from the image using the given boundary.
        
        Args:
            image: Input image
            boundary: Tuple of ((start_h, start_w), (end_h, end_w))
            
        Returns:
            Extracted tile
        """
        (start_h, start_w), (end_h, end_w) = boundary
        return image[start_h:end_h, start_w:end_w]
    
    def merge_tiles(self, tiles: List[np.ndarray], boundaries: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
                   output_shape: Tuple[int, int]) -> np.ndarray:
        """
        Merge processed tiles back into a single image.
        
        Args:
            tiles: List of processed tiles
            boundaries: List of tile boundaries
            output_shape: Shape of the output image
            
        Returns:
            Merged image
        """
        output = np.zeros(output_shape, dtype=tiles[0].dtype)
        overlap_h, overlap_w = self.overlap
        
        for tile, ((start_h, start_w), (end_h, end_w)) in zip(tiles, boundaries):
            # Skip empty tiles
            if end_h <= start_h or end_w <= start_w:
                continue
                
            # Calculate overlap regions
            if start_h > 0:
                start_h += overlap_h
            if start_w > 0:
                start_w += overlap_w
            if end_h < output_shape[0]:
                end_h -= overlap_h
            if end_w < output_shape[1]:
                end_w -= overlap_w
                
            # Skip if region is empty after overlap adjustment
            if end_h <= start_h or end_w <= start_w:
                continue
                
            # Calculate source region
            src_start_h = overlap_h if start_h > 0 else 0
            src_end_h = tile.shape[0] - (overlap_h if end_h < output_shape[0] else 0)
            src_start_w = overlap_w if start_w > 0 else 0
            src_end_w = tile.shape[1] - (overlap_w if end_w < output_shape[1] else 0)
            
            # Skip if source region is empty
            if src_end_h <= src_start_h or src_end_w <= src_start_w:
                continue
                
            # Copy tile to output
            output[start_h:end_h, start_w:end_w] = tile[src_start_h:src_end_h, src_start_w:src_end_w]
            
        return output 