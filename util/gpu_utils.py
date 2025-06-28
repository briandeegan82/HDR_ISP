import cv2
import numpy as np
from typing import Tuple

class GPUAccelerator: # Corrected class name back to GPUAccelerator
    """
    A class for performing GPU-accelerated image processing operations
    using OpenCV's CUDA module, with CPU fallbacks.
    """
    def __init__(self):
        """
        Initializes the GPUAccelerator and checks for CUDA availability.
        """
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        print(f"CUDA available: {self.cuda_available}")

    def bilateral_filter_gpu(self, 
                           src: np.ndarray, 
                           d: int, 
                           sigma_color: float, 
                           sigma_space: float) -> np.ndarray:
        """
        Performs bilateral filtering with GPU acceleration if available.
        Falls back to CPU implementation if CUDA is not available.

        Args:
            src (np.ndarray): Input image (single channel, float32 or uint8)
            d (int): Diameter of each pixel neighborhood
            sigma_color (float): Filter sigma in the color space
            sigma_space (float): Filter sigma in the coordinate space

        Returns:
            np.ndarray: The bilateral filtered image
        """
        if not self.cuda_available:
            print("CUDA not available, falling back to CPU bilateral filter.")
            return cv2.bilateralFilter(src, d, sigma_color, sigma_space)
        
        try:
            # Ensure input is in the correct format
            if src.dtype != np.float32:
                src = src.astype(np.float32)
            
            # Upload to GPU
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src)
            
            # Apply bilateral filter on GPU
            gpu_dst = cv2.cuda.bilateralFilter(gpu_src, d, sigma_color, sigma_space)
            
            # Download result
            dst = gpu_dst.download()
            return dst
            
        except Exception as e:
            print(f"GPU bilateral filter failed ({e}), falling back to CPU.")
            return cv2.bilateralFilter(src, d, sigma_color, sigma_space)

    def resize_gpu(self, 
                  src: np.ndarray, 
                  dsize: Tuple[int, int], 
                  interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Performs image resizing with GPU acceleration.

        Args:
            src (np.ndarray): Input image
            dsize (Tuple[int, int]): Output size (width, height)
            interpolation (int): Interpolation method

        Returns:
            np.ndarray: The resized image
        """
        if not self.cuda_available:
            print("CUDA not available, falling back to CPU resize.")
            return cv2.resize(src, dsize, interpolation=interpolation)
        
        try:
            # Upload to GPU
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src)
            
            # Apply resize on GPU
            gpu_dst = cv2.cuda.resize(gpu_src, dsize, interpolation=interpolation)
            
            # Download result
            dst = gpu_dst.download()
            return dst
            
        except Exception as e:
            print(f"GPU resize failed ({e}), falling back to CPU.")
            return cv2.resize(src, dsize, interpolation=interpolation)

    def matrix_multiply_gpu(self, 
                           src: np.ndarray, 
                           matrix: np.ndarray) -> np.ndarray:
        """
        Performs matrix multiplication with GPU acceleration.

        Args:
            src (np.ndarray): Input image (reshaped to 2D if needed)
            matrix (np.ndarray): Transformation matrix

        Returns:
            np.ndarray: The transformed image
        """
        if not self.cuda_available:
            print("CUDA not available, falling back to CPU matrix multiplication.")
            # Reshape if needed for matrix multiplication
            original_shape = src.shape
            if len(original_shape) == 3:
                src_2d = src.reshape(-1, original_shape[-1])
                result_2d = np.dot(src_2d, matrix.T)
                return result_2d.reshape(original_shape)
            else:
                return np.dot(src, matrix.T)
        
        try:
            # For now, fall back to CPU implementation
            # OpenCV CUDA matrix operations are complex and may not be available
            print("GPU matrix multiplication not implemented, using CPU fallback.")
            original_shape = src.shape
            if len(original_shape) == 3:
                src_2d = src.reshape(-1, original_shape[-1])
                result_2d = np.dot(src_2d, matrix.T)
                return result_2d.reshape(original_shape)
            else:
                return np.dot(src, matrix.T)
            
        except Exception as e:
            print(f"GPU matrix multiplication failed ({e}), falling back to CPU.")
            # Reshape if needed for matrix multiplication
            original_shape = src.shape
            if len(original_shape) == 3:
                src_2d = src.reshape(-1, original_shape[-1])
                result_2d = np.dot(src_2d, matrix.T)
                return result_2d.reshape(original_shape)
            else:
                return np.dot(src, matrix.T)

    def histogram_equalization_gpu(self, src: np.ndarray) -> np.ndarray:
        """
        Performs histogram equalization with GPU acceleration.

        Args:
            src (np.ndarray): Input image (grayscale)

        Returns:
            np.ndarray: The histogram equalized image
        """
        if not self.cuda_available:
            print("CUDA not available, falling back to CPU histogram equalization.")
            return cv2.equalizeHist(src)
        
        try:
            # Ensure input is in the correct format
            if src.dtype != np.uint8:
                src = cv2.convertScaleAbs(src)
            
            # Upload to GPU
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src)
            
            # Apply histogram equalization on GPU
            gpu_dst = cv2.cuda.equalizeHist(gpu_src)
            
            # Download result
            dst = gpu_dst.download()
            return dst
            
        except Exception as e:
            print(f"GPU histogram equalization failed ({e}), falling back to CPU.")
            return cv2.equalizeHist(src)

    def clahe_gpu(self, 
                  src: np.ndarray, 
                  clip_limit: float = 2.0, 
                  tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Performs Contrast Limited Adaptive Histogram Equalization (CLAHE).
        Attempts to use GPU acceleration if CUDA is available; otherwise,
        falls back to the CPU implementation.

        Args:
            src (np.ndarray): The input image (expects grayscale or will convert).
                              Should be 8-bit (uint8).
            clip_limit (float): The threshold for contrast limiting.
            tile_grid_size (Tuple[int, int]): Size of the grid for histogram equalization.
                                              Input image will be divided into this many small blocks.

        Returns:
            np.ndarray: The CLAHE processed image.
        """
        # Ensure the input image is in a suitable format for CLAHE (grayscale, uint8)
        # CLAHE typically operates on single-channel 8-bit images.
        if len(src.shape) == 3:
            # Convert to grayscale if it's a color image
            processed_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            processed_src = src.copy() # Work on a copy if already grayscale

        if processed_src.dtype != np.uint8:
            # Convert to 8-bit unsigned integer if not already
            processed_src = cv2.convertScaleAbs(processed_src)

        if not self.cuda_available:
            print("CUDA not available, falling back to CPU CLAHE.")
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(processed_src)
        
        try:
            # --- GPU CLAHE Implementation ---
            print("Attempting to use GPU CLAHE...")
            
            # Upload the image to GPU memory
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(processed_src)

            # Create the CUDA CLAHE object
            cuda_clahe = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

            # Apply CLAHE on the GPU
            gpu_dst = cuda_clahe.apply(gpu_src)

            # Download the result back to CPU memory
            dst = gpu_dst.download()
            print("GPU CLAHE successful.")
            return dst

        except Exception as e:
            # Fallback to CPU if GPU operation fails
            print(f"GPU CLAHE failed ({e}), falling back to CPU CLAHE.")
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(processed_src)

    def filter2d_gpu(self, src: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Performs 2D convolution with GPU acceleration if available.
        Falls back to CPU implementation if CUDA is not available.

        Args:
            src (np.ndarray): Input image (single channel, float32 or uint8)
            kernel (np.ndarray): 2D convolution kernel (float32)

        Returns:
            np.ndarray: The filtered image
        """
        if not self.cuda_available:
            print("CUDA not available, falling back to CPU filter2D.")
            return cv2.filter2D(src, -1, kernel)
        try:
            # Ensure input is in the correct format
            if src.dtype != np.float32:
                src = src.astype(np.float32)
            if kernel.dtype != np.float32:
                kernel = kernel.astype(np.float32)
            # Upload to GPU
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(src)
            # Create CUDA filter
            gpu_kernel = cv2.cuda_GpuMat()
            gpu_kernel.upload(kernel)
            gpu_filter = cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, kernel)
            gpu_dst = gpu_filter.apply(gpu_src)
            # Download result
            dst = gpu_dst.download()
            return dst
        except Exception as e:
            print(f"GPU filter2D failed ({e}), falling back to CPU.")
            return cv2.filter2D(src, -1, kernel)

# Create global instance
gpu_accelerator = GPUAccelerator()

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Create a dummy image for testing CLAHE
    # CLAHE works best on grayscale, 8-bit images.
    img = np.random.randint(0, 100, (512, 512), dtype=np.uint8)
    img[100:200, 100:200] += 100 # Add a low-contrast region

    print("\n--- Testing clahe_gpu function ---")
    # Call the method on the correctly named instance
    processed_img = gpu_accelerator.clahe_gpu(img, clip_limit=3.0, tile_grid_size=(8, 8)) 

    # Optional: Compare with pure CPU version
    print("\n--- Comparing with pure CPU CLAHE ---")
    cpu_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cpu_processed_img = cpu_clahe.apply(img)
    print("Pure CPU CLAHE applied.")

    if gpu_accelerator.cuda_available: # Use the correct instance name here too
        print("\nNote: GPU and CPU results might have minor differences due to floating-point precision.")
    else:
        print("\nCUDA was not available, so both processed images should be identical (both using CPU).")
        print(f"Are CPU and GPU/fallback results identical? {np.array_equal(processed_img, cpu_processed_img)}")

    # You could save or display these images here if you have matplotlib or want to write to disk.
    # cv2.imwrite("original.png", img)
    # cv2.imwrite("processed_clahe_gpu.png", processed_img)
    # cv2.imwrite("processed_clahe_cpu.png", cpu_processed_img)