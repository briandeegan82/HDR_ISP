import cv2
import numpy as np

def fast_bilateral_filter(image, sigma_space, sigma_color, downsample_factor=4):
    """
    Apply a fast approximation of the bilateral filter.

    Parameters:
        image (numpy.ndarray): Input image (grayscale or color).
        sigma_space (float): Spatial standard deviation for the Gaussian kernel.
        sigma_color (float): Range standard deviation for the intensity kernel.
        downsample_factor (int): Factor by which to downsample the image.

    Returns:
        numpy.ndarray: Filtered image.
    """
    # Downsample the image
    small_image = cv2.resize(image, None, fx=1/downsample_factor, fy=1/downsample_factor, interpolation=cv2.INTER_LINEAR)

    # Apply bilateral filter on the downsampled image
    small_filtered = cv2.bilateralFilter(small_image, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Upsample the filtered image to the original size
    filtered_image = cv2.resize(small_filtered, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    return filtered_image


def durand_tone_mapping_fast(hdr_image, sigma_space=10, sigma_color=0.4, contrast_factor=5, downsample_factor=4):
    """
    Apply Durand's tone mapping algorithm with a fast bilateral filter approximation.

    Parameters:
        hdr_image (numpy.ndarray): Input HDR image (float32 or float64).
        sigma_space (float): Spatial standard deviation for bilateral filtering.
        sigma_color (float): Color standard deviation for bilateral filtering.
        contrast_factor (float): Scaling factor for compressing the base layer.
        downsample_factor (int): Downsampling factor for the fast bilateral filter.

    Returns:
        numpy.ndarray: Tone-mapped LDR image (normalized to [0, 1]).
    """
    # Step 1: Convert to logarithmic domain
    log_hdr = np.log1p(hdr_image)

    # Step 2: Apply fast bilateral filtering to separate base and detail layers
    base_layer = fast_bilateral_filter(log_hdr, sigma_space, sigma_color, downsample_factor)

    # Step 3: Compress the base layer
    compressed_base = base_layer / contrast_factor

    # Step 4: Recombine the layers
    detail_layer = log_hdr - base_layer
    log_ldr = compressed_base + detail_layer

    # Step 5: Convert back to linear domain
    ldr_image = np.expm1(log_ldr)

    # Normalize the result to the range [0, 1]
    ldr_image = (ldr_image - np.min(ldr_image)) / (np.max(ldr_image) - np.min(ldr_image))

    return ldr_image


def load_hdr_image(file_path):
    """
    Load an HDR image from a file (e.g., OpenEXR format).

    Parameters:
        file_path (str): Path to the HDR image file.

    Returns:
        numpy.ndarray: HDR image as a NumPy array.
    """
    hdr_image = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if hdr_image is None:
        raise ValueError(f"Could not load HDR image from {file_path}")
    return hdr_image


def save_ldr_image(ldr_image, file_path):
    """
    Save an LDR image to a file (e.g., PNG format).

    Parameters:
        ldr_image (numpy.ndarray): LDR image (normalized to [0, 1]).
        file_path (str): Path to save the LDR image.
    """
    ldr_image_8bit = (ldr_image * 255).astype(np.uint8)
    cv2.imwrite(file_path, ldr_image_8bit)


def main():
    # Path to the input HDR image (OpenEXR format)
    hdr_file_path = 'input_hdr.exr'

    # Path to save the output LDR image (PNG format)
    ldr_file_path = 'output_ldr_fast.png'

    # Load the HDR image
    hdr_image = load_hdr_image(hdr_file_path)

    # Apply Durand's tone mapping with fast bilateral filter
    ldr_image = durand_tone_mapping_fast(hdr_image, sigma_space=10, sigma_color=0.4, contrast_factor=5, downsample_factor=4)

    # Save the tone-mapped LDR image
    save_ldr_image(ldr_image, ldr_file_path)

    print(f"Tone-mapped image saved to {ldr_file_path}")


if __name__ == "__main__":
    main()