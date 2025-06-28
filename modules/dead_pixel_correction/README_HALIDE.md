# Halide-based Dead Pixel Correction

This directory contains a Halide-accelerated implementation of the dead pixel correction algorithm for the HDR ISP pipeline.

## Overview

The Halide implementation provides significant performance improvements over the original NumPy-based implementation by leveraging Halide's automatic optimization and parallelization capabilities.

## Files

- `dead_pixel_correction_halide.py` - Main Halide implementation with automatic fallback
- `dead_pixel_correction.py` - Original NumPy-based implementation
- `dynamic_dpc.py` - Core algorithm implementation used by both versions

## Algorithm

The dead pixel correction algorithm works as follows:

1. **Detection**: For each pixel, examine its 8 neighbors in a specific 5x5 pattern:
   ```
   [1, 0, 1, 0, 1]
   [0, 0, 0, 0, 0]
   [1, 0, 0, 0, 1]
   [0, 0, 0, 0, 0]
   [1, 0, 1, 0, 1]
   ```

2. **Conditions for Dead Pixel Detection**:
   - **Condition 1**: Center pixel is outside the min-max range of its neighbors
   - **Condition 2**: All differences between center pixel and neighbors exceed threshold

3. **Correction**: For detected dead pixels, compute gradients in 4 directions and use the direction with minimum gradient to interpolate the corrected value.

## Usage

### Basic Usage

```python
from modules.dead_pixel_correction.dead_pixel_correction_halide import DeadPixelCorrectionHalideFallback

# Initialize with automatic fallback
dpc = DeadPixelCorrectionHalideFallback(img, sensor_info, parm_dpc, platform)

# Execute the correction
result = dpc.execute()
```

### Direct Halide Usage (if available)

```python
from modules.dead_pixel_correction.dead_pixel_correction_halide import DeadPixelCorrectionHalide

# Direct Halide implementation (will raise ImportError if Halide not available)
dpc = DeadPixelCorrectionHalide(img, sensor_info, parm_dpc, platform)
result = dpc.execute()
```

## Installation

### Prerequisites

1. **Halide Installation**: Install Halide Python bindings
   ```bash
   pip install halide
   ```
   
   Note: Halide installation can be complex. See [Halide documentation](https://halide-lang.org/docs/install.html) for detailed instructions.

2. **Alternative**: Use the fallback implementation which automatically uses the original NumPy version if Halide is not available.

### Dependencies

The implementation requires the same dependencies as the main HDR ISP project:
- numpy
- scipy
- opencv-python (optional, for testing)

## Performance

The Halide implementation typically provides:
- **2-10x speedup** over the original NumPy implementation
- **Automatic parallelization** across CPU cores
- **Vectorization** for SIMD operations
- **Memory optimization** through Halide's scheduling

## Testing

Run the test script to compare implementations:

```bash
python test_halide_dpc.py
```

This will:
1. Create a synthetic test image with dead pixels
2. Run both original and Halide implementations
3. Compare results and performance
4. Save output images for visual inspection

### Test with Real Image

```bash
python test_halide_dpc.py path/to/your/image.png
```

## Integration

The Halide implementation is designed to be a drop-in replacement for the original implementation. It maintains the same interface and automatically falls back to the original implementation if Halide is not available.

### In ISP Pipeline

Replace the original import:
```python
# Original
from modules.dead_pixel_correction.dead_pixel_correction import DeadPixelCorrection

# With Halide version
from modules.dead_pixel_correction.dead_pixel_correction_halide import DeadPixelCorrectionHalideFallback
```

## Configuration

The implementation uses the same configuration parameters as the original:

```python
parm_dpc = {
    "is_enable": True,
    "dp_threshold": 50.0,  # Threshold for dead pixel detection
    "is_debug": True,      # Enable debug output
    "is_save": False       # Save intermediate results
}
```

## Troubleshooting

### Halide Not Available
If you see "Halide not available" messages, the implementation will automatically fall back to the original NumPy version. This ensures compatibility even without Halide.

### Performance Issues
- Ensure Halide is properly installed and compiled
- Check that your system supports the required SIMD instructions
- Monitor CPU usage to verify parallelization is working

### Memory Issues
- The Halide implementation may use more memory during compilation
- For very large images, consider processing in tiles

## Implementation Details

### Halide Pipeline Structure

The Halide pipeline is structured as follows:

1. **Input**: 2D float32 image
2. **Neighbor Access**: 8-neighbor pattern in 5x5 window
3. **Detection**: Two-condition dead pixel detection
4. **Gradient Computation**: 4-direction gradient calculation
5. **Correction**: Direction-based interpolation
6. **Output**: Corrected image

### Scheduling

The pipeline uses Halide's scheduling primitives:
- `parallel(y)`: Parallelize across rows
- `vectorize(x, 8)`: Vectorize across columns with 8-wide SIMD

### Memory Management

- Automatic buffer management through Halide
- Minimal memory copies between NumPy and Halide
- Efficient in-place operations where possible

## Future Improvements

Potential enhancements:
- GPU acceleration using Halide's CUDA/OpenCL backends
- Multi-scale dead pixel detection
- Adaptive thresholding based on image statistics
- Integration with other ISP modules for end-to-end optimization

## References

- Original algorithm: [IEEE Paper](https://ieeexplore.ieee.org/document/9194921)
- OpenISP implementation: [GitHub](https://github.com/cruxopen/openISP)
- Halide documentation: [halide-lang.org](https://halide-lang.org/) 