# Demosaic and Tone Mapping Performance Profile (Fast Mode)

## Summary

- Configuration: `/home/brian/ISP_ws/brilliantISP/config/nicos_config.yaml`
- Input image: `/home/brian/ISP_ws/brilliantISP/in_frames/normal/avl_image.raw`
- Measured runs per variant: 3
- Warm-up runs per variant: 1
- Mode: Direct module profiling (not full pipeline)
- Platform: CPU only (GPU disabled)

## Demosaic Algorithm Performance

**Fastest**: `hamilton_adams_opt` at 1.495s
**Slowest**: `ahd_opt` at 6.355s
**Speedup**: 4.25x (fastest vs slowest)

### Detailed Results

| Algorithm | Mean | Stdev | Min | Max | Relative |
| --- | --- | --- | --- | --- | --- |
| hamilton_adams_opt | 1.495s | 0.006s | 1.490s | 1.501s | baseline |
| bilinear | 1.758s | 0.066s | 1.709s | 1.833s | 1.18x |
| vng_opt | 1.898s | 0.141s | 1.783s | 2.055s | 1.27x |
| malvar | 2.131s | 0.016s | 2.113s | 2.141s | 1.43x |
| ppg_opt | 2.342s | 0.018s | 2.329s | 2.363s | 1.57x |
| lmmse_fast | 2.462s | 0.022s | 2.439s | 2.484s | 1.65x |
| ahd_opt | 6.355s | 0.022s | 6.330s | 6.371s | 4.25x |


## Tone Mapping Operator Performance

**Fastest**: `aces` at 0.044s
**Slowest**: `reinhard_integer` at 0.071s
**Speedup**: 1.62x (fastest vs slowest)

### Detailed Results

| Tone Mapper | Mean | Stdev | Min | Max | Relative |
| --- | --- | --- | --- | --- | --- |
| aces | 0.044s | 0.001s | 0.043s | 0.044s | baseline |
| hable | 0.045s | 0.001s | 0.043s | 0.045s | 1.02x |
| hable_integer | 0.068s | 0.006s | 0.062s | 0.072s | 1.55x |
| aces_integer | 0.069s | 0.005s | 0.063s | 0.072s | 1.57x |
| reinhard_integer | 0.071s | 0.005s | 0.065s | 0.074s | 1.62x |


## Notes

- Fast mode: Profiles only the demosaic/tone mapping module execution
- All measurements are CPU-only (GPU acceleration disabled)
- Results exclude pipeline overhead (loading, white balance, CCM, etc.)
- Using optimized and fast variants where available
- Results are input-dependent and may vary with different images
