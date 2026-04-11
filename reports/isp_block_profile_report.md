# ISP Block Execution Time Profile

## Summary
- Mean end-to-end wall time: `4.464s`
- Run-to-run stdev: `0.046s`
- Slowest measured block in base config: `Bayer Noise Reduction` at `1.875s`
- Enabled in base config: `Dead Pixel Correction, Black Level Correction, Companding / PWC, Digital Gain, Lens Shading Correction, Bayer Noise Reduction, Auto White Balance, White Balance, Tone Mapping, Demosaic, Color Correction Matrix, Auto Exposure, Linear 16-bit to 8-bit Conversion, Color Space Conversion, Sharpen, RGB Conversion, Gamma Correction, YUV Conversion Format`
- Disabled in base config but profiled separately: `Crop, OECF, LDCI, 2D Noise Reduction, Scale`

## Methodology
- Config: `/home/brian/ISP_ws/brilliantISP/config/svs_cam.yml`
- Input: `/home/brian/ISP_ws/brilliantISP/in_frames/hdr_mode/frame_0460_fsin_38361194647660880.raw`
- Measured runs per variant: `3`
- Warm-up runs per variant: `1`
- Benchmark path: `BrilliantISP.run_pipeline(visualize_output=False)`
- Side outputs disabled during profiling: histogram plots, tone-curve plots, and module save outputs
- The inline 16-bit to 8-bit conversion step is included in coverage but left unmeasured because the current code does not emit a dedicated high-resolution timing event for it

## Base Config Profile
| Block | Status | Mean | Stdev | Min | Max | Share of Pipeline |
| --- | --- | --- | --- | --- | --- | --- |
| Crop | Disabled | n/a | n/a | n/a | n/a | n/a |
| Dead Pixel Correction | Enabled | 1.231s | 0.014s | 1.223s | 1.247s | 27.6% |
| Black Level Correction | Enabled | 0.012s | 0.001s | 0.011s | 0.012s | 0.3% |
| Companding / PWC | Enabled | 0.013s | 0.001s | 0.012s | 0.013s | 0.3% |
| OECF | Disabled | n/a | n/a | n/a | n/a | n/a |
| Digital Gain | Enabled | 0.008s | 0.001s | 0.007s | 0.008s | 0.2% |
| Lens Shading Correction | Enabled | 0.052s | 0.001s | 0.052s | 0.053s | 1.2% |
| Bayer Noise Reduction | Enabled | 1.875s | 0.030s | 1.845s | 1.905s | 42.0% |
| Auto White Balance | Enabled | 0.111s | 0.002s | 0.109s | 0.113s | 2.5% |
| White Balance | Enabled | 0.009s | 0.001s | 0.008s | 0.009s | 0.2% |
| Tone Mapping | Enabled | 0.175s | 0.001s | 0.175s | 0.176s | 3.9% |
| Demosaic | Enabled | 0.360s | 0.001s | 0.359s | 0.361s | 8.1% |
| Color Correction Matrix | Enabled | 0.025s | 0.005s | 0.022s | 0.031s | 0.6% |
| Auto Exposure | Enabled | 0.148s | 0.001s | 0.148s | 0.149s | 3.3% |
| Linear 16-bit to 8-bit Conversion | Inline step | n/a | n/a | n/a | n/a | n/a |
| Color Space Conversion | Enabled | 0.239s | 0.008s | 0.232s | 0.247s | 5.4% |
| LDCI | Disabled | n/a | n/a | n/a | n/a | n/a |
| Sharpen | Enabled | 0.046s | 0.001s | 0.046s | 0.047s | 1.0% |
| 2D Noise Reduction | Disabled | n/a | n/a | n/a | n/a | n/a |
| RGB Conversion | Enabled | 0.069s | 0.003s | 0.066s | 0.072s | 1.5% |
| Gamma Correction | Enabled | 0.014s | 0.001s | 0.013s | 0.014s | 0.3% |
| Scale | Disabled | n/a | n/a | n/a | n/a | n/a |
| YUV Conversion Format | Enabled, not timed | n/a | n/a | n/a | n/a | n/a |

## Disabled Block Spot Checks
| Block | Block Mean | Block Stdev | Variant Wall Time | Delta vs Base |
| --- | --- | --- | --- | --- |
| Crop | 0.000s | 0.000s | 0.111s | -4.352s |
| OECF | 0.001s | 0.000s | 4.426s | -0.037s |
| LDCI | 0.060s | 0.001s | 4.533s | +0.069s |
| 2D Noise Reduction | 7.744s | 0.154s | 12.275s | +7.812s |
| Scale | 0.011s | 0.001s | 4.544s | +0.081s |

## Notes
- Variant wall times are not additive; each disabled-block spot check re-runs the pipeline with that single block enabled on top of the base config.
- Block totals do not sum to the full pipeline wall time because the pipeline also includes orchestration, array conversions, logging, object construction, and non-instrumented glue code.
- The `crop` spot check changes the image size from 1920x1536 to 300x300, so its wall-time delta mostly reflects the reduced downstream workload rather than the crop operation itself.
- `yuv_conversion_format` is configured on, but in this profile it bypassed conversion after logging `Invalid input for YUV conversion: RGB image format.`, so no execution-time sample was emitted.
- Results are input-dependent. Different RAW frames, dimensions, algorithms, or hardware backends will change the timing profile.
