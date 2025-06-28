"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
from pathlib import Path
import numpy as np
import yaml
import rawpy
from matplotlib import pyplot as plt
import tifffile as tiff
import os

import util.utils as util

from modules.crop.crop import Crop
from modules.dead_pixel_correction.dead_pixel_correction_halide import (
    DeadPixelCorrectionHalideFallback as DPC,
)
from modules.black_level_correction.black_level_correction_numba_fallback import (
    BlackLevelCorrectionNumbaFallback as BLC,
)
from modules.pwc_generation.pwc_generation_numba_fallback import (
    PiecewiseCurveNumbaFallback as PWC,
)
from modules.oecf.oecf import OECF
from modules.digital_gain.digital_gain import DigitalGain as DG
from modules.lens_shading_correction.lens_shading_correction import (
    LensShadingCorrection as LSC,
)
from modules.bayer_noise_reduction.bayer_noise_reduction import (
    BayerNoiseReduction as BNR,
)
from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance as AWB
from modules.white_balance.white_balance import WhiteBalance as WB
from modules.hdr_durand.hdr_durand_fast import HDRDurandToneMapping as HDR
from modules.demosaic.demosaic_opencv_cuda_fallback import DemosaicOpenCVCUDAFallback as Demosaic
from modules.color_correction_matrix.color_correction_matrix_cuda_fallback import ColorCorrectionMatrixCUDAFallback as CCM
from modules.gamma_correction.gamma_correction_numba_fallback import GammaCorrectionNumbaFallback as GC
from modules.auto_exposure.auto_exposure_cuda_fallback import AutoExposureCUDAFallback as AE
from modules.color_space_conversion.color_space_conversion_fallback import ColorSpaceConversionFallback as CSC
from modules.ldci.ldci import LDCI
from modules.sharpen.sharpen import Sharpening as SHARP
from modules.noise_reduction_2d.noise_reduction_2d import NoiseReduction2d as NR2D
from modules.rgb_conversion.rgb_conversion import RGBConversion as RGBC
from modules.scale.scale import Scale
from modules.yuv_conv_format.yuv_conv_format import YUVConvFormat as YUV_C

# Numba warm-up for gamma correction
try:
    from modules.gamma_correction.gamma_correction_numba_fallback import generate_gamma_lut_numba, apply_gamma_numba
    import numpy as np
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint16)
    dummy_lut = generate_gamma_lut_numba(12)
    _ = apply_gamma_numba(dummy_img, dummy_lut)
except Exception:
    pass

# Dummy profile decorator for normal runs (no-op if not using kernprof)
try:
    profile
except NameError:
    def profile(func):
        return func

class InfiniteISP:
    """
    Infinite-ISP Pipeline
    """

    def __init__(self, data_path, config_path, memory_mapped_data=None):
        """
        Constructor: Initialize with config and raw file path
        and Load configuration parameter from yaml file
        
        Args:
            data_path (str): Path to the data directory
            config_path (str): Path to the configuration file
            memory_mapped_data (numpy.memmap, optional): Memory-mapped array for large files
        """
        self.data_path = data_path
        self.memory_mapped_data = memory_mapped_data
        self._module_instances = {}  # Module instance pool
        self.load_config(config_path)

    def _get_module(self, module_class, *args, **kwargs):
        """
        Get or create module instance from the pool
        
        Args:
            module_class: The class of the module to get/create
            *args: Arguments to pass to the module constructor
            **kwargs: Keyword arguments to pass to the module constructor
            
        Returns:
            An instance of the requested module
        """
        if module_class not in self._module_instances:
            self._module_instances[module_class] = module_class(*args, **kwargs)
        return self._module_instances[module_class]

    def load_config(self, config_path):
        """
        Load config information to respective module parameters
        """
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as file:
            c_yaml = yaml.safe_load(file)

            # Extract workspace info
            self.platform = c_yaml["platform"]
            self.raw_file = self.platform["filename"]
            self.render_3a = self.platform["render_3a"]

            # Extract basic sensor info
            self.sensor_info = c_yaml["sensor_info"]

            # Get isp module params
            self.parm_dpc = c_yaml["dead_pixel_correction"]
            self.parm_cmpd = c_yaml["companding"]
            self.parm_dga = c_yaml["digital_gain"]
            self.parm_lsc = c_yaml["lens_shading_correction"]
            self.parm_bnr = c_yaml["bayer_noise_reduction"]
            self.parm_blc = c_yaml["black_level_correction"]
            self.parm_oec = c_yaml["oecf"]
            self.parm_wbc = c_yaml["white_balance"]
            self.parm_awb = c_yaml["auto_white_balance"]
            self.parm_dem = c_yaml["demosaic"]
            self.parm_ae = c_yaml["auto_exposure"]
            self.parm_ccm = c_yaml["color_correction_matrix"]
            self.parm_gmc = c_yaml["gamma_correction"]
            self.param_durand = c_yaml["hdr_durand"]
            self.parm_csc = c_yaml["color_space_conversion"]
            self.parm_cse = c_yaml["color_saturation_enhancement"]
            self.parm_ldci = c_yaml["ldci"]
            self.parm_sha = c_yaml["sharpen"]
            self.parm_2dn = c_yaml["2d_noise_reduction"]
            self.parm_rgb = c_yaml["rgb_conversion"]
            self.parm_sca = c_yaml["scale"]
            self.parm_cro = c_yaml["crop"]
            self.parm_yuv = c_yaml["yuv_conversion_format"]
            self.c_yaml = c_yaml

            self.platform["rgb_output"] = self.parm_rgb["is_enable"]

        # add rgb_output_conversion module

    def load_raw(self):
        """
        Load raw image from provided path using memory mapping for large files
        """
        # Load raw image file information
        path_object = Path(self.data_path, self.raw_file)
        raw_path = str(path_object.resolve())
        self.in_file = path_object.stem
        self.out_file = "Out_" + self.in_file

        self.platform["in_file"] = self.in_file
        self.platform["out_file"] = self.out_file

        width = self.sensor_info["width"]
        height = self.sensor_info["height"]
        bit_depth = self.sensor_info["bit_depth"]
        
        # Calculate file size to determine if memory mapping should be used
        file_size = path_object.stat().st_size
        use_mmap = file_size > 100 * 1024 * 1024  # Use mmap for files > 100MB

        # Load Raw
        if self.memory_mapped_data is not None:
            # Use provided memory-mapped data
            if bit_depth > 8:
                self.raw = self.memory_mapped_data.reshape((height, width))
            else:
                self.raw = self.memory_mapped_data.reshape((height, width)).astype(np.uint16)
        elif path_object.suffix == ".raw":
            if use_mmap:
                # Use memory mapping for large raw files
                if bit_depth > 8:
                    self.raw = np.memmap(raw_path, dtype='>u2', mode='r', shape=(height, width))
                else:
                    self.raw = np.memmap(raw_path, dtype=np.uint8, mode='r', shape=(height, width)).astype(np.uint16)
            else:
                # Direct loading for smaller files
                if bit_depth > 8:
                    self.raw = np.fromfile(raw_path, dtype='>u2').reshape((height, width))
                else:
                    self.raw = np.fromfile(raw_path, dtype=np.uint8).reshape((height, width)).astype(np.uint16)
        elif path_object.suffix == ".tiff":
            # Load tiff file with memory mapping for large files
            if use_mmap:
                img = tiff.imread(raw_path, aszarr=True)
                if img.ndim == 3:
                    self.raw = img[:, :, 0]
                else:
                    self.raw = img
            else:
                img = tiff.imread(raw_path)
                if img.ndim == 3:
                    self.raw = img[:, :, 0]
                else:
                    self.raw = img
        else:
            # For other formats, use rawpy with memory mapping if possible
            if use_mmap:
                with rawpy.imread(raw_path) as raw:
                    self.raw = np.memmap(raw_path, dtype=raw.raw_image.dtype, mode='r', shape=raw.raw_image.shape)
            else:
                with rawpy.imread(raw_path) as raw:
                    self.raw = raw.raw_image.copy()  # Use copy to ensure data is loaded into memory

    @profile
    def run_pipeline(self, visualize_output=True, save_intermediate=False):
        """
        Simulation of ISP-Pipeline
        
        Args:
            visualize_output (bool): Whether to visualize and save the final output
            save_intermediate (bool): Whether to save intermediate images at each stage
        """
        # Create output directory for intermediate images if needed
        if save_intermediate:
            intermediate_dir = Path("out_frames/intermediate")
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            stage_count = 0

        timings = []

        # =====================================================================
        # Cropping
        t0 = time.time()
        crop = self._get_module(Crop, self.raw, self.platform, self.sensor_info, self.parm_cro)
        img = crop.execute()
        timings.append(("Crop", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_crop.png"))
            stage_count += 1

        # =====================================================================
        #  Dead pixels correction
        t0 = time.time()
        dpc = self._get_module(DPC, img, self.sensor_info, self.parm_dpc, self.platform)
        img = dpc.execute()
        timings.append(("Dead Pixel Correction", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_dead_pixel_correction.png"))
            stage_count += 1

        # =====================================================================
        # Black level correction
        t0 = time.time()
        blc = self._get_module(BLC, img, self.platform, self.sensor_info, self.parm_blc)
        img = blc.execute()
        timings.append(("Black Level Correction", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_black_level_correction.png"))
            stage_count += 1

        # =====================================================================
        # decompanding
        t0 = time.time()
        cmpd = self._get_module(PWC, img, self.platform, self.sensor_info, self.parm_cmpd)
        img = cmpd.execute()
        timings.append(("Decompanding", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_decompanding.png"))
            stage_count += 1

        # =====================================================================
        # OECF
        t0 = time.time()
        oecf = self._get_module(OECF, img, self.platform, self.sensor_info, self.parm_oec)
        img = oecf.execute()
        timings.append(("OECF", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_oecf.png"))
            stage_count += 1

        # =====================================================================
        # Digital Gain
        t0 = time.time()
        dga = self._get_module(DG, img, self.platform, self.sensor_info, self.parm_dga)
        img, self.dga_current_gain = dga.execute()
        timings.append(("Digital Gain", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_digital_gain.png"))
            stage_count += 1

        # =====================================================================
        # Lens shading correction
        t0 = time.time()
        lsc = self._get_module(LSC, img, self.platform, self.sensor_info, self.parm_lsc)
        img = lsc.execute()
        timings.append(("Lens Shading Correction", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_lens_shading_correction.png"))
            stage_count += 1

        # =====================================================================
        # Bayer noise reduction
        t0 = time.time()
        bnr = self._get_module(BNR, img, self.sensor_info, self.parm_bnr, self.platform)
        img = bnr.execute()
        timings.append(("Bayer Noise Reduction", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_bayer_noise_reduction.png"))
            stage_count += 1

        # =====================================================================
        # Auto White Balance
        t0 = time.time()
        awb = self._get_module(AWB, img, self.sensor_info, self.parm_awb)
        self.awb_gains = awb.execute()
        timings.append(("Auto White Balance", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_auto_white_balance.png"))
            stage_count += 1

        # =====================================================================
        # White Balance
        t0 = time.time()
        wb = self._get_module(WB, img, self.platform, self.sensor_info, self.parm_wbc)
        img = wb.execute()
        timings.append(("White Balance", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_white_balance.png"))
            stage_count += 1

        # =====================================================================
        # HDR Durand Tone Mapping
        if self.param_durand["is_enable"]:
            print("[Pipeline] HDR Durand start")
            t0 = time.time()
            hdr = self._get_module(HDR, img, self.platform, self.sensor_info, self.param_durand)
            img = hdr.execute()
            # Explicit GPU sync if using CuPy
            try:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
                print("[Pipeline] CuPy GPU sync after HDR Durand")
            except ImportError:
                pass
            print(f"[Pipeline] HDR Durand end: {time.time() - t0:.4f}s")
            timings.append(("HDR Durand", time.time() - t0))
            print("[Pipeline] After HDR Durand, before CCM")
            t_durand_ccm = time.time()
            if save_intermediate:
                util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_hdr_durand.png"))
                stage_count += 1

        # =====================================================================
        # CFA interpolation
        t0 = time.time()
        dem = self._get_module(Demosaic, img, self.platform, self.sensor_info, self.parm_dem)
        img = dem.execute()
        timings.append(("Demosaic", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_demosaic.png"))
            stage_count += 1

        # =====================================================================
        # Color Correction Matrix
        print("[Pipeline] CCM start")
        t_ccm = time.time()
        ccm = self._get_module(CCM, img, self.platform, self.sensor_info, self.parm_ccm)
        img = ccm.execute()
        print(f"[Pipeline] CCM end: {time.time() - t_ccm:.4f}s (elapsed since Durand: {time.time() - t_durand_ccm:.4f}s)")
        timings.append(("Color Correction Matrix", time.time() - t_ccm))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_color_correction_matrix.png"))
            stage_count += 1

        # =====================================================================
        # Gamma Correction
        t0 = time.time()
        gmc = self._get_module(GC, img, self.platform, self.sensor_info, self.parm_gmc)
        img = gmc.execute()
        timings.append(("Gamma Correction", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_gamma_correction.png"))
            stage_count += 1

        # =====================================================================
        # Auto Exposure
        t0 = time.time()
        ae = self._get_module(AE, img, self.sensor_info, self.parm_ae)
        self.ae_feedback = ae.execute()
        timings.append(("Auto Exposure", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_auto_exposure.png"))
            stage_count += 1

        # =====================================================================
        # Color Space Conversion
        t0 = time.time()
        csc = self._get_module(CSC, img, self.platform, self.sensor_info, self.parm_csc, self.parm_cse)
        img = csc.execute()
        timings.append(("Color Space Conversion", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_color_space_conversion.png"))
            stage_count += 1

        # =====================================================================
        # LDCI
        t0 = time.time()
        ldci = self._get_module(LDCI, img, self.platform, self.sensor_info, self.parm_ldci, self.parm_csc["conv_standard"])
        img = ldci.execute()
        timings.append(("LDCI", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_ldci.png"))
            stage_count += 1

        # =====================================================================
        # Sharpening
        t0 = time.time()
        sha = self._get_module(SHARP, img, self.platform, self.sensor_info, self.parm_sha, self.parm_csc["conv_standard"])
        img = sha.execute()
        timings.append(("Sharpening", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_sharpening.png"))
            stage_count += 1

        # =====================================================================
        # 2D Noise Reduction
        if self.parm_2dn["is_enable"]:
            t0 = time.time()
            nr2d = self._get_module(NR2D, img, self.platform, self.parm_2dn)
            img = nr2d.execute()
            timings.append(("2D Noise Reduction", time.time() - t0))
            if save_intermediate:
                util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_2d_noise_reduction.png"))
                stage_count += 1

        # =====================================================================
        # RGB Conversion
        t0 = time.time()
        rgbc = self._get_module(RGBC, img, self.platform, self.sensor_info, self.parm_rgb, self.parm_csc)
        img = rgbc.execute()
        timings.append(("RGB Conversion", time.time() - t0))
        if save_intermediate:
            util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_rgb_conversion.png"))
            stage_count += 1

        # =====================================================================
        # Scale
        if self.parm_sca["is_enable"]:
            t0 = time.time()
            sca = self._get_module(Scale, img, self.platform, self.parm_sca)
            img = sca.execute()
            timings.append(("Scale", time.time() - t0))
            if save_intermediate:
                util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_scale.png"))
                stage_count += 1

        # =====================================================================
        # YUV Conversion Format
        if self.parm_yuv["is_enable"]:
            t0 = time.time()
            yuv = self._get_module(YUV_C, img, self.platform, self.sensor_info, self.parm_yuv)
            img = yuv.execute()
            timings.append(("YUV Conversion Format", time.time() - t0))
            if save_intermediate:
                util.save_image(img, os.path.join(intermediate_dir, f"{stage_count:02d}_yuv_conversion.png"))
                stage_count += 1

        # Print timings for all stages
        print("\n--- Pipeline Stage Timings ---")
        for name, t in timings:
            print(f"{name:28}: {t:.4f} s")
        print("-----------------------------\n")
        return img

    def execute(self, img_path=None, save_intermediate=False):
        """
        Start execution of Infinite-ISP
        
        Args:
            img_path (str, optional): Path to the input image. If None, uses the path from config.
            save_intermediate (bool): Whether to save intermediate images at each stage.
        """
        if img_path is not None:
            self.raw_file = img_path
            self.c_yaml["platform"]["filename"] = self.raw_file

        self.load_raw()

        # Print Logs to mark start of pipeline Execution
        print(50 * "-" + "\nLoading RAW Image Done......\n")
        print("Filename: ", self.in_file)

        # Note Initial Time for Pipeline Execution
        start = time.time()
        
        # Generate timestamp for output filename
        timestamp = time.strftime("_%Y%m%d_%H%M%S")

        if not self.render_3a:
            # Run ISP-Pipeline once
            final_img = self.run_pipeline(visualize_output=True, save_intermediate=save_intermediate)
            # Save final output
            output_dir = Path("out_frames")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.out_file}{timestamp}.png"
            util.save_image(final_img, output_path)
            print(f"Final output saved to: {output_path}")
        else:
            # Run ISP-Pipeline till Correct Exposure with AWB gains
            final_img = self.execute_with_3a_statistics(save_intermediate)
            # Save final output
            output_dir = Path("out_frames")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.out_file}{timestamp}.png"
            util.save_image(final_img, output_path)
            print(f"Final output saved to: {output_path}")

        #util.display_ae_statistics(self.ae_feedback, self.awb_gains)

        # Print Logs to mark end of pipeline Execution
        print(50 * "-" + "\n")

        # Calculate pipeline execution time
        print(f"\nPipeline Elapsed Time: {time.time() - start:.3f}s")

    def load_3a_statistics(self, awb_on=True, ae_on=True):
        """
        Update 3A Stats into WB and DG modules parameters
        """
        # Update 3A in c_yaml too because it is output config
        if awb_on is True and self.parm_dga["is_auto"] and self.parm_awb["is_enable"]:
            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"]["r_gain"] = float(
                self.awb_gains[0]
            )
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"]["b_gain"] = float(
                self.awb_gains[1]
            )
        if ae_on is True and self.parm_dga["is_auto"] and self.parm_ae["is_enable"]:
            self.parm_dga["ae_feedback"] = self.c_yaml["digital_gain"][
                "ae_feedback"
            ] = self.ae_feedback
            self.parm_dga["current_gain"] = self.c_yaml["digital_gain"][
                "current_gain"
            ] = self.dga_current_gain

    def execute_with_3a_statistics(self, save_intermediate=False):
        """
        Execute Infinite-ISP with AWB gains and correct exposure
        
        Args:
            save_intermediate (bool): Whether to save intermediate images at each stage
        """
        # Maximum Iterations depend on total permissible gains
        max_dg = len(self.parm_dga["gain_array"])

        # Run ISP-Pipeline
        self.run_pipeline(visualize_output=False, save_intermediate=save_intermediate)
        self.load_3a_statistics()
        while not (
            (self.ae_feedback == 0)
            or (self.ae_feedback == -1 and self.dga_current_gain == max_dg)
            or (self.ae_feedback == 1 and self.dga_current_gain == 0)
            or self.ae_feedback is None
        ):
            self.run_pipeline(visualize_output=False, save_intermediate=save_intermediate)
            self.load_3a_statistics()

        self.run_pipeline(visualize_output=True, save_intermediate=save_intermediate)

    def update_sensor_info(self, sensor_info, update_blc_wb=False):
        """
        Update sensor_info in config files
        """
        self.sensor_info["width"] = self.c_yaml["sensor_info"]["width"] = sensor_info[0]

        self.sensor_info["height"] = self.c_yaml["sensor_info"]["height"] = sensor_info[
            1
        ]

        self.sensor_info["bit_depth"] = self.c_yaml["sensor_info"][
            "bit_depth"
        ] = sensor_info[2]

        self.sensor_info["bayer_pattern"] = self.c_yaml["sensor_info"][
            "bayer_pattern"
        ] = sensor_info[3]

        if update_blc_wb:
            self.parm_blc["r_offset"] = self.c_yaml["black_level_correction"][
                "r_offset"
            ] = sensor_info[4][0]
            self.parm_blc["gr_offset"] = self.c_yaml["black_level_correction"][
                "gr_offset"
            ] = sensor_info[4][1]
            self.parm_blc["gb_offset"] = self.c_yaml["black_level_correction"][
                "gb_offset"
            ] = sensor_info[4][2]
            self.parm_blc["b_offset"] = self.c_yaml["black_level_correction"][
                "b_offset"
            ] = sensor_info[4][3]

            self.parm_blc["r_sat"] = self.c_yaml["black_level_correction"][
                "r_sat"
            ] = sensor_info[5]
            self.parm_blc["gr_sat"] = self.c_yaml["black_level_correction"][
                "gr_sat"
            ] = sensor_info[5]
            self.parm_blc["gb_sat"] = self.c_yaml["black_level_correction"][
                "gb_sat"
            ] = sensor_info[5]
            self.parm_blc["b_sat"] = self.c_yaml["black_level_correction"][
                "b_sat"
            ] = sensor_info[5]

            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"][
                "r_gain"
            ] = sensor_info[6][0]
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"][
                "b_gain"
            ] = sensor_info[6][2]

            # if sensor_info[7] is not None:
            #     self.parm_ccm["corrected_red"] = sensor_info[7][0,0:3]
            #     self.parm_ccm["corrected_green"] = sensor_info[7][1,0:3]
            #     self.parm_ccm["corrected_blue"] = sensor_info[7][2,0:3]
