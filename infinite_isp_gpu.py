"""
File: infinite_isp_gpu.py
Description: GPU-accelerated ISP pipeline execution
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

import util.utils as util

from modules.crop.crop import Crop
from modules.dead_pixel_correction.dead_pixel_correction import (
    DeadPixelCorrection as DPC,
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
# GPU-accelerated modules
from modules.bayer_noise_reduction.bayer_noise_reduction_gpu import BayerNoiseReductionGPU as BNR
from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance as AWB
from modules.white_balance.white_balance import WhiteBalance as WB
from modules.hdr_durand.hdr_durand_gpu import HDRDurandToneMappingGPU as HDR
from modules.demosaic.demosaic_opencv_cuda_fallback import DemosaicOpenCVCUDAFallback as Demosaic
from modules.color_correction_matrix.color_correction_matrix_cuda_fallback import ColorCorrectionMatrixCUDAFallback as CCM
from modules.gamma_correction.gamma_correction import GammaCorrection as GC
from modules.auto_exposure.auto_exposure import AutoExposure as AE
from modules.color_space_conversion.color_space_conversion_gpu import ColorSpaceConversionGPU as CSC
from modules.ldci.ldci import LDCI
from modules.sharpen.sharpen_gpu import SharpeningGPU as SHARP
from modules.noise_reduction_2d.noise_reduction_2d import NoiseReduction2d as NR2D
from modules.rgb_conversion.rgb_conversion import RGBConversion as RGBC
from modules.scale.scale_gpu import ScaleGPU as Scale
from modules.yuv_conv_format.yuv_conv_format import YUVConvFormat as YUV_C


class InfiniteISPGPU:
    """
    GPU-accelerated Infinite-ISP Pipeline
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

    def run_pipeline(self, visualize_output=True, save_intermediate=False):
        """
        GPU-accelerated simulation of ISP-Pipeline
        
        Args:
            visualize_output (bool): Whether to visualize and save the final output
            save_intermediate (bool): Whether to save intermediate images at each stage
        """
        # Create output directory for intermediate images if needed
        if save_intermediate:
            intermediate_dir = Path("out_frames/intermediate_gpu")
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            stage_count = 0

        total_start_time = time.time()

        # =====================================================================
        # Cropping
        crop = self._get_module(Crop, self.raw, self.platform, self.sensor_info, self.parm_cro)
        img = crop.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_crop.png")
            stage_count += 1

        # =====================================================================
        #  Dead pixels correction
        dpc = self._get_module(DPC, img, self.sensor_info, self.parm_dpc, self.platform)
        img = dpc.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_dead_pixel_correction.png")
            stage_count += 1

        # =====================================================================
        # Black level correction
        blc = self._get_module(BLC, img, self.platform, self.sensor_info, self.parm_blc)
        img = blc.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_black_level_correction.png")
            stage_count += 1

        # =====================================================================
        # decompanding
        cmpd = self._get_module(PWC, img, self.platform, self.sensor_info, self.parm_cmpd)
        img = cmpd.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_decompanding.png")
            stage_count += 1

        # =====================================================================
        # OECF
        oecf = self._get_module(OECF, img, self.platform, self.sensor_info, self.parm_oec)
        img = oecf.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_oecf.png")
            stage_count += 1

        # =====================================================================
        # Digital Gain
        dga = self._get_module(DG, img, self.platform, self.sensor_info, self.parm_dga)
        img, self.dga_current_gain = dga.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_digital_gain.png")
            stage_count += 1

        # =====================================================================
        # Lens shading correction
        lsc = self._get_module(LSC, img, self.platform, self.sensor_info, self.parm_lsc)
        img = lsc.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_lens_shading_correction.png")
            stage_count += 1

        # =====================================================================
        # GPU-accelerated Bayer noise reduction
        bnr = self._get_module(BNR, img, self.sensor_info, self.parm_bnr, self.platform)
        img = bnr.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_bayer_noise_reduction_gpu.png")
            stage_count += 1

        # =====================================================================
        # Auto White Balance
        awb = self._get_module(AWB, img, self.sensor_info, self.parm_awb)
        self.awb_gains = awb.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_auto_white_balance.png")
            stage_count += 1

        # =====================================================================
        # White Balance
        wb = self._get_module(WB, img, self.platform, self.sensor_info, self.parm_wbc)
        img = wb.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_white_balance.png")
            stage_count += 1

        # =====================================================================
        # GPU-accelerated HDR Durand Tone Mapping
        if self.param_durand["is_enable"]:
            hdr = self._get_module(HDR, img, self.platform, self.sensor_info, self.param_durand)
            img = hdr.execute()
            if save_intermediate:
                util.save_image(img, intermediate_dir / f"{stage_count:02d}_hdr_durand_gpu.png")
                stage_count += 1

        # =====================================================================
        # GPU-accelerated CFA interpolation
        dem = self._get_module(Demosaic, img, self.platform, self.sensor_info, self.parm_dem)
        img = dem.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_demosaic_gpu.png")
            stage_count += 1

        # =====================================================================
        # Color Correction Matrix
        ccm = self._get_module(CCM, img, self.platform, self.sensor_info, self.parm_ccm)
        img = ccm.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_color_correction_matrix.png")
            stage_count += 1

        # =====================================================================
        # Gamma Correction
        gmc = self._get_module(GC, img, self.platform, self.sensor_info, self.parm_gmc)
        img = gmc.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_gamma_correction.png")
            stage_count += 1

        # =====================================================================
        # Auto Exposure
        ae = self._get_module(AE, img, self.sensor_info, self.parm_ae)
        self.ae_feedback = ae.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_auto_exposure.png")
            stage_count += 1

        # =====================================================================
        # GPU-accelerated Color Space Conversion
        csc = self._get_module(CSC, img, self.platform, self.sensor_info, self.parm_csc, self.parm_cse)
        img = csc.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_color_space_conversion_gpu.png")
            stage_count += 1

        # =====================================================================
        # LDCI
        ldci = self._get_module(LDCI, img, self.platform, self.sensor_info, self.parm_ldci, self.parm_csc["conv_standard"])
        img = ldci.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_ldci.png")
            stage_count += 1

        # =====================================================================
        # GPU-accelerated Sharpening
        sha = self._get_module(SHARP, img, self.platform, self.sensor_info, self.parm_sha, self.parm_csc["conv_standard"])
        img = sha.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_sharpening_gpu.png")
            stage_count += 1

        # =====================================================================
        # 2D Noise Reduction
        if self.parm_2dn["is_enable"]:
            nr2d = self._get_module(NR2D, img, self.sensor_info, self.parm_2dn, self.platform, self.parm_csc["conv_standard"])
            img = nr2d.execute()
            if save_intermediate:
                util.save_image(img, intermediate_dir / f"{stage_count:02d}_2d_noise_reduction.png")
                stage_count += 1

        # =====================================================================
        # RGB Conversion
        rgbc = self._get_module(RGBC, img, self.platform, self.sensor_info, self.parm_rgb, self.parm_csc)
        img = rgbc.execute()
        if save_intermediate:
            util.save_image(img, intermediate_dir / f"{stage_count:02d}_rgb_conversion.png")
            stage_count += 1

        # =====================================================================
        # GPU-accelerated Scale
        if self.parm_sca["is_enable"]:
            sca = self._get_module(Scale, img, self.platform, self.sensor_info, self.parm_sca, self.parm_csc["conv_standard"])
            img = sca.execute()
            if save_intermediate:
                util.save_image(img, intermediate_dir / f"{stage_count:02d}_scale_gpu.png")
                stage_count += 1

        # =====================================================================
        # YUV Conversion Format
        if self.parm_yuv["is_enable"]:
            yuv = self._get_module(YUV_C, img, self.platform, self.sensor_info, self.parm_yuv)
            img = yuv.execute()
            if save_intermediate:
                util.save_image(img, intermediate_dir / f"{stage_count:02d}_yuv_conversion.png")
                stage_count += 1

        total_time = time.time() - total_start_time
        print(f"Total GPU-accelerated pipeline execution time: {total_time:.3f}s")

        return img

    def execute(self, img_path=None, save_intermediate=False):
        """
        Start execution of GPU-accelerated Infinite-ISP
        
        Args:
            img_path (str, optional): Path to save the final output image
            save_intermediate (bool): Whether to save intermediate images
        """
        print("Starting GPU-accelerated Infinite-ISP pipeline...")
        
        # Load raw image
        self.load_raw()
        
        # Run the pipeline
        output_img = self.run_pipeline(save_intermediate=save_intermediate)
        
        # Save final output
        if img_path:
            util.save_image(output_img, img_path)
            print(f"Final output saved to: {img_path}")
        
        return output_img

    def load_3a_statistics(self, awb_on=True, ae_on=True):
        """
        Load 3A statistics for AWB and AE
        """
        # This would load statistics from a separate file
        # For now, we'll use default values
        if awb_on:
            self.awb_gains = [1.0, 1.0, 1.0]  # Default gains
        if ae_on:
            self.ae_feedback = 1.0  # Default exposure feedback

    def execute_with_3a_statistics(self, save_intermediate=False):
        """
        Execute pipeline with pre-loaded 3A statistics
        """
        print("Starting GPU-accelerated Infinite-ISP pipeline with 3A statistics...")
        
        # Load raw image
        self.load_raw()
        
        # Load 3A statistics
        self.load_3a_statistics()
        
        # Run the pipeline
        output_img = self.run_pipeline(save_intermediate=save_intermediate)
        
        return output_img

    def update_sensor_info(self, sensor_info, update_blc_wb=False):
        """
        Update sensor information and optionally update BLC/WB parameters
        """
        self.sensor_info.update(sensor_info)
        if update_blc_wb:
            # Update BLC and WB parameters based on new sensor info
            pass 