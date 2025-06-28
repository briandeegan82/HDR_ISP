"""
File: clahe_opencv_cuda.py
Description: OpenCV CUDA CLAHE implementation for LDCI
Author: 10xEngineers
------------------------------------------------------------
"""
import time
import numpy as np
import cv2
from util.utils import save_output_array_yuv


class CLAHEOpenCVCUDA:
    """
    OpenCV CUDA Contrast Limited Adaptive Histogram Equalization
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
        
        # Check OpenCV CUDA availability
        self.opencv_cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

    def apply_clahe_opencv_cuda(self):
        """
        Apply OpenCV CUDA CLAHE to the Y channel
        """
        total_start = time.time()
        try:
            yuv_input = self.yuv.copy()
            y_channel = yuv_input[:, :, 0].astype(np.uint8)
            if yuv_input.dtype == np.float32:
                y_channel = np.round(255 * y_channel).astype(np.uint8)

            # Use a standard tileGridSize (8, 8)
            tile_grid_size = (8, 8)
            opencv_clip_limit = max(1.0, min(4.0, self.clip_limit / (self.wind * self.wind / 256)))
            clahe = cv2.createCLAHE(
                clipLimit=opencv_clip_limit,
                tileGridSize=tile_grid_size
            )
            y_enhanced = clahe.apply(y_channel)

            # Ensure output shape matches input
            if y_enhanced.shape != y_channel.shape:
                print(f"[DEBUG] CLAHE output shape {y_enhanced.shape} != input shape {y_channel.shape}, resizing...")
                y_enhanced = cv2.resize(y_enhanced, (y_channel.shape[1], y_channel.shape[0]), interpolation=cv2.INTER_NEAREST)

            print(f"[DEBUG] y_channel.shape: {y_channel.shape}, y_enhanced.shape: {y_enhanced.shape}")
            out_ceh = np.empty_like(yuv_input)
            print(f"[DEBUG] out_ceh.shape: {out_ceh.shape}")
            try:
                out_ceh[:, :, 0] = y_enhanced
                print(f"[DEBUG] Successfully assigned y_enhanced to out_ceh[:, :, 0]")
            except Exception as e:
                print(f"[DEBUG] Error assigning y_enhanced to out_ceh[:, :, 0]: {e}")
                print(f"[DEBUG] y_enhanced.shape: {y_enhanced.shape}, out_ceh[:, :, 0].shape: {out_ceh[:, :, 0].shape}")
                raise
            out_ceh[:, :, 1] = yuv_input[:, :, 1]
            out_ceh[:, :, 2] = yuv_input[:, :, 2]
            return out_ceh
        except Exception as e:
            print(f"\nError in OpenCV CLAHE processing: {str(e)}")
            print(f"Image shape: {self.yuv.shape}")
            print(f"Window size: {self.wind}")
            print(f"Clip limit: {self.clip_limit}")
            print(f"Tile grid size: (8, 8)")
            raise


class LDCIOpenCVCUDA:
    """
    OpenCV CUDA-optimized Local Dynamic Contrast Enhancement
    """

    def __init__(self, yuv, platform, sensor_info, parm_ldci, conv_std=None):
        print(f"[DEBUG] LDCIOpenCVCUDA __init__ called with yuv.shape={getattr(yuv, 'shape', None)}, type={type(yuv)}")
        self.yuv = yuv
        self.img = yuv
        self.enable = parm_ldci["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ldci = parm_ldci
        self.is_save = parm_ldci["is_save"]
        self.platform = platform
        self.conv_std = conv_std  # Accept but do not use

    def apply_ldci(self):
        """
        Applying OpenCV CUDA LDCI module to the given image
        """
        #print("\nOpenCV CUDA LDCI Processing:")
        #print(f"  Window size: {self.parm_ldci['wind']}")
        #print(f"  Clip limit: {self.parm_ldci['clip_limit']}")
        
        clahe_start = time.time()
        clahe = CLAHEOpenCVCUDA(self.yuv, self.platform, self.sensor_info, self.parm_ldci)
        out_ceh = clahe.apply_clahe_opencv_cuda()
        clahe_time = time.time() - clahe_start
        #print(f"  OpenCV CUDA CLAHE processing time: {clahe_time:.3f}s")

        return out_ceh

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_start = time.time()
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_ldci_opencv_cuda_",
                self.platform,
                self.conv_std,
            )
            save_time = time.time() - save_start
            #print(f"  Save time: {save_time:.3f}s")

    def execute(self):
        """
        Executing OpenCV CUDA LDCI module according to user choice
        """
        print(f"[DEBUG] LDCIOpenCVCUDA execute() called, enable={self.enable}")
        print(f"[DEBUG] self.yuv.shape: {self.yuv.shape}, self.img.shape: {self.img.shape}")

        if self.enable is True:
            print(f"[DEBUG] LDCI is enabled, calling apply_ldci()")
            total_start = time.time()
            try:
                s_out = self.apply_ldci()
                print(f"[DEBUG] apply_ldci() returned shape: {s_out.shape}")
                total_time = time.time() - total_start
                #print(f"  Total processing time: {total_time:.3f}s")
                self.img = s_out
                print(f"[DEBUG] self.img updated to shape: {self.img.shape}")
            except Exception as e:
                print(f"[DEBUG] Error in apply_ldci(): {e}")
                raise

        print(f"[DEBUG] Calling save()")
        self.save()
        print(f"[DEBUG] Returning self.img with shape: {self.img.shape}")
        return self.img 