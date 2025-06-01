"""
File: ldci.py
Description: Implements the contrast adjustment in the yuv domain
Author: 10xEngineers
------------------------------------------------------------
"""
import time
from util.utils import save_output_array_yuv
from modules.ldci.clahe import CLAHE


class LDCI:
    """
    Local Dynamic Contrast Enhancement
    """

    def __init__(self, yuv, platform, sensor_info, parm_ldci, conv_std):
        self.yuv = yuv
        self.img = yuv
        self.enable = parm_ldci["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ldci = parm_ldci
        self.is_save = parm_ldci["is_save"]
        self.platform = platform
        self.conv_std = conv_std

    def apply_ldci(self):
        """
        Applying LDCI module to the given image
        """
        #print("\nLDCI Processing:")
        #print(f"  Window size: {self.parm_ldci['wind']}")
        #print(f"  Clip limit: {self.parm_ldci['clip_limit']}")
        
        clahe_start = time.time()
        clahe = CLAHE(self.yuv, self.platform, self.sensor_info, self.parm_ldci)
        out_ceh = clahe.apply_clahe()
        clahe_time = time.time() - clahe_start
        #print(f"  CLAHE processing time: {clahe_time:.3f}s")

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
                "Out_ldci_",
                self.platform,
                self.conv_std,
            )
            save_time = time.time() - save_start
            #print(f"  Save time: {save_time:.3f}s")

    def execute(self):
        """
        Executing LDCI module according to user choice
        """
        print("\nLDCI Module:")
        print(f"  Enabled: {self.enable}")

        if self.enable is True:
            total_start = time.time()
            s_out = self.apply_ldci()
            total_time = time.time() - total_start
            print(f"  Total processing time: {total_time:.3f}s")
            self.img = s_out

        self.save()
        return self.img
