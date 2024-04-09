import os
import numpy as np
from scipy import ndimage
import cupy as cp
import cupyx
from cupyx.scipy import ndimage as cp_ndimage
import glob
import nilearn as nl
import nilearn.image
from pathlib import Path
import time


TRAIN_DATA_DIR = Path(r"D:\datasets\brain_mrt\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
start_time = 0

seg_scans_paths = glob.glob(str(TRAIN_DATA_DIR/r"*\*seg.nii"))

seg_scans_lis = [np.clip(nl.image.load_img(path).get_fdata(), 0, 1).astype(np.ubyte) for path in seg_scans_paths]

seg_scans_stacked = np.stack(seg_scans_lis, axis=3)
# print(seg_scans_stacked.min(), seg_scans_stacked.max())
# print(seg_scans_stacked.shape)

seg_scans_stacked = seg_scans_stacked.sum(axis=3)
# print(seg_scans_stacked.min(), seg_scans_stacked.max())
# print(seg_scans_stacked.shape)

print("Starting calculation with numpy...")
# start_time = time.perf_counter()
center_of_mass = ndimage.center_of_mass(seg_scans_stacked)
center_of_mass = [round(val) for val in center_of_mass]
print(f"\nResult:")
print(f"\tcenter of mass: {center_of_mass}")
# print(f"\ttime elapsed: {time.perf_counter() - start_time}")

# Output: center of mass: (117.04233055606413, 127.01660535542032, 80.69418995100384)
num_labeled_all = seg_scans_stacked.sum()

dmin0, dmax0 = center_of_mass[0] - 64, center_of_mass[0] + 64   #
dmin1, dmax1 = center_of_mass[1] - 64, center_of_mass[1] + 64   # percentage lost: 3.0852%
dmin2, dmax2 = center_of_mass[2] - 64, center_of_mass[2] + 64   #
# dmin0, dmax0 = 56, 184    #
# dmin1, dmax1 = 56, 184    # percentage lost: 3.89223%
# dmin2, dmax2 = 13, 141    #

num_labeled_cropped = seg_scans_stacked[dmin0:dmax0, dmin1:dmax1, dmin2:dmax2].sum()

print(f"\tnumber of positive labels: all: {num_labeled_all} | cropped: {num_labeled_cropped}")
print(f"\tnumber of labels lost: {num_labeled_all-num_labeled_cropped}")
print(f"\tpercentage lost: {(1-num_labeled_cropped/num_labeled_all)*100:.5f}%")

# very slow, conversion to cp.ndarray on gpu expensive
#
# print("\nStarting calculation with cupy...")
# start_time = time.perf_counter()
# seg_scans_stacked_cp = cp.array(seg_scans_stacked)
# center_of_mass = cp_ndimage.center_of_mass(seg_scans_stacked_cp)
# print(f"Result:")
# print(f"\tcenter of mass: {center_of_mass}")
# print(f"\ttime elapsed: {time.perf_counter() - start_time}")








