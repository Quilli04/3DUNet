from pathlib import Path
import numpy as np
import os
import glob

NPY_FILES_DIR = Path(r"D:\datasets\brain_mrt\npy_files")

scan_file_paths = glob.glob(str(NPY_FILES_DIR/"*/scans/*.npy"))

for path in scan_file_paths:
    arr = np.load(path)
    if arr.shape == (3, 128, 128, 128):
        continue
    arr = arr.transpose(3, 0, 1, 2)
    np.save(path, arr)
