from pathlib import Path
import os


TRAIN_DIR = Path(r"D:\datasets\brain_mrt\npy_files\train")
VAL_DIR = Path(r"D:\datasets\brain_mrt\npy_files\val")

DIR = TRAIN_DIR

scan_names = sorted(os.listdir(DIR/"scans"))
mask_names = sorted(os.listdir(DIR/"masks"))

for i, (scan_name, mask_name) in enumerate(zip(scan_names, mask_names)):
    old_scan_path = DIR/"scans"/scan_name
    old_mask_path = DIR/"masks"/mask_name
    # idx = mask_name.split(".")[0].split("_")[-1]
    new_scan_name = f"BraTS20_Training_{i+1:03}.npy"
    new_mask_name = f"BraTS20_Training_Mask_{i+1:03}.npy"

    new_scan_path = DIR/"scans"/new_scan_name
    new_mask_path = DIR/"masks"/new_mask_name

    os.rename(old_scan_path, new_scan_path)
    os.rename(old_mask_path, new_mask_path)




