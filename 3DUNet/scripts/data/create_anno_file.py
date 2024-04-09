from pathlib import Path
import pandas as pd
import os

TRAIN_DIR = Path(r"D:\datasets\brain_mrt\npy_files\train")
VAL_DIR = Path(r"D:\datasets\brain_mrt\npy_files\val")

DIR = VAL_DIR

scan_names = sorted(os.listdir(DIR/"scans"))
mask_names = sorted(os.listdir(DIR/"masks"))

# df = pd.DataFrame(list(zip(scan_names, mask_names)), columns=["scan", "mask"])
df = pd.DataFrame({"scan": scan_names, "mask": mask_names})
df.to_csv(DIR/"annotations.csv", index=False, mode="w")

