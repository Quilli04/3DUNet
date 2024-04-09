import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path


TRAIN_DIR = Path(r"D:\datasets\brain_mrt\npy_files\train")
VAL_DIR = Path(r"D:\datasets\brain_mrt\npy_files\val")

DIR = VAL_DIR

scan_names = sorted(glob.glob("scans/*", root_dir=DIR))
mask_names = sorted(glob.glob("masks/*", root_dir=DIR))

for scan_name, mask_name in zip(scan_names, mask_names):
    scans = np.moveaxis(np.load(DIR/scan_name), 0, -1)
    # assert scans.max() <= 1
    # assert scans.min() >= 0
    mask = np.load(DIR/mask_name)
    # assert set(np.unique(mask)).issubset({0, 1, 2, 3})
    # print(scans.max(), scans.min(), set(np.unique(mask)))

    fig, axs = plt.subplots(ncols=4, figsize=(16, 8))

    axs[0].imshow(scans[:, 64, :, 0], cmap="grey")
    axs[1].imshow(scans[:, :, 64, 1], cmap="grey")
    axs[2].imshow(scans[:, :, 64, 2], cmap="grey")
    axs[3].imshow(mask[:, :, 64], cmap="grey")

    plt.show()

