from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import os
import glob
import nilearn as nl
import nilearn.image
import numpy as np

TRAIN_DATA_DIR = Path(r"D:\datasets\brain_mrt\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
SAVE_DIR = Path(r"D:\datasets\brain_mrt\ndarrays\train")

train_data_folder_names = sorted(glob.glob("BraTS20_Training_*", root_dir=TRAIN_DATA_DIR))

scaler = MinMaxScaler()
com = (117, 127, 81)    # center_of_mass
dmin0, dmax0 = com[0] - 64, com[0] + 64   #
dmin1, dmax1 = com[1] - 64, com[1] + 64   # percentage of labels lost: 3.0852%
dmin2, dmax2 = com[2] - 64, com[2] + 64   #

for i, train_data_folder_name in enumerate(train_data_folder_names):
    data_names = sorted(os.listdir(TRAIN_DATA_DIR / train_data_folder_name))
    scan_names = [scan for scan in data_names if any(s in scan for s in ("flair.nii", "t1ce.nii", "t2.nii"))]
    mask_name = [mask for mask in data_names if "seg.nii" in mask][0]

    scan_arrays = [nl.image.load_img(TRAIN_DATA_DIR / train_data_folder_name / scan_name).get_fdata().astype("float32") for scan_name in scan_names]
    mask_array = nl.image.load_img(TRAIN_DATA_DIR / train_data_folder_name / mask_name).get_fdata().astype("float32")

    scan_arrays = [scaler.fit_transform(X=scan.reshape(-1, 1)).reshape(scan.shape) for scan in scan_arrays]

    final_scans_array = np.stack(scan_arrays, axis=0)[dmin0:dmax0, dmin1:dmax1, dmin2:dmax2, :]

    final_mask_array = mask_array[dmin0:dmax0, dmin1:dmax1, dmin2:dmax2]
    final_mask_array[final_mask_array == 4] = 3
    # masks are saved with shape: 128x128x128, one-hot in dataset __get_item__

    # np.save(SAVE_DIR / "scans" / (train_data_folder_name + ".npy"), final_scans_array)
    np.save(SAVE_DIR / "masks" / (train_data_folder_name + ".npy"), final_mask_array)
    # print(f"{i+1}/{len(train_data_folder_names)}")



