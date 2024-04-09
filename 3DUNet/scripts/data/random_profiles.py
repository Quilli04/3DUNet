from pathlib import Path
import os
import random
import nilearn as nl
import nilearn.image
import matplotlib.pyplot as plt


TRAIN_DATASET_DIR = Path(r"D:\datasets\brain_mrt\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
train_folder_names = os.listdir(TRAIN_DATASET_DIR)

while True:
    scans_folder_name = random.choice(train_folder_names)
    scans_folder_path = TRAIN_DATASET_DIR / scans_folder_name
    scans_paths = [scans_folder_path / img_scantype_name for img_scantype_name in os.listdir(scans_folder_path)]

    scans_arrays = {str(scan_path).split('_')[-1].split('.')[0]: nl.image.load_img(scan_path).get_fdata() for scan_path in scans_paths}
    scans_profiles = {scan_type: scan_array[..., scan_array.shape[-1]//2] for scan_type, scan_array in scans_arrays.items()}

    fig, axs = plt.subplots(ncols=5, figsize=(15, 7))

    for (scan_type, scan_profile), ax in zip(scans_profiles.items(), axs):
        ax.imshow(scan_profile, cmap="grey")
        ax.set_title(scan_type)

    plt.show()
