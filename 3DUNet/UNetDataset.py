import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import numpy as np
import pandas as pd


class UNetDataset(Dataset):

    def __init__(self, split_dir, num_classes):

        if type(split_dir) is str:
            split_dir = Path(split_dir)

        self.annotations = pd.read_csv(split_dir/"annotations.csv")
        self.scans_dir = split_dir/"scans"
        self.masks_dir = split_dir/"masks"
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        scan_path = self.scans_dir/self.annotations["scan"][index]
        mask_path = self.masks_dir/self.annotations["mask"][index]

        scan = np.load(str(scan_path))
        mask = np.load(str(mask_path))

        scan = torch.tensor(scan)
        mask = torch.tensor(mask, dtype=torch.long)

        mask = one_hot(mask, self.num_classes).permute(3, 0, 1, 2).contiguous()

        # scan.shape: (num_scan_types, 128, 128, 128), mask.shape: (num_classes, 128, 128, 128)
        return scan, mask



