import glob
import random
from pathlib import Path
from utils.singletons import *
import numpy as np
import torch

from UNetUtils import load_model, draw_segmentation_masks

torch.set_default_device("cuda")

MODEL_NAME = "UNet_v2"
DATASET_NAME = "BraTS2020"

model_cfg = model_cfgs[MODEL_NAME]
dataset = datasets[DATASET_NAME]

train_dir = Path(dataset.train_dir)
val_dir = Path(dataset.val_dir)

DIR = val_dir

scan_names = sorted(glob.glob("scans/*", root_dir=DIR))
mask_names = sorted(glob.glob("masks/*", root_dir=DIR))

idx = random.randint(0, len(scan_names)-1)
# idx = 10
scan = np.load(DIR/scan_names[idx])
mask = np.load(DIR/mask_names[idx])

model = load_model(model_cfg, "best_val")

with torch.no_grad():
    pred_mask = model(torch.tensor(scan).unsqueeze(0))

pred_mask = pred_mask.cpu().detach().numpy()
pred_mask = np.squeeze(np.argmax(pred_mask, axis=1))

cls_names = dataset.labels[1:]
colors = dataset.colors
mask_axis_labels = ("ground truth", "UNet_v2")
scan_axis_labels = dataset.scan_types

draw_segmentation_masks(scan, [mask, pred_mask], cls_names, colors, 128, mask_axis_labels, scan_axis_labels)

# green, gold, aqua




