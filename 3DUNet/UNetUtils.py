import torch
from torch.cuda.amp import GradScaler
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from tqdm import tqdm
from statistics import mean
from glob import glob
from pathlib import Path
from utils.paths import root_dir
from UNetModel import UNet


def train_step(model, dataloader, loss_fn, optimizer, scaler: GradScaler):

    losses = []

    for i, (scans, masks) in enumerate(dataloader):

        # with torch.autograd.detect_anomaly(check_nan=True):

        preds = model(scans)
        loss = loss_fn(preds, masks)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        # loss.backward()
        #
        # optimizer.step()

        optimizer.zero_grad()

        losses.append(loss.item())

        if i % 10 == 9:
            print(f"\nBatch no. {i-8}-{i+1}")
            print(f"Loss: {mean(losses[-10:])}")

    return mean(losses)


def val_step(model, dataloader, loss_fn):

    losses = []

    for scans, masks in tqdm(dataloader):

        with torch.no_grad():
            preds = model(scans)
            loss = loss_fn(preds, masks)

        losses.append(loss.item())

    return mean(losses)


def load_checkpoint(checkpoint_dir, checkpoint_name, model, optimizer, scaler):

    try:
        checkpoint = torch.load(checkpoint_dir/checkpoint_name)
    except FileNotFoundError:
        print("\nNo matching checkpoint name found!")
        return

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])


def load_model(model_cfg, mode="best_val"):

    model_param_cfg = model_cfg.model_param_cfg
    dataset_cfg = model_cfg.dataset

    model = UNet(model_param_cfg, dataset_cfg.num_cls)
    checkpoint_dir = model_cfg.checkpoint_dir

    checkpoint_name = glob(mode+"*", root_dir=checkpoint_dir)[0]

    model = UNet(model_param_cfg, dataset_cfg.num_cls)

    checkpoint = torch.load(checkpoint_dir/checkpoint_name)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def load_yaml_file(cfg_path):

    with open(cfg_path) as stream:
        cfg_file = yaml.safe_load(stream)

    return cfg_file


def load_optimizer(model_parameters, cfg):

    name, hyperparameters = cfg
    return getattr(torch.optim, name)(model_parameters, **hyperparameters)


def write_losses_to_csv(file_path: Path, losses, cols):

    if len(losses) != len(cols):
        raise ValueError("length of losses and cols do not match")

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as file:
            file.write(",".join(cols))

    line = ",".join([f"{loss:.6}" for loss in losses])

    with file_path.open("a+") as file:
        file.write("\n" + line)


class SaveCheckpoints:

    def __init__(self, checkpoint_dir):

        self.checkpoint_dir = checkpoint_dir

        self.best_train_name = None
        self.best_train_loss = None

        self.best_val_name = None
        self.best_val_loss = None

        try:
            self.best_train_name = glob("best_train_*.pt", root_dir=checkpoint_dir)[0]
            self.best_train_loss = float(Path(self.best_train_name).stem.split("_")[-2])
        except IndexError:
            pass

        try:
            self.best_val_name = glob("best_val_*.pt", root_dir=checkpoint_dir)[0]
            self.best_val_loss = float(Path(self.best_val_name).stem.split("_")[-1])
        except IndexError:
            pass

    def __call__(self, model, optimizer, scaler, train_loss, val_loss, *args, **kwargs):

        save_checkpoint(self.checkpoint_dir, "latest.pt", model, optimizer, scaler)

        if (self.best_train_name is None and self.best_train_loss is None) or (self.best_train_loss > train_loss):

            if self.best_train_name and self.best_train_loss:
                os.remove(self.checkpoint_dir/self.best_train_name)

            self.best_train_loss = train_loss
            self.best_train_name = f"best_train_{self.best_train_loss:.6}_{val_loss:.6}.pt"
            save_checkpoint(self.checkpoint_dir, self.best_train_name, model, optimizer, scaler)

        if (self.best_val_name is None and self.best_val_loss is None) or (self.best_val_loss > val_loss):

            if self.best_val_name and self.best_val_loss:
                os.remove(self.checkpoint_dir/self.best_val_name)

            self.best_val_loss = val_loss
            self.best_val_name = f"best_val_{train_loss:.6}_{self.best_val_loss:.6}.pt"
            save_checkpoint(self.checkpoint_dir, self.best_val_name, model, optimizer, scaler)


def save_checkpoint(checkpoint_dir, checkpoint_name, model, optimizer, scaler):

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict()
    }

    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, checkpoint_dir/checkpoint_name)


def get_num_former_epochs(model_dir):

    try:
        with open(model_dir/"training"/"losses.txt") as f:
            return len(f.readlines())-1
    except FileNotFoundError:
        return 0


def draw_segmentation_masks(scans: np.ndarray, masks: list[np.ndarray], cls_names, colors, img_size,
                            mask_axis_labels, scan_axis_labels, depth=0.5, alpha=0.33):
    # scans.shape: (no. scans, img_size, img_size, img_size) - 3 scans were stacked: flair, t1ce, t2
    # masks.shape: (img_size, img_size, img_size)

    if len(cls_names) == len(colors):
        num_cls = len(cls_names)
    else:
        raise ValueError

    rgb_pixels = [np.array(mcolors.to_rgb(color)) for color in colors]
    rgba_pixels = [np.array(mcolors.to_rgba(color, alpha)) for color in colors]

    num_masks = len(masks)
    num_scans = scans.shape[0]

    # select 2D slice from 3D scan
    slice_no = int(img_size*depth)
    scans, masks = scans[..., slice_no], [mask[..., slice_no] for mask in masks]

    # convert scan to "RGB" image, imshow allows 0-1 float input (and 0-255 int)
    rgb_scans = np.expand_dims(scans, axis=-1).repeat(3, axis=-1)                       # shape: (no. scans, img_size, img_size, RGB)

    # for each mask a list of boolean masks for each class/color
    cond_lists = [[np.expand_dims(mask == i, -1) for i in range(1, num_cls+1)] for mask in masks]    # list of shape: (img_size, img_size, 1)

    # for each color a (unicolored) dyed stack of the 3 scans
    choose_lists = [(rgb_pixel*alpha + (1-alpha)*rgb_scans) for rgb_pixel in rgb_pixels]          # list of shape: (no. scans, img_size, img_size, RGB)

    # create final stacked scans colored corresponding to each mask
    colored_scans = [np.select(cond_list, choose_lists, rgb_scans)
                     for cond_list in cond_lists]

    # create array with only colored regions and alpha value/
    colored_masks = [np.select(cond_list, rgba_pixels, 0) for cond_list in cond_lists]

    fig, axs = plt.subplots(nrows=1+num_masks, ncols=1+num_scans, layout="constrained")

    # plot raw masks
    for i, (cm, label) in enumerate(zip(colored_masks, mask_axis_labels)):
        axs[1+i, 0].imshow(cm)
        axs[1+i, 0].set_ylabel(label)

    # plot raw scans
    for i, (scan, label) in enumerate(zip(scans, scan_axis_labels)):
        axs[0, 1+i].imshow(scan, cmap="grey", vmin=0, vmax=1)
        axs[0, 1+i].set_xlabel(label)
        axs[0, 1+i].xaxis.set_label_position("top")

    # plot colored scans row-wise
    for i, stacked_scan in enumerate(colored_scans):
        for j, scan in enumerate(stacked_scan):
            axs[i+1, j+1].imshow(scan)

    # axis settings
    for axs_row in axs:
        for ax in axs_row:
            ax.set_xticks([])
            ax.set_yticks([])

    # create legend
    patches = [mpatches.Patch(color=color, label=cls_name) for color, cls_name in zip(colors, cls_names)]
    axs[0, 0].set_axis_off()
    axs[0, 0].legend(handles=patches, bbox_to_anchor=(0, 1), loc="upper left")

    plt.show()
