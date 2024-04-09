# from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt
# import shutil
# import glob
# import random
# import nilearn as nl
# import nilearn.image
# import nilearn.plotting
# import nibabel as nib
import numpy as np
# import cupy as cp
# import pandas as pd
# import skimage
# import matplotlib.pyplot as plt
# import torch
# from UNetModel import UNet
# import yaml
# from scipy import ndimage
# from sklearn.preprocessing import MinMaxScaler
from UNetModel import UNet
# import torch
import yaml


# x = torch.arange(4*10*2).view(4, 10, 2)
# y = x.permute(2, 0, 1)
#
# # View works on contiguous tensors
# print(x.is_contiguous())
# print(x.view(-1))
#
# # Reshape works on non-contiguous tensors (contiguous() + view)
# print(y.is_contiguous())
# try:
#     print(y.view(-1))
# except RuntimeError as e:
#     print(e)
# print(y.reshape(-1))
# print(y.contiguous().view(-1))

# orig = np.arange(2*3*4)
# reshaped = orig.reshape((2, 3, 4))
# tranposed = reshaped.transpose(1, 2, 0)
# print(orig.base)
# print(reshaped.base)
# print(tranposed.base)
# print(tranposed.data.contiguous)
# print(np.ascontiguousarray(tranposed).reshape(-1))
# np.save(r"./arr.npy", tranposed)
# l = np.load(r"./arr.npy")
# print(f"loaded arr: {l.data.contiguous}")
# print(l.reshape(-1))

# with open(r"C:\Users\joni\PycharmProjects\UNet\cfg\models\unet_s.yaml") as f:
#     cfg_file = yaml.safe_load(f)
# nc = 4
#
# model = UNet(cfg_file, nc)
# input = torch.randn((2, 3, 128, 128, 128))
#
# with torch.no_grad():
#     output = model(input)
#
# print(f"Output shape: {output.shape}")





































