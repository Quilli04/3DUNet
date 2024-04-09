import torch
from torch.nn import Conv3d, BatchNorm3d, MaxPool3d, Upsample, SiLU, Softmax, Module


class CBS(Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.conv = Conv3d(**kwargs, padding="same")
        self.bn = BatchNorm3d(kwargs["out_channels"])
        self.silu = SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))


class CBSBlock(Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.cs1 = CBS(**kwargs)
        kwargs["in_channels"] = kwargs["out_channels"]
        self.cs2 = CBS(**kwargs)

    def forward(self, x):
        return self.cs2(self.cs1(x))


class UpConv(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.upsample = Upsample(scale_factor=2, mode="nearest")
        self.conv = Conv3d(**kwargs, padding="same")

    def forward(self, x):
        return self.conv(self.upsample(x))


class Head(Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["out_channels"] = kwargs["num_classes"]
        del kwargs["num_classes"]
        self.conv = Conv3d(**kwargs, kernel_size=1, padding="same")

    def forward(self, x):
        return self.conv(x)

