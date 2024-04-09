import torch
from torch.nn import Module, ModuleList
import yaml
import UNetLayers
from UNetLayers import *


class UNet(Module):

    def __init__(self, cfg_file, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cfg_file = cfg_file
        self.build_model_from_cfg()

    def build_model_from_cfg(self):
        self.downward_path = ModuleList()
        self.bottleneck = ModuleList()
        self.upward_path = ModuleList()
        self.head = ModuleList()

        for lis, cfg_block in zip((self.downward_path, self.bottleneck, self.upward_path, self.head),
                                    self.cfg_file.layers):
            for module_name, kwargs in cfg_block:
                module = getattr(UNetLayers, module_name)
                if module is Head:
                    kwargs["num_classes"] = self.num_classes
                lis.append(module(**kwargs))

    def forward(self, x):
        concatenations = []

        for module in self.downward_path:
            if type(module) is CBSBlock:
                x = module(x)
                concatenations.append(x)
            elif type(module) is MaxPool3d:
                x = module(x)

        for module in self.bottleneck:
            x = module(x)

        for module in self.upward_path:
            if type(module) is CBSBlock:
                x = torch.concat((x, concatenations.pop(-1)), dim=1)
                x = module(x)
            elif type(module) is UpConv:
                x = module(x)

        for module in self.head:
            x = module(x)

        return x








        
