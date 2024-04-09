import torch
import yaml

from UNetModel import UNet


def _print_params(it, depth: int):
    total = 0
    for n, m in it:
        num_params = sum([p.numel() for p in m.parameters() if p.requires_grad])
        name = n if type(m) is torch.nn.ModuleList else type(m).__name__

        if depth == 0:
            print()

        print("\t"*depth + f"{name}: {num_params:,}")
        if list(m.named_children()):
            total += _print_params(m.named_children(), depth+1)
        else:
            total += num_params
    return total


def print_params(model: torch.nn.Module):
    total = _print_params(model.named_children(), 0)
    print(f"\n Total number of parameters: {total}")


cfg_path = r"/cfg/models/unet_l.yaml"
num_classes = 4

with open(cfg_path) as file:
    cfg = yaml.safe_load(file)

model = UNet(cfg, 4)
print_params(model)

