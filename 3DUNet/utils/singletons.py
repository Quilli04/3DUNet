from utils.paths import *
from UNetUtils import load_yaml_file
import glob


class _ModelSingleton:

    def __init__(self, model_name):
        self.model_dir = models_dir / model_name
        self.cfg = load_yaml_file(models_dir / model_name / "cfg.yaml")
        self.train_cfg = self.cfg["training"]

        self.model_param_cfg = model_param_cfgs[self.cfg["model"]]
        self.dataset = datasets[self.cfg["dataset"]]

        self.checkpoint_dir = self.model_dir / "training" / "checkpoints"
        self.losses_path = self.model_dir / "training" / "losses.txt"


class _ModelCfgSingleton:

    def __init__(self, model_cfg_name):
        self.cfg = load_yaml_file(model_cfgs_dir / (model_cfg_name + ".yaml"))
        self.name = self.cfg["name"]
        self.layers = self.cfg["layers"]


class _DatasetSingleton:

    def __init__(self, dataset_name):
        self.cfg = load_yaml_file(dataset_cfgs_dir / (dataset_name + ".yaml"))
        self.train_dir = self.cfg["train"]
        self.val_dir = self.cfg["val"]
        self.name = self.cfg["name"]
        self.num_cls = self.cfg["num_classes"]
        self.labels = self.cfg["labels"]
        self.img_size = self.cfg["img_size"]
        self.scan_types = self.cfg["scan_types"]
        self.colors = self.cfg["colors"]


model_cfgs = {}
model_param_cfgs = {}
datasets = {}

_model_names = glob.glob("*", root_dir=models_dir)
_model_cfg_names = glob.glob("*.yaml", root_dir=model_cfgs_dir)
_model_cfg_names = [_mcn.split(".")[0] for _mcn in _model_cfg_names]
_dataset_names = glob.glob("*.yaml", root_dir=dataset_cfgs_dir)
_dataset_names = [_dn.split(".")[0] for _dn in _dataset_names]

for _mcn in _model_cfg_names:
    model_param_cfgs[_mcn] = _ModelCfgSingleton(_mcn)

for _dn in _dataset_names:
    datasets[_dn] = _DatasetSingleton(_dn)

for _mn in _model_names:
    model_cfgs[_mn] = _ModelSingleton(_mn)
