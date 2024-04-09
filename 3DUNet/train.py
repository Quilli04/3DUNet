import torch.nn.functional
from torch.utils.data import DataLoader
from UNetDataset import UNetDataset
from UNetModel import UNet
from UNetLoss import DiceLoss
from UNetUtils import *
from utils.paths import *
import utils.singletons as singletons
import datetime


MODEL_NAME = "UNet_v2"
model_cfg = singletons.model_cfgs[MODEL_NAME]

model_dir = model_cfg.model_dir
train_cfg = model_cfg.train_cfg

model_param_cfg = model_cfg.model_param_cfg
dataset_cfg = model_cfg.dataset

checkpoint_dir = model_cfg.checkpoint_dir
losses_path = model_cfg.losses_path

SHUTDOWN = False

EPOCHS = 25
BATCH_SIZE = train_cfg["batch_size"]
USE_AMP = train_cfg["use_amp"]

torch.set_default_device(device="cuda")

model = UNet(model_param_cfg, dataset_cfg.num_cls)
loss_fn = DiceLoss()
optimizer = load_optimizer(model.parameters(), train_cfg["optimizer"])
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

load_checkpoint(checkpoint_dir, "latest.pt", model, optimizer, scaler)

train_dataset = UNetDataset(dataset_cfg.train_dir, dataset_cfg.num_cls)
val_dataset = UNetDataset(dataset_cfg.val_dir, dataset_cfg.num_cls)

train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, generator=torch.Generator(device="cuda"))
val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

checkpoint_saver = SaveCheckpoints(checkpoint_dir)
num_former_epochs = get_num_former_epochs(model_dir)

print(f"\nStarting training for {EPOCHS} epochs...")


# for i in range(1, EPOCHS+1):
#
#     print(f"\n\n----- Epoch no. {num_former_epochs + i} -----")
#
#     train_loss = train_step(model, train_dataloader, loss_fn, optimizer, scaler)
#
#     print("\nValidating model...")
#     val_loss = val_step(model, val_dataloader, loss_fn)
#
#     print(f"\nEpoch {num_former_epochs + i} summary:")
#     print(f"\nTrain loss: {train_loss}")
#     print(f"Val loss: {val_loss}")
#
#     write_losses_to_csv(losses_path, losses=(train_loss, val_loss), cols=("train_loss", "val_loss"))
#     checkpoint_saver(model, optimizer, scaler, train_loss, val_loss)
#
# print(f"\n Finished training: {datetime.datetime.now()}")
#
# if SHUTDOWN:
#     os.system("shutdown /s /t 1")





