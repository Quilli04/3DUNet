import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

LOSSES_PATH = Path(r"/models/UNet_v2/training/losses.txt")

df = pd.read_csv(LOSSES_PATH)

plt.plot(df["train_loss"], color="red", label="train loss")
plt.plot(df["val_loss"], color="blue", label="val loss")
plt.xlabel("# iterations")
plt.legend()
plt.ylim([0, 4])
plt.grid(True)
plt.show()

