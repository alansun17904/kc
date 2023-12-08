import os
import sys
import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


DIST_DATA_DIR = sys.argv[1]
ADV_DATA_DIR = sys.argv[2]
ATTACK_METHOD_PREFIX = sys.argv[3]
MODEL_NAME = sys.argv[4]
TOTAL_LAYERS = len(os.listdir(Path(DATA_DIR) / MODEL_NAME))

N_COLS = 4
N_ROWS = TOTAL_LAYERS // N_COLS + (TOTAL_LAYERS % N_COLS > 0)
subplot_height = 3
GRID_SIZE = 100


# load the discontinuities data file
eps = delt = np.linspace(0,1,GRID_SIZE)
ticks = [round(v,1) for v in np.linspace(0,1,10)]

# load the adversarial attacks file
df = pd.read_csv(Path(ADV_DATA_DIR) / f"{ATTACK_METHOD_PREFIX}-{MODEL_NAME}.csv"

# note that the first dimension corresponds to a specific delta-value then
# the second dimension correspodns to a specific epsilon-value.
fig, axs = plt.subplots(
    N_ROWS,
    N_COLS,
    sharex=True,
    sharey=True,
    figsize=(
        N_ROWS * subplot_height,
        N_COLS * subplot_height
    ),
)

plt.tight_layout()


