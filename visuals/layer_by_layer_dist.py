import os
import sys
import tqdm
import torch
import numpy as np
import matplotlib
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


DATA_DIR = sys.argv[1]
MODEL_NAME = sys.argv[2]
TOTAL_LAYERS = len(os.listdir(Path(DATA_DIR) / MODEL_NAME))

N_COLS = 4
N_ROWS = TOTAL_LAYERS // N_COLS + (TOTAL_LAYERS % N_COLS > 0)
subplot_height = 3
GRID_SIZE = 100


# load the data file
eps = delt = np.linspace(0,1,GRID_SIZE)
ticks = [round(v,1) for v in np.linspace(0,1,10)]

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

for l in range(TOTAL_LAYERS):
    fname = Path(DATA_DIR) / MODEL_NAME / f"L{l}.pth"
    if not fname.exists():
        print(fname, "does not exist continuing with next layer.")
        continue
    disconts = torch.load(fname)
    disconts = [v.cpu() for v in disconts]
    grid = np.zeros((GRID_SIZE,GRID_SIZE))
    for i in tqdm.tqdm(range(GRID_SIZE), desc=f"Generating layer {l+1}"):
        # i = indexing delta
        # j = indexing eps
        i_scaled = i * (len(disconts) // GRID_SIZE)
        for j in range(GRID_SIZE):
            grid[j][i] = np.log(
                (disconts[i_scaled] >= eps[j]).sum() + 1
            )
    grid = np.flip(grid, axis=0)
    ax = sns.heatmap(
        grid,
        ax=axs.flat[l],
        cbar=False,
    )
    ax.set_title(f"Layer {l+1}", pad=0)
    ax.set_yticks([v*GRID_SIZE for v in ticks])
    ax.set_xticks([v*GRID_SIZE for v in ticks])
    ax.tick_params(which='both', width=0.25, length=1)
    ax.set_yticklabels(ticks[::-1])
    ax.set_xticklabels(ticks)
    ax.set_aspect('equal')
    # only label the y axis for the left-hand most plots
    if l % N_COLS == 0:
        ax.set_ylabel("$\epsilon$")
    if l // N_ROWS >= N_ROWS - 1:
        ax.set_xlabel("$\delta$")

# make the colorbar based on the last heatmap
cbar = plt.colorbar(ax.get_children()[0], ax=axs, orientation='vertical', fraction=0.05, pad=0.1)

for i in range(TOTAL_LAYERS, len(axs.flat)):
    axs.flat[i].set_axis_off()

plt.savefig(f"data/k0/figures/{MODEL_NAME}.pdf")

