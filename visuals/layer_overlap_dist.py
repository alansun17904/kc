import os
import sys
import tqdm
import torch
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor


DATA_DIR = sys.argv[1]
MODEL_NAME = sys.argv[2]
TOTAL_LAYERS = len(os.listdir(Path(DATA_DIR) / MODEL_NAME))

GRID_SIZE = 100
eps = delt = np.linspace(0, 1, GRID_SIZE)
ticks = [round(v, 1) for v in np.linspace(0, 1, 10)]

# note that the first dimension corresponds to a specific delta-value then
# the second dimension correspodns to a specific epsilon-value.


def compute_overlap_disconts(layer_1, layer_2):
    global GRID_SIZE
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for i in tqdm.tqdm(range(GRID_SIZE)):
        i_scaled = i * (len(layer_1) // GRID_SIZE)
        l1i, l2i = layer_1[i_scaled].cpu(), layer_2[i_scaled].cpu()
        for j in range(GRID_SIZE):
            l1_disconts = np.nonzero(np.where(l1i >= eps[j], 1, 0))[0]
            l2_disconts = np.nonzero(np.where(l2i >= eps[j], 1, 0))[0]
            if len(l1_disconts) == len(l2_disconts) == 0:
                grid[j][i] = 1
                continue
            grid[j][i] = len(np.intersect1d(l1_disconts, l2_disconts)) / max(
                len(np.union1d(l1_disconts, l2_disconts)), 1
            )
    grid = np.flip(grid, axis=0)
    return grid


def plot_grid(grid, l1_index, l2_index):
    global MODEL_NAME, GRID_SIZE
    ax = sns.heatmap(grid, cbar=False)
    ax.set_title(
        f"Overlap of Discontinuities Between Layers {l1_index+1}, {l2_index+1}"
    )
    ax.set_yticks([v * GRID_SIZE for v in ticks])
    ax.set_xticks([v * GRID_SIZE for v in ticks])
    ax.set_yticklabels(ticks[::-1])
    ax.set_xticklabels(ticks)
    ax.set_aspect("equal")
    ax.set_ylabel("$\epsilon$")
    ax.set_xlabel("$\delta$")
    plt.savefig(f"data/k0/figures/{MODEL_NAME}-{l1_index+1}-{l2_index+1}.pdf")


def compute_plot_discont_overlap(l1_index, l2_index):
    global DATA_DIR, MODEL_NAME
    l1_disconts = torch.load(Path(DATA_DIR) / MODEL_NAME / f"L{l1_index}.pth")
    l2_disconts = torch.load(Path(DATA_DIR) / MODEL_NAME / f"L{l2_index}.pth")
    grid = compute_overlap_disconts(l1_disconts, l2_disconts)
    plot_grid(grid, l1_index, l2_index)


with ProcessPoolExecutor() as executor:
    for l in range(TOTAL_LAYERS):
        for j in range(TOTAL_LAYERS):
            # find the overlap between the discontinuities in layer l and j
            if j >= l:
                break
            # compute_plot_discont_overlap(l, j)
            executor.submit(compute_plot_discont_overlap, l, j)
