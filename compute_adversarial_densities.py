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
