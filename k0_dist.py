import os
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent import futures
from numba import jit
from numba import prange
from numba_progress import ProgressBar

print('start')

class ConstantIterator:
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        return self

    def __next__(self):
        return self.value


DATA_DIR = Path(sys.argv[1])
MODEL_NAME = sys.argv[2]
LAYER_INDEX = int(sys.argv[3])

labels = pickle.load(open(DATA_DIR / "labels.pkl", "rb"))

# convert all of the labels to ints 
labels = [int(v) for v in labels]

# load the activation data and the predictions
activations = pickle.load(open(DATA_DIR / f"{MODEL_NAME}-k0.pkl", "rb"))
logits = torch.cat([v[0] for v in activations])  # load all of the logits
acts = []
for i in range(len(activations[0][1])):  # load activations and organize them by layer
    layer_act = []
    for v in activations:
        layer_act.append(v[1][i])
    acts.append(torch.cat(layer_act))

# apply softmax to the logits to get raw probabilities
def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), axis=1).reshape(-1,1)

print('stuff 2')

# we only care about the probabilites that correspond to the correct label
predicted_probs = softmax(logits)[torch.arange(len(logits)),labels]
predicted_probs = predicted_probs.cpu().detach().numpy()

# norm the activations 
normed_acts = acts[LAYER_INDEX] / np.max(np.linalg.norm(acts[LAYER_INDEX], axis=1).reshape(-1, 1))
normed_acts = normed_acts.cpu().detach().numpy()

# Decorate the function with @jit to enable Just-In-Time compilation
# @jit(nopython=True)
def calculate_mags(normed_acts):
    num_samples = len(normed_acts)
    mags = []

    for i in tqdm(range(num_samples)):
        stuff = np.linalg.norm(normed_acts - normed_acts[i], axis=1)
        to_append = [(stuff[j], j) for j in range(len(stuff))]
        to_append.sort()
        mags.append(to_append)

    return mags

# Call the optimized function
mags = calculate_mags(normed_acts)

# create our cache
# cache[0] -- all of the distances within delta
# cache[1] -- all of the points greater than delta
# cache[2] -- indices of things in cache[0]
# cache[3] -- loss of all of the indices
# cache[4] -- count
# cache[5] -- sum
# cache[6] -- index in cache[1]
cache = [[[], mags[i], [], [], 0, 0, 0] for i in range(len(normed_acts))]

#@jit(nopython=True, cache=True)
def check_k0(x, x_loss, cache_idx, mags, normed_acts, loss, delta=0.1, i=0):
    global cache

    if cache[cache_idx][6] < len(cache[cache_idx][1]):
        for idx in range(cache[cache_idx][6], len(cache[cache_idx][1])):
            item = cache[cache_idx][1][idx][0]
            if item < delta:
                # cache[cache_idx][0].append(item)
                # cache[cache_idx][3].append(loss[idx])
                # cache[cache_idx][2].append(idx)
                cache[cache_idx][4] += 1
                cache[cache_idx][5] += np.abs(x_loss - loss[cache[cache_idx][1][idx][1]]) / (1 + item)
                if idx == len(cache[cache_idx][1]) - 1:
                    cache[cache_idx][6] = idx + 1
            else:
                # new_cache_1.append(item)
                cache[cache_idx][6] = idx
                break
    # cache[cache_idx][1] = new_cache_1
    if cache[cache_idx][4] == 0:# len(cache[cache_idx][0]) == 0:
        # is this safe and intended?
        return 0
    scored_mean = cache[cache_idx][5] / cache[cache_idx][4]#np.mean(
    #    (np.abs(x_loss - np.array(cache[cache_idx][3])) / (1 + np.array(cache[cache_idx][0]))).numpy()
    #)
    return scored_mean 

print('start calculating k0')

# get the number of discontinuities as a function of delta
deltas = np.linspace(0.001, 0.999, 25)

model_dir = DATA_DIR / f"{MODEL_NAME}-k0-l{LAYER_INDEX + 1}-ep"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

def parallel_check_k0(enumerated, delta):
    global predicted_probs
    global mags
    global normed_acts
    i, row = enumerated
    value = check_k0(row, predicted_probs[i], i, mags[i], normed_acts, predicted_probs, delta, i) 
    
    return i, value

enumerated = []
enumerated_vals = []
enumerated_idx = []

for i, k in enumerate(normed_acts):
    enumerated_vals.append(k)
    enumerated_idx.append(i)
    enumerated.append((i, k))

discontinuity = np.zeros(len(predicted_probs))

#@jit(nopython=True, cache=True)
def check_k0_numba(delta, progress_proxy):
    global predicted_probs
    global mags
    global normed_acts
    global enumerated_vals
    global enumerated_idx

    for i in range(len(enumerated_idx)):
        i = enumerated_idx[i]
        row = enumerated_vals[i]
        value = check_k0(row, predicted_probs[i], i, mags[i], normed_acts, predicted_probs, delta, i)
        discontinuity[i] = value
        progress_proxy.update(1)

count = 1
for delta in tqdm(deltas):
    # do parallel processing if delta < 0.3 -- this is faster parallel, whereas later on it's faster sequential due to overhead
    
    if delta <= 0.85:
        with futures.ProcessPoolExecutor() as pool:
            for i, value in tqdm(pool.map(parallel_check_k0, enumerated, ConstantIterator(delta)), desc=f"delta={delta:.2f}"):
                discontinuity[i] = value
    #with ProgressBar(total=15000) as progress:
    #    check_k0_numba(delta, progress)
    else:
        for i in tqdm(range(len(enumerated))):
            i, value = parallel_check_k0(enumerated[i], delta)
            discontinuity[i] = value
    pickle.dump(
        discontinuity,
        open(model_dir / f"{MODEL_NAME}-k0-delta-{delta}.pkl", "wb+")
    )
    print(f'finished {delta}')
    count += 1
    # witih ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = [executor.submit(check_k0, x, normed_acts, predicted_probs, delta) for i, x in enumerate(normed_acts)]
    #     for future in as_completed(futures):
    #         i, result = future.result()
    #         discontinuity[i] = result
    #         pbar.update(1)
    # discontinuities.append(discontinuity)
