import os
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import cdist
from concurrent import futures


@torch.no_grad()
def cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    print(x1_norm.shape, x2_norm.shape, x1.shape, x2.shape)
    res = torch.addmm(
        x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
    ).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res


@torch.no_grad()
def compute_cdist_layer(x, fname):
    """For a given layer, compute the pairwise distance between all points
    in that layer, and return this as a matrix.
    :param x: (n x d) vectors from the hidden layer
    :param fname: storage filename
    """
    x = x.to("cuda")
    bcdist = cdist_l2(x, x)
    torch.save((bcdist / torch.max(bcdist)).cpu(), fname)


def precompute_pdistance(activations, cache_dir, name):
    """Compute the pairwise distance between activations
    for each layer, then store in `fname` (Path obj.)"""
    acts = []
    for i in range(len(activations[0])):
        layer_act = []
        for v in activations:
            layer_act.append(v[i])
        acts.append(torch.cat(layer_act))
    for i, act in tqdm(enumerate(acts)):
        compute_cdist_layer(act, cache_dir / f"{name}-{i+1}.pth")


@torch.no_grad()
def precompute_performance(performance, cache_dir, name):
    """Compute the pairwise L1 distance between performances
    across all of the data points, then store in name"""
    torch.save(
        torch.cdist(performance, performance, p=1), cache_dir / f"{name}-label-dist.pth"
    )


@torch.no_grad()
def discontinuity_score_delta(pdistance, ldistance, delta):
    """Given the pairwise distance between points, the pairwise L1
    distance between the performance, and the delta values find the
    discontinuity score of each point with respect to a delta ball
    around that point.

    :return: nx1 vector with the point score.
    """
    ldistance = ldistance.to("cuda")
    pdistance = pdistance.to("cuda")
    in_delta_ball = torch.where(
        pdistance < delta, ldistance, torch.zeros(ldistance.shape).to("cuda")
    )
    point_score = in_delta_ball / (1 + pdistance)
    return torch.mean(point_score, axis=1)


@torch.no_grad()
def discontinuity_tf_eps(point_scores, eps):
    return (
        torch.nonzero(
            torch.where(
                point_scores >= eps,
                torch.ones(point_scores.shape).to("cuda"),
                torch.zeros(point_scores.shape).to("cuda"),
            )
        )
        .cpu()
        .numpy()
    )


def main():
    DATA_DIR = Path(sys.argv[1])
    SENT_DIR = DATA_DIR / "sentiment-analysis"
    MODEL_NAME = sys.argv[2]
    CACHE_DIR = DATA_DIR / "dist-cache" / f"{MODEL_NAME}-cdist"
    DIST_DIR = DATA_DIR / "discontinuities"

    LAYER_INDEX = None if len(sys.argv) < 4 else int(sys.argv[3])
    DIST_LAYER_STORE = DIST_DIR / MODEL_NAME

    print(f"Model: {MODEL_NAME}; Layer: {LAYER_INDEX + 1}")
    if not os.path.isdir(CACHE_DIR):  # if no cache then pre-compute distances
        os.mkdir(CACHE_DIR)
        print(f"{CACHE_DIR} created.")

        # load all activations
        activations = pickle.load(open(SENT_DIR / f"{MODEL_NAME}-k0.pkl", "rb"))

        print(len(activations))
        print(len(activations[0]))

        precompute_pdistance(activations, CACHE_DIR, f"{MODEL_NAME}-cdist")

        # load the labels
        labels = pickle.load(open(SENT_DIR / "labels.pkl", "rb"))
        logits = torch.cat([v[0] for v in activations])  # load all of the logits

        # apply softmax to the logits to get raw probabilities
        def softmax(x):
            return torch.exp(x) / torch.sum(torch.exp(x), axis=1).reshape(-1, 1)

        # we only care about the probabilites that correspond to the correct label
        predicted_probs = softmax(logits)[torch.arange(len(logits)), labels]
        precompute_performance(predicted_probs.reshape(-1, 1), CACHE_DIR, MODEL_NAME)
        sys.exit(0)

    # load all euclidean distances from the cached file
    if not os.path.isdir(DIST_LAYER_STORE):
        os.mkdir(DIST_LAYER_STORE)

    pdist = torch.load(CACHE_DIR / f"{MODEL_NAME}-cdist-{LAYER_INDEX+1}.pth")
    ldist = torch.load(CACHE_DIR / f"{MODEL_NAME}-label-dist.pth")

    l = list(range(1000))
    for i, d in tqdm(enumerate(torch.linspace(0, 1, 1000)), total=1000):
        l[i] = discontinuity_score_delta(pdist, ldist, d)
        # a = list(range(1000))
        # for j, e in enumerate(torch.linspace(0,1,1000)):
        #     disconts = discontinuity_tf_eps(scores, e)
        #     a[j] = disconts
        # l[i] = a
    torch.save(l, DIST_LAYER_STORE / f"L{LAYER_INDEX}.pth")


if __name__ == "__main__":
    main()
