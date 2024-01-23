import tqdm
import torch
import torch.nn as nn


class SolutionSemanticProjection(nn.Module):
    def __init__(self, hidden_dimension):
        self.linear = nn.Linear(hidden_dimension, hidden_dimension)

    def forward(x):
        return self.linear(x)


def cluster_loss(x, labels, lambda_1, lambda_2):
    in_cls = x[torch.nonzero(labels)]
    out_cls = x[torch.nonzero(1 - labels)]

    p_ctr = torch.mean(in_cls, axis=0)
    n_ctr = torch.mean(out_cls, axis=0)

    # get the distance between in-class examples
    in_dist = torch.sum(torch.norm(in_cls - p_ctr, dim=0))

    # get the distance between inter-class examples
    out_dist = torch.sum(torch.norm(in_cls - n_ctr, dim=0))

    return lambda_1 * in_dist - lambda_2 * out_dist


def train(x, labels, model, criterion, optimizer, batch_size, epochs):
    model.train()

    # shuffle the data and labels
    permutation = torch.randperm(x.shape[0])

    for epoch in range(epochs):
        for i in tqdm.tqdm(range(0, x.shape[0], batch_size), desc="Epoch {epoch+1}"):
            optimizer.zero_grad()
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = x[indices], y[indices]

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    a = torch.randn((5, 768))
    l = torch.LongTensor([1, 0, 1, 0, 0])
    l = l.reshape(5, 1)
    print(loss(a, l, 1, 1))
