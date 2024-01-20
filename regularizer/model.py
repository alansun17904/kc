import torch
import numpy as np
import torch.nn as nn
import transformers


class KnowledgeContinuousModel(nn.Module):
    """Wrapper for any HuggingFace neural network with the proposed
    regularizer from our paper.
    """
    def __init__(self, model, alpha, beta):
        """Initialization for a regularized neural network
        :param model: the language model to be wrapped 
        :param alpha: hyperparameter for the beta distribution
        :param beta: hyperparameter for the beta distribution
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, input_ids, labels, attention_mask=None):
        x = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # choose a random layer using the beta distribution and get
        # the hidden activations from that hidden layer
        layer = int(
            len(x.hidden_states) * np.random.beta(self.alpha, self.beta)
        )
        return torch.mean(x.hidden_states[layer],axis=1), x.logits


