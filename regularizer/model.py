import torch
import numpy as np
import torch.nn as nn
from transformers import PreTrainedModel


class KnowledgeContinuousModel(nn.Module):
    """Wrapper for any HuggingFace neural network with the proposed
    regularizer from our paper.
    """

    def __init__(self, model, alpha, beta, is_encoder_decoder=False, determinisitic=False):
        """Initialization for a regularized neural network
        :param model: the language model to be wrapped
        :param alpha: hyperparameter for the beta distribution
        :param beta: hyperparameter for the beta distribution
        """
        super().__init__()
        self.inference = False
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.determinisitic = determinisitic
        self.is_encoder_decoder = is_encoder_decoder

    def toggle_inference(self):
        self.inference = not self.inference

    def forward(self, input_ids, labels, inputs_embeds=None, attention_mask=None, determinisitc_idx=None):
        x = self.model(
            input_ids=None if inputs_embeds is not None else input_ids,
            labels=labels,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=(not self.inference),
        )
        if self.inference:
            return x
        # choose a random layer using the beta distribution and get
        # the hidden activations from that hidden layer
        if determinisitc_idx is not None:
            if self.is_encoder_decoder:
                return x.encoder_hidden_states[0], x.logits
            return x.hidden_states[0], x.logits
        elif self.determinisitic:
            return None, x.logits

        if not self.is_encoder_decoder:
            layer = int(len(x.hidden_states) * np.random.beta(self.alpha, self.beta))
            return torch.mean(x.hidden_states[layer], axis=1), x.logits

        # extract all both the encoder and decoder layers
        num_encoder_hs, num_decoder_hs = (
            len(x.encoder_hidden_states),
            len(x.decoder_hidden_states),
        )
        layer = int(
            (num_encoder_hs + num_decoder_hs) * np.random.beta(self.alpha, self.beta)
        )
        if layer >= num_encoder_hs:
            return (
                torch.mean(x.decoder_hidden_states[layer - num_encoder_hs], axis=1),
                x.logits,
            )
        else:
            return torch.mean(x.encoder_hidden_states[layer], axis=1), x.logits
