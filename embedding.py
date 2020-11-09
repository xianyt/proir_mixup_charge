#! -*- coding: utf-8 -*-
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_dropout=0.3):
        super(EmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if embed_dropout > 0:
            self.embed_drop = nn.Dropout(embed_dropout)
        else:
            self.embed_drop = None

    def load_weights(self, weights):
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        if self.embed_drop:
            embed = self.embed_drop(embed)

        return embed
