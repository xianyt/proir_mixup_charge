#! -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from embedding import EmbeddingLayer


class RnnEncoder(nn.Module):
    def __init__(self, rnn_type, vocab_size, embed_dim, rnn_dim,
                 dropout, layer_num, bidir):
        super(RnnEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.rnn_type = rnn_type
        self.bidir = bidir
        self.layer_num = layer_num
        self.rnn_dim = rnn_dim

        self.embedding = EmbeddingLayer(vocab_size, embed_dim)

        rnn_drop = dropout if self.layer_num > 1 else 0
        rnn = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn(embed_dim, rnn_dim, dropout=rnn_drop,
                       num_layers=layer_num, bidirectional=bidir)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def output_dim(self):
        return (self.rnn_dim * 2 + self.embed_dim) if self.bidir else self.rnn_dim + self.embed_dim

    def load_embed_weights(self, weights):
        self.embedding.load_weights(weights)

    def forward(self, inp, mask, inp_len):

        inp = self.embedding(inp)

        if self.bidir:
            inp_len_sort, idx = torch.sort(inp_len, descending=True)
            inp_pack = torch.index_select(inp, dim=1, index=idx)

            inp_pack = pack_padded_sequence(inp_pack, inp_len_sort)
            outp, _ = self.rnn(inp_pack)
            outp, _ = pad_packed_sequence(outp)

            _, rev_idx = torch.sort(idx, descending=False)
            outp = torch.index_select(outp, dim=1, index=rev_idx)
        else:
            outp, _ = self.rnn(inp)

        outp = torch.cat([outp, inp], dim= -1)

        if self.dropout:
            outp = self.dropout(outp)

        return outp
