#! -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.functional import pairwise_distance
from ignite.utils import to_onehot
from torch.distributions import Beta, Categorical

from struct_att import StructAttention
from utils import max_pooling, mean_pooling


class StructAttMixup(nn.Module):

    def __init__(self, cfg, encoder: nn.Module):
        super(StructAttMixup, self).__init__()
        self.encoder = encoder
        self.num_class = cfg.num_class
        self.summary_type = cfg.summary_type

        self.multi_label = cfg.multi_label

        self.encoder_outp_dim = encoder.output_dim()

        self.label_count = torch.tensor(cfg.label_count, dtype=torch.float).to(cfg.device)
        self.prior = self.label_count / self.label_count.sum()

        self.mixup_type = cfg.mixup_type

        pool_dims = []
        self.strut_att = None
        for pool in self.summary_type:
            if pool == 'max' or pool == 'mean' or pool == 'first' or pool == 'last':
                pool_dims.append(self.encoder_outp_dim)
            elif pool == 'struct_att':
                self.strut_att = StructAttention(self.encoder_outp_dim, cfg.attention_dim, cfg.attention_head)
                pool_dims.append(self.encoder_outp_dim * cfg.attention_head)
            elif pool == 'none':
                pool_dims.append(self.encoder_outp_dim)
            else:
                raise Exception('unsupported pooling type "%s".' % pool)

        self.drop = nn.Dropout(cfg.dropout)
        concentration = cfg.mixup_beta_concentration
        self.beta_dist = Beta(torch.tensor([concentration]), torch.tensor([concentration]))

        self.hidden_dim = sum(pool_dims)
        self.normalize = nn.LayerNorm(self.hidden_dim)

        self.prototypes = nn.Parameter(torch.rand(cfg.num_class, self.hidden_dim), requires_grad=False)

        self.cls = nn.Linear(self.hidden_dim, cfg.num_class)

    def load_embed_weights(self, weights):
        self.embedding.load_weights(weights)

    def prior_mixup(self, labels, shuf_labels):
        shuf_prior = self.prior[shuf_labels]
        label_prior = self.prior[labels]
        lam_y = (1. - torch.tanh(label_prior / (label_prior + shuf_prior)))
        return lam_y

    def forward(self, inputs, labels=None):
        """
        :param inputs: [bsz, max_seq_leng]
        :param labels: [bsz, num_class]
        :return:
        """
        inputs = inputs.t()
        mask = (inputs > 0).float()
        inputs_len = (inputs > 0).int().sum(dim=0)

        hidden = self.encoder(inputs, mask, inputs_len)

        pool_values = []
        for pool in self.summary_type:
            if pool == 'max':
                val = max_pooling(hidden, mask)
                pool_values.append(val)
            elif pool == 'mean':
                val = mean_pooling(hidden, inputs_len, mask)
                pool_values.append(val)
            elif pool == 'first':
                seq_len, bsz, dim = hidden.size()
                val = hidden[0, :, :].view(bsz, -1).contiguous()
                pool_values.append(val)
            elif pool == 'last':
                seq_len, bsz, dim = hidden.size()
                val = hidden[-1, :, :].view(bsz, -1).contiguous()
                pool_values.append(val)
            elif pool == 'struct_att':
                val, att = self.strut_att(hidden, mask)
                bsz, head_num, dim = val.size()
                val = val.contiguous().view(bsz, -1)
                pool_values.append(val)
            elif pool == 'none':
                pool_values.append(hidden)

        if len(self.summary_type) == 1:
            hidden = pool_values[0]
        else:
            hidden = torch.cat(pool_values, dim=-1).contiguous()

        # [bsz, hid_dim]
        bsz, hid_dim = hidden.size()
        # logits = self.cls(self.dropout(hidden))
        hidden = self.normalize(hidden)
        logits = self.cls(hidden)

        if self.training:
            # Mixup
            indices = torch.randperm(bsz, device=logits.device)
            shuf_labels = torch.index_select(labels, 0, indices)
            shuf_hidden = torch.index_select(hidden, 0, indices)

            if self.mixup_type == 'mixup':
                lam = self.beta_dist.sample(sample_shape=(bsz, 1))
                lam = lam.to(inputs.device)
                lam_x, lam_y = lam, lam

            elif self.mixup_type == 'prior_mix':
                lam_x = self.beta_dist.sample(sample_shape=(bsz,))
                lam_x = lam_x.to(inputs.device)
                lam_y = self.prior_mixup(labels, shuf_labels)
                lam_y = 2. * lam_x * lam_y / (lam_x + lam_y)

            else:
                raise Exception('Unsupported mixup type %s' % self.mixup_type)

            mix_hidden = lam_x * hidden + (1 - lam_x) * shuf_hidden

            if not self.multi_label:
                onehot_label = to_onehot(labels, self.num_class)
                onehot_shuf_label = to_onehot(shuf_labels, self.num_class)
            else:
                onehot_label = labels
                onehot_shuf_label = shuf_labels

            lam_y = lam_y.unsqueeze(-1)
            mix_labels = lam_y * onehot_label + (1 - lam_y) * onehot_shuf_label

            mix_logits = self.cls(mix_hidden)

            return logits, mix_logits, mix_labels

        return logits, hidden
