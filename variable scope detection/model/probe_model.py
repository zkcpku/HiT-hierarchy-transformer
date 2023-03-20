import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from einops import rearrange

from model.TiT import TiTClf
from model.transformer import TransformerClf


class ProbeModel(nn.Module):
    def __init__(self, encoder, nhid=256):
        super(ProbeModel, self).__init__()
        self.encoder = encoder
        self.nhid = nhid
        self.qa_outputs = nn.Linear(nhid, nhid)
        self.freeze_encoder()

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, inps, edge_labels):
        sequence_output = self.encoder.forward_encoder(inps)

        tmp=torch.tanh(self.qa_outputs(sequence_output))
        # (bs, seq_len, hidden_size)
        score=torch.einsum("abc,adc->abd",tmp.float(),sequence_output.float()).sigmoid()
        # (bs, seq_len, seq_len)
        loss_fct = CrossEntropyLoss()
        scores=torch.cat(((1-score)[:,:,:,None],score[:,:,:,None]),-1)
        # import pdb; pdb.set_trace()
        masked_graph_loss=loss_fct(scores.view(-1, 2), edge_labels.view(-1))

        return masked_graph_loss, score
