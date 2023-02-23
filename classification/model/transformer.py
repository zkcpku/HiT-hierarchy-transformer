import math
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from einops import rearrange
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logger = logging.getLogger(__name__)
# https://pytorch.org/tutorials/advanced/ddp_pipeline.html?highlight=transformer


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M ".format(round(model_size / 1e6)) + str(model_size)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # Need (S, N) format for encoder.
        src = src.t()
        src = self.encoder(src) * math.sqrt(self.ninp)
        return self.pos_encoder(src)


class Decoder(nn.Module):
    def __init__(self, ntoken, ninp, dropout):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(ninp, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, inp):
        # Need batch dimension first for output of pipeline.
        return self.linear(inp)


class TransformerClf(nn.Module):
    def __init__(self, ntokens, ncls, emsize, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.ntokens = ntokens
        self.emsize = emsize
        self.nhead = nhead
        self.nhid = nhid
        self.dropout = dropout
        self.nlayers = nlayers
        encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_block = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.encoder = Encoder(ntokens, emsize, dropout)
        self.decoder = Decoder(ncls, emsize, dropout)

        logger.info(f"Model size of HiTClf {get_model_size(self)}.")
    def forward(self, inp, labels = None):
        # (batch, seq_len)
        src = self.encoder(inp)
        src = self.transformer_block(src)
        # print(src.shape)
        src = src[0]
        # (batch, seq_len, ntoken)
        logits = self.decoder(src)
        prob = F.softmax(logits, dim=1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

def unit_test():
    ntokens = 10
    ninp = 12
    nhid = 12
    nhead = 4
    nlayers = 2
    emsize = 12
    dropout = 0.5
    ncls = 4
    model = TransformerClf(ntokens, ncls, emsize, nhead, nhid, nlayers, dropout)
    # print(model)
    src = torch.randint(ntokens, (8, 25))
    print(src.shape)
    tgt = model(src)
    print(tgt.shape)

    label = torch.randint(ncls, (8,))
    loss,prob = model(src, label)
    print(loss)
    print(prob.shape)
    y_preds = prob.argmax(dim=1)
    print(y_preds.shape)
    print(y_preds)


if __name__ == '__main__':
    unit_test()
    pass



        
        
