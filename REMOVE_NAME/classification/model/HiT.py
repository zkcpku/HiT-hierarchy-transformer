import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from einops import rearrange
from model.transformer import Encoder, Decoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAD_ID = 0


logger = logging.getLogger(__name__)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M ".format(round(model_size / 1e6)) + str(model_size)


class HiTClf(nn.Module):
    def __init__(self, ntokens, ncls, emsizes, nhead, nhid, nlayers, path_nlayers=2, dropout=0.2):
        super().__init__()
        self.ntokens = ntokens
        self.emsizes = emsizes
        self.emsize = sum(emsizes)
        self.nhead = nhead
        self.nhid = nhid
        self.dropout = dropout
        self.nlayers = nlayers
        encoder_layers = nn.TransformerEncoderLayer(self.emsize, nhead, nhid, dropout)
        self.transformer_block = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.encoders = [Encoder(ntokens[i], emsizes[i], dropout) for i in range(len(ntokens))]
        self.encoders = nn.ModuleList(self.encoders)
        self.decoder = Decoder(ncls, self.emsize, dropout)
        
        path_encoder_layers = nn.TransformerEncoderLayer(self.emsizes[1], nhead, nhid, dropout)
        self.path_encoder = nn.TransformerEncoder(path_encoder_layers, path_nlayers)
        logger.info(f"Model size of HiTClf {get_model_size(self)}.")

    def forward(self, inps, labels = None):
        # (batch, seq_len)
        code_inp, path_inp = inps
        path_inp_shape = path_inp.shape
        code_padding_mask = code_inp.eq(PAD_ID)
        path_padding_mask = path_inp.eq(PAD_ID)

        code_encode = self.encoders[0](code_inp)
        path_inp = rearrange(path_inp, 'b s p -> (b s) p')
        path_encode = self.encoders[1](path_inp)
        
        path_padding_mask = path_padding_mask.reshape(-1, path_inp_shape[-1])
        path_encode = self.path_encoder(path_encode, src_key_padding_mask=path_padding_mask)
        # (seq_len, batch, nhid)
        path_padding_mask = path_padding_mask.permute(1, 0)       # this is GPU memory consuming
        fill_mask = path_padding_mask.unsqueeze(2).expand_as(path_encode).float()
        path_encode = path_encode * (1 - fill_mask)
        path_encode = path_encode.sum(dim=0) / ((1 - fill_mask).sum(dim=0))

        path_encode = rearrange(path_encode, '(b s) e -> b s e', b=path_inp_shape[0])
        path_encode = torch.einsum('b s e -> s b e', path_encode)
        src = torch.cat([code_encode,path_encode], dim=-1)
        src = self.transformer_block(src, src_key_padding_mask=code_padding_mask)
        code_padding_mask = code_padding_mask.permute(1, 0)
        fill_mask = code_padding_mask.unsqueeze(2).expand_as(src).float()
        src = src * (1 - fill_mask)
        src = src.sum(dim=0) / ((1 - fill_mask).sum(dim=0))     # average pooling with padding_mask
        # (batch, hidden_size)
        logits = self.decoder(src)
        prob = F.softmax(logits, dim=1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


class HiT_simply(nn.Module):
    def __init__(self, ntokens, ncls, emsizes, nhead, nhid, nlayers, path_nlayers=2,dropout=0.2):
        super().__init__()
        self.ntokens = ntokens
        self.emsizes = emsizes
        self.emsize = sum(emsizes)
        self.nhead = nhead
        self.nhid = nhid
        self.dropout = dropout
        self.nlayers = nlayers
        encoder_layers = nn.TransformerEncoderLayer(self.emsize, nhead, nhid, dropout)
        self.transformer_block = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.encoders = [Encoder(ntokens[i], emsizes[i], dropout) for i in range(len(ntokens))]
        self.encoders = nn.ModuleList(self.encoders)
        self.decoder = Decoder(ncls, self.emsize, dropout)
        
        path_encoder_layers = nn.TransformerEncoderLayer(self.emsizes[1], nhead, nhid, dropout)
        self.path_encoder = nn.TransformerEncoder(path_encoder_layers, path_nlayers)

    def forward(self, inps, labels = None):
        # (batch, seq_len)
        code_inp, path_inp = inps
        path_inp_shape = path_inp.shape

        code_encode = self.encoders[0](code_inp)
        path_inp = rearrange(path_inp, 'b s p -> (b s) p')
        path_encode = self.encoders[1](path_inp)
        path_encode = self.path_encoder(path_encode)[0]
        # (b*s),embsize1
        path_encode = rearrange(path_encode, '(b s) e -> b s e',b = path_inp_shape[0])
        path_encode = torch.einsum('b s e -> s b e',path_encode)
        # import pdb;pdb.set_trace()
        src = torch.cat([code_encode,path_encode], dim=-1)
        # https://zhuanlan.zhihu.com/p/107586681
        # src_key_padding_mask
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


class HiT_Conv1d(nn.Module):
    def __init__(self, ntokens, ncls, emsizes, nhead, nhid, nlayers, path_len = 20,window_sizes = [2,3,4,5],dropout=0.2):
        super().__init__()
        self.ntokens = ntokens
        self.emsizes = emsizes
        self.emsize = sum(emsizes)
        self.nhead = nhead
        self.nhid = nhid
        self.dropout = dropout
        self.nlayers = nlayers
        encoder_layers = nn.TransformerEncoderLayer(self.emsize, nhead, nhid, dropout)
        self.transformer_block = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.encoders = [Encoder(ntokens[i], emsizes[i], dropout) for i in range(len(ntokens))]
        self.encoders = nn.ModuleList(self.encoders)
        self.decoder = Decoder(ncls, self.emsize, dropout)
        
        self.path_encoder = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.emsizes[1],out_channels=self.emsizes[1],kernel_size=h),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=path_len-h+1))
            for h in window_sizes
        ])
        self.path_linear = nn.Linear(self.emsizes[1]*len(window_sizes),self.emsizes[1])

    def forward(self, inps, labels = None):
        # (batch, seq_len)
        code_inp, path_inp = inps
        path_inp_shape = path_inp.shape

        code_encode = self.encoders[0](code_inp)
        path_inp = rearrange(path_inp, 'b s p -> (b s) p')
        path_encode = self.encoders[1](path_inp)
        # b*s, p, e
        path_encode = torch.einsum('p b d -> b d p',path_encode)
        # p, e, b*s
        # import pdb;pdb.set_trace()
        path_encode = [conv(path_encode) for conv in self.path_encoder]
        path_encode = torch.cat(path_encode,dim=1)
        path_encode = path_encode.squeeze()
        path_encode = self.path_linear(path_encode)
        
        # (b*s),embsize1
        path_encode = rearrange(path_encode, '(b s) e -> b s e',b = path_inp_shape[0])
        path_encode = torch.einsum('b s e -> s b e',path_encode)
        # import pdb;pdb.set_trace()
        src = torch.cat([code_encode,path_encode], dim=-1)
        # https://zhuanlan.zhihu.com/p/107586681
        # src_key_padding_mask
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
    ntokens = (10,5)
    ninp = 12
    nhid = 12
    nhead = 4
    nlayers = 2
    emsize = (12,16)
    dropout = 0.5
    ncls = 4
    model = HiT_Conv1d(ntokens, ncls, emsize, nhead, nhid, nlayers, dropout = dropout)
    model = model.cuda()
    # print(model)
    src = [torch.randint(ntokens[0], (8,25)).cuda(), torch.randint(ntokens[1],(8,25,20)).cuda()]
    # src = [torch.randint(ntokens[i], (8, 256)).cuda() for i in range(len(emsize))]
    print([src_.shape for src_ in src])
    tgt = model(src)
    print(tgt.shape)

    label = torch.randint(ncls, (8,)).cuda()
    loss,prob = model(src, label)
    print(loss)
    print(prob.shape)
    y_preds = prob.argmax(dim=1)
    print(y_preds.shape)
    print(y_preds)


if __name__ == '__main__':
    unit_test()
    pass

