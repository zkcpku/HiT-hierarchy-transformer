import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import LayerNorm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, args, length, hidden, pad_idx=None, init=False):
        super().__init__()
        self.embedding = nn.Embedding(length, hidden, padding_idx=pad_idx)

        self.args = args
        if init:
            self._init_embedding_weights(self.embedding)

    def _init_embedding_weights(self, embedding):
        if embedding is not None:
            init.uniform_(embedding.weight, -0.1, 0.1)
            if embedding.padding_idx is not None:
                embedding.weight.data[embedding.padding_idx].zero_()

    def forward(self, idx):
        embeddings = self.embedding(idx)
        return embeddings


class TreePositionalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_ary = self.args.max_ary
        self.hidden = self.args.hidden
        self.h = args.attn_heads
        self.d_k = args.hidden // args.attn_heads
        self.action_size = args.action_size
        self.length = self.max_ary + 2  # +2 for pad and overflow flag
        self.embedding = PositionalEmbedding(args, self.length, self.action_size, pad_idx=0, init=True)
        self.linear = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(2)])
        self.norm = LayerNorm(args.hidden)
        self.cat_linear = nn.Linear(self.args.max_depth * self.action_size, self.hidden)
        self.spl_embedding = nn.ModuleList(
            [PositionalEmbedding(args, self.args.max_code_length, self.h, init=False) for _ in range(3)])

    def forward(self, actions, spl):
        bs, l, depth = actions.shape

        embedding = self.embedding(actions).view(bs, l, -1)
        final = self.cat_linear(embedding)
        norm_embedding = self.norm(final)
        U_Q, U_K = [l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                    zip(self.linear, [norm_embedding] * 2)]
        score = torch.einsum('bhik,bhjk->bhij', U_Q, U_K)

        if self.args.spl:
            if self.args.two_spl:
                X, Y = [l(x).transpose(2, 3).transpose(1, 2) for l, x in zip(self.spl_embedding, spl[:2])]
                if self.args.two_spl_type == 'mul':
                    Z = X * Y
                elif self.args.two_spl_type == 'add':
                    Z = X + Y
                else:
                    raise Exception
            else:
                Z = self.spl_embedding[-1](spl[-1]).transpose(2, 3).transpose(1, 2)
            score = score * Z
        return score


class LeafPositionalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden = self.args.hidden
        self.h = args.attn_heads
        self.d_k = args.hidden // args.attn_heads
        self.embedding = PositionalEmbedding(args, self.args.max_code_length, self.hidden, pad_idx=0,
                                             init=self.args.leaf_init)
        self.linear = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(2)])
        self.norm = LayerNorm(args.hidden)

    def forward(self, leaf_PE):
        bs, l = leaf_PE.shape
        norm_embedding = self.norm(self.embedding(leaf_PE))
        U_Q, U_K = [l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                    zip(self.linear, [norm_embedding] * 2)]

        if self.args.leaf_PE_mask:
            mask = (leaf_PE == 0).unsqueeze(1).repeat(1, self.h, 1).unsqueeze(-1)  # bs,l,h,1
            U_Q = U_Q.masked_fill(mask, 0.0)
            U_K = U_Q.masked_fill(mask, 0.0)

        score = torch.einsum('bhik,bhjk->bhij', U_Q, U_K)
        return score
