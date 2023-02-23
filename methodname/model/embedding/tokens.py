from torch import nn
import os
import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm
from positional import PositionalEmbedding
import math
from torch.nn import init


# TODO create two embedding for bfs and dfs
class TokenEmbedding(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.args = args

        self.embedding = nn.Embedding(self.vocab_size, self.args.embedding_size, padding_idx=self.vocab.pad_index)
        self._init_embedding_weights(self.embedding)

    def _init_embedding_weights(self, embedding):
        if embedding is not None:
            init.uniform_(embedding.weight, -0.1, 0.1)
            if embedding.padding_idx is not None:
                embedding.weight.data[embedding.padding_idx].zero_()


class LeftEmbedding(TokenEmbedding):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.args = args
        if self.args.old_PE:
            self.p = PositionalEmbedding(self.args.max_code_length, self.args.hidden)
        if args.embedding_size * args.subtokens != args.hidden:
            self.in_ = nn.Linear(args.embedding_size * args.subtokens, args.hidden)
        else:
            self.in_ = None
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, content):
        '''
        :param named:
        :param content: bs,max_code_length,subtokens
        '''
        bs, l, _ = content.shape
        c_1 = self.embedding(content).view(bs, l, -1)

        if self.args.embedding_mul:
            c_1 *= math.sqrt(self.args.embedding_size)
        if self.in_:
            c_1 = self.in_(c_1)
        if self.args.old_PE:
            c_1 = c_1 + self.p(c_1)
        c_1 = self.dropout(c_1)
        return c_1


class RightEmbedding(TokenEmbedding):
    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        # self.out = nn.Linear(args.embedding_size, self.vocab_size)

        if args.embedding_size != args.hidden:
            self.in_ = nn.Linear(args.embedding_size, args.hidden)
        else:
            self.in_ = None

        # if args.embedding_size != args.hidden:
        #     self.out_ = nn.Linear(args.hidden, args.embedding_size)
        # else:
        #     self.out_ = None
        self.p = PositionalEmbedding(args.max_target_len, args.embedding_size)
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, f_source):
        '''

        :param f_source: bs,max_target_len
        :return:bs,max_target_len,hidden
        '''
        c_1 = self.embedding(f_source)

        if self.args.embedding_mul:
            c_1 *= math.sqrt(self.args.embedding_size)
        if self.p:
            c_1 = c_1 + self.p(c_1)
        if self.in_:
            c_1 = self.in_(c_1)
        c_1 = self.dropout(c_1)
        return c_1

    # def prob(self, data):
    #     if self.out_:
    #         data = self.out_(data)
    #     return self.out(data)
