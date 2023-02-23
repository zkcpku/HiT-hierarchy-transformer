import sys
import os
from torch import nn

sys.path.append()
from .embedding import LeftEmbedding, RightEmbedding
from .encoder import Encoder
import torch
import math
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args, s_vocab, t_vocab):
        super().__init__()
        self.args = args
        self.left_embedding = LeftEmbedding(args, s_vocab)
        self.right_embedding = RightEmbedding(args, t_vocab)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=args.hidden, nhead=args.attn_heads,
                                                        dim_feedforward=args.d_ff_fold * args.hidden,
                                                        dropout=args.dropout, activation=self.args.activation)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=args.decoder_layers)
        self.encoder = Encoder(args)
        # self.encoder = nn.TransformerEncoder()
        self.softmax = nn.LogSoftmax(dim=-1)
        if self.args.uni_vocab:
            self.right_embedding.embedding.weight = self.left_embedding.embedding.weight
        if self.args.weight_tying:
            self.right_embedding.out.weight = self.right_embedding.embedding.weight

        if self.args.pointer:
            self.query_linear = nn.Linear(self.args.hidden, self.args.hidden)
            self.sentinel = nn.Parameter(torch.rand(1, self.args.hidden))
            self.key_linear = nn.Linear(self.args.hidden, self.args.hidden * self.args.subtokens)
            if self.args.activation == 'gelu':
                self.activation = torch.nn.GELU()
            elif self.args.activation == 'relu':
                self.activation = torch.nn.ReLU()
            if self.args.pointer_type == 'add':
                self.additive_attention_W = nn.Linear(self.args.hidden * 2, self.args.hidden)
                self.additive_attention_v = nn.Parameter(torch.rand(self.args.hidden))

    def encode(self, data):
        content = data['content']
        content_mask = (torch.sum(content, dim=-1) == 0).to(content.device)
        if self.args.tree:
            actions = data['actions']
        else:
            actions = None

        if self.args.spl:
            if self.args.two_spl:
                X, Y = data['X'], data['Y']
                Z = None
            else:
                X, Y = None, None
                Z = data['Z']
        else:
            X, Y, Z = None, None, None
        spl = [X, Y, Z]

        if self.args.leaf_PE:
            leaf_PE = data['leaf_PE']
        else:
            leaf_PE = None

        content_ = self.left_embedding(content)
        # bs, max_code_length, hidden
        mask_ = content_mask.unsqueeze(1).repeat(1, content_mask.size(1), 1).unsqueeze(1)
        # bs, 1,max_code_length,max_code_length

        memory = self.encoder(content_, mask_, actions, spl, leaf_PE)
        # bs, max_code_length, hidden

        # memory_key_padding_mask = (content_mask == 0)

        return memory, content_mask
        # b * l * d

    def pointer(self, out, feature, memory, memory_key_padding_mask, content_e, voc_len):
        voc_len = torch.max(voc_len).item()
        bs, src_len, tgt_len = memory.shape[0], memory.shape[1], feature.shape[1]
        pointer_key = torch.cat((memory, self.sentinel.unsqueeze(0).expand(bs, -1, -1)), dim=1)  # bs,src_len,hid
        pointer_query = self.activation((self.query_linear(feature)))  # bs,tgt,hid
        if self.args.pointer_type == 'mul':
            pointer_atten = torch.einsum('bth,bsh->bts', pointer_query, pointer_key) / math.sqrt(self.args.hidden)
        elif self.args.pointer_type == 'add':
            pointer_query = pointer_query.unsqueeze(2).repeat(1, 1, pointer_key.shape[1], 1)  # bs,tgt,src_len,hid
            pointer_key = pointer_key.unsqueeze(1).repeat(1, tgt_len, 1, 1)  # bs,tgt,src_len,hid
            pointer_atten = self.activation(
                self.additive_attention_W(torch.cat([pointer_query, pointer_key], dim=-1)))  # bs,tgt,src_len,hid
            pointer_atten = torch.einsum('btsh,h->bts', pointer_atten, self.additive_attention_v)  # bs,tgt,src_len
        else:
            pointer_atten = None
        mask = torch.cat((memory_key_padding_mask, torch.ones(bs, 1).to(memory_key_padding_mask.device) == 0),
                         dim=-1).unsqueeze(1)  # bs,1,s
        pointer_atten = pointer_atten.masked_fill(mask, -1e9)
        pointer_atten = F.log_softmax(pointer_atten, dim=-1)
        pointer_gate = pointer_atten[:, :, -1].unsqueeze(-1)  # b,t,1
        pointer_atten = pointer_atten[:, :, :-1]  # b,t,s
        M = torch.zeros((bs, voc_len, src_len))
        # print(content_e.shape)
        # print(bs, voc_len, src_len)
        M[torch.arange(bs).unsqueeze(-1).expand(bs, src_len).reshape(-1),
          content_e.view(-1),
          torch.arange(src_len).repeat(bs)] = 1
        pointer_atten_p = torch.einsum('bts,bvs->btv', pointer_atten.exp(), M.to(pointer_atten.device))
        pointer_atten_log = (pointer_atten_p + torch.finfo(torch.float).eps).log()
        pointer_atten_log = pointer_atten_log - torch.log1p(
            -pointer_gate.exp() + torch.finfo(torch.float).eps)  # norm
        # pointer_atten_log: bs,max_target_len,extend_vocab_size. extend_vocab_size >= vocab_size

        # Avoid having -inf in attention scores as they produce NaNs during backward pass
        pointer_atten_log[pointer_atten_log == float('-inf')] = torch.finfo(torch.float).min
        if torch.isnan(pointer_atten_log).any():
            raise Exception("NaN in final pointer attention!")

        out = torch.cat((out, torch.zeros(bs, tgt_len, voc_len - out.shape[-1]).fill_(float('-inf')).to(out.device)),
                        dim=-1)  # not 0 , should -inf

        p = torch.stack(
            [out + pointer_gate, pointer_atten_log + (1 - pointer_gate.exp() + torch.finfo(torch.float).eps).log()],
            dim=-2)  # bs,tgt_len,2,extend_voc
        out = torch.logsumexp(p, dim=-2)
        # print(out.shape)
        return out

    def decode(self, memory, f_source, memory_key_padding_mask, content_e=None, voc_len=None):
        '''

        :param voc_len: a list of voc len
        :param content_e: extended vocab mapped content for pointer
        :param memory_key_padding_mask: ==0 ->True   ==1->False
        :param memory: # bs, max_code_length, hidden
        :param f_source: # bs,max_target_len
        :return:
        '''
        f_source_ = self.right_embedding(f_source)
        # bs,max_target_len,hidden

        f_len = f_source.shape[-1]
        tgt_mask = (torch.ones(f_len, f_len).tril_() == 0).to(memory.device)
        memory_key_padding_mask = memory_key_padding_mask.to(memory.device)
        tgt_key_padding_mask = (f_source == 0).to(memory.device)
        feature = self.decoder(f_source_.permute(1, 0, 2), memory.permute(1, 0, 2), tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        feature = feature.permute(1, 0, 2)
        # bs,max_target_len,hidden

        out = self.softmax(
            self.right_embedding.prob(feature))
        # bs,max_target_len,vocab_size
        if self.args.pointer:
            bs, l, _ = memory.shape
            memory = self.key_linear(memory).view(bs, l, self.args.subtokens, -1) \
                .view(bs, l * self.args.subtokens, -1)
            content_e = content_e.view(bs, -1)
            key_padding_mask = (content_e == 0).to(content_e.device)
            out = self.pointer(out, feature, memory, key_padding_mask, content_e, voc_len)
        return out

    def forward(self, data):
        f_source = data['f_source']
        memory, memory_key_padding_mask = self.encode(data)
        if self.args.pointer:
            out = self.decode(memory, f_source, memory_key_padding_mask, data['content_e'], data['voc_len'])
        else:
            out = self.decode(memory, f_source, memory_key_padding_mask)
        return out



if __name__ == '__main__':
    print('test')