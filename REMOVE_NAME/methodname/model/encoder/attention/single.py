import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class RelationAwareAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, query, key, value, mask=None, tree_score=None, leaf_score=None):
        """
        :param query: bs, head,max_code_length, hidden//head
        :param key:
        :param value:bs, h,max_code_length, hidden
        :param mask:bs, 1,max_code_length,max_code_length
        :param dropout:
        :param kernel: bs,1,max_code_length,max_code_length
        :param score: bs,h,max_code_length,max_code_length
        :return:
        """
        sqrt = 1
        if self.args.tree:
            sqrt += 1
        if self.args.leaf_PE:
            sqrt += 1

        w_score = torch.einsum('bhik,bhjk->bhij', query, key)
        if tree_score is not None:
            w_score = w_score + tree_score  # should set sqrt=2
        if leaf_score is not None:
            w_score = w_score + leaf_score  # should set sqrt=2

        w_score = w_score / math.sqrt(sqrt * query.size(-1))
        if mask is not None:
            w_score = w_score.masked_fill(mask, -1e9)
        p_attn = F.softmax(w_score, dim=-1)

        p_attn = self.dropout(p_attn)
        attn_sum = torch.einsum('bhij,bhjk->bhik', p_attn, value)

        return attn_sum, p_attn
