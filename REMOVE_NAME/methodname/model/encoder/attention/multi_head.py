import torch.nn
import torch.nn as nn
from .single import RelationAwareAttention


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden
        self.h = args.attn_heads
        assert self.d_model % self.h == 0
        self.d_k = args.hidden // args.attn_heads
        self.args = args

        self.linear_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(3)])

        self.output_linear = nn.Linear(self.d_model, self.d_model)
        self.attention = RelationAwareAttention(args)

    def forward(self, query, key, value, mask=None, tree_score=None, leaf_score=None):
        '''
        :param query: bs, max_code_length, hidden
        :param key: bs, max_code_length, hidden
        :param value: bs, max_code_length, hidden
        :param mask:bs, 1,max_code_length,max_code_length
        :param kernel: bs,max_code_length,max_code_length
        :return:
        '''
        batch_size, max_code_length = query.size(0), query.size(1)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, tree_score=tree_score, leaf_score=leaf_score)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
