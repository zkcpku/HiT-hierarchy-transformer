import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward
from .position import TreePositionalEmbedding, LeafPositionalEmbedding
import torch
from torch.nn import LayerNorm


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, args):
        super().__init__()
        self.attention = MultiHeadedAttention(args)
        self.feed_forward = PositionwiseFeedForward(args)
        self.input_sublayer = SublayerConnection(args)
        self.output_sublayer = SublayerConnection(args)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, content, mask, tree_score, leaf_score):
        '''
        :param kernel: bs,max_code_length,max_code_length
        :param content: bs, max_code_length, hidden
        :param mask: bs, 1,max_code_length,max_code_length
        :param score: bs, h,max_code_length,max_code_length
        :param score: bs, h,max_code_length,max_code_length
        :return:
        '''
        x = self.input_sublayer(content,
                                lambda _x: self.attention.forward(_x, _x, _x, mask=mask, tree_score=tree_score,
                                                                  leaf_score=leaf_score))
        x = self.output_sublayer(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """f
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.transformer_blocks = nn.ModuleList([TransformerBlock(args) for _ in range(args.layers)])
        self.max_code_length = args.max_code_length
        self.h = args.attn_heads
        self.d_k = args.hidden // args.attn_heads
        if self.args.tree:
            self.tree_position_embedding = TreePositionalEmbedding(args)

        if self.args.leaf_PE:
            self.leaf_position_embedding = LeafPositionalEmbedding(args)

    def forward(self, content, mask, actions, spl, leaf_PE):
        '''
        :param kernel: bs,max_code_length,max_code_length
        :param content: bs, max_code_length, hidden
        :param mask: bs, 1,max_code_length,max_code_length
        :param level_idx: bs, max_code_length
        :param post_idx: bs, max_code_length
        :return:
        '''
        if self.args.tree:
            tree_score = self.tree_position_embedding(actions, spl)
        else:
            tree_score = None

        if self.args.leaf_PE:
            leaf_score = self.leaf_position_embedding(leaf_PE)
        else:
            leaf_score = None
        for transformer in self.transformer_blocks:
            content = transformer(content, mask, tree_score, leaf_score)
        return content
        # B * L * H
