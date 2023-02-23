import torch.nn as nn
from torch.nn import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, args):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(args.hidden)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
