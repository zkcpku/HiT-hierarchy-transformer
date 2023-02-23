import torch.nn as nn
from torch.nn import ReLU, GELU


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, args):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(args.hidden, args.hidden * args.e_ff_fold)
        self.w_2 = nn.Linear(args.hidden * args.e_ff_fold, args.hidden)
        self.dropout = nn.Dropout(args.dropout)
        self.activation = GELU() if args.activation == 'gelu' else ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
