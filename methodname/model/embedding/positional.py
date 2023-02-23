import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, length, dim):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(length, dim).float()
        pe.require_grad = False

        position = torch.arange(0, length).float().unsqueeze(1)
        div_term = (torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        bs, len, dim = x.shape
        return self.pe[:, :len, :dim]


if __name__ == '__main__':
    torch.manual_seed(2)
    a = PositionalEmbedding(2, 10)
    torch.manual_seed(2)
    b = PositionalEmbedding(2, 20)
    print(a.pe[:, :2, :10])
    print(b.pe[:, :2, :10])
# tensor([[[0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00,
#           1.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 1.0000e+00],
#          [8.4147e-01, 5.4030e-01, 1.5783e-01, 9.8747e-01, 2.5116e-02,
#           9.9968e-01, 3.9811e-03, 9.9999e-01, 6.3096e-04, 1.0000e+00]]])
# tensor([[[0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000,
#           0.0000, 1.0000],
#          [0.8415, 0.5403, 0.3877, 0.9218, 0.1578, 0.9875, 0.0631, 0.9980,
#           0.0251, 0.9997]]])
