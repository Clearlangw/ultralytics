import torch
from torch import nn


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert hasattr(x, "__iter__")
        y = x[0]
        for e in x[1:]:
            y = y + e
        return y
