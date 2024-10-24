import torch
from torch import nn


class GetFrom(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        assert hasattr(x, "__iter__")
        return x[self.idx]

def parse_GetFrom_args(ch, f, n, m, args):
    idx = args[0]
    if hasattr(ch[f], "__iter__"):
        c1, c2 = ch[f][idx], ch[f][idx]
    else:
        c1, c2 = ch[f], ch[f]
    return c1, c2, n, args
