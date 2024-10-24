import torch
from torch import nn


class Split(nn.Module):
    def __init__(
        self, 
        dim=1,
        split_list=[3, 1],
    ):
        super().__init__()
        self.dim = dim
        self.split_list = split_list

    def forward(self, x):
        x = torch.split(x, self.split_list, self.dim)
        return x

def parse_Split_args(ch, f, n, m, args):
    c1 = ch[f]
    c2 = ch[f]
    dim, split_list = args
    if not hasattr(split_list, "__iter__"):
        split_list = [split_list]
        args = [dim, split_list]
    if dim == 1:
        c2 = split_list
    return c1, c2, n, args
