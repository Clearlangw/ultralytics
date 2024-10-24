from .split import Split, parse_Split_args
from .get_from import GetFrom, parse_GetFrom_args
from .identity import Identity
from .add import Add
from .CALNet import GPTcross as CALNet_GPTcross
from .CALNet import parse_CALNet_GPTcross_method

from .mamba_blocks import *
from .mamba_blocks import parse_args_method as parse_mamba_args
from collections import ChainMap


def _default_parse_args_method(ch, f, n, m, args):
    if hasattr(f, "__iter__"):
        return ch[f[0]], ch[f[0]], n, args
    else:
        return ch[f], ch[f], n, args

parse_args_method = dict([
    (Split, parse_Split_args),
    (GetFrom, parse_GetFrom_args),
    (Identity, _default_parse_args_method),
    (Add, _default_parse_args_method),
    (CALNet_GPTcross, parse_CALNet_GPTcross_method),
])
parse_args_method = dict(
    ChainMap(
        parse_args_method,
        parse_mamba_args,
    )
)


__all__ = [
    'Split',
    'GetFrom',
    'Identity',
    'Add',
    'CALNet_GPTcross',
    *mamba_blocks.__all__,
]