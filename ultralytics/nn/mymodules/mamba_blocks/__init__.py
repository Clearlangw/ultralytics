from .cmsa import CMSABlock
from .dcfm import DCFMBlock
from .cbam import CBAM
from .dssf import DSSFBlock, SSCSBlock
from .commdiff_mamba import CommDiffMambaBlock
from collections import ChainMap

def _default_parse_args_method(ch, f, n, m, args):
    if hasattr(f, "__iter__"):
        return ch[f[0]], ch[f[0]], n, [ch[f[0]], *args]
    else:
        return ch[f], ch[f], n, [ch[f], *args]

parse_args_method = dict([
    (CMSABlock, _default_parse_args_method),
    (DCFMBlock, _default_parse_args_method),
    (CBAM, _default_parse_args_method),
    (DSSFBlock, _default_parse_args_method),
    (SSCSBlock, _default_parse_args_method),
    (CommDiffMambaBlock, _default_parse_args_method),
])
parse_args_method = dict(
    ChainMap(
        parse_args_method,
    )
)

__all__ = [
    'CMSABlock',
    'DCFMBlock',
    'CBAM',
    'DSSFBlock',
    'SSCSBlock',
    'CommDiffMambaBlock',
]