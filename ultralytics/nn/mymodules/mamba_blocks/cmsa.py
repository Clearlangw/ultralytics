import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any, Dict
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat, einsum
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
import itertools
try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit


class CrossSS2DScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, S, C, H, W = x.shape
        ctx.shape = (B, S, C, H, W)
        xs = x.new_empty((B, S, 4, C, H * W))
        xs[:, :, 0] = x.flatten(3, 4)
        xs[:, :, 1] = x.transpose(dim0=3, dim1=4).flatten(3, 4)
        xs[:, :, 2:4] = torch.flip(xs[:, :, 0:2], dims=[-1])
        xs = rearrange(
            xs, 
            'batch stream scan_direction channel (height width) -> batch scan_direction channel height width stream',
            batch = B,
            stream = S,
            channel = C,
            scan_direction = 4,
            height = H,
            width = W
        ).contiguous()
        return xs # (B, 4, C, S, H, W)
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, S, C, H, W = ctx.shape
        L = H * W
        ys = rearrange(
            ys,
            'batch scan_direction channel height width stream -> batch stream scan_direction channel (height width)',
            batch = B,
            stream = S,
            channel = C,
            scan_direction = 4,
            height = H,
            width = W
        )
        ys = ys[:, :, 0:2] + ys[:, :, 2:4].flip(dims=[-1]).view(B, S, 2, C, L)
        y = ys[:, :, 0].contiguous() + ys[:, :, 1].view(B, S, C, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, S, C, L)
        return y.view(B, S, C, H, W) # (B, S, C, H, W)


class CrossSS2DMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, D, C, H, W, S = ys.shape
        ctx.shape = (B, D, C, H, W, S)
        ys = rearrange(
            ys,
            'batch scan_direction channel height width stream -> batch stream scan_direction channel (height width)',
            batch = B,
            stream = S,
            channel = C,
            scan_direction = D,
            height = H,
            width = W
        )
        ys = ys.view(B, S, D, C, -1)
        ys = ys[:, :, 0:2] + ys[:, :, 2:4].flip(dims=[-1]).view(B, S, 2, C, -1)
        y = ys[:, :, 0].contiguous().view(B, S, C, H, W) + ys[:, :, 1].view(B, S, C, W, H).transpose(dim0=3, dim1=4).contiguous()
        return y # (B, S, C, H, W)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, D, C, H, W, S = ctx.shape
        B, S, C, H, W = x.shape
        L = H * W

        xs = x.new_empty((B, S, 4, C, H * W))
        xs[:, :, 0] = x.flatten(3, 4)
        xs[:, :, 1] = x.transpose(dim0=3, dim1=4).flatten(3, 4)
        xs[:, :, 2:4] = torch.flip(xs[:, :, 0:2], dims=[-1])
        xs = rearrange(
            xs, 
            'batch stream scan_direction channel (height width) -> batch scan_direction channel height width stream',
            batch = B,
            stream = S,
            channel = C,
            scan_direction = D,
            height = H,
            width = W
        ).contiguous()
        return xs


def cross_ss2d_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    out_norm: torch.nn.Module=None,
    nrows = -1,
    delta_softplus = True,
    to_dtype=True,
    force_fp32=True,
):
    batch, stream, channel, height, width = x.shape
    parallel, d_state = A_logs.shape
    stream, scan_direction, channel, low_rank = dt_projs_weight.shape
    unistream_length = height * width
    multistream_length = stream * unistream_length
    params = d_state + d_state + low_rank

    assert x.shape == (batch, stream, channel, height, width), f'expexted_shape = {(batch, stream, channel, height, width)}, get {x.shape}'
    assert x_proj_weight.shape == (scan_direction, stream, params, channel), f'expexted_shape = {(scan_direction, stream, params, channel)}, get {x_proj_weight.shape}'
    if x_proj_bias is not None:
        assert x_proj_bias.squeeze().shape == (scan_direction, params, stream), f'expexted_shape = {(scan_direction, params, stream)}, get {x_proj_bias.squeeze().shape}'
    assert dt_projs_weight.shape == (stream, scan_direction, channel, low_rank), f'expexted_shape = {(stream, scan_direction, channel, low_rank)}, get {dt_projs_weight.shape}'
    assert A_logs.shape == (parallel, d_state), f'expexted_shape = {(parallel, d_state)}, get {A_logs.shape}'
    assert parallel == scan_direction * channel, f'parallel_mamba_number = {parallel}, expected = {scan_direction * channel}'
    assert dt_projs_bias.numel() == parallel, f'expexted_length = {parallel}, get {dt_projs_bias.numel()}'
    if Ds is not None:
        assert Ds.numel() == parallel, f'expexted_length = {parallel}, get {Ds.numel()}'

    if nrows < 1:
        if channel % 4 == 0:
            nrows = 4
        elif channel % 3 == 0:
            nrows = 3
        elif channel % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    
    xs = CrossSS2DScan.apply(x) # (batch scan_direction channel height width stream)

    x_dbl = einsum(
        xs,
        x_proj_weight,
        'batch scan_direction channel height width stream, scan_direction stream params channel -> batch scan_direction params height width stream'
    ).reshape(batch, scan_direction, params, unistream_length, stream)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, scan_direction, params, 1, stream)
    dts, Bs, Cs = torch.split(x_dbl, [low_rank, d_state, d_state], dim=2)
    dts = einsum(
        dts,
        dt_projs_weight,
        'batch scan_direction low_rank unistream_length stream, stream scan_direction channel low_rank -> batch scan_direction channel unistream_length stream'
    ).reshape(batch, parallel, multistream_length)
    xs = xs.view(batch, parallel, multistream_length)
    As = -torch.exp(A_logs.to(torch.float)).view(parallel, d_state) # (scan_direction * channel, d_state)
    Bs = Bs.contiguous().view(batch, scan_direction, d_state, multistream_length)
    Cs = Cs.contiguous().view(batch, scan_direction, d_state, multistream_length)
    Ds = Ds.to(torch.float).view(parallel)
    delta_bias = dt_projs_bias.to(torch.float).view(parallel)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    selective_scan = partial(selective_scan_fn, backend="mamba")
    
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(batch, scan_direction, channel, height, width, stream)
    
    y: torch.Tensor = CrossSS2DMerge.apply(ys).view(batch, stream, channel, height * width)
    y = y.transpose(dim0=2, dim1=3).contiguous() # (batch, stream, unistream_length, channel)
    y = out_norm(y).view(batch, stream, height, width, channel)

    return (y.to(x.dtype) if to_dtype else y)
    
class CMSABlock(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        # dwconv ===============
        dw_kernel_size=3,
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        assert dw_kernel_size % 2 == 1
        assert dw_kernel_size > 1

        # softmax | sigmoid | norm ===========================
        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type =======================================
        self.scan_direction = 4

        # in proj =======================================
        self.in_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs),
                )
                for _ in range(2)
            ]
        )

        self.act = nn.SiLU()
        
        # conv =======================================
        self.conv2d = nn.Conv2d(
            in_channels=d_expand*2,
            out_channels=d_expand*2,
            groups=d_expand*2,
            bias=conv_bias,
            kernel_size=dw_kernel_size,
            padding=(dw_kernel_size - 1) // 2,
            **factory_kwargs,
        )

        # patch mark
        self.patch_mark_generator = nn.Conv2d(
            in_channels=2*d_expand,
            out_channels=d_expand,
            groups=1,
            bias=True,
            kernel_size=1,
            padding=0,
            **factory_kwargs
        )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj_weight = nn.Parameter(
            repeat(
                nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight,
                'channel params -> scan_direction stream channel params',
                scan_direction=self.scan_direction,
                stream=2,
            ).contiguous()
        )

        # dt proj ============================
        dt_projs = self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(
            repeat(
                dt_projs.weight,
                'channel low_rank -> stream scan_direction channel low_rank',
                scan_direction=self.scan_direction,
                stream=2,
            ).contiguous()
        )
        self.dt_projs_bias = nn.Parameter(
            repeat(
                dt_projs.bias,
                'channel -> scan_direction channel',
                scan_direction=self.scan_direction,
            ).contiguous()
        )
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, scan_direction=self.scan_direction, merge=True) # (scan_direction * channel, d_state)
        self.Ds = self.D_init(d_inner, scan_direction=self.scan_direction, merge=True) # (scan_direction, channel)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, scan_direction=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "d_state -> channel d_state",
            channel=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if scan_direction > 0:
            A_log = repeat(
                A_log, "channel d_state -> scan_direction channel d_state", 
                scan_direction=scan_direction
            )
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, scan_direction=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if scan_direction > 0:
            D = repeat(D, 'channel -> scan_direction channel', scan_direction=scan_direction)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        nrows = 1
        if self.ssm_low_rank:
            B, S, _, H, W = x.shape
            x = self.in_rank(x.flatten(1, 2)).reshape((B, S, -1, H, W)).contiguous()
        x = cross_ss2d_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x
    
    def forward(self, input, **kwargs):
        input0, input1 = input
        xz = torch.stack(
            [
                in_proj(input.permute(0, 2, 3, 1))
                for in_proj, input 
                in zip(self.in_proj, [input0, input1])
            ],
            dim=1
        ) # (batch stream height width channel)
        x, z = xz.chunk(2, dim=-1)
        z = self.act(z.clone())
        x = x.permute(0, 1, 4, 2, 3).contiguous() # (batch stream channel height width)
        x = self.conv2d(x.flatten(1, 2)).view(*x.shape)
        # x = self.conv2d(x.flatten(0, 1)).view(*x.shape)
        patch_mark = self.patch_mark_generator(x.flatten(1, 2)) # (batch channel height width)
        x = x + patch_mark.unsqueeze(dim=1)
        y = self.forward_core(x)
        y = y * z
        out = self.dropout(self.out_proj(y))
        out0, out1 = out[:, 0, ...].permute(0, 3, 1, 2), out[:, 1, ...].permute(0, 3, 1, 2)
        return (input0 + out0).contiguous(), (input1 + out1).contiguous()

if __name__ == '__main__': 
    batch = 3
    stream = 2
    channel = 16
    height = 12
    width = 9
    scan_direction = 4

    d_state = 4
    low_rank = 2
    param = d_state + d_state + low_rank
    d_model = 96

    def test_cross_ss2d_selective_scan():
        x = torch.randn((batch, stream, channel, height, width)).to('cuda:0')
        x_proj_weight = torch.randn((scan_direction, stream, param, channel)).to('cuda:0')
        x_proj_bias = torch.randn(((1, scan_direction, param, stream, 1))).to('cuda:0')
        dt_projs_weight = torch.randn((stream, scan_direction, channel, low_rank)).to('cuda:0')
        dt_projs_bias = torch.randn((scan_direction, channel)).to('cuda:0')
        A_logs = torch.randn((4 * channel, d_state)).to('cuda:0')
        Ds = torch.randn((scan_direction, channel)).to('cuda:0')
        out_norm = nn.LayerNorm(channel).to('cuda:0')

        y = cross_ss2d_selective_scan(
            x,
            x_proj_weight,
            x_proj_bias,
            dt_projs_weight,
            dt_projs_bias,
            A_logs,
            Ds,
            out_norm
        )
        return y

    def test_CMSABlock():
        cmsa = CMSABlock(d_model=d_model).to('cuda:0')
        x1 = torch.randn((batch, d_model, height, width)).to('cuda:0')
        x2 = torch.randn((batch, d_model, height, width)).to('cuda:0')
        y1, y2 = cmsa([x1, x2])
        return y1, y2

    y = test_cross_ss2d_selective_scan()
    print(y.shape)
    l = y.sum()
    l.backward()
    print('finish')

    y1, y2 = test_CMSABlock()
    print(y1.shape, y2.shape)
    l = torch.cat([e.sum().reshape((1,)) for e in [y1, y2]], dim=0).sum()
    l.backward()
    print('finish')
