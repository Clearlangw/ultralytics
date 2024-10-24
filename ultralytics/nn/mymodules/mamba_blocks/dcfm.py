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


class DisparitySelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, S, C, H, W = x.shape
        L = H * W
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, 2, H * W))
        xs[:, 0, :, 0] = x[:, 0].flatten(2, 3)
        xs[:, 1, :, 0] = x[:, 1].flatten(2, 3)
        xs[:, 0, :, 1] = x[:, 2].flatten(2, 3)
        xs[:, 1, :, 1] = x[:, 2].flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.contiguous().view(B, 4, C, 2, H, W)
        return xs # (B, 4, C, 2, H, W)
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys.view(B, 4, C, 2, L)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1])
        y = ys.new_empty((B, 3, C, L))
        y[:, 0:2] = ys[:, 0:2, :, 0]
        y[:, 2] = ys[:, 0:2, :, 1].sum(dim=1)
        return y.contiguous().view(B, 3, C, H, W) # (B, S, C, H, W)


class DisparityMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, D, C, S, H, W = ys.shape
        ctx.shape = (B, D, C, S, H, W)
        ys = ys.view(B, D, C, S, H * W)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1])
        y = ys[:, :, :, 0, :].contiguous().view(B, 2, C, H, W)
        return y # (B, S, C, H, W)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, D, C, S, H, W = ctx.shape
        B, S, C, H, W = x.shape
        L = H * W

        xs = x.new_zeros((B, D, C, S, H * W))
        xs[:, 0, :, 0] = x[:, 0].flatten(2, 3)
        xs[:, 1, :, 0] = x[:, 1].flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.contiguous().view(B, D, C, S, H, W)
        return xs # (B, D, C, S, H, W)


def disparity_guided_selective_scan(
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
    scan_direction, channel, low_rank = dt_projs_weight.shape
    length = height * width
    params = d_state + d_state + low_rank

    assert x.shape == (batch, stream, channel, height, width), f'expexted_shape = {(batch, stream, channel, height, width)}, get {x.shape}'
    assert x_proj_weight.shape == (scan_direction, params, channel), f'expexted_shape = {(scan_direction, params, channel)}, get {x_proj_weight.shape}'
    if x_proj_bias is not None:
        assert x_proj_bias.squeeze().shape == (scan_direction, params), f'expexted_shape = {(scan_direction, params)}, get {x_proj_bias.squeeze().shape}'
    assert dt_projs_weight.shape == (scan_direction, channel, low_rank), f'expexted_shape = {(scan_direction, channel, low_rank)}, get {dt_projs_weight.shape}'
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
    
    xs = DisparitySelectiveScan.apply(x) # (batch scan_direction channel stream height width)

    x_dbl = einsum(
        xs,
        x_proj_weight,
        'batch scan_direction channel stream height width, scan_direction params channel -> batch scan_direction params stream height width'
    ).reshape(batch, scan_direction, params, 2 * length)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, scan_direction, params, 1)
    dts, Bs, Cs = torch.split(x_dbl, [low_rank, d_state, d_state], dim=2)
    dts = einsum(
        dts,
        dt_projs_weight,
        'batch scan_direction low_rank length, scan_direction channel low_rank -> batch scan_direction channel length'
    ).reshape(batch, parallel, 2 * length)
    xs = xs.view(batch, parallel, 2 * length)
    As = -torch.exp(A_logs.to(torch.float)).view(parallel, d_state) # (scan_direction * channel, d_state)
    Bs = Bs.contiguous().view(batch, scan_direction, d_state, 2 * length)
    Cs = Cs.contiguous().view(batch, scan_direction, d_state, 2 * length)
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
    ).view(batch, scan_direction, channel, 2, height, width)
    
    y: torch.Tensor = DisparityMerge.apply(ys).view(batch, 2, channel, height * width)
    y = y.transpose(dim0=2, dim1=3).contiguous() # (batch, 2, length, channel)
    y = out_norm(y).view(batch, 2, height, width, channel)

    return (y.to(x.dtype) if to_dtype else y)


class CAB(nn.Module):
    def __init__(
        self,
        in_channels,
        channel_first: False,
        **kwargs,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.channel_attention_generator = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
        )
        self.act = nn.Sigmoid()
        self.channel_first = channel_first

    def forward(self, x: torch.Tensor):
        x = self.silu(x)
        if self.channel_first:
            B, C, H, W = x.shape
            avg_x = x.flatten(2, 3).clone().mean(dim=-1)
            max_x = x.flatten(2, 3).clone().max(dim=-1)[0]
        else:
            B, H, W, C = x.shape
            avg_x = x.flatten(1, 2).clone().mean(dim=-2)
            max_x = x.flatten(1, 2).clone().max(dim=-2)[0]

        channel_attention = self.channel_attention_generator(avg_x) + self.channel_attention_generator(max_x)
        channel_attention = self.act(channel_attention).contiguous() + 1

        if self.channel_first:
            return x.clone() * channel_attention.view(B, C, 1, 1)
        else:
            return x.clone() * channel_attention.view(B, 1, 1, C)
    
class DCFMBlock(nn.Module):
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
        self.in_norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_expand, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()
        
        # conv =======================================
        self.conv2d = nn.Conv2d(
            in_channels=d_expand * 3,
            out_channels=d_expand * 3,
            groups=d_expand * 3,
            bias=conv_bias,
            kernel_size=dw_kernel_size,
            padding=(dw_kernel_size - 1) // 2,
            **factory_kwargs,
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
                'channel params -> scan_direction channel params',
                scan_direction=self.scan_direction,
            ).contiguous()
        )

        # dt proj ============================
        dt_projs = self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(
            repeat(
                dt_projs.weight,
                'channel low_rank -> scan_direction channel low_rank',
                scan_direction=self.scan_direction,
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
        self.out_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs),
                    nn.Dropout(dropout) if dropout > 0. else nn.Identity()
                )
                for _ in range(2)
            ]
        )

        # CAB ========================================
        self.cab = nn.ModuleList(
            [
                CAB(in_channels=d_model, channel_first=False)
                for _ in range(2)
            ]
        )

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
        x = disparity_guided_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x
    
    def forward(self, input, **kwargs):
        input0, input1 = input
        x = torch.stack(
            [input0.permute(0, 2, 3, 1), input1.permute(0, 2, 3, 1)],
            dim=1
        ) # (batch stream height width channel)
        x = self.in_norm(x)
        z = torch.stack(
            [
                cab(x[:, idx])
                for idx, cab in enumerate(self.cab)
            ],
            dim=1
        )

        x = torch.cat(
            [
                x,
                x[:, 0:1] - x[:, 1:2]
            ], 
            dim=1
        ).contiguous()
        x = self.in_proj(x).permute(0, 1, 4, 2, 3).contiguous() # (batch stream channel height width)
        x = self.conv2d(x.flatten(1, 2)).view(*x.shape)
        x = self.act(x).contiguous()
        y = self.forward_core(x)
        out0 = self.out_proj[0](y[:, 0]) * z[:, 0]
        out1 = self.out_proj[1](y[:, 1]) * z[:, 1]
        out0, out1 = out0.permute(0, 3, 1, 2), out1.permute(0, 3, 1, 2)
        return (input0 + out0).contiguous(), (input1 + out1).contiguous()
    

if __name__ == '__main__': 
    batch = 3
    channel = 16
    height = 12
    width = 9
    scan_direction = 4

    d_state = 4
    low_rank = 2
    param = d_state + d_state + low_rank
    d_model = 96

    def test_cross_ss2d_selective_scan():
        x = torch.randn((batch, 3, channel, height, width)).to('cuda:0')
        x_proj_weight = torch.randn((scan_direction, param, channel)).to('cuda:0')
        x_proj_bias = torch.randn(((1, scan_direction, param, 1))).to('cuda:0')
        dt_projs_weight = torch.randn((scan_direction, channel, low_rank)).to('cuda:0')
        dt_projs_bias = torch.randn((scan_direction, channel)).to('cuda:0')
        A_logs = torch.randn((4 * channel, d_state)).to('cuda:0')
        Ds = torch.randn((scan_direction, channel)).to('cuda:0')
        out_norm = nn.LayerNorm(channel).to('cuda:0')

        y = disparity_guided_selective_scan(
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

    def test_DCFMBlock():
        dcfm = DCFMBlock(d_model=d_model).to('cuda:0')
        x0 = torch.randn((batch, height, width, d_model)).to('cuda:0')
        x1 = torch.randn((batch, height, width, d_model)).to('cuda:0')
        y0, y1 = dcfm([x0, x1])
        return y0, y1
    
    y = test_cross_ss2d_selective_scan()
    print(y.shape)
    l = y.sum()
    l.backward()
    print('finish')

    y0, y1 = test_DCFMBlock()
    print(y0.shape, y1.shape)
    l = torch.cat([e.sum().reshape((1,)) for e in [y0, y1]], dim=0).sum()
    l.backward()
    print('finish')
