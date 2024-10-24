import torch
from torch import nn
from einops import einsum, rearrange, reduce, repeat
from functools import partial
import math


try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit


def scan_tensor(x: torch.Tensor, scan_mode, forward=True, *args, **kwargs):
    in_pattern = 'batch stream signal channel height width'
    if scan_mode == 'interwave':
        out_pattern = 'stream batch channel (height width signal)'
    elif scan_mode == 'joint':
        out_pattern = 'stream batch channel (signal height width)'
    else:
        raise NotImplementedError

    if forward==True:
        x = rearrange(x, f'{in_pattern}->{out_pattern}', *args, **kwargs)
        return x
    else:
        x = rearrange(x, f'{out_pattern}->{in_pattern}', *args, **kwargs)
        return x


class CommDiffSS2DScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, common_scan_mode, differential_scan_mode, shape):
        assert x.shape == shape
        B, S, C, H, W = shape
        ctx.shape = (B, S, C, H, W)
        ctx.scan_mode = (common_scan_mode, differential_scan_mode)
        axis_info = {
            'batch': B,
            'signal': S,
            'channel': C,
            'height': H,
            'width': W,
        }
        transpose_axis_info = {
            'batch': B,
            'signal': S,
            'channel': C,
            'height': W,
            'width': H,
        }
        xs = x.new_empty((4, S + 1, B, C, H * W * S))
        x_comm_0 = scan_tensor(x.view((B, 1, S, C, H, W)), common_scan_mode, forward=True, **axis_info)
        x_comm_1 = scan_tensor(x.view((B, 1, S, C, H, W)).transpose(-1, -2), common_scan_mode, forward=True, **transpose_axis_info)

        x_dif = x[:, :, None] - x[:, None, :]
        x_dif[:, torch.arange(S-1), torch.arange(S-1)] = x_dif[:, torch.arange(S-1), S-1]
        x_dif[:, :, S-1] = x
        x_diff_0 = scan_tensor(x_dif, differential_scan_mode, forward=True, **axis_info)
        x_diff_1 = scan_tensor(x_dif.transpose(-1, -2), differential_scan_mode, forward=True, **transpose_axis_info)

        xs[0] = torch.cat([x_diff_0, x_comm_0], dim=0).view((S + 1, B, C, H * W * S))
        xs[1] = torch.cat([x_diff_1, x_comm_1], dim=0).view((S + 1, B, C, H * W * S))
        xs[2:] = torch.flip(xs[:2], dims=[-1])
        xs = rearrange(
            xs, 
            'scan_direction stream batch channel unistream_length -> batch scan_direction stream channel unistream_length',
            scan_direction = 4,
            stream = S + 1,
            batch = B,
            channel = C,
            unistream_length = H * W * S,
        ).contiguous()
        return xs # (B, 4, S + 1, C, S * H * W)
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, S, C, H, W = ctx.shape
        assert ys.shape == (B, 4, S + 1, C, S * H * W)
        common_scan_mode, differential_scan_mode = ctx.scan_mode
        axis_info = {
            'batch': B,
            'signal': S,
            'channel': C,
            'height': H,
            'width': W,
        }
        transpose_axis_info = {
            'batch': B,
            'signal': S,
            'channel': C,
            'height': W,
            'width': H,
        }
        ys = rearrange(
            ys,
            'batch scan_direction stream channel unistream_length -> scan_direction stream batch channel unistream_length',
            scan_direction = 4,
            stream = S + 1,
            batch = B,
            channel = C,
            unistream_length = H * W * S,
        )
        ys = ys[:2] + ys[2:].flip(dims=[-1])
        ys_diff_0, ys_comm_0 = torch.split(ys[0], [S, 1], dim=0)
        ys_diff_1, ys_comm_1 = torch.split(ys[1], [S, 1], dim=0)

        y_dif = scan_tensor(ys_diff_0, differential_scan_mode, forward=False, **axis_info) \
            + scan_tensor(ys_diff_1, differential_scan_mode, forward=False, **transpose_axis_info).transpose(-1, -2)
        y = y_dif[:, :, S-1]
        y_dif[:, torch.arange(S-1), S-1] = y_dif[:, torch.arange(S-1), torch.arange(S-1)]
        y_dif[:, torch.arange(S), torch.arange(S)] = 0
        y = y + y_dif.sum(dim=2) - y_dif.sum(dim=1)
        y_com = scan_tensor(ys_comm_0, common_scan_mode, forward=False, **axis_info).view((B, S, C, H, W)) \
            + scan_tensor(ys_comm_1, common_scan_mode, forward=False, **transpose_axis_info).view((B, S, C, W, H)).transpose(-1, -2)
        return y + y_com, None, None, None # (B, S, C, H, W)


class CommDiffSS2DMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, common_scan_mode, differential_scan_mode, shape):
        B, S, C, H, W = shape
        assert ys.shape == (B, 4, S + 1, C, S * H * W)
        ctx.shape = shape
        ctx.scan_mode = (common_scan_mode, differential_scan_mode)
        axis_info = {
            'batch': B,
            'signal': S,
            'channel': C,
            'height': H,
            'width': W,
        }
        transpose_axis_info = {
            'batch': B,
            'signal': S,
            'channel': C,
            'height': W,
            'width': H,
        }
        ys = rearrange(
            ys,
            'batch scan_direction stream channel unistream_length -> scan_direction stream batch channel unistream_length',
            scan_direction = 4,
            stream = S + 1,
            batch = B,
            channel = C,
            unistream_length = H * W * S,
        )
        ys = ys[:2] + ys[2:].flip(dims=[-1])
        ys_diff_0, ys_comm_0 = torch.split(ys[0], [S, 1], dim=0)
        ys_diff_1, ys_comm_1 = torch.split(ys[1], [S, 1], dim=0)
        
        y_dif = scan_tensor(ys_diff_0, differential_scan_mode, forward=False, **axis_info) \
            + scan_tensor(ys_diff_1, differential_scan_mode, forward=False, **transpose_axis_info).transpose(-1, -2)
        y = y_dif[:, :, S-1]
        
        y_dif[:, torch.arange(S-1), S-1] = y_dif[:, torch.arange(S-1), torch.arange(S-1)]
        y_dif[:, torch.arange(S), torch.arange(S)] = 0
        y_dif_extend = y_dif.sum(dim=2)

        y_com = scan_tensor(ys_comm_0, common_scan_mode, forward=False, **axis_info) \
            + scan_tensor(ys_comm_1, common_scan_mode, forward=False, **transpose_axis_info).transpose(-1, -2)

        return y, y_com.view((B, S, C, H, W)), y_dif_extend # (B, S, C, H, W)
    
    @staticmethod
    def backward(ctx, x_dif: torch.Tensor, x_com: torch.Tensor, x_dif_extend: torch.Tensor):
        B, S, C, H, W = ctx.shape
        common_scan_mode, differential_scan_mode = ctx.scan_mode
        axis_info = {
            'batch': B,
            'signal': S,
            'channel': C,
            'height': H,
            'width': W,
        }
        transpose_axis_info = {
            'batch': B,
            'signal': S,
            'channel': C,
            'height': W,
            'width': H,
        }
        xs_com_0 = scan_tensor(x_com.view(B, 1, S, C, H, W), common_scan_mode, forward=True, **axis_info)
        xs_com_1 = scan_tensor(x_com.view(B, 1, S, C, H, W).transpose(-1, -2), common_scan_mode, forward=True, **transpose_axis_info)
        
        x_dif_mat = x_dif.new_zeros((B, S, S, C, H, W))
        
        x_dif_mat += x_dif_extend[:, :, None]
        x_dif_mat[:, torch.arange(S), torch.arange(S)] = 0
        x_dif_mat[:, torch.arange(S-1), torch.arange(S-1)] = x_dif_mat[:, torch.arange(S-1), S-1]

        x_dif_mat[:, :, S-1] += x_dif
        xs_dif_0 = scan_tensor(x_dif_mat, differential_scan_mode, forward=True, **axis_info)
        xs_dif_1 = scan_tensor(x_dif_mat.transpose(-1, -2), differential_scan_mode, forward=True, **transpose_axis_info)
        xs = x_dif.new_zeros((4, S + 1, B, C, H * W * S))
        xs[0] = torch.cat([xs_dif_0, xs_com_0], dim=0)
        xs[1] = torch.cat([xs_dif_1, xs_com_1], dim=0)
        xs[2:] = torch.flip(xs[:2], dims=[-1])
        xs = rearrange(
            xs, 
            'scan_direction stream batch channel unistream_length -> batch scan_direction stream channel unistream_length',
            scan_direction = 4,
            stream = S + 1,
            batch = B,
            channel = C,
            unistream_length = H * W * S,
        ).contiguous()
        return xs, None, None, None # (B, 4, S + 1, C, S * H * W)


def commdiff_ss2d_selective_scan(
    # selective args
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
    # common/differential args
    common_scan_mode: str='interwave',
    differential_scan_mode: str='interwave',
):
    assert all(scan_mode in ['interwave', 'joint'] for scan_mode in [common_scan_mode, differential_scan_mode])
    batch, signal, channel, height, width = x.shape
    parallel, d_state = A_logs.shape
    scan_direction, stream, _, low_rank = dt_projs_weight.shape
    length = height * width
    unistream_length = signal * length
    params = d_state + d_state + low_rank

    assert x.shape == (batch, signal, channel, height, width), f'expected_shape = {(batch, signal, channel, height, width)}, get {x.shape}'
    assert x_proj_weight.shape == (scan_direction, stream, params, channel), f'expected_shape = {(scan_direction, stream, params, channel)}, get {x_proj_weight.shape}'
    if x_proj_bias is not None:
        assert x_proj_bias.squeeze().shape == (scan_direction, stream, params), f'expected_shape = {(scan_direction, stream, params)}, get {x_proj_bias.squeeze().shape}'
    assert dt_projs_weight.shape == (scan_direction, stream, channel, low_rank), f'expected_shape = {(scan_direction, stream, channel, low_rank)}, get {dt_projs_weight.shape}'
    assert A_logs.shape == (parallel, d_state), f'expected_shape = {(parallel, d_state)}, get {A_logs.shape}'
    assert parallel == scan_direction * stream * channel, f'parallel_mamba_number = {parallel}, expected = {scan_direction * stream * channel}'
    assert dt_projs_bias.numel() == parallel, f'expected_length = {parallel}, get {dt_projs_bias.numel()}'
    assert stream == signal + 1, f'expected_stream = {signal + 1}, get {stream}'
    if Ds is not None:
        assert Ds.numel() == parallel, f'expected_length = {parallel}, get {Ds.numel()}'

    if nrows < 1:
        if channel % 4 == 0:
            nrows = 4
        elif channel % 3 == 0:
            nrows = 3
        elif channel % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    
    shape = x.shape
    xs = CommDiffSS2DScan.apply(x, common_scan_mode, differential_scan_mode, shape) # (B, 4, S, C, S * H * W)

    x_dbl = einsum(
        xs,
        x_proj_weight,
        'batch scan_direction stream channel unistream_length, scan_direction stream params channel -> batch scan_direction stream params unistream_length'
    )
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, scan_direction, stream, params, 1)
    dts, Bs, Cs = torch.split(x_dbl, [low_rank, d_state, d_state], dim=3)
    dts = einsum(
        dts,
        dt_projs_weight,
        'batch scan_direction stream low_rank unistream_length, scan_direction stream channel low_rank -> batch scan_direction stream channel unistream_length'
    ).reshape(batch, parallel, unistream_length)
    xs = xs.view(batch, parallel, unistream_length)
    As = -torch.exp(A_logs.to(torch.float)).view(parallel, d_state) # (scan_direction * stream * channel, d_state)
    Bs = Bs.contiguous().view(batch, scan_direction * stream, d_state, unistream_length)
    Cs = Cs.contiguous().view(batch, scan_direction * stream, d_state, unistream_length)
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
    ).view(batch, scan_direction, stream, channel, unistream_length)
    
    y_dif, y_com, y_dif_extend = CommDiffSS2DMerge.apply(ys, common_scan_mode, differential_scan_mode, shape) # (B, S, C, H, W)
    y_dif = out_norm(rearrange(y_dif, 'batch signal channel height width -> batch signal height width channel'))
    y_com = out_norm(rearrange(y_com, 'batch signal channel height width -> batch signal height width channel'))
    y_dif_extend = out_norm(rearrange(y_dif_extend, 'batch signal channel height width -> batch signal height width channel'))
    if to_dtype:
        y_dif = y_dif.to(x.dtype)
        y_com = y_com.to(x.dtype)
        y_dif_extend = y_dif_extend.to(x.dtype)

    return y_dif, y_com, y_dif_extend


class CommDiffMambaBlock(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        common_scan_mode='interwave',
        differential_scan_mode='interwave',
        differential_add_scale=float("inf"),
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
        assert all(scan_mode in ['interwave', 'joint'] for scan_mode in [common_scan_mode, differential_scan_mode])
        self.common_scan_mode = common_scan_mode
        self.differential_scan_mode = differential_scan_mode
        self.differential_add_scale = differential_add_scale
        self.selective_scan = partial(commdiff_ss2d_selective_scan, common_scan_mode=self.common_scan_mode, differential_scan_mode=self.differential_scan_mode)

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
                stream=2+1,
            ).contiguous()
        )

        # dt proj ============================
        dt_projs = self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(
            repeat(
                dt_projs.weight,
                'channel low_rank -> scan_direction stream channel low_rank',
                scan_direction=self.scan_direction,
                stream=2+1,
            ).contiguous()
        )
        self.dt_projs_bias = nn.Parameter(
            repeat(
                dt_projs.bias,
                'channel -> scan_direction stream channel',
                scan_direction=self.scan_direction,
                stream=2+1,
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
                A_log, "channel d_state -> scan_direction stream channel d_state", 
                scan_direction=scan_direction,
                stream=2+1,
            )
            if merge:
                A_log = A_log.flatten(0, 2)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, scan_direction=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if scan_direction > 0:
            D = repeat(
                D, 
                'channel -> scan_direction stream channel', 
                scan_direction=scan_direction,
                stream=2+1,
            )
            if merge:
                D = D.flatten(0, 2)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        nrows = 1
        if self.ssm_low_rank:
            B, S, _, H, W = x.shape
            x = self.in_rank(x.flatten(1, 2)).reshape((B, S, -1, H, W)).contiguous()
        x_dif, x_com, x_dif_extend = self.selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, force_fp32=self.training,
        )
        if self.ssm_low_rank:
            x_dif, x_com, x_dif_extend = self.out_rank(x_dif), self.out_rank(x_com), self.out_rank(x_dif_extend)
        return x_dif, x_com, x_dif_extend
    
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
        y_dif, y_com, y_dif_extend = self.forward_core(x)
        y = y_com + y_dif * 0 + y_dif_extend * 0
        y = y * z
        out = self.dropout(self.out_proj(y))
        out0, out1 = out[:, 0, ...].permute(0, 3, 1, 2), out[:, 1, ...].permute(0, 3, 1, 2)
        return (input0 + out0).contiguous(), (input1 + out1).contiguous()



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

if __name__ == '__main__': 
    def test_commdiff_ss2d_selective_scan():
        x = torch.randn((batch, stream, channel, height, width)).to('cuda:0')
        x_proj_weight = torch.randn((scan_direction, stream + 1, param, channel)).to('cuda:0')
        x_proj_bias = torch.randn(((scan_direction, stream + 1, param, 1))).to('cuda:0')
        dt_projs_weight = torch.randn((scan_direction, stream + 1, channel, low_rank)).to('cuda:0')
        dt_projs_bias = torch.randn((scan_direction, stream + 1, channel)).to('cuda:0')
        A_logs = torch.randn((scan_direction * (stream + 1) * channel, d_state)).to('cuda:0')
        Ds = torch.randn((scan_direction, (stream + 1), channel)).to('cuda:0')
        out_norm = nn.LayerNorm(channel).to('cuda:0')

        y1, y2, y3 = commdiff_ss2d_selective_scan(
            x,
            x_proj_weight,
            x_proj_bias,
            dt_projs_weight,
            dt_projs_bias,
            A_logs,
            Ds,
            out_norm,
            common_scan_mode='interwave',
            differential_scan_mode='joint',
        )
        return y1 + y2 + y3

    def test_CommDiffMambaBlock():
        comm_diff = CommDiffMambaBlock(d_model=d_model).to('cuda:0')
        x1 = torch.randn((batch, d_model, height, width)).to('cuda:0')
        x2 = torch.randn((batch, d_model, height, width)).to('cuda:0')
        y1, y2 = comm_diff([x1, x2])
        return y1, y2

    y = test_commdiff_ss2d_selective_scan()
    print(y.shape)
    l = y.sum()
    l.backward()
    print('finish')

    y1, y2 = test_CommDiffMambaBlock()
    print(y1.shape, y2.shape)
    l = torch.cat([e.sum().reshape((1,)) for e in [y1, y2]], dim=0).sum()
    l.backward()
    print('finish')
