import torch
from torch import nn
from math import ceil


class PowerMean(nn.Module):
    def __init__(self, dim, p):
        super().__init__()
        self.p = p
        self.dim = dim
    
    def forward(self, x):
        if self.p == 1:
            return x.mean(dim=self.dim, keepdims=True)
        x_max = x.max(dim=self.dim, keepdims=True)[0]
        if self.p == float("inf"):
            return x_max
        x = ((x / x_max)**self.p).mean(dim=self.dim, keepdims=True)
        return (x ** (1 / self.p)) * x_max


def get_power_mean_func(power_index, dim=-1):
    assert power_index > 0
    return PowerMean(dim=dim, p=power_index)


class CAM(nn.Module):
    def __init__(
        self, 
        channels,
        reduction_ratio = 16,
        power_mean = [float(1), float("inf")],
        bias=False,
        factor=1,
    ):
        super().__init__()
        self.channels = channels
        self.hidden = ceil(self.channels / reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.hidden, bias=bias),
            nn.ReLU(),
            nn.Linear(self.hidden, self.channels, bias=bias),
        )
        self.power_func = []
        for power_index in (power_mean if hasattr(power_mean, "__iter__") else [power_mean]):
            self.power_func.append(get_power_mean_func(power_index, dim=-1))
        self.factor = factor
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_reduction = torch.cat(
            [
                power_func(x.view(B, C, -1))
                for power_func in self.power_func
            ],
            dim=-1
        ) # B C len(power_func)
        att = self.mlp(x_reduction.permute(0, 2, 1)) # B len(power_func) C
        y = (att.sum(dim=1, keepdims=False) / self.factor).sigmoid().view((B, C, 1, 1))
        return (y + 1) * x

class SAM(nn.Module):
    def __init__(
        self, 
        channels,
        kernel_size=7,
        power_mean = [float(1), float("inf")],
        bias=False,
        factor=1,
    ):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(
            in_channels=len(power_mean), 
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1)//2,
            bias=bias
        )
        self.power_func = []
        for power_index in (power_mean if hasattr(power_mean, "__iter__") else [power_mean]):
            self.power_func.append(get_power_mean_func(power_index, dim=-3))
        self.factor = factor
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_reduction = torch.cat(
            [
                power_func(x)
                for power_func in self.power_func
            ],
            dim=-3
        ) # B len(power_func) H W
        att = self.conv(x_reduction) # B 1 H W
        y = (att / self.factor).sigmoid().view((B, 1, H, W))
        return (y + 1) * x

class CBAM(nn.Module):
    def __init__(
        self,
        channels,
        channel_attention_reduction_ratio = 16,
        channel_attention_power_mean = [float(1), float("inf")],
        channel_attention_bias=False,
        channel_attention_factor=1,
        spatial_attention_kernel_size=7,
        spatial_attention_power_mean = [float(1), float("inf")],
        spatial_attention_bias=False,
        spatial_attention_factor=1,
        channel_attention_first=True,
    ):
        super().__init__()
        channel_attention = CAM(
            channels=channels,
            reduction_ratio=channel_attention_reduction_ratio,
            power_mean=channel_attention_power_mean,
            bias=channel_attention_bias,
            factor=channel_attention_factor,
        )
        spatial_attention = SAM(
            channels=channels,
            kernel_size=spatial_attention_kernel_size,
            power_mean=spatial_attention_power_mean,
            bias=spatial_attention_bias,
            factor=spatial_attention_factor,
        )
        if channel_attention_first:
            self.att = nn.Sequential(channel_attention, spatial_attention)
        else:
            self.att = nn.Sequential(spatial_attention, channel_attention)
    
    def forward(self, x):
        return self.att(x)

if __name__ == '__main__':
    x = torch.randn((4, 16, 5, 5))
    cbam = CBAM(channels=16)
    y = cbam(x)
    print(y.shape)
    loss = y.sum()
    loss.backward()
    print(loss)
