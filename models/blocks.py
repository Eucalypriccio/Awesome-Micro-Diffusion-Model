"""U-Net 基础模块：ResBlock、Downsample、Upsample、SelfAttention。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块：Conv1 -> GroupNorm(经时间嵌入 γ/β 调制) -> SiLU -> Conv2 -> GroupNorm -> SiLU，
    与输入跳跃相加（通道数不一致时跳跃连接用 1x1 卷积对齐）。
    """

    def __init__(self, in_channels, out_channels, time_dim, num_groups):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        # 每个残差块专属的两层 MLP：time_dim -> time_dim -> 2*out_channels（一半 γ 一半 β）
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, 2 * out_channels),
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(x)
        # 时间嵌入生成每个通道的 scale(γ) 与 shift(β)，作用在 GroupNorm 之后
        scale, shift = self.time_mlp(time_emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm1(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(self.norm2(h))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    """MaxPool 2x2（stride 2），分辨率减半，不改变通道数。"""

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=2)


class Upsample(nn.Module):
    """最近邻插值 x2；smooth=True 时再经 3x3 平滑卷积（通道数不变，CPU 上较贵）。"""

    def __init__(self, channels, smooth=True):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1) if smooth else None

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.conv is not None else x


class SelfAttention(nn.Module):
    """瓶颈自注意力：把 H*W 个像素视为 token，按 channel 切多头，输出与输入残差相加。"""

    def __init__(self, channels, num_heads, num_groups):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels({channels}) 必须整除 num_heads({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)   # 一次算出 Q/K/V
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = torch.chunk(self.qkv(h), 3, dim=1)   # 各 [B, C, H, W]
        # 按 channel 切头：[B, C, H, W] -> [B, heads, H*W, head_dim]
        q = q.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W).transpose(-1, -2)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)
        out = attn @ v                                  # [B, heads, H*W, head_dim]
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)
