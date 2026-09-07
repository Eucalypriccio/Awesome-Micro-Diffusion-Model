"""时间步嵌入：正弦位置编码 + 两层全连接。"""
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入：f_i = 10000^(-2i/d)，偶数位 sin(t*f_i)，奇数位 cos(t*f_i)。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] 整数时间步
        half = self.dim // 2
        freqs = 10000 ** (-torch.arange(half, device=t.device, dtype=torch.float32) * 2 / self.dim)
        args = t.float()[:, None] * freqs[None, :]        # [B, half]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # [B, dim]


class TimeEmbedding(nn.Module):
    """正弦嵌入后经两层 MLP，得到全网络共享的时间向量（各残差块再各自投影）。"""

    def __init__(self, dim):
        super().__init__()
        self.sinusoid = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoid(t))
