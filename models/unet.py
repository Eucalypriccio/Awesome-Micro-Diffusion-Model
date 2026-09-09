"""U-Net 组装：编码器（残差块 + MaxPool 降采样）-> 瓶颈（ResBlock+注意力+ResBlock）
-> 解码器（上采样 + 跳跃连接拼接 + 残差块）-> 1x1 输出卷积。

以默认配置（base=24, multipliers=(1,2,4)）为例，数据流：
  [B,1,28,28] -> init_conv -> L0: 24@28 -> pool -> L1: 48@14 -> pool -> L2: 96@7
  -> bottleneck: 96@7 -> dec L2: concat(96)+96 -> 96@7 -> up -> dec L1: concat(48)+96
  -> 48@14 -> up -> dec L0: concat(24)+48 -> 24@28 -> out_conv -> [B,1,28,28]
"""
import torch
import torch.nn as nn

from models.blocks import Downsample, ResBlock, SelfAttention, Upsample
from models.time_embedding import TimeEmbedding


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=24, channel_multipliers=(1, 2, 4),
                 num_res_blocks=1, time_dim=128, num_groups=8, num_heads=4,
                 upsample_smooth=False, conditional=True, num_classes=10):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]   # 如 [32, 64, 128]
        num_levels = len(channels)

        self.time_embedding = TimeEmbedding(time_dim)
        # 数字标签 embedding：num_classes 个真实标签 + 索引 num_classes 表示空标签 ∅
        self.conditional = conditional
        self.num_classes = num_classes
        if conditional:
            self.label_embed = nn.Embedding(num_classes + 1, time_dim)
        self.init_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # 编码器：第 i 层输出通道 channels[i]，跳跃连接保存每层降采样前的输出
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_channels = channels[0]
        for i, ch in enumerate(channels):
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = prev_channels if j == 0 else ch
                blocks.append(ResBlock(in_ch, ch, time_dim, num_groups))
            prev_channels = ch
            self.encoder_blocks.append(blocks)
            # 最后一层不再降采样（已到达瓶颈分辨率）
            self.downsamples.append(Downsample() if i < num_levels - 1 else nn.Identity())

        # 瓶颈：ResBlock -> SelfAttention -> ResBlock
        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_dim, num_groups)
        self.mid_attn = SelfAttention(channels[-1], num_heads, num_groups)
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_dim, num_groups)

        # 解码器：自深向浅，进入每层的特征与该层编码器输出按 channel 拼接
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(num_levels)):
            ch = channels[i]
            # 最深层输入来自瓶颈（channels[-1]），其余来自下一层的上采样输出（channels[i+1]）
            in_from_below = channels[i + 1] if i < num_levels - 1 else channels[-1]
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = (in_from_below + ch) if j == 0 else ch   # 首个块额外接收拼接的跳跃连接
                blocks.append(ResBlock(in_ch, ch, time_dim, num_groups))
            self.decoder_blocks.append(blocks)
            self.upsamples.append(Upsample(ch, smooth=upsample_smooth) if i > 0 else nn.Identity())

        self.out_conv = nn.Conv2d(channels[0], in_channels, kernel_size=1)   # 1x1 卷积恢复通道数

    def forward(self, x, t, labels=None):
        time_emb = self.time_embedding(t)
        if self.conditional:
            # 标签向量与时间向量同维相加，复用残差块的 γ/β 注入通路；
            # labels 为 None 时按空标签 ∅ 处理（即无条件模式）
            if labels is None:
                labels = torch.full((x.shape[0],), self.num_classes,
                                    device=x.device, dtype=torch.long)
            time_emb = time_emb + self.label_embed(labels)
        h = self.init_conv(x)

        skips = []
        for blocks, down in zip(self.encoder_blocks, self.downsamples):
            for block in blocks:
                h = block(h, time_emb)
            skips.append(h)      # 保存降采样前的输出（跳跃连接）
            h = down(h)

        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        for blocks, up in zip(self.decoder_blocks, self.upsamples):
            h = torch.cat([h, skips.pop()], dim=1)
            for block in blocks:
                h = block(h, time_emb)
            h = up(h)

        return self.out_conv(h)


def build_model(config):
    """按配置构建 UNet（train / sample / check 复用）。"""
    return UNet(
        in_channels=config.in_channels,
        base_channels=config.base_channels,
        channel_multipliers=tuple(config.channel_multipliers),
        num_res_blocks=config.num_res_blocks,
        time_dim=config.time_dim,
        num_groups=config.num_groups,
        num_heads=config.num_heads,
        upsample_smooth=config.upsample_smooth,
        conditional=config.conditional,
        num_classes=config.num_classes,
    )
