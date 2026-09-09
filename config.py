"""全部超参数集中管理。命令行参数可覆盖部分字段（见 main.py）。"""
from dataclasses import dataclass


@dataclass
class Config:
    # 数据
    data_dir: str = "./data"
    image_size: int = 28          # MNIST 28x28
    in_channels: int = 1          # 灰度图

    # 模型（U-Net）—— 经 CPU 实测调优（见 plan.md 3.4）：base24 / 2块每层 / 纯插值上采样
    base_channels: int = 24
    channel_multipliers: tuple = (1, 2, 4)   # 三个分辨率层级 28/14/7 -> 24/48/96
    num_res_blocks: int = 2       # 每个层级的残差块数
    time_dim: int = 128           # 时间嵌入维度
    num_groups: int = 8           # GroupNorm 组数（需整除 24/48/96）
    num_heads: int = 4            # 瓶颈自注意力头数（96 通道 -> head_dim 24）
    upsample_smooth: bool = False # True 时上采样后接 3x3 平滑卷积（约占 19% 算力，CPU 上去掉）

    # 条件生成（扩展二：指定数字采样，classifier-free guidance）
    conditional: bool = True      # 是否带数字标签条件（旧的无条件存档在采样时自动兼容）
    num_classes: int = 10         # 数字 0-9；空标签 ∅ 用索引 num_classes 表示
    label_drop_prob: float = 0.1  # 训练时把标签替换为 ∅ 的概率
    guidance_w: float = 2.0       # 采样时的默认引导强度（1=普通条件采样，0=无条件）

    # 扩散过程
    T: int = 1000                 # 总扩散步数
    schedule: str = "linear"      # linear / cosine
    beta_start: float = 1e-4      # 线性调度端点
    beta_end: float = 0.02
    cosine_s: float = 0.008       # 余弦调度偏移量

    # 训练
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    grad_clip: float = 1.0
    ema_decay: float = 0.999      # 权重指数滑动平均；采样用 EMA 权重，质量更稳
    log_interval: int = 100       # 每多少 step 打印一次
    sample_every: int = 0         # 每多少 epoch 采样监控一次（0=关闭，CPU 上较慢）
    seed: int = 42

    # 采样
    num_samples: int = 64         # 8x8 网格

    # 路径
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
