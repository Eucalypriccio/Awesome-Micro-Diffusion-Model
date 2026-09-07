"""辅助工具包：设备、随机种子、参数量统计、EMA 权重、存图、计时、loss 曲线。"""
import os
import random

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def get_device():
    """有 GPU 用 GPU，否则 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EmaWeights:
    """模型权重的指数滑动平均（EMA）：shadow = decay * shadow + (1-decay) * param。

    DDPM 论文的标准做法：训练照常进行，采样时用 EMA 权重，生成质量更稳定。
    每步仅一次 lerp，CPU 上耗时可忽略。
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, shadow):
        """从 checkpoint 恢复 EMA 权重。"""
        self.shadow = {k: v.detach().clone() for k, v in shadow.items()}


def save_image_grid(images, path, nrow=8):
    """images 取值 [-1,1]，先映射回 [0,1] 再拼成网格保存。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images = (images.clamp(-1, 1) + 1) / 2
    grid = make_grid(images.cpu(), nrow=nrow)
    save_image(grid, path)


def format_time(seconds):
    """秒 -> '1h23m45s' / '12m30s' 易读格式。"""
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"


def save_loss_curve(losses, path):
    """训练结束后保存 loss 曲线图。"""
    import matplotlib
    matplotlib.use("Agg")   # 无显示环境下也能存图
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("training loss")
    plt.grid(True)
    plt.savefig(path, dpi=150)
    plt.close()
