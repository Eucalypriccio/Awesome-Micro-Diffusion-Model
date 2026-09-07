"""采样生成：加载 checkpoint，从纯噪声 ancestral sampling，保存 8x8 样本网格。"""
import os

import torch

from config import Config
from diffusion import build_diffusion
from models.unet import build_model
from utils import get_device, save_image_grid, set_seed


def sample(config: Config, ckpt_path, num_samples=None, out_path=None):
    set_seed(config.seed)
    device = get_device()
    print(f"device: {device}")

    # 模型结构与调度以 checkpoint 中保存的配置为准，保证与训练现场一致
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_config = Config(**checkpoint["config"])
    # channels_last：CPU 上卷积更快（1000 步采样全是前向，收益明显）
    model = build_model(saved_config).to(device, memory_format=torch.channels_last)
    # 优先使用 EMA 权重（生成质量更稳），旧 checkpoint 没有则退回原始权重
    if checkpoint.get("ema_state") is not None:
        model.load_state_dict(checkpoint["ema_state"])
        print("using EMA weights")
    else:
        model.load_state_dict(checkpoint["model_state"])
        print("using raw weights (no EMA in checkpoint)")
    diffusion = build_diffusion(saved_config).to(device)

    n = num_samples or config.num_samples
    shape = (n, saved_config.in_channels, saved_config.image_size, saved_config.image_size)
    print(f"sampling {n} images from pure noise ({saved_config.T} steps, {saved_config.schedule} schedule) ...")
    images = diffusion.sample_loop(model, shape, device)

    out = out_path or os.path.join(config.output_dir, "samples_final.png")
    save_image_grid(images, out, nrow=min(8, n))
    print(f"saved {n} samples as grid -> {out}")
    return out
