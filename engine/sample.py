"""采样生成：加载 checkpoint，从纯噪声采样（DDPM/DDIM），保存 8x8 样本网格。

支持传入多个采样步数做"质量-耗时"对比：每个步数使用相同的初始噪声（固定种子），
生成一张网格图，最后打印耗时汇总表。
"""
import os
import time

import torch

from config import Config
from diffusion import build_diffusion
from models.unet import build_model
from utils import get_device, save_image_grid, set_seed


def sample(config: Config, ckpt_path, num_samples=None, out_path=None,
           sample_steps_list=None, eta=1.0):
    device = get_device()
    print(f"device: {device}")

    # 模型结构与调度以 checkpoint 中保存的配置为准，保证与训练现场一致
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_config = Config(**checkpoint["config"])
    # channels_last：CPU 上卷积更快（采样全是前向，收益明显）
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
    steps_list = sample_steps_list or [saved_config.T]   # 默认走满训练时的全部步数

    results = []
    for steps in steps_list:
        # 每个步数重置一次种子：相同的初始噪声 x_T，质量差异只来自步数/eta
        set_seed(config.seed)
        print(f"sampling {n} images | steps={steps} eta={eta} ({saved_config.schedule} schedule) ...")
        start_time = time.perf_counter()
        images = diffusion.sample_loop(model, shape, device, sample_steps=steps, eta=eta)
        elapsed = time.perf_counter() - start_time

        out = out_path or os.path.join(config.output_dir, f"samples_step_{steps}_eta_{int(eta)}.png")
        save_image_grid(images, out, nrow=min(8, n))
        print(f"saved {n} samples as grid -> {out} ({elapsed:.1f}s)\n")
        results.append((steps, elapsed))

    if len(results) > 1:   # 多个步数时打印耗时汇总，配合各网格图做质量-耗时对比
        print("sampling steps summary:")
        for steps, elapsed in results:
            print(f"  steps={steps:5d} | total {elapsed:6.1f}s | {1000 * elapsed / steps:.1f} ms/step")
    return out
