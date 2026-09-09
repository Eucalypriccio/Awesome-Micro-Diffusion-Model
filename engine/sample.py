"""采样生成：加载 checkpoint，从纯噪声采样（DDPM/DDIM），保存样本网格。

支持传入多个采样步数做"质量-耗时"对比：每个步数使用相同的初始噪声（固定种子），
生成一张网格图，最后打印耗时汇总表。
支持指定数字采样（扩展二）：--digit 0-9 生成对应数字，--digit all 生成 0-9 各 8 张
（每行一个数字），用 classifier-free guidance（--guidance-w）调节条件强度。
"""
import os
import time

import torch

from config import Config
from diffusion import build_diffusion
from models.unet import build_model
from utils import get_device, save_image_grid, set_seed


def sample(config: Config, ckpt_path, num_samples=None, out_path=None,
           sample_steps_list=None, eta=1.0, digit=None, guidance_w=None):
    device = get_device()
    print(f"device: {device}")

    # 模型结构与调度以 checkpoint 中保存的配置为准，保证与训练现场一致
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_config = Config(**checkpoint["config"])
    # 旧存档（无条件模型）没有 label_embed 权重：据 state_dict 自动判定并兼容
    state_for_check = checkpoint.get("ema_state") or checkpoint["model_state"]
    saved_config.conditional = any("label_embed" in k for k in state_for_check.keys())

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

    # 指定数字 -> labels；未指定则为无条件采样
    n = num_samples or config.num_samples
    labels = None
    w = None
    if digit is not None:
        if not saved_config.conditional:
            print("该 checkpoint 是无条件模型（无 label_embed），不支持 --digit；"
                  "请用条件配置重新训练后再试")
            return None
        w = guidance_w if guidance_w is not None else saved_config.guidance_w
        if digit == "all":   # 0-9 各 8 张，按行排列：每行一个数字
            n = 80
            labels = torch.arange(saved_config.num_classes, device=device).repeat_interleave(8)
        else:
            labels = torch.full((n,), int(digit), device=device, dtype=torch.long)

    shape = (n, saved_config.in_channels, saved_config.image_size, saved_config.image_size)
    steps_list = sample_steps_list or [saved_config.T]   # 默认走满训练时的全部步数

    results = []
    for steps in steps_list:
        # 每个步数重置一次种子：相同的初始噪声 x_T，质量差异只来自步数/eta/条件
        set_seed(config.seed)
        desc = f"digit={digit} w={w}" if labels is not None else "unconditional"
        print(f"sampling {n} images | {desc} | steps={steps} eta={eta} "
              f"({saved_config.schedule} schedule) ...")
        start_time = time.perf_counter()
        images = diffusion.sample_loop(model, shape, device, sample_steps=steps, eta=eta,
                                       labels=labels, guidance_w=w)
        elapsed = time.perf_counter() - start_time

        if out_path:
            out = out_path
        elif labels is not None:
            out = os.path.join(config.output_dir, f"samples_digit{digit}_step{steps}_w{w}.png")
        else:
            out = os.path.join(config.output_dir, f"samples_step_{steps}_eta_{int(eta)}.png")
        save_image_grid(images, out, nrow=min(8, n))
        print(f"saved {n} samples as grid -> {out} ({elapsed:.1f}s)\n")
        results.append((steps, elapsed))

    if len(results) > 1:   # 多个步数时打印耗时汇总，配合各网格图做质量-耗时对比
        print("sampling steps summary:")
        for steps, elapsed in results:
            print(f"  steps={steps:5d} | total {elapsed:6.1f}s | {1000 * elapsed / steps:.1f} ms/step")
    return out
