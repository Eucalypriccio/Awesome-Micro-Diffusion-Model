"""训练循环：随机采 t -> 闭式加噪 -> U-Net 预测噪声 -> MSE -> 反向传播更新。"""
import os
import time
from dataclasses import asdict

import torch

from config import Config
from datasets import get_mnist_dataloader
from diffusion import build_diffusion
from models.unet import build_model
from utils import (EmaWeights, count_parameters, format_time, get_device,
                   save_image_grid, save_loss_curve, set_seed)


def train(config: Config, resume_path=None):
    device = get_device()
    print(f"device: {device}")

    # 断点续训：先读存档，模型结构/噪声调度以存档配置为准（必须与权重匹配）；
    # 训练参数（目标轮数/batch_size）与全部路径取当前命令的设置
    resume_ckpt = None
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"找不到 checkpoint: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved_config = Config(**resume_ckpt["config"])
        saved_config.epochs = config.epochs
        saved_config.batch_size = config.batch_size
        saved_config.sample_every = config.sample_every
        saved_config.data_dir = config.data_dir
        saved_config.checkpoint_dir = config.checkpoint_dir
        saved_config.output_dir = config.output_dir
        config = saved_config

    set_seed(config.seed)

    dataloader = get_mnist_dataloader(config, train=True)
    # channels_last：CPU 上 oneDNN 对该格式的卷积更快（本机实测提速约 25%）
    model = build_model(config).to(device, memory_format=torch.channels_last)
    num_params = count_parameters(model)
    print(f"model parameters: {num_params / 1e6:.2f}M (limit 5M)")
    assert num_params < 5_000_000, "参数量超过 5M 限制！"

    diffusion = build_diffusion(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()
    ema = EmaWeights(model, decay=config.ema_decay)   # 采样时使用 EMA 权重

    # 断点续训：恢复模型/EMA/优化器/epoch/loss 历史/RNG，从下一 epoch 继续
    start_epoch = 1
    epoch_losses = []
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model_state"])
        if resume_ckpt.get("ema_state") is not None:
            ema.load_state_dict(resume_ckpt["ema_state"])
        else:   # 旧存档没有 EMA：以加载后的模型权重重新初始化
            ema = EmaWeights(model, decay=config.ema_decay)
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        for state in optimizer.state.values():   # 优化器状态搬到当前设备
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        start_epoch = resume_ckpt["epoch"] + 1
        epoch_losses = resume_ckpt.get("epoch_losses", [])
        if resume_ckpt.get("rng_state") is not None:
            torch.set_rng_state(resume_ckpt["rng_state"])
        print(f"resumed from {resume_path}: finished epoch {resume_ckpt['epoch']} "
              f"(loss {resume_ckpt['loss']:.4f}), continue to epoch {config.epochs}")
        if start_epoch > config.epochs:
            print("目标 epochs 已达到，无需续训（可用 --epochs 增大目标轮数）")
            return resume_path

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    steps_per_epoch = len(dataloader)
    train_start = time.perf_counter()
    checkpoint = None
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        for step, (x0, labels) in enumerate(dataloader):
            x0 = x0.to(device=device, memory_format=torch.channels_last)
            labels = labels.to(device)
            # classifier-free guidance：以一定概率把标签替换为空标签 ∅（索引 num_classes），
            # 让同一个网络同时学会条件与无条件两种预测
            if config.conditional:
                drop_mask = torch.rand(labels.shape[0], device=device) < config.label_drop_prob
                labels[drop_mask] = config.num_classes
            # 每个样本随机采一个时间步 t 和噪声，闭式一步到位加噪（无梯度）
            t = torch.randint(0, config.T, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t, noise)

            pred_noise = model(x_t, t, labels if config.conditional else None)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            ema.update(model)

            loss_sum += loss.item()
            if step % config.log_interval == 0:
                elapsed = time.perf_counter() - train_start
                print(f"[Epoch {epoch:02d}/{config.epochs}] step {step:03d}/{steps_per_epoch} "
                      f"| loss {loss.item():.4f} | lr {config.lr:.1e} | elapsed {format_time(elapsed)}",
                      flush=True)

        epoch_loss = loss_sum / steps_per_epoch
        epoch_losses.append(epoch_loss)
        epoch_time = time.perf_counter() - epoch_start
        eta = epoch_time * (config.epochs - epoch)
        print(f"[Epoch {epoch:02d}/{config.epochs}] done | avg loss {epoch_loss:.4f} "
              f"| epoch time {format_time(epoch_time)} | ETA {format_time(eta)}", flush=True)

        # 每 epoch 保存一次最新 checkpoint（含配置与 EMA 权重，采样时可直接还原现场）
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch_losses": epoch_losses,      # 续训时恢复 loss 曲线
            "rng_state": torch.get_rng_state(),  # 续训时恢复随机序列
            "config": asdict(config),
            "loss": epoch_loss,
        }
        torch.save(checkpoint, os.path.join(config.checkpoint_dir, "unet_latest.pt"))

        # 可选：每 sample_every 个 epoch 采样 16 张监控质量（CPU 较慢，默认关闭）
        if config.sample_every > 0 and epoch % config.sample_every == 0:
            images = diffusion.sample_loop(
                model, (16, config.in_channels, config.image_size, config.image_size),
                device, log_interval=200)
            save_image_grid(images, os.path.join(config.output_dir, f"samples_epoch{epoch:03d}.png"), nrow=4)

    final_path = os.path.join(config.checkpoint_dir, f"{config.out_ckpt}.pt")
    torch.save(checkpoint, final_path)
    total_time = time.perf_counter() - train_start
    print(f"training finished in {format_time(total_time)} | final checkpoint: {final_path}")

    save_loss_curve(epoch_losses, os.path.join(config.output_dir, f"loss_curve_{config.epochs}.png"))
    return final_path
