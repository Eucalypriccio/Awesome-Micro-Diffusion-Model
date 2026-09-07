"""训练速度基准测试：定位 CPU 瓶颈（python benchmark.py，不改动训练代码）"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from datasets import get_mnist_dataloader
from diffusion import build_diffusion
from models.unet import build_model


def time_train_step(model, optimizer, loss_fn, x_t, t, noise, iters):
    def step():
        loss = loss_fn(model(x_t, t), noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for _ in range(2):   # warmup
        step()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    return (time.perf_counter() - start) / iters


class InterpOnlyUpsample(nn.Module):
    """纯最近邻插值上采样（去掉平滑卷积），用于评估候选配置。"""

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


def strip_smooth_conv(model):
    """把 U-Net 里的 Upsample 替换为纯插值（仅基准测试用）。"""
    for i, m in enumerate(model.upsamples):
        if isinstance(m, nn.Conv2d) or m.__class__.__name__ == "Upsample":
            model.upsamples[i] = InterpOnlyUpsample()
    return model


def main():
    config = Config()
    device = torch.device("cpu")
    logical = os.cpu_count()
    print(f"CPU 逻辑核心数: {logical} | torch 线程数: {torch.get_num_threads()} "
          f"| torch {torch.__version__} | mkldnn: {torch.backends.mkldnn.enabled}")

    model = build_model(config).to(device)
    diffusion = build_diffusion(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()

    B = config.batch_size
    x0 = torch.rand(B, 1, 28, 28) * 2 - 1
    t = torch.randint(0, config.T, (B,))
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise)

    # ---------- 1. 数据加载耗时占比 ----------
    dataloader = get_mnist_dataloader(config)
    it = iter(dataloader)
    start = time.perf_counter()
    for _ in range(20):
        next(it)
    data_time = (time.perf_counter() - start) / 20

    # ---------- 2. 前向 / 完整 step 耗时 ----------
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(x_t, t)
        start = time.perf_counter()
        for _ in range(5):
            model(x_t, t)
        fwd_time = (time.perf_counter() - start) / 5
    model.train()

    step_time = time_train_step(model, optimizer, loss_fn, x_t, t, noise, iters=5)

    steps_per_epoch = 60000 // B
    print("\n===== 各环节耗时 =====")
    print(f"数据加载: {data_time * 1000:.1f} ms/step ({data_time / step_time * 100:.1f}%)")
    print(f"纯前向:   {fwd_time * 1000:.0f} ms/step")
    print(f"完整 step(前向+反向+优化器): {step_time * 1000:.0f} ms")
    print(f"=> 每 epoch ({steps_per_epoch} steps): {step_time * steps_per_epoch / 60:.1f} min | "
          f"20 epochs 预计: {step_time * steps_per_epoch * 20 / 3600:.1f} h")

    # ---------- 3. 各模块算力分布（MACs，batch=1 前向） ----------
    macs = {}

    def make_hook(name):
        def hook(module, inp, out):
            # MACs = B*H*W*Cout * (Cin/groups) * k^2 = out.numel() * (Cin/groups) * k^2
            m = out.numel() * (module.in_channels // module.groups) \
                * module.kernel_size[0] * module.kernel_size[1]
            parts = name.split(".")
            group = ".".join(parts[:2]) if len(parts) > 1 and parts[1].isdigit() else parts[0]
            macs[group] = macs.get(group, 0) + m
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        model(x_t[:1], t[:1])
    for h in hooks:
        h.remove()

    total_macs = sum(macs.values())
    res_of = {"encoder_blocks.0": "28x28", "decoder_blocks.2": "28x28", "upsamples.1": "28x28",
              "encoder_blocks.1": "14x14", "decoder_blocks.1": "14x14", "upsamples.0": "14x14",
              "encoder_blocks.2": "7x7", "decoder_blocks.0": "7x7", "mid_block1": "7x7",
              "mid_block2": "7x7", "mid_attn": "7x7"}
    print(f"\n===== 各模块 MACs（batch=1 前向，总计 {total_macs / 1e6:.0f}M = {2 * total_macs / 1e9:.2f} GFLOPs） =====")
    print(f"（训练一步 ≈ 3x 前向 x batch {B} ≈ {3 * 2 * total_macs * B / 1e9:.0f} GFLOPs）")
    for name, m in sorted(macs.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {res_of.get(name, ''):6s} {m / 1e6:7.1f}M  ({m / total_macs * 100:4.1f}%)")

    # ---------- 4. channels_last 对比 ----------
    print("\n===== channels_last 内存格式对比 =====")
    try:
        model_cl = build_model(config).to(memory_format=torch.channels_last)
        opt_cl = torch.optim.Adam(model_cl.parameters(), lr=config.lr)
        x_cl = x_t.to(memory_format=torch.channels_last)
        cl_time = time_train_step(model_cl, opt_cl, loss_fn, x_cl, t, noise, iters=3)
        print(f"NCHW: {step_time * 1000:.0f} ms | channels_last: {cl_time * 1000:.0f} ms "
              f"({cl_time / step_time * 100:.0f}%)")
        del model_cl, opt_cl, x_cl
    except Exception as e:
        print(f"channels_last 测试失败: {e}")

    # ---------- 5. torch 线程数对比 ----------
    print("\n===== torch 线程数对比 =====")
    default_threads = torch.get_num_threads()
    for k in dict.fromkeys([default_threads, logical]):
        torch.set_num_threads(k)
        tm = time_train_step(model, optimizer, loss_fn, x_t, t, noise, iters=3)
        print(f"threads={k:2d}: {tm * 1000:.0f} ms/step")
    torch.set_num_threads(default_threads)

    # ---------- 6. 候选瘦身配置实测 ----------
    print("\n===== 候选配置实测（NCHW，未含 channels_last 收益） =====")
    candidates = [
        ("C0 当前: base32, 2块/层, 平滑卷积", dict(base_channels=32, num_res_blocks=2), False),
        ("C1: base24, 2块/层, 去平滑卷积", dict(base_channels=24, num_res_blocks=2), True),
        ("C2: base24, 1块/层, 去平滑卷积", dict(base_channels=24, num_res_blocks=1), True),
    ]
    for label, overrides, strip in candidates:
        cfg = Config(**{**config.__dict__, **overrides})
        m = build_model(cfg).to(device)
        if strip:
            m = strip_smooth_conv(m)
        opt = torch.optim.Adam(m.parameters(), lr=config.lr)
        n_params = sum(p.numel() for p in m.parameters())
        tm = time_train_step(m, opt, loss_fn, x_t, t, noise, iters=3)
        print(f"{label}: {n_params / 1e6:.2f}M params | {tm * 1000:.0f} ms/step | "
              f"epoch {tm * steps_per_epoch / 60:.1f} min | 20ep {tm * steps_per_epoch * 20 / 3600:.1f} h | "
              f"15ep {tm * steps_per_epoch * 15 / 3600:.1f} h")
        del m, opt


if __name__ == "__main__":
    main()
