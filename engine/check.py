"""自检（python main.py check）：
1. 参数量 < 5M
2. 噪声调度合法性（beta 范围、alpha_bar 单调递减、线性调度末端约等于 0）
3. 加噪 / 网络前向 / 单步去噪的形状断言
4. 单 batch 过拟合测试（验证梯度通路）
"""
import torch

from config import Config
from datasets import get_mnist_dataloader
from diffusion import build_diffusion, get_beta_schedule
from models.unet import build_model
from utils import EmaWeights, count_parameters, get_device, set_seed


def run_check(config: Config):
    device = get_device()
    print(f"[check] device: {device}")

    # ---------- 1. 参数量 ----------
    print("\n[check 1/4] 模型参数量 ...")
    model = build_model(config).to(device)
    num_params = count_parameters(model)
    print(f"  parameters: {num_params / 1e6:.2f}M (limit 5M)")
    assert num_params < 5_000_000, "参数量超过 5M！"

    # ---------- 2. 调度合法性 ----------
    print("\n[check 2/4] 噪声调度合法性 ...")
    for name in ["linear", "cosine"]:
        betas = get_beta_schedule(name, config.T, config)
        alpha_bar = torch.cumprod(1 - betas, dim=0)
        assert (betas > 0).all() and (betas < 1).all(), f"{name}: beta 超出 (0,1)"
        assert (alpha_bar[1:] <= alpha_bar[:-1] + 1e-12).all(), f"{name}: alpha_bar 非单调递减"
        print(f"  {name}: beta in [{betas.min():.6f}, {betas.max():.6f}], "
              f"alpha_bar_T = {alpha_bar[-1]:.6f}")
        if name == "linear":
            assert alpha_bar[-1] < 0.01, "线性调度末端 alpha_bar 应接近 0"

    # ---------- 3. 形状断言 ----------
    print("\n[check 3/4] 加噪 / 网络前向 / 单步去噪形状 ...")
    diffusion = build_diffusion(config).to(device)
    x0 = torch.randn(4, config.in_channels, config.image_size, config.image_size, device=device)
    t = torch.randint(0, config.T, (4,), device=device)
    x_t = diffusion.q_sample(x0, t, torch.randn_like(x0))
    assert x_t.shape == x0.shape, "x_t 形状与 x0 不一致"
    pred = model(x_t, t)
    assert pred.shape == x0.shape, "预测噪声形状与 x0 不一致"
    x_prev = diffusion.p_sample(model, x_t, t_index=5, prev_t_index=4)
    assert x_prev.shape == x0.shape, "单步去噪输出形状不一致"
    print(f"  x0 {tuple(x0.shape)} -> x_t {tuple(x_t.shape)} -> pred {tuple(pred.shape)} "
          f"-> x_prev {tuple(x_prev.shape)}  [OK]")

    # EMA 冒烟测试：update 后 shadow 键集合与模型一致且数值被修改
    ema = EmaWeights(model, decay=0.99)
    before = next(iter(ema.state_dict().values())).clone()
    ema.update(model)
    after = next(iter(ema.state_dict().values()))
    assert set(ema.state_dict().keys()) == set(model.state_dict().keys())
    print("  EMA update [OK]")

    # ---------- 4. 单 batch 过拟合 ----------
    print("\n[check 4/4] 单 batch 过拟合测试 (200 steps) ...")
    set_seed(config.seed)
    dataloader = get_mnist_dataloader(config, train=True)
    x0_batch, _ = next(iter(dataloader))
    x0_batch = x0_batch[:32].to(device)   # 取小 batch，加快自检
    overfit_model = build_model(config).to(device)
    optimizer = torch.optim.Adam(overfit_model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss()
    overfit_model.train()
    losses = []
    for step in range(200):
        t = torch.randint(0, config.T, (x0_batch.shape[0],), device=device)
        noise = torch.randn_like(x0_batch)
        x_t = diffusion.q_sample(x0_batch, t, noise)
        loss = loss_fn(overfit_model(x_t, t), noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"  step {step + 1:3d}/200 | loss {loss.item():.4f}", flush=True)
    head_avg = sum(losses[:20]) / 20
    tail_avg = sum(losses[-20:]) / 20
    print(f"  first-20 avg {head_avg:.4f} -> last-20 avg {tail_avg:.4f}")
    assert tail_avg < head_avg * 0.7, "loss 下降不明显，梯度通路可能有问题"

    print("\n[check] 全部自检通过 [OK]")
