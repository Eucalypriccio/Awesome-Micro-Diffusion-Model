"""噪声调度：线性 / 余弦，输出 beta 序列（float64 保证累乘精度）。"""
import torch


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """线性调度：beta 从 beta_start 到 beta_end 线性插值。"""
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def cosine_beta_schedule(T, s=0.008):
    """余弦调度（Nichol & Dhariwal）：
    alpha_bar_t = f(t)/f(0)，f(t) = cos^2(((t/T + s)/(1+s)) * pi/2)，beta_t = 1 - alpha_bar_t/alpha_bar_{t-1}
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * (torch.pi / 2)) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.999)   # 截断防止末端数值奇异


def get_beta_schedule(name, T, config):
    if name == "linear":
        return linear_beta_schedule(T, config.beta_start, config.beta_end)
    if name == "cosine":
        return cosine_beta_schedule(T, config.cosine_s)
    raise ValueError(f"unknown schedule: {name}")
