"""扩散过程：闭式前向加噪 q_sample、反向单步 p_sample、完整采样循环 sample_loop。

约定：时间步内部采用 0-indexed，t = 0 对应 README 公式中的第 1 步（噪声最少），
t = T-1 对应第 T 步（最接近纯噪声）。
"""
import time

import torch
import torch.nn as nn

from .schedules import get_beta_schedule


def _extract(coefficients, t, ndim):
    """从 [T] 的系数向量中按时间步 t 取 [B] 个值，reshape 成可广播的 [B,1,1,...]"""
    out = coefficients.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (ndim - 1)))


class GaussianDiffusion(nn.Module):
    """DDPM 高斯扩散。系数在 __init__ 中预计算为 buffer，随 .to(device) 一起迁移。"""

    def __init__(self, betas):
        super().__init__()
        betas = betas.to(torch.float64)      # float64 累乘，避免 1000 步精度损失
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.T = betas.shape[0]
        self.register_buffer("betas", betas.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar).float())
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar).float())

    def q_sample(self, x0, t, noise):
        """闭式一步到位加噪（训练用，纯数值运算不产生梯度）：
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        sqrt_ab = _extract(self.sqrt_alpha_bar, t, x0.ndim)
        sqrt_1_ab = _extract(self.sqrt_one_minus_alpha_bar, t, x0.ndim)
        return sqrt_ab * x0 + sqrt_1_ab * noise

    @torch.no_grad()
    def p_sample(self, model, x, t_index, prev_t_index):
        """反向单步：由 x_t 采样 x_prev。

        使用广义后验 q(x_prev | x_t, x0)：prev_t_index = t_index - 1 时
        严格退化为 README 推导的 ancestral sampling（均值/方差公式一致）；
        传入跳跃的 prev_t_index 即可支持少步采样（扩展内容预留）。
        """
        t = torch.full((x.shape[0],), t_index, device=x.device, dtype=torch.long)
        alpha_bar_t = _extract(self.alpha_bar, t, x.ndim)
        if prev_t_index < 0:   # 已到第 0 步，约定 alpha_bar_{-1} = 1
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        else:
            alpha_bar_prev = _extract(self.alpha_bar, torch.full_like(t, prev_t_index), x.ndim)

        pred_noise = model(x, t)
        # 由预测噪声反解 x0：x0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        # 后验均值系数与方差（与 README boxed 公式代数等价）
        beta_gen = 1 - alpha_bar_t / alpha_bar_prev
        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_gen / (1 - alpha_bar_t)
        coef_x = torch.sqrt(alpha_bar_t / alpha_bar_prev) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        mean = coef_x0 * pred_x0 + coef_x * x
        variance = ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_gen).clamp(min=0.0)

        noise = torch.randn_like(x) if prev_t_index >= 0 else torch.zeros_like(x)   # 最后一步不加噪声
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample_loop(self, model, shape, device, timesteps=None, log_interval=100):
        """完整采样：从纯噪声 x_T 逐步去噪到 x_0，输出 clamp 到 [-1,1]。

        timesteps 默认是完整的 T-1 ... 0；传入降序子序列即可少步采样（扩展预留）。
        """
        model.eval()
        if timesteps is None:
            timesteps = list(range(self.T - 1, -1, -1))
        x = torch.randn(shape, device=device)
        start_time = time.perf_counter()
        for i, t_index in enumerate(timesteps):
            prev_t_index = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.p_sample(model, x, t_index, prev_t_index)
            if log_interval and (i + 1) % log_interval == 0:
                elapsed = time.perf_counter() - start_time
                print(f"  sampling {i + 1}/{len(timesteps)} | elapsed {elapsed:.1f}s", flush=True)
        total_time = time.perf_counter() - start_time
        print(f"  sampling done: {len(timesteps)} steps in {total_time:.1f}s "
              f"({1000 * total_time / len(timesteps):.1f} ms/step)", flush=True)
        return x.clamp(-1, 1)


def build_diffusion(config):
    """按配置构建 GaussianDiffusion（train / sample / check 复用）。"""
    betas = get_beta_schedule(config.schedule, config.T, config)
    return GaussianDiffusion(betas)
