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


def make_timestep_subsequence(T, steps):
    """在 [0, T-1] 上均匀取 steps 个时间步（含端点），返回降序列表（用于 DDIM 少步采样）。"""
    if steps > T:
        raise ValueError(f"sample_steps({steps}) 不能超过总扩散步数 T({T})")
    seq = torch.linspace(0, T - 1, steps).round().long().tolist()
    return seq[::-1]


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
    def p_sample(self, model, x, t_index, prev_t_index, eta=1.0, labels=None, guidance_w=None):
        """DDIM 反向单步：由 x_t 计算 x_prev（公式见 README "少步采样与 DDIM" 一节）。

        x_prev = sqrt(ᾱ_prev) * x0_pred + sqrt(1-ᾱ_prev-σ²) * ε̂ + σ z
        σ = eta * sqrt((1-ᾱ_prev)/(1-ᾱ_t)) * sqrt(1-ᾱ_t/ᾱ_prev)

        eta=1 时与广义后验（DDPM ancestral sampling）系数代数等价；
        eta=0 时为确定性 DDIM。prev_t_index 可跳跃（子序列少步采样）。
        labels 与 guidance_w 同时给出时，ε̂ 为 classifier-free guidance 合成：
        ε̂ = ε(∅) + w * (ε(y) - ε(∅))（见 README "指定数字生成" 一节）。
        """
        t = torch.full((x.shape[0],), t_index, device=x.device, dtype=torch.long)
        alpha_bar_t = _extract(self.alpha_bar, t, x.ndim)
        if prev_t_index < 0:   # 已到第 0 步，约定 alpha_bar_{-1} = 1
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        else:
            alpha_bar_prev = _extract(self.alpha_bar, torch.full_like(t, prev_t_index), x.ndim)

        if labels is None or guidance_w is None:
            pred_noise = model(x, t, labels)   # UNet 中 labels=None 即空标签（无条件模式）
        else:
            # classifier-free guidance：条件/无条件各前向一次，按 w 合成
            eps_uncond = model(x, t, None)
            eps_cond = model(x, t, labels)
            pred_noise = eps_uncond + guidance_w * (eps_cond - eps_uncond)
        # 由预测噪声反解 x0：x0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        # 仅 eta < 1 时截断到数据范围 [-1,1]：
        # - eta<1（确定性/半确定性）无噪声掩护，高噪声区除以极小的 sqrt(ᾱ_t) 会把预测偏差
        #   放大数百倍导致轨迹发散（实测 eta=0 首步 x0_pred 达 800+），截断是必需的；
        # - eta=1（ancestral）注入的噪声本就能冲掉发敞，截断反而会把边界像素提前钉死在 ±1，
        #   在低噪声区冻结成孤立噪点（同权重同种子对照：孤立亮点 13 vs 1）。
        if eta < 1.0:
            pred_x0 = pred_x0.clamp(-1, 1)

        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) \
            * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
        direction = torch.sqrt((1 - alpha_bar_prev - sigma ** 2).clamp(min=0.0)) * pred_noise
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + direction

        noise = torch.randn_like(x) if prev_t_index >= 0 else torch.zeros_like(x)   # 最后一步不加噪声
        return x_prev + sigma * noise

    @torch.no_grad()
    def sample_loop(self, model, shape, device, sample_steps=None, eta=1.0,
                    labels=None, guidance_w=None, log_interval=100):
        """完整采样：从纯噪声 x_T 逐步去噪到 x_0，输出 clamp 到 [-1,1]。

        sample_steps 默认等于 T（走满全部时间步）；传入更小的步数时在 [0,T-1] 上
        均匀取子序列跳步采样（DDIM，见 README 对应小节）。eta 控制随机性。
        labels（指定数字）与 guidance_w 同时给出时按 classifier-free guidance 采样。
        """
        model.eval()
        timesteps = make_timestep_subsequence(self.T, sample_steps or self.T)
        x = torch.randn(shape, device=device)
        start_time = time.perf_counter()
        for i, t_index in enumerate(timesteps):
            prev_t_index = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.p_sample(model, x, t_index, prev_t_index, eta=eta,
                              labels=labels, guidance_w=guidance_w)
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
