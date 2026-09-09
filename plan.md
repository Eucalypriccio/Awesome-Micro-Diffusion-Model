# 实施计划 plan.md — Awesome-Micro-Diffusion-Model

> 配套文档：README.md（实验要求与原理推导，已校对修正）
> 目标：在 MNIST 上从零训练一个参数量 ≤5M 的 DDPM 迷你扩散模型，CPU 训练 ≤2h，最终从纯噪声采样输出 8×8 样本网格。

---

## 1. 模型架构设计（U-Net，去噪网络 $\epsilon_\theta(x_t, t)$）

### 1.1 总体结构

输入为 `[B, 1, 28, 28]`（MNIST，归一化到 [-1, 1]），输出同形状预测噪声。三个分辨率层级：28×28 → 14×14 → 7×7，编码器 / 瓶颈 / 解码器镜像对称，跳跃连接按 channel 维拼接（concat）。

> 以下为 CPU 实测调优后的配置（初版 base32/2块每层/带平滑卷积约 6h 超预算 → 精简版 base24/1块每层 40min 但 64 张样本约 10 张难辨认 → 当前版恢复 2 块每层并加 EMA，分析见 §3.4）。

| 位置 | 分辨率 | 通道数 | 组成 |
|---|---|---|---|
| 输入卷积 | 28×28 | 24 | 3×3 conv，1→24 |
| 编码器 L0 | 28×28 | 24 | ResBlock(24→24) × 2，MaxPool 2×2 |
| 编码器 L1 | 14×14 | 48 | ResBlock(24→48)，ResBlock(48→48)，MaxPool 2×2 |
| 编码器 L2 | 7×7 | 96 | ResBlock(48→96)，ResBlock(96→96) |
| 瓶颈 | 7×7 | 96 | ResBlock(96) → SelfAttention(96, 4头) → ResBlock(96) |
| 解码器 L2 | 7×7 | 96 | concat 编码器 L2 输出(96) → ResBlock(192→96)，ResBlock(96→96) |
| 解码器 L1 | 14×14 | 48 | 最近邻上采样 ×2，concat L1 输出(48) → ResBlock(144→48)，ResBlock(48→48) |
| 解码器 L0 | 28×28 | 24 | 最近邻上采样 ×2，concat L0 输出(24) → ResBlock(72→24)，ResBlock(24→24) |
| 输出层 | 28×28 | 1 | 1×1 conv，24→1 |

设计要点：

- **通道配置**：base_channels = 24，层级倍增 (1, 2, 4)；每个层级 2 个残差块。参数量约 **1.85M**（满足 ≤5M）。
- **GroupNorm**：num_groups = 8（能整除 24/48/96）。
- **注意力**：只在 7×7 瓶颈处（49 个 token，计算量极小），4 个头，head_dim = 24。
- **下采样**：MaxPool 2×2（stride 2），与 README 原理一致；28→14→7 均为整数。
- **上采样**：纯最近邻插值 ×2，再 concat 同层编码器输出（初版的 3×3 平滑卷积约占 19% 算力，已去除；`config.upsample_smooth=True` 可恢复）。
- **channels_last**：训练与采样均使用该内存格式（CPU oneDNN 实测提速约 25%）。

### 1.2 残差块 ResBlock（全网络统一复用）

```
def res_block(x, time_emb):
    h = conv1(x)                      # 3x3, cin -> cout
    scale, shift = mlp(time_emb)      # 该块专属 Linear: time_dim -> 2*cout，按 channel 切半
    h = group_norm1(h) * (1 + scale) + shift   # 时间嵌入以 gamma/beta 方式注入
    h = silu(h)
    h = conv2(h)                      # 3x3, cout -> cout
    h = silu(group_norm2(h))
    return h + shortcut(x)            # cin != cout 时 shortcut 为 1x1 conv，否则恒等
```

- 与 README 对应：`Conv1 → GroupNorm+SiLU → Conv2 → GroupNorm+SiLU` + 跳跃连接 = 一个残差块。
- 时间嵌入注入方式与 README"时间步嵌入"一节一致：每个残差块拥有专属 MLP，把统一的 time_dim 维向量投影出本块通道数的 γ（scale）和 β（shift）。

### 1.3 时间步嵌入

- 正弦嵌入：维度 time_dim = 128，$f_i = 10000^{-2i/d}$，偶数位 sin、奇数位 cos（README 公式）。
- 两层 MLP：`Linear(128→128) → SiLU → Linear(128→128)`，得到全网络共享的 time_emb，再经各残差块专属 MLP 产生各自的 γ、β。

### 1.4 瓶颈自注意力

- 输入 `[B, 128, 7, 7]` → 视为 49 个 token、embedding 维度 128。
- 先做一次 GroupNorm，再算 Q/K/V（各 128→128），4 头切分（每头 32 维），$O=\mathrm{softmax}(QK^\top/\sqrt{32})V$，拼接后过输出投影 $W_O$，最后与输入做残差相加。

---

## 2. 扩散过程设计（schedule + 前向/反向）

预计算并缓存（register_buffer 或 numpy 数组，随模型/模块存取）：

- $\beta_t$：T = 1000
  - **线性调度**：$\beta_1=10^{-4},\ \beta_T=0.02$ 线性插值
  - **余弦调度**：$\bar\alpha_t = f(t)/f(0)$，$f(t)=\cos^2(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2})$，$s=0.008$；$\beta_t = 1-\bar\alpha_t/\bar\alpha_{t-1}$，截断 ≤ 0.999
- 派生量：$\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod\alpha_i$、$\sqrt{\bar\alpha_t}$、$\sqrt{1-\bar\alpha_t}$、后验方差 $\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$、均值系数 $\frac{1}{\sqrt{\alpha_t}}$ 与 $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}$

前向加噪（训练用，闭式一步到位，无梯度）：

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon\sim\mathcal N(0,I)$$

反向单步（ancestral sampling）：

$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\Big)+z\sqrt{\tilde\beta_t},\quad t=T,\dots,1$$

$t=1$ 时 $z=0$（不加噪声）。采样从 $x_T\sim\mathcal N(0,I)$ 开始，循环 T 步，输出 clamp 到 [-1,1] 后映射回 [0,1] 存图。

---

## 3. 训练设计

### 3.1 数据

- `torchvision.datasets.MNIST(root="./data", train=True, download=True)`，60,000 张训练图。
- 变换：`ToTensor()`（→[0,1]）后 `x = x * 2 - 1`（→[-1,1]）。
- DataLoader：batch_size = 128，shuffle = True，num_workers = 0（Windows/CPU 稳妥），drop_last = True。每 epoch 约 469 步。

### 3.2 超参数

| 项 | 值 | 说明 |
|---|---|---|
| T（扩散步数） | 1000 | 训练时 $t$ 在 1..1000 均匀随机采样 |
| 调度 | linear（默认）/ cosine | 命令行 `--schedule` 切换，两种都要能跑 |
| 优化器 | Adam，lr = 1e-3 | 小模型 + MNIST，收敛快；不加权重衰减 |
| 损失 | MSE($\epsilon_\theta$, $\epsilon$) | 预测噪声 |
| batch_size | 128 | CPU 可承受 |
| epochs | 20 | 见 3.4 时间预算（已实测校准） |
| EMA | decay = 0.999 | 权重指数滑动平均，采样用 EMA 权重，质量更稳（DDPM 论文做法，几乎零成本） |
| 梯度裁剪 | clip_grad_norm = 1.0 | 一行代码，防偶发梯度爆炸 |
| 随机种子 | 42 | torch / numpy / random 全部固定 |

### 3.3 训练循环（每个 step）

1. 取 batch $x_0$；2. 随机采 $t$、采噪声 $\epsilon$；3. 闭式加噪得 $x_t$（无梯度）；4. 正弦嵌入 + MLP 得 time_emb；5. U-Net 前向得 $\epsilon_\theta$；6. MSE loss；7. `zero_grad()` → `backward()` → 梯度裁剪 → `step()`；8. 更新 EMA 权重。

日志规范（print，不依赖第三方进度条库）：

- 每 100 step 一行：`[Epoch 03/20] step 0300/0469 | loss 0.0231 (epoch avg 0.0267) | lr 1.0e-03 | elapsed 45s`
- 每 epoch 结束一行汇总：平均 loss、本 epoch 耗时、预计剩余时间（ETA）。
- checkpoint：每 epoch 结束保存 `checkpoints/unet_latest.pt`（含 model state_dict、**EMA state_dict**、优化器状态、epoch、loss 历史、RNG 状态、超参数配置），训练结束另存 `unet_final.pt`。采样时优先加载 EMA 权重。
- 断点续训：`python main.py train --resume [ckpt路径]`（默认 `unet_latest.pt`）——恢复模型/EMA/优化器/epoch/loss 历史/RNG，从下一 epoch 继续；模型结构与调度以存档配置为准（必须与权重匹配），训练参数（`--epochs` 可延长目标轮数）与全部输出路径取当前命令的设置。
- 可选 `--sample_every 5`：每 5 个 epoch 用当前模型采样 16 张存 `outputs/`，监控生成质量演进（CPU 上 1000 步采样较慢，默认关闭）。

### 3.4 CPU 时间预算（≤2h 约束，已实测校准）

实测环境：16 逻辑核（torch 用 8 线程最优，16 线程反而更慢）、torch 2.14+cpu、mkldnn 开启。

**初版配置（base32 / 2块每层 / 平滑卷积）实测超预算**：

- 单步算力 ≈ 230 GFLOPs（前向 0.60 GFLOPs/图 × 3 × batch 128），CPU 实际吞吐约 100 GFLOPS → 2.3 s/步 → 每 epoch 约 18 min → 20 epoch ≈ **6 h**。
- 瓶颈是纯卷积算力（前向+反向占 66% 时间），数据加载仅 1.4%，线程/库配置无异常。
- MACs 分布（300M/图）：解码器 43%（拼接后 2C→C 卷积最贵）、两个上采样平滑卷积 19%、编码器 27%、瓶颈 11%（注意力仅 1.1%）。28×28 处一次 3×3 卷积成本是 7×7 处的 16 倍，高分辨率层是重灾区。
- 教训：≤5M 是**参数量**约束，不约束**算力**；本模型参数不多但高分辨率卷积多，估算时间必须看 FLOPs。

**优化手段（按收益排序，均为本机实测）**：

| 手段 | 收益 |
|---|---|
| channels_last 内存格式 | −25% 时间 |
| base 32→24（通道 24/48/96） | −44% MACs |
| 2 块 → 1 块每层 | −29% MACs |
| 去掉上采样平滑卷积（纯插值） | −19% MACs |

**精简配置（base24 / 1块每层 / 纯插值 / channels_last，1.23M 参数）实际训练**：约 340 ms/步 → 15 epoch ≈ **40 min**。但 64 张样本中约 10 张难辨认——容量与训练量均不足。

**当前配置（base24 / 2块每层 / 纯插值 / channels_last / EMA，约 1.85M 参数）估算**：约 500 ms/步 → 每 epoch ≈ 4 min → **20 epoch ≈ 1.3 h**，满足 ≤2h。质量手段按收益排序：EMA（几乎零成本，采样稳定性提升最明显）> 恢复 2 块每层（+45% 时间，各分辨率容量翻倍）> epochs 15→20。base32 与平滑卷积不恢复（分别 +76% / +19% 算力，性价比低）。

若实测超时：用 `--epochs 15` 降到约 1 h，其余不动。复现基准：`python benchmark.py`。

### 3.5 最终采样输出（实验要求）

训练结束后：从 $\mathcal N(0,I)$ 采 64 张纯噪声 → ancestral sampling 完整 T 步 → clamp(-1,1) → (x+1)/2 → `torchvision.utils.make_grid` 拼成 **8×8 网格**保存为 `outputs/samples_step_{步数}_eta_{η}.png`，同时打印采样总耗时与平均每步耗时。扩展一已实现：`--sample-steps` 可传多个步数逐一出图，并打印耗时汇总表（相同初始噪声，公平对比）。

---

## 4. 项目架构（代码文件清单）

```
Awesome-Micro-Diffusion-Model/
├── README.md                  # 实验要求 + 原理（已完成校对）
├── plan.md                    # 本文件
├── requirements.txt           # torch, torchvision, numpy, matplotlib
├── main.py                    # CLI 入口：python main.py train / sample / check
├── benchmark.py               # 性能基准工具（python benchmark.py）
├── config.py                  # 全部超参数集中定义（dataclass），argparse 可覆盖
├── datasets/                  # 数据集准备
│   ├── __init__.py            # 对外导出 get_mnist_dataloader
│   └── mnist.py               # MNIST 下载、[-1,1] 归一化、DataLoader 构建
├── models/                    # 模型架构
│   ├── __init__.py
│   ├── time_embedding.py      # 正弦时间嵌入 + 两层 MLP
│   ├── blocks.py              # ResBlock、Downsample(MaxPool)、Upsample(插值)、SelfAttention
│   └── unet.py                # UNet 组装（编码器/瓶颈/解码器/跳跃连接/输出层）
├── diffusion/                 # 扩散过程
│   ├── __init__.py            # 对外再导出调度函数与 GaussianDiffusion
│   ├── schedules.py           # linear / cosine β 调度
│   └── gaussian.py            # 系数预计算、q_sample、p_sample、sample_loop
├── engine/                    # 执行流程
│   ├── __init__.py
│   ├── train.py               # 训练循环、日志、梯度裁剪、EMA、checkpoint
│   ├── sample.py              # 加载 checkpoint、采样 64 张、拼 8×8 网格存 PNG
│   └── check.py               # 自检（参数量/调度/形状/EMA/单batch过拟合）
├── utils/
│   └── __init__.py            # 设备检测、固定种子、参数量统计、EMA 权重、存图、计时、loss 曲线
├── data/                      # MNIST（自动下载，注意与 datasets/ 源码包区分）
├── checkpoints/               # 模型权重
└── outputs/                   # 生成样本、loss 曲线
```

职责划分对应实验要求："数据准备 / 模型架构 / 训练 / 测试（采样生成）/ 辅助工具" 各自独立成文件；ResBlock、调度系数等可复用单元只实现一次。

`main.py` 三个子命令：

- `python main.py train --schedule linear --epochs 20` — 训练
- `python main.py sample --ckpt checkpoints/unet_final.pt` — 生成 8×8 网格
- `python main.py check` — 快速自检（见 §5.2）

环境管理：`python -m venv .venv` → 激活 → `pip install -r requirements.txt`。只用原生 PyTorch / numpy / torchvision / matplotlib，不引入 diffusers 等高级库。

---

## 5. 验证标准与实施顺序

### 5.1 实施顺序

1. 建 `.venv` + `requirements.txt`，跑通 `dataset.py`（打印 batch 形状、存一张原图网格）
2. `schedules.py` + `diffusion.py` 前向加噪（可视化同一张图 t=0/250/500/750/999 的加噪过程）
3. `models/`（time_embedding → blocks → unet），`main.py check` 验证参数量 < 5M、前后向形状正确
4. `train.py` 完整训练（linear 调度），观察 loss 正常下降
5. `sample.py` 生成最终 8×8 网格
6. 改用 `--schedule cosine` 复训，对比两种调度的样本质量
7. （后续扩展）少步采样对比、指定数字条件生成

### 5.2 自检项（`main.py check`）

- 参数量打印且 < 5M
- 调度系数合法性：$\beta_t\in(0,1)$、$\bar\alpha_t$ 单调递减、线性调度 $\bar\alpha_T\approx0$
- 形状断言：任意 $t$ 下 $x_t$、$\epsilon_\theta$ 与 $x_0$ 同形状 `[B,1,28,28]`
- 单 batch 过拟合测试：固定一个 batch 训练 200 step，loss 明显下降（验证梯度通路正确）

### 5.3 交付标准

- 训练日志符合 §3.3 规范，CPU 总时长 ≤ 2h
- `outputs/samples_final.png` 为 8×8 网格，数字形态可辨认
- linear / cosine 两种调度均可通过命令行复现训练与采样

---

## 6. 扩展预留（本期不实现，设计时留好接口）

1. **不同采样步数对比（已实现）**：采用 DDIM 统一形式（η 参数化；可证明 η=1 与最初的广义后验实现系数代数等价，η=0 为确定性采样）。`python main.py sample --sample-steps 1000 200 50 20 --eta 0` 逐一出图并打印耗时汇总；理论已写入 README "少步采样与 DDIM" 一节。
2. **指定数字生成（条件扩散）**：`nn.Embedding(10, time_dim)` 把数字标签嵌入与 time_emb 相加后注入各残差块；训练时标签取自 DataLoader，采样时终端输入 0-9 指定生成类别。模型 forward 预留可选 `labels` 参数位。
