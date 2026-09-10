# 从零训练迷你扩散模型 Awesome-Micro-Diffusion-Model

## Quick Start

在仓库根目录执行：

```
python install -e .
.venv\Scripts\activate

python main.py check                                   # 自检（参数量/调度/形状/过拟合）

python main.py train --schedule linear --epochs 20     # 训练（线性调度）
python main.py train --schedule cosine --epochs 20     # 训练（余弦调度）
python main.py train --resume --epochs 20              # 断点续训（默认接 checkpoints\unet_latest.pt，--epochs 延长目标轮数）

python main.py sample --ckpt checkpoints\unet_final.pt
# 基础采样：64 张 8x8 网格（默认 1000 步 ancestral、EMA 权重）

python main.py sample --ckpt checkpoints\unet_final.pt --sample-steps 1000 200 100 50 20 --eta 1
python main.py sample --ckpt checkpoints\unet_final.pt --sample-steps 1000 200 100 50 20 --eta 0
# 扩展一：不同采样步数对比（相同初始噪声，逐一出图并打印耗时汇总）

python main.py sample --ckpt checkpoints\unet_final_0909.pt --digit all --guidance-w 2.0
# 扩展二：指定数字采样（0-9 各 8 张拼网格）
python main.py sample --ckpt checkpoints\unet_final_0909.pt --digit 8 --guidance-w 0
python main.py sample --ckpt checkpoints\unet_final_0909.pt --digit 8 --guidance-w 1
python main.py sample --ckpt checkpoints\unet_final_0909.pt --digit 8 --guidance-w 4
# guidance 强度对比（w=0 退化为无条件采样）

python benchmark.py                                    # 性能基准（FLOPs/耗时分析）
```

注：`unet_final_0909.pt` 为条件模型（支持 `--digit`）；`unet_final.pt` 为更早的无条件模型。

## 原理

前向加噪，反向去噪
一位画家，从一幅完工的画开始观察
- 前向：向这幅画泼五颜六色的颜料，破坏它
- 反向：从一片狼藉中，一步步去掉泼上去的颜料

### forward diffusion
逐步向原始图像添加高斯噪声，最终原始图像变为纯粹的随机噪声图

原始图像 $`x_0`$，总采样步数 $`T`$，在每一步 $`t`$，根据上一步的图像 $`x_{t-1}`$ 生成这一步的带噪图像：
$$\boxed{x_{t} = x_{t-1}\sqrt{1-\beta_t} + \epsilon_t \sqrt{\beta_t}}$$
要理解这个公式，首先需要理解这里的图像被表示成什么形态

实际上这里的图像是高维空间中的一个向量。例如，对于 $`256\times 256 \times 3`$ 的图像，它就是一个 196608 维的向量，每个元素都是一个灰度值（如 0-255, 8bit）。
噪声向量 $`\epsilon_t \in \mathbb{R}^{1\times 196608}`$ 就是一个和图像同维的向量，它的每一个元素值独立地从标准正态分布 $`\mathcal{N}(0,1)`$ 中抽样 
既然 $`\epsilon_t \sim \mathcal{N}(0,1)`$，那么由正态分布的性质很容易得到，$`x_t \sim \mathcal{N}(x_{t-1}\sqrt{1-\beta_t},\beta_t)`$

> [!NOTE]
> **正态分布的性质 1**
> $`X\sim \mathcal{N}(\mu, \sigma^2)`$, $`Y = aX+b`$, thus $`Y\sim \mathcal{N}(a\mu+b, \sigma^2 a^2)`$

$`\beta_t`$ 就是噪声方差，控制每步加噪的步长
$`\beta_t`$ 每步都在变化，根据预先设计好的方差调度方法来变化
- 线性调度：预设 $`\beta_1 = 10^{-4}, \beta_T = 0.02`$，中间线性变化，即 $`\beta_t = \beta_1 + \frac{t-1}{T-1} (\beta_T - \beta_1)`$
- 余弦调度：先定义 $`\bar{\alpha}_t = \dfrac{f(t)}{f(0)}`$，其中 $`f(t)=\cos^2\left(\dfrac{t/T+s}{1+s}\cdot\dfrac{\pi}{2}\right)`$（$`s`$ 为小的偏移量，常取 $`0.008`$），再令 $`\beta_t = 1-\dfrac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}`$（通常截断到不超过 $`0.999`$，防止数值奇异）

**从逐步加噪到一步加噪**
以上的加噪过程是一步步进行的，但从数学形式中，可以得到一步到位的公式
记 $`\alpha_t = 1-\beta_t`$
第一步：$`x_1 = x_0 \sqrt{\alpha_1} + \epsilon_1\sqrt{1-\alpha_1}`$
第二步：$`x_2 = x_1 \sqrt{\alpha_2} + \epsilon_2\sqrt{1-\alpha_2}`$

将 $`x_1`$ 代入 $`x_2`$，得到
$$x_2 = x_0\sqrt{\alpha_1 \alpha_2} + \epsilon_1 \sqrt{\alpha_2 - \alpha_1 \alpha_2} + \epsilon_2\sqrt{1-\alpha_2}$$
关键是，$`\epsilon_1,\epsilon_2`$ 是独立同分布的标准正态分布噪声
依旧是正态分布的性质，$`x_2 \sim \mathcal{N}(x_0\sqrt{\alpha_1 \alpha_2}, 1-\alpha_1 \alpha_2)`$
令 $`\bar{\alpha}_t = \prod_{i=1}^t \alpha_i`$（即第 1 步到第 $`t`$ 步所有 $`\alpha`$ 的连乘），则
$$\boxed{x_t = x_0\sqrt{\bar{\alpha}_t} + \epsilon \sqrt{1-\bar{\alpha}_t}}$$

> [!NOTE]
> **正态分布的性质 2**
> $`X,Y`$ 相互独立，$`X\sim \mathcal{N}(\mu_1, \sigma^2_1),Y\sim \mathcal{N}(\mu_2, \sigma^2_2)`$，则 $`X+Y \sim \mathcal{N}(\mu_1+\mu_2, \sigma^2_1 + \sigma^2_2)`$

### backward denoising
从纯噪声图开始，一步步预测并去除每一步添加的噪声

目标是根据 $`x_t`$ 和 $`t`$，推导 $`x_{t-1}`$ 的分布

从逐步加噪以及一步加噪公式出发：
$$x_{t} = x_{t-1}\sqrt{\alpha_t} + \epsilon_t \sqrt{1-\alpha_t}, \quad x_t = x_0\sqrt{\bar{\alpha}_t} + \epsilon_{0\rightarrow t} \sqrt{1-\bar{\alpha}_t},\quad x_{t-1} = x_0\sqrt{\bar{\alpha}_{t-1}} + \epsilon_{0\rightarrow t-1} \sqrt{1-\bar{\alpha}_{t-1}}$$
在给定 $`x_0`$ 的条件下，$`x_t = A`$ 和 $`x_{t-1} = B`$ 服从二维正态分布
计算两者的协方差
$$\sum_{AB} = \text{Cov}(A,B) = \text{Cov}(B,B\sqrt{\alpha_t} + \epsilon_t \sqrt{1-\alpha_t})$$
由协方差的性质可知
$$\text{Cov}(B,B\sqrt{\alpha_t} + \epsilon_t \sqrt{1-\alpha_t}) = \sqrt{\alpha_t} \text{Cov}(B,B) + \sqrt{1-\alpha_t}\text{Cov}(B,\epsilon_t)$$
而 $`x_{t-1}`$ 与 $`\epsilon`$ 相互独立，则
$$\sum_{AB} = \sqrt{\alpha_t} (1-\bar{\alpha}_{t-1})$$
目标是给定 A 求 $`B`$ 的分布，而二维正态分布的条件分布依然是正态分布：
$$B|A \sim \mathcal{N}(\tilde{\mu}_t,\tilde{\beta}_t)$$
其中
$$\tilde{\mu}_t = \mu_B + \frac{\sigma_B}{\sigma_A} \rho (A-\mu_A) = x_0\sqrt{\bar{\alpha}_{t-1}} + \frac{\sqrt{1-\bar{\alpha}_{t-1}}}{\sqrt{1-\bar{\alpha}_t}} \cdot \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{\sqrt{1-\bar{\alpha}_{t-1}}\sqrt{1-\bar{\alpha}_t}}\cdot (x_t - x_0\sqrt{\bar{\alpha}_t}) $$
再带入 $`x_t = x_0\sqrt{\bar{\alpha}_t} + \epsilon_{0\rightarrow t} \sqrt{1-\bar{\alpha}_t}`$（解出 $`x_0`$ 代回），得到：
$$\boxed{\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_{0\rightarrow t}\right)}$$
再计算方差：
$$\tilde{\beta}_t = (1-\rho^2)\sigma_B^2 = \left(1 - \frac{\alpha_t (1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \right) \cdot (1-\bar{\alpha}_{t-1}) = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$
至此我们得到了想要的 $`x_{t-1}`$ 分布

去噪公式即为：
$$\boxed{x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_{\theta}(x_t;t)\right) + z\sqrt{\tilde{\beta}_t}}$$
整个公式就是 ancestral sampling 的一步：从后验分布 $`\mathcal{N}(\tilde{\mu}_t,\tilde{\beta}_t)`$ 中采样 $`x_{t-1}`$，其中第二项 $`z\sqrt{\tilde{\beta}_t}`$（$`z\sim\mathcal{N}(0,1)`$）提供随机性（$`t=1`$ 时不再加噪声）

> [!NOTE]
> **协方差与 Pearson 相关系数**
> $`\text{Cov}(X,Y) = \mathbb{E}[(X-\mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]`$
> $`\rho = \text{Cov}(\frac{X-\mathbb{E}[X]}{\sqrt{\mathbb{D}[X]}},\frac{Y-\mathbb{E}[Y]}{\sqrt{\mathbb{D}[Y]}})=\frac{\text{Cov}(X,Y)}{\sqrt{\mathbb{D}[X]}\sqrt{\mathbb{D}[Y]}}`$

### 少步采样与 DDIM

#### 1. 突破口：训练只约束“边缘分布”

要理解 DDIM 的加速采样，首先得看穿 DDPM 训练目标的本质。扩散模型的训练损失，实际上只约束**单点边缘分布**，即：
$$q(x_t|x_0)=\mathcal{N}(x_0\sqrt{\bar{\alpha}_t},\ 1-\bar{\alpha}_t)$$

这个公式描述的是一张干净图片 $`x_0`$ 在第 $`t`$ 步时，被独立高斯噪声污染后的“快照”分布。它完全不涉及 $`x_t`$ 与 $`x_{t-1}`$ 之间是如何一步一步演变过来的。

用通俗的话说：**训练过程只检查“第 $`t`$ 步的图长什么样”，至于这张图是从 $`t-1`$ 步爬楼梯过来的，还是从起点空降过来的，损失函数根本不关心。**

这给了我们极大的设计自由：只要保证最终算出来的边缘分布符合上式，我们完全可以重新规划从 $`x_T`$ 到 $`x_0`$ 的“行走路径”，从而绕开必须走满 $`T`$ 步的限制。

#### 2. 重新设计路径：非马尔可夫前向过程

DDPM 的前向过程是马尔可夫链：$`x_t`$ 只依赖上一步 $`x_{t-1}`$，像爬楼梯一样，必须一阶一阶地走。

DDIM 则构造了一族**非马尔可夫**的前向过程。它允许 $`x_t`$ 同时依赖于 $`x_0`$ 和上一步的结果，甚至可以直接“跳步”——从 $`x_{\tau_i}`$ 直接转移到 $`x_{\tau_{i-1}}`$（其中 $`\tau`$ 是降序子序列）。只要新过程的边缘分布仍然等于 $`\mathcal{N}(x_0\sqrt{\bar{\alpha}_t},\ 1-\bar{\alpha}_t)`$，这种路径在数学上就是完全合法的。

**直观理解**：DDPM 规定必须爬楼梯（一步一阶）；DDIM 说我可以坐电梯直达任意楼层，只要到达时窗外的风景（边缘分布）和爬楼梯看到的完全一样，那我就可以随意设计路线。

#### 3. 反向更新的核心工程构造

现在我们要设计一步反向更新：从 $`x_{\tau_i}`$ 跳到 $`x_{\tau_{i-1}}`$。关键问题是：这一步的公式应该长什么样？

**（一）锚定“预测的干净图像”**

手里有当前噪声图 $`x_{\tau_i}`$ 和网络预测的噪声 $`\epsilon_\theta`$。利用边缘分布公式，我们可以反解出一个**预测的干净图像**：
$$\hat{x}_0 = \frac{x_{\tau_i} - \sqrt{1-\bar{\alpha}_{\tau_i}}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_{\tau_i}}}$$

既然我们唯一确定的是：$`x_{\tau_{i-1}}`$ 必须落在以 $`\hat{x}_0`$ 为中心、方差为 $`1-\bar{\alpha}_{\tau_{i-1}}`$ 的边缘分布上。那么它的结构**必须**长成“原图方向 + 噪声方向”的样子：
$$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}}\,\hat{x}_0 + \text{（某种噪声）}$$

**（二）噪声项的设计：为何拆成 $`c\cdot\epsilon_\theta(x_{\tau_i}) + \sigma_i z`$？**

这里的“某种噪声”**不是推导出来的，而是巧妙设计出来的**。它被刻意拆成两项，原因有三：

- **顺应代数结构**：将 $`\hat{x}_0`$ 代回上式后，展开项里天然会掉下来一个与 $`\epsilon_\theta`$ 成比例的系数项。因此，把噪声的第一部分写成 $`c\cdot\epsilon_\theta`$ 是为了方便合并同类项。

- **方差预算的“切蛋糕”逻辑（最本质）**：  
  在给定 $`\hat{x}_0`$ 的条件下，$`x_{\tau_{i-1}}`$ 的总噪声方差预算是固定的：
  $$\text{总预算} = 1-\bar{\alpha}_{\tau_{i-1}}$$

  我们手里恰好有两股独立的噪声来源：
  - 来源 A（历史方向）：上一步残留的噪声方向 $`\epsilon_\theta`$，跟着它能保持图像轮廓的连续性。
  - 来源 B（新随机探索）：新采的高斯噪声 $`z\sim\mathcal{N}(0,1)`$，用于弥补模型预测误差，增加生成多样性。

  因为 $`\epsilon_\theta`$ 和 $`z`$ 相互独立，方差具有可加性，所以必须把总预算切成两块：
  $$c^2 + \sigma_i^2 = 1-\bar{\alpha}_{\tau_{i-1}}$$

- **保持高斯性的最简操作**：两个独立高斯变量的线性组合依然是高斯分布，这使得我们可以继续用简单的均值和方差来描述整个分布，避免数学复杂化。

**灵魂比喻**：  
$`c\cdot\epsilon_\theta`$ 是你手里的**指南针**，指向出口的大致方向（确定性引导）。  
$`\sigma_i z`$ 是你随机迈出的**小碎步**，用于探索捷径（随机性探索）。  
两者线性相加，就是在“方向引导”和“随机探索”之间做出的最简单平衡。

#### 4. 系数确定与 $`\eta`$ 的调节作用

根据方差预算公式 $`c^2+\sigma_i^2 = 1-\bar{\alpha}_{\tau_{i-1}}`$，可以解出噪声方向系数：
$$c = \sqrt{1-\bar{\alpha}_{\tau_{i-1}}-\sigma_i^2}$$

这个式子揭示了一个核心事实：**一旦选定 $`\sigma_i`$，$`c`$ 就被唯一确定；而 $`\sigma_i`$ 本身是自由的**——这完美呼应了“训练只约束边缘分布，所以转移规则不唯一”的观察。

为了统一调控这个自由度，DDIM 引入系数 $`\eta\in[0,1]`$，令 $`\sigma_i = \eta \cdot \sigma_i^{DDPM}`$，其中 $`\sigma_i^{DDPM}`$ 是广义后验分布的标准差：
$$\sigma_i^{DDPM} = \sqrt{\frac{1-\bar{\alpha}_{\tau_{i-1}}}{1-\bar{\alpha}_{\tau_i}}}\sqrt{1-\frac{\bar{\alpha}_{\tau_i}}{\bar{\alpha}_{\tau_{i-1}}}}$$

将上述关系代回更新式，得到统一的 DDIM 反向更新公式：
$$\boxed{
x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{\tau_{i-1}}-\sigma_i^2}\;\epsilon_\theta(x_{\tau_i},\tau_i) + \sigma_i z
}$$

- 当 $`\eta=1`$ 时，整式完全退化为 DDPM 的祖先采样（子序列取完整时即标准 DDPM 公式）。
- 当 $`\eta=0`$ 时，$`\sigma_i=0`$，无新噪声注入，整个过程**完全确定**——相同的初始噪声 $`x_T`$ 必然生成相同的图，这正是 “Implicit” 一词的由来。

#### 5. 少步采样为何倾向 $`\eta=0`$ 的确定性模式

当采样步数 $`S`$ 很小时，子序列相邻时间步之间的噪声级差会变得很大，此时 $`\hat{x}_0`$ 的预测误差会被显著放大。

- 如果采用 $`\eta=1`$（DDPM 模式），每步还要额外注入一份全新的随机噪声 $`z`$。这相当于在已经偏离的路径上再“乱抖一下”，导致预测误差逐级累积，最终生成结果明显模糊。
- 如果采用 $`\eta=0`$（确定性 DDIM），则完全消除了随机扰动这一误差来源。虽然每一步都完全依赖模型预测，但至少没有额外的干扰项去放大错误。

**直观理解**：步子跨得越大（步数越少），就越要攥紧指南针（$`\eta=0`$），尽量减少瞎蹦跶；只有当步数足够多（$`S`$ 接近 $`T`$）时，偶尔掷骰子（$`\eta=1`$）探索一下才不至于跑偏，此时两者差别不大。

### U-Net
$$SiLU(x) = x \cdot \frac{1}{1+e^{-x}}$$

以输入图像通道数为 $`C`$ 为例

**编码器**

$$Convolution1 \rightarrow GroupNorm + SiLU \rightarrow Convolution2 \rightarrow GroupNorm+SiLU$$

在此基础上，把残差块的输入通过一条跳跃连接直接加到输出上（输入输出通道数不一致时用 1x1 卷积对齐），整体才构成一个残差块；时间步嵌入也注入到每个残差块中（见后文）

每个层级（若干残差块）结束后，通过 MaxPool 降采样，max-pooling 导致降维

- 卷积层：kernel 大小 3x3xC，步长为 1，padding 为 1，有 M 个卷积核
- 最大池化层：kernel 大小为 2x2，步长为 2

如何理解卷积核？
一个卷积核处理所有输入通道，将结果相加，输出一个通道
将它想象为一个 3 维的长方体，长和宽是 $`3\times 3`$（kernel size），高就是输入通道数 $`C`$，不同层高有不同的权重，但是整个 kernel 只有一个 bias
设图像的第 $`i`$ 个通道为 $`\mathbf{X}_i \in \mathbb{R}^{16 \times 16}`$，第 $`j`$ 个卷积核的偏置为 $`b_j`$，第 $`i`$ 层权重为 $`\mathbf{W}_{j,i}\in \mathbb{R}^{3 \times 3}`$，则经过处理后输出为：
$$Y_j = \sum_{i=1}^C \mathbf{X}_i * \mathbf{W}_{j,i} + b_j \in \mathbb{R}^{16 \times 16 \times 1}$$
再将所有卷积核的输出叠起来得到卷积层的输出 $`\in \mathbb{R}^{16 \times 16 \times M}`$

**Group Nomalization**
按 channel 划分组，计算组内所有像素值的均值和方差，对每个像素值先进行归一化：
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2+\varepsilon}}$$
每个 channel 都有各自的 $`\gamma`$（scale） 和 $`\beta`$（shift），再进行计算：
$$y_i = \gamma \hat{x}_i + \beta$$

> [!NOTE]
> **输出尺寸计算**
> $$O = \frac{I - K + 2P}{S} + 1$$


**解码器**
为了将经过编码器的图像维度放大，进行最邻近插值/双线性插值

以最邻近插值为例，输出图像位置为 $`(i,j)`$ 的像素值，等于输入图像位置为 $`(\lfloor \frac{i}{2} \rfloor,\lfloor \frac{j}{2} \rfloor)`$ 的像素值（位置坐标从 0 开始）
$$
\begin{bmatrix}
A & B \\
C & D \\
\end{bmatrix}
\rightarrow
\begin{bmatrix}
A & A & B & B \\
A & A & B & B \\
C & C & D & D \\
C & C & D & D \\
\end{bmatrix}
$$
接着进行平滑卷积，然后*跳跃连接*，叠加同一层编码器的输出，然后进行（顺序和编码器不同）：

$$GroupNorm + SiLU \rightarrow Convolution 1 \rightarrow GroupNorm+SiLU \rightarrow Convolution2$$

最终还要经过一个输出层，它的卷积核大小是 1x1，总共有 $`C`$ 个，将 channel 数恢复到输入图像的 channel 数

**跳跃连接**
解码器在插值+平滑卷积之后，图像尺寸就和同一层的编码器经过所有残差块，但还没有做 maxpooling 的输出尺寸一样了。接收后按 channel 维度堆叠
注意，并不是接收编码器的最终输出

**时间步嵌入**
注入到每一个残差块中
设时间嵌入向量的维度为 $`d`$（通常与 U-Net 第一层卷积核的通道数一致）
对于向量的第 $`i(i=0,1,...,\frac{d}{2}-1)`$ 个维度，首先计算频率系数：
$$f_i = (10000)^{-\frac{2i}{d}}$$
10000 是经验常数
应用正余弦函数，得到所有元素值：
$$\vec{emb}[2i] = \sin(t\times f_i)，\vec{emb}[2i+1] = \cos(t\times f_i)$$
最后通过一个两层全连接网络得到最终的向量表示
$$\vec{emb}_t = \text{Linear}(\text{SiLU}(\text{Linear}(\vec{emb})))$$

这个时间嵌入向量用于计算 GroupNorm 中使用到的 $`\gamma`$ 和 $`\beta`$

- 不同的通道，有不同的 $`\gamma`$ 和 $`\beta`$
- 不同的残差块，处理的通道数不同
- 每个残差块接收的时间嵌入向量维度都一样

所以，每一个残差块，都拥有完全独立、专属于自己的两层线性层（MLP）
统一的 $`d`$ 维时间嵌入向量输入时，第一层先投影到内部隐藏维度，第二层再投影到该残差块的通道数

**瓶颈**
接收编码器的最终输出，输出同维度结果给解码器

$$ResNetBlock\times 2 \rightarrow MultiHeadSelfAttention \rightarrow ResNetBlock\times 2$$

设编码器的最终输出大小为 $`16\times 16\times 1024`$，有 1024 个通道，每个像素视作一个 token，它的 embedding 维度即为 1024，输入就可以看作 256 个 token 的 embedding 矩阵
多头注意力按 channel 来切分，假设 $`h=8`$，$`256\times 1024`$ 被切分为 8 个 $`256\times 128`$ 矩阵
每个头有自己独立的 $`W_Q,W_K,W_V`$ 权重，输出独立的 $`Q,K,V`$ 值，再经过自注意力计算：
$$O_i = \text{softmax}\left( \frac{Q_i K_i^\mathsf{T}}{\sqrt{d_k}} \right)V_i \in \mathbb{R}^{256\times 128},\quad i=1,2,...,8$$
最终将所有头的结果按 channel 拼接，得到：
$$O = [O_1,O_2,...,O_8]$$
再通过一个可学习的输出权重 $`W_O`$ 进行融合得到最终输出：
$$O_{bottleneck} = O \cdot W_O$$

### EMA 权重指数滑动平均

训练中，参数 $`\theta`$ 的每一步更新都带有 batch 抽样带来的随机性，参数轨迹是一条抖动的曲线。最后时刻的参数只是这条轨迹上一个随机的"快照"——恰好停在哪，带点运气成分；而轨迹上最近一段位置的平均，往往落在损失曲面更平坦、更靠近盆地中心的地方，生成质量更稳定。

EMA（Exponential Moving Average，指数滑动平均）为此维护一份**影子参数** $`\theta_{ema}`$，训练的每一步之后做一次插值：

$$\boxed{\theta_{ema} \leftarrow \rho\,\theta_{ema} + (1-\rho)\,\theta}$$

$`\rho`$ 是衰减系数（本项目取 $`\rho=0.999`$）。影子参数**不参与梯度下降**，只是被动地跟踪 $`\theta`$。

如何理解这个公式？
把递推逐步展开，$`\theta_{ema}`$ 实际上是参数历史轨迹的指数加权平均：越近的参数权重越大，越早的按 $`\rho`$ 的几何级数衰减，有效窗口约为 $`\frac{1}{1-\rho}`$ 步。$`\rho=0.999`$ 对应窗口约 1000 步（约 2 个 epoch），窗口内的高频抖动被平均掉——相当于对参数轨迹做低通滤波，输出一条平滑的轨迹。

使用方式：
- 训练：照常反向传播更新 $`\theta`$，每步结束后按上式更新一次 $`\theta_{ema}`$（一次向量插值，CPU 上耗时不到 1ms）
- 采样：把 $`\theta_{ema}`$ 载入模型（本项目 checkpoint 同时保存两套权重，采样时优先加载 EMA 权重）

> [!NOTE]
> **$`\rho`$ 与训练长度的关系**
> 有效窗口 $`\frac{1}{1-\rho}`$ 应明显小于总训练步数，否则影子参数还停留在初始化附近，"平均"失去意义。本项目训练约 $`20\times469\approx 10^4`$ 步，$`\rho=0.999`$（窗口 $`10^3`$ 步）是合适的折中；DDPM 原文训练几十万步，使用 $`\rho=0.9999`$。

### 指定数字生成（条件扩散与 Classifier-Free Guidance）

至此训练的模型是**无条件**的：从纯噪声出发，生成哪个数字全凭运气。希望像给画家下订单一样，指定生成数字 $`y\in\{0,1,...,9\}`$。

**条件注入：把标签变成向量**

和时间步 $`t`$ 一样，数字标签 $`y`$ 也要先变成向量才能送进网络。最简单的方式是一张可学习的查找表（embedding table）：10 个数字各对应一个 $`d`$ 维向量，向量值随训练自动学习：

$$\vec{emb}_y = \text{EmbeddingTable}[y] \in \mathbb{R}^d$$

取与时间嵌入相同的维度 $`d`$，直接相加，然后沿用原来的注入通路（每个残差块的专属 MLP 生成 $`\gamma,\beta`$）：

$$\vec{emb} = \vec{emb}_t + \vec{emb}_y$$

模型从 $`\epsilon_\theta(x_t,t)`$ 变为 $`\epsilon_\theta(x_t,t,y)`$；前向加噪、损失函数（MSE）、反向采样公式**全部不变**——条件只是网络额外接收的一份信息。

**Classifier-Free Guidance（无分类器引导）**

只注入条件，模型对条件的服从往往不够"坚决"（生成的数字有时看起来并不像指定的那个）。希望有一个旋钮，能在采样时调节条件的强度。

从贝叶斯公式出发：$`p(y|x) \propto p(x|y)\,/\,p(x)`$，两边取对数再对 $`x_t`$ 求梯度（score）：

$$\nabla_{x_t}\log p(x_t|y) = \nabla_{x_t}\log p(x_t) + \nabla_{x_t}\log p(y|x_t)$$

即：条件分布的 score = 无条件分布的 score + 一个"分类器"指出的方向。人为给分类器方向加一个权重 $`w`$（guidance scale）：

$$\nabla_{x_t}\log p_w(x_t|y) = \nabla_{x_t}\log p(x_t) + w\,\nabla_{x_t}\log p(y|x_t)$$

$`w=1`$ 时就是正常的条件采样；$`w>1`$ 时沿着"更像数字 $`y`$"的方向走得更远。

问题：上式需要额外训练一个分类器 $`p(y|x_t)`$。**classifier-free 的关键技巧**：不用单独训练分类器——由贝叶斯分解，$`\nabla\log p(y|x_t) = \nabla\log p(x_t|y) - \nabla\log p(x_t)`$，等号右边两项恰好是**同一个去噪网络在有/无条件下的两种输出**！让网络同时学会这两种模式：训练时以一定概率（如 10%）把真实标签替换成一个特殊的"空标签" $`\varnothing`$，于是同一个网络既能给出 $`\epsilon_\theta(x_t,t,y)`$，也能给出 $`\epsilon_\theta(x_t,t,\varnothing)`$。

代入贝叶斯分解：

$$\nabla_{x_t}\log p_w(x_t|y) = \nabla_{x_t}\log p(x_t) + w\left[\nabla_{x_t}\log p(x_t|y) - \nabla_{x_t}\log p(x_t)\right]$$

而噪声预测与 score 只相差一个负系数：$`\epsilon_\theta(x_t,t,\cdot) \approx -\sqrt{1-\bar{\alpha}_t}\,\nabla_{x_t}\log p(x_t|\cdot)`$（score 指向数据分布的高密度方向，噪声指向其反方向）。代入并把负系数提出，得到采样时每步实际使用的引导噪声预测：

$$\boxed{\hat{\epsilon} = \epsilon_\theta(x_t,t,\varnothing) + w\left[\epsilon_\theta(x_t,t,y) - \epsilon_\theta(x_t,t,\varnothing)\right]}$$

读法：无条件预测 + $`w`$ × 条件相对无条件的"修正方向"。

- $`w=1`$：$`\hat{\epsilon}=\epsilon_\theta(x_t,t,y)`$，退化为普通条件采样
- $`w>1`$：条件强度加大，指定数字的特征更鲜明；过大（如 >10）会导致画面失真、对比度异常
- $`w=0`$：$`\hat{\epsilon}=\epsilon_\theta(x_t,t,\varnothing)`$，退化为无条件采样

实现上，每个采样步需要条件、无条件**两次前向传播**（或把两份输入拼成一个 batch 一次前向）。得到 $`\hat{\epsilon}`$ 后，DDPM / DDIM 的采样公式照常使用（把公式中的 $`\epsilon_\theta`$ 换成 $`\hat{\epsilon}`$ 即可）。

> [!NOTE]
> **空标签 $`\varnothing`$ 的实现**
> embedding 表多开一行（共 11 行），用索引 10 固定表示 $`\varnothing`$；训练时每张图以概率 $`p_{drop}=0.1`$ 把标签换成 $`\varnothing`$。因此同一份网络权重同时学到条件与无条件两种预测模式。

**对训练流程的改动**（承接前面的 9 步，改动极小）：第 1 步取 batch 时同时取出标签 $`y`$；以 10% 概率把标签替换为 $`\varnothing`$；第 5 步前向传播多传入标签 embedding。其余步骤不变。

## 训练

1. **输入**：取一张图 $`x_0`$（MNIST，形状 `[1, 28, 28]`，已归一化到 [-1, 1]）
2. **随机采样**：抽 $`t=50`$，随机抽取噪声向量 $`\epsilon`$（形状同图像）
3. **一步加噪**：代一步加噪公式（此步纯粹是数值运算，不产生梯度）
4. **生成时间嵌入**：将标量 `50` 编码为高维向量 `time_emb`。
5. **前向传播（有梯度）**：将 `x_50` 和 `time_emb` 送入 U-Net。输出预测噪声 $`\epsilon_{\theta}`$（形状 `[1, 28, 28]`）
6. **计算损失**：$`L=MSE(\epsilon_\theta,\epsilon)`$，得到一个标量数字
7. **反向传播（有梯度）**：执行 `loss.backward()`，计算损失函数关于 U-Net 中每一个参数的梯度
8. **更新参数**：优化器（如 Adam）拿着这些梯度，执行 `optimizer.step()`，微调 U-Net 里的参数（注意：每个 step 的反向传播之前要先 `optimizer.zero_grad()` 清空上一轮累积的梯度）
9. **更新影子参数（EMA）**：执行 $`\theta_{ema} \leftarrow 0.999\,\theta_{ema} + 0.001\,\theta`$。影子参数不参与梯度，仅被动跟踪；采样时加载它（见"EMA 权重指数滑动平均"一节）