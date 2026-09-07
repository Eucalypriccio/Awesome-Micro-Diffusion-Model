# 从零训练迷你扩散模型 Awesome-Micro-Diffusion-Model

## Quick Start

在仓库根目录执行：

```
.venv\Scripts\activate
python main.py check
python main.py train --schedule linear --epochs 20
python main.py train --resume                # 断点续训（默认接 checkpoints\unet_latest.pt，可 --epochs 延长）
python main.py sample --ckpt checkpoints\unet_final.pt
```

## 实验
- 前向加噪：噪声调度，一步到位的闭式加噪
- 去噪网络：构建 U-Net（参数量 $\leq 5\text{M}$），以 MSE 预测噪声并训练
- 反向采样：实现 ancestral sampling，从纯噪声逐步去噪
- 输出：推理时从标准正态分布随机采样 8x8 张纯噪声图，输出得到 8x8 样本网格
- 数据集 MNIST `torchvision.datasets.MNIST()`
	- 训练集 60,000 张
	- 测试集 10,000 张
	- 每张图片大小 28x28x1
- 支持 CPU/GPU（`torch.cuda.is_available()` 自动检测）
- 支持线性调度和余弦调度
- 扩展内容（后续完成）
	- 不同采样步数对生成质量与耗时的影响
	- 完成指定数字采样（即支持文字 embedding），比如在终端输入数字 0-9，生成对应的图像

要求：
- 控制模型大小，CPU 训练小时不超过 2h
- 配置好训练规范，几个 epoch，什么优化器，训练情况打印
- 代码规范
	- 使用原生 PyTorch, numpy, sklearn, transformers 等库实现，不借助其他高级库
	- 做好 Python 环境管理，`.venv, requirements.txt`
	- 变量名要清晰易懂无歧义
	- 使用最基础、直观的语法
	- 将数据集准备、模型架构、训练、测试以及其他辅助性代码分开管理，一份代码只关注一类功能
	- 可复用的模块、函数直接封装，不要重复完整实现
	- 代码关键部分写上简洁、易懂的注释

## 原理

前向加噪，反向去噪
一位画家，从一幅完工的画开始观察
- 前向：向这幅画泼五颜六色的颜料，破坏它
- 反向：从一片狼藉中，一步步去掉泼上去的颜料

### forward diffusion
逐步向原始图像添加高斯噪声，最终原始图像变为纯粹的随机噪声图

原始图像 $x_0$，总采样步数 $T$，在每一步 $t$，根据上一步的图像 $x_{t-1}$ 生成这一步的带噪图像：
$$\boxed{x_{t} = x_{t-1}\sqrt{1-\beta_t} + \epsilon_t \sqrt{\beta_t}}$$
要理解这个公式，首先需要理解这里的图像被表示成什么形态

实际上这里的图像是高维空间中的一个向量。例如，对于 $256\times 256 \times 3$ 的图像，它就是一个 196608 维的向量，每个元素都是一个灰度值（如 0-255, 8bit）。
噪声向量 $\epsilon_t \in \mathbb{R}^{1\times 196608}$ 就是一个和图像同维的向量，它的每一个元素值独立地从标准正态分布 $\mathcal{N}(0,1)$ 中抽样 
既然 $\epsilon_t \sim \mathcal{N}(0,1)$，那么由正态分布的性质很容易得到，$x_t \sim \mathcal{N}(x_{t-1}\sqrt{1-\beta_t},\beta_t)$

> [!note] 正态分布的性质 1
> $X\sim \mathcal{N}(\mu, \sigma^2)$, $Y = aX+b$, thus $Y\sim \mathcal{N}(a\mu+b, \sigma^2 a^2)$

$\beta_t$ 就是噪声方差，控制每步加噪的步长
$\beta_t$ 每步都在变化，根据预先设计好的方差调度方法来变化
- 线性调度：预设 $\beta_1 = 10^{-4}, \beta_T = 0.02$，中间线性变化，即 $\beta_t = \beta_1 + \frac{t-1}{T-1} (\beta_T - \beta_1)$
- 余弦调度：先定义 $\bar{\alpha}_t = \dfrac{f(t)}{f(0)}$，其中 $f(t)=\cos^2\left(\dfrac{t/T+s}{1+s}\cdot\dfrac{\pi}{2}\right)$（$s$ 为小的偏移量，常取 $0.008$），再令 $\beta_t = 1-\dfrac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$（通常截断到不超过 $0.999$，防止数值奇异）

**从逐步加噪到一步加噪**
以上的加噪过程是一步步进行的，但从数学形式中，可以得到一步到位的公式
记 $\alpha_t = 1-\beta_t$
第一步：$x_1 = x_0 \sqrt{\alpha_1} + \epsilon_1\sqrt{1-\alpha_1}$
第二步：$x_2 = x_1 \sqrt{\alpha_2} + \epsilon_2\sqrt{1-\alpha_2}$

将 $x_1$ 待入 $x_2$，得到
$$x_2 = x_0\sqrt{\alpha_1 \alpha_2} + \epsilon_1 \sqrt{\alpha_2 - \alpha_1 \alpha_2} + \epsilon_2\sqrt{1-\alpha_2}$$
关键是，$\epsilon_1,\epsilon_2$ 是独立同分布的标准正态分布噪声
依旧是正态分布的性质，$x_2 \sim \mathcal{N}(x_0\sqrt{\alpha_1 \alpha_2}, 1-\alpha_1 \alpha_2)$
令 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$（即第 1 步到第 $t$ 步所有 $\alpha$ 的连乘），则
$$\boxed{x_t = x_0\sqrt{\bar{\alpha}_t} + \epsilon \sqrt{1-\bar{\alpha}_t}}$$

> [!note] 正态分布的性质 2
> $X,Y$ 相互独立，$X\sim \mathcal{N}(\mu_1, \sigma^2_1),Y\sim \mathcal{N}(\mu_2, \sigma^2_2)$，则 $X+Y \sim \mathcal{N}(\mu_1+\mu_2, \sigma^2_1 + \sigma^2_2)$

### backward denoising
从纯噪声图开始，一步步预测并去除每一步添加的噪声

目标是根据 $x_t$ 和 $t$，推导 $x_{t-1}$ 的分布

从逐步加噪以及一步加噪公式出发：
$$x_{t} = x_{t-1}\sqrt{\alpha_t} + \epsilon_t \sqrt{1-\alpha_t}, \quad x_t = x_0\sqrt{\bar{\alpha}_t} + \epsilon_{0\rightarrow t} \sqrt{1-\bar{\alpha}_t},\quad x_{t-1} = x_0\sqrt{\bar{\alpha}_{t-1}} + \epsilon_{0\rightarrow t-1} \sqrt{1-\bar{\alpha}_{t-1}}$$
在给定 $x_0$ 的条件下，$x_t = A$ 和 $x_{t-1} = B$ 服从二维正态分布
计算两者的协方差
$$\sum_{AB} = \text{Cov}(A,B) = \text{Cov}(B,B\sqrt{\alpha_t} + \epsilon_t \sqrt{1-\alpha_t})$$
由协方差的性质可知
$$\text{Cov}(B,B\sqrt{\alpha_t} + \epsilon_t \sqrt{1-\alpha_t}) = \sqrt{\alpha_t} \text{Cov}(B,B) + \sqrt{1-\alpha_t}\text{Cov}(B,\epsilon_t)$$
而 $x_{t-1}$ 与 $\epsilon$ 相互独立，则
$$\sum_{AB} = \sqrt{\alpha_t} (1-\bar{\alpha}_{t-1})$$
目标是给定 A 求 $B$ 的分布，而二维正态分布的条件分布依然是正态分布：
$$B|A \sim \mathcal{N}(\tilde{\mu}_t,\tilde{\beta}_t)$$
其中
$$\tilde{\mu}_t = \mu_B + \frac{\sigma_B}{\sigma_A} \rho (A-\mu_A) = x_0\sqrt{\bar{\alpha}_{t-1}} + \frac{\sqrt{1-\bar{\alpha}_{t-1}}}{\sqrt{1-\bar{\alpha}_t}} \cdot \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{\sqrt{1-\bar{\alpha}_{t-1}}\sqrt{1-\bar{\alpha}_t}}\cdot (x_t - x_0\sqrt{\bar{\alpha}_t}) $$
再带入 $x_t = x_0\sqrt{\bar{\alpha}_t} + \epsilon_{0\rightarrow t} \sqrt{1-\bar{\alpha}_t}$（解出 $x_0$ 代回），得到：
$$\boxed{\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_{0\rightarrow t}\right)}$$
再计算方差：
$$\tilde{\beta}_t = (1-\rho^2)\sigma_B^2 = \left(1 - \frac{\alpha_t (1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} \right) \cdot (1-\bar{\alpha}_{t-1}) = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$
至此我们得到了想要的 $x_{t-1}$ 分布

去噪公式即为：
$$\boxed{x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_{\theta}(x_t;t)\right) + z\sqrt{\tilde{\beta}_t}}$$
整个公式就是 ancestral sampling 的一步：从后验分布 $\mathcal{N}(\tilde{\mu}_t,\tilde{\beta}_t)$ 中采样 $x_{t-1}$，其中第二项 $z\sqrt{\tilde{\beta}_t}$（$z\sim\mathcal{N}(0,1)$）提供随机性（$t=1$ 时不再加噪声）

> [!note] 协方差与 Pearson 相关系数
> $\text{Cov}(X,Y) = \mathbb{E}[(X-\mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$
> $\rho = \text{Cov}(\frac{X-\mathbb{E}[X]}{\sqrt{\mathbb{D}[X]}},\frac{Y-\mathbb{E}[Y]}{\sqrt{\mathbb{D}[Y]}})=\frac{\text{Cov}(X,Y)}{\sqrt{\mathbb{D}[X]}\sqrt{\mathbb{D}[Y]}}$

### U-Net
$SiLU(x) = x \cdot \frac{1}{1+e^{-x}}$

以输入图像通道数为 $C$ 为例

**编码器**

$Convolution1 \rightarrow GroupNorm + SiLU \rightarrow Convolution2 \rightarrow GroupNorm+SiLU$

在此基础上，把残差块的输入通过一条跳跃连接直接加到输出上（输入输出通道数不一致时用 1x1 卷积对齐），整体才构成一个残差块；时间步嵌入也注入到每个残差块中（见后文）

每个层级（若干残差块）结束后，通过 MaxPool 降采样，max-pooling 导致降维

- 卷积层：kernel 大小 3x3xC，步长为 1，padding 为 1，有 M 个卷积核
- 最大池化层：kernel 大小为 2x2，步长为 2

如何理解卷积核？
一个卷积核处理所有输入通道，将结果相加，输出一个通道
将它想象为一个 3 维的长方体，长和宽是 $3\times 3$（kernel size），高就是输入通道数 $C$，不同层高有不同的权重，但是整个 kernel 只有一个 bias
设图像的第 $i$ 个通道为 $\mathbf{X}_i \in \mathbb{R}^{16 \times 16}$，第 $j$ 个卷积核的偏置为 $b_j$，第 $i$ 层权重为 $\mathbf{W}_{j,i}\in \mathbb{R}^{3 \times 3}$，则经过处理后输出为：
$$Y_j = \sum_{i=1}^C \mathbf{X}_i * \mathbf{W}_{j,i} + b_j \in \mathbb{R}^{16 \times 16 \times 1}$$
再将所有卷积核的输出叠起来得到卷积层的输出 $\in \mathbb{R}^{16 \times 16 \times M}$

**Group Nomalization**
按 channel 划分组，计算组内所有像素值的均值和方差，对每个像素值先进行归一化：
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2+\varepsilon}}$$
每个 channel 都有各自的 $\gamma$（scale） 和 $\beta$（shift），再进行计算：
$$y_i = \gamma \hat{x}_i + \beta$$

> [!note] 输出尺寸计算
> $$O = \frac{I - K + 2P}{S} + 1$$


**解码器**
为了将经过编码器的图像维度放大，进行最邻近插值/双线性插值

以最邻近插值为例，输出图像位置为 $(i,j)$ 的像素值，等于输入图像位置为 $(\lfloor \frac{i}{2} \rfloor,\lfloor \frac{j}{2} \rfloor)$ 的像素值（位置坐标从 0 开始）
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

$GroupNorm + SiLU \rightarrow Convolution 1 \rightarrow GroupNorm+SiLU \rightarrow Convolution2$

最终还要经过一个输出层，它的卷积核大小是 1x1，总共有 $C$ 个，将 channel 数恢复到输入图像的 channel 数

**跳跃连接**
解码器在插值+平滑卷积之后，图像尺寸就和同一层的编码器经过所有残差块，但还没有做 maxpooling 的输出尺寸一样了。接收后按 channel 维度堆叠
注意，并不是接收编码器的最终输出

**时间步嵌入**
注入到每一个残差块中
设时间嵌入向量的维度为 $d$（通常与 U-Net 第一层卷积核的通道数一致）
对于向量的第 $i(i=0,1,...,\frac{d}{2}-1)$ 个维度，首先计算频率系数：
$$f_i = (10000)^{-\frac{2i}{d}}$$
10000 是经验常数
应用正余弦函数，得到所有元素值：
$$\vec{emb}[2i] = \sin(t\times f_i)，\vec{emb}[2i+1] = \cos(t\times f_i)$$
最后通过一个两层全连接网络得到最终的向量表示
$$\vec{emb}_t = \text{Linear}(\text{SiLU}(\text{Linear}(\vec{emb})))$$

这个时间嵌入向量用于计算 GroupNorm 中使用到的 $\gamma$ 和 $\beta$

- 不同的通道，有不同的 $\gamma$ 和 $\beta$
- 不同的残差块，处理的通道数不同
- 每个残差块接收的时间嵌入向量维度都一样

所以，每一个残差块，都拥有完全独立、专属于自己的两层线性层（MLP）
统一的 $d$ 维时间嵌入向量输入时，第一层先投影到内部隐藏维度，第二层再投影到该残差块的通道数

**瓶颈**
接收编码器的最终输出，输出同维度结果给解码器

$ResNetBlock\times 2 \rightarrow MultiHeadSelfAttention \rightarrow ResNetBlock\times 2$

设编码器的最终输出大小为 $16\times 16\times 1024$，有 1024 个通道，每个像素视作一个 token，它的 embedding 维度即为 1024，输入就可以看作 256 个 token 的 embedding 矩阵
多头注意力按 channel 来切分，假设 $h=8$，$256\times 1024$ 被切分为 8 个 $256\times 128$ 矩阵
每个头有自己独立的 $W_Q,W_K,W_V$ 权重，输出独立的 $Q,K,V$ 值，再经过自注意力计算：
$$O_i = \text{softmax}\left( \frac{Q_i K_i^\mathsf{T}}{\sqrt{d_k}} \right)V_i \in \mathbb{R}^{256\times 128},\quad i=1,2,...,8$$
最终将所有头的结果按 channel 拼接，得到：
$$O = [O_1,O_2,...,O_8]$$
再通过一个可学习的输出权重 $W_O$ 进行融合得到最终输出：
$$O_{bottleneck} = O \cdot W_O$$

## 训练

1. **输入**：取一张图 $x_0$（MNIST，形状 `[1, 28, 28]`，已归一化到 [-1, 1]）
2. **随机采样**：抽 $t=50$，随机抽取噪声向量 $\epsilon$（形状同图像）
3. **一步加噪**：代一步加噪公式（此步纯粹是数值运算，不产生梯度）
4. **生成时间嵌入**：将标量 `50` 编码为高维向量 `time_emb`。
5. **前向传播（有梯度）**：将 `x_50` 和 `time_emb` 送入 U-Net。输出预测噪声 $\epsilon_{\theta}$（形状 `[1, 28, 28]`）
6. **计算损失**：$L=MSE(\epsilon_\theta,\epsilon)$，得到一个标量数字
7. **反向传播（有梯度）**：执行 `loss.backward()`，计算损失函数关于 U-Net 中每一个参数的梯度
8. **更新参数**：优化器（如 Adam）拿着这些梯度，执行 `optimizer.step()`，微调 U-Net 里的参数（注意：每个 step 的反向传播之前要先 `optimizer.zero_grad()` 清空上一轮累积的梯度）