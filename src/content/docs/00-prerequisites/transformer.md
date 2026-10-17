---
title: 02｜Transformer：从零推导到源码（深度版）
description: 面向非计算机专业的博士级教程，含矩阵手算、多头注意力、RoPE、Mask、KV Cache、梯度和 PyTorch
sidebar:
  order: 2
---

# Transformer：一步也不跳的深度教程

> 这不是术语速览。阅读目标是：**能从输入张量手算 Attention，自己写 PyTorch，解释梯度如何回传，并读懂 π₀ 的双专家信息流**。每节先讲直觉，再讲公式、示例及工程意义。

![QKV 和 Attention 的运算流程](/Embodied-AI-Roadmap/diagrams/prerequisites-attention.svg)

**完整主线**：Token/Embedding → QKV 投影 → 相似度矩阵 → Mask → Softmax → V 加权 → 多头拼接 → MLP/残差/归一化 → 位置编码 → KV Cache → 训练与 π₀ 应用。

## 01｜先修数学：矩阵、点积、Softmax

**向量**是一列有顺序的数字。给定 $a=[1,0]$、$b=[0,1]$，点积 $a^\top b=0$；给定 $c=[1,0]$，则 $a^\top c=1$。在同等长度下，这能反映方向一致性。

**矩阵乘法**遵守中间维匹配原则：$X\in\mathbb R^{L\times d}$、$W\in\mathbb R^{d\times h}$，所以 $XW\in\mathbb R^{L\times h}$。例如 5 个 token、每个 8 维，线性投影到 4 维，形状从 [5,8] 变为 [5,4]；**不会改变 token 数量**。

**Softmax** 把匹配分数转换成权重：

$$p_i=\frac{e^{s_i}}{\sum_j e^{s_j}}.$$

得分 [2,0] 时，$e^2\approx7.389$，所以两个权重分别为 $7.389/8.389\approx0.881$ 和 $1/8.389\approx0.119$。注意力权重不是类别概率，也不是反向传播梯度。

## 02｜从 Token、Embedding 到 QKV

Tokenizer 将文本变为 token ID，Embedding 查表将每个 ID 映射成向量。图像可以先切 patch，再投影成视觉 token；机器人状态和动作也可以转换成向量表示。

输入 $X\in\mathbb R^{B\times L\times d}$，经过三组**独立可学习**的线性权重：

$$Q=XW_Q,\quad K=XW_K,\quad V=XW_V.$$

**类比检索资料**：Q 表示当前要找什么，K 表示每个候选提供何种索引，V 表示实际取回的信息。Q/K 用于算匹配分数，V 用于输出加权特征，因此三者作用不同。

Self-Attention 的 Q/K/V 常来自同一个 X 的不同投影；Cross-Attention 中 Q 和 K/V 可以来自不同来源。不要把两者混为一谈。

## 03｜Attention 完整手算：从得分到输出

取一个 Query、两个 Key：

$$Q=[1,0],\ K=\begin{bmatrix}1&0\\0&1\end{bmatrix},\
V=\begin{bmatrix}10&0\\0&10\end{bmatrix}.$$

步骤一：$QK^\top=[1,0]$。

步骤二：$d_k=2$，缩放分数 $S=[1/\sqrt2,0]\approx[0.7071,0]$。

步骤三：Softmax 得 $P\approx[0.670,0.330]$。

步骤四：乘上 V 得 $O=[6.70,3.30]$。

**这个输出仍是隐藏特征，不是机器人动作，也不是一句语言回答。**

完整公式：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

为何除以 $\sqrt{d_k}$？在各维独立、零均值、单位方差的简化假设下，点积方差大致随维度增长。缩放可避免大维度点积造成 Softmax 过度饱和，但不意味着一定不会梯度不稳定。

## 04｜对照维度查每个计算环节

| 阶段 | 张量形状 |
| --- | --- |
| 输入 | [B,L,d_model] |
| Q | [B,Lq,d_k] |
| K | [B,Lk,d_k] |
| V | [B,Lk,d_v] |
| QK 的转置乘法 | [B,Lq,Lk] |
| Softmax 权重 | [B,Lq,Lk] |
| 加权 V | [B,Lq,d_v] |

对有 100 个 token 的单头 Self-Attention，注意力矩阵是 [100,100]。假设 B=2、8 个头、L=100，每头的注意力矩阵组合为 [2,8,100,100]。它不是模型输出的动作长度，而是查询 token 对键 token 的匹配关系。

## 05｜Multi-Head、MLP、Norm、残差：一个 Block 真正包含什么

多头注意力不是把同一个结果复制八遍，而是不同头有不同的 Q/K/V 投影：

$$\mathrm{MHA}(X)=\mathrm{Concat}(head_1,\ldots,head_h)W_O.$$

若隐藏维度 512、头数 8，常见每头宽度 64。**一个头可能学习物体关系，另一个可能关注长程位置**，但没有机制保证每头都有一个固定的人类可解释语义。

典型 Pre-Norm Transformer Block：

$$H=X+\mathrm{MHA}(\mathrm{Norm}(X)),$$

$$Y=H+\mathrm{MLP}(\mathrm{Norm}(H)).$$

Attention 用于 token 间交互；MLP 对单个 token 的特征做非线性变换（例如 512→2048→512），通常不改变 token 数；残差让信号和梯度有更直接的路径。

LayerNorm 对特征做中心化及尺度归一；RMSNorm 主要按均方根做尺度归一。现代模型还可能使用 SwiGLU 等激活。不能仅把 Transformer 理解为 QKV 计算。

## 06｜位置编码和 RoPE，为什么 token 顺序重要

“杯子在托盘左边”和“托盘在杯子左边”拥有近似相同的词集合，但语义不同。仅仅把词作为无位置信息的集合，无法充分表达顺序关系。

RoPE 在 Q/K 的成对维度上施加与 token 位置有关的旋转，二维旋转矩阵是：

$$R(m\theta)=\begin{bmatrix}\cos(m\theta)&-\sin(m\theta)\\\sin(m\theta)&\cos(m\theta)\end{bmatrix}.$$

通过位置旋转，Q/K 的匹配分数能够反映相对位置差。实际还有多组频率、不同 head 维度和缓存位置偏移，不能把“一个旋转矩阵”当成完整实现。

## 07｜Mask：真正决定谁可以看谁

| Query \\ Key | Prefix P1 | Prefix P2 | Action A1 | Action A2 |
| --- | --- | --- | --- | --- |
| Prefix P1 | ✓ | ✓ | × | × |
| Prefix P2 | ✓ | ✓ | × | × |
| Action A1 | ✓ | ✓ | ✓ | ✓ |
| Action A2 | ✓ | ✓ | ✓ | ✓ |

这是 π₀ 的**教学简化版**，不包含全部 padding、状态 token 和实现分组。Mask 在算分时把不允许的位置加上负无穷大，使对应 Softmax 权重变为零。

- Causal Mask：自回归 token 不看未来。
- Padding Mask：不读取填充 token。
- Block Mask：Prefix/Suffix 区域具有不同可见性。

如果某个 Query 被屏蔽掉**所有** Key，Softmax 可能产生 NaN，必须在实现里处理。

**π₀ 的关键**：视觉语言 Prefix 不应受到每次重新采样的随机动作污染，而动作 Suffix 要能读取视觉语言条件。真实 Mask 要结合官方源代码验证，不能仅靠教学图。

## 08｜KV Cache、反向传播和训练

**KV Cache** 在重复推理时保存已计算且条件未变的 K/V，避免重新处理历史或固定 token。文本自回归场景最典型；π₀ 的 Prefix 在动作 Flow 积分时通常保持固定，因此有缓存的机会。但 Action Expert 需要随着带噪动作变化反复前向，**不是整个 Transformer 只算一次**。

**反向传播**不是让 token 倒序运行。假设最终 Loss 依赖 $O=PV$，梯度先从输出传给 P/V，再经 Softmax 回到分数 $S$，再经 $S=QK^\top/\sqrt d$ 到 Q/K，最后回到 $W_Q,W_K,W_V$。若某一模块被冻结，它的权重不会被优化器更新。

注意区分**前向注意力权重**（用于数据融合）与**损失梯度**（用于训练参数）；它们不是同一种对象。

## 09｜最小可运行 PyTorch：自己实现多头注意力

下面不是伪代码，它实现 QKV 投影、拆头、缩放注意力、重新拼头和梯度回传：

~~~python
import math
import torch
from torch import nn

class TinyMHA(nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, allow=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def to_heads(z):
            return z.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = map(to_heads, (q, k, v))
        score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if allow is not None:
            score = score.masked_fill(~allow[:, None], -1e9)
        weights = torch.softmax(score, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        return self.out(out), weights

x = torch.randn(2, 4, 8)
model = TinyMHA()
result, weights = model(x)
print(result.shape)   # [2,4,8]
print(weights.shape)  # [2,2,4,4]
result.square().mean().backward()
print(model.qkv.weight.grad.norm())
~~~

**建议亲手改**：把 heads 改为 1；把 token 数改为 10；再加入布尔 allow 矩阵。逐步检查每个 tensor 的 shape。

## 10｜怎么映射到 π₀：结构、源码、动作生成

| 本章概念 | 在 π₀ 中的角色 | 应读哪里 |
| --- | --- | --- |
| 视觉 Token | 图像经视觉编码器进入 Prefix | embed_prefix |
| 语言 Token | 语言嵌入形成上下文 | embed_prefix |
| Linear Projection | State/Noisy Action 的动作侧表示 | embed_suffix |
| Attention | 视觉语言侧与动作侧交互 | gemma.py |
| Mask | 控制 Prefix/Suffix 的信息可见性 | make_attn_mask |
| KV Cache | 多次积分时固定条件的复用 | sample_actions |
| 梯度 | Flow Matching Loss 回传训练权重 | compute_loss |

π₀ 的两侧可以有不同 Transformer 参数，不是典型 Router MoE；PaliGemma 不必先解码出一段文本，动作专家可直接利用视觉语言条件。

要记住：**模型结构、损失目标和推理算法是三件不同的事**。Transformer 提供表征和计算，Flow Matching 定义连续动作的训练监督，数值积分把模型速度场变成动作块。

## 11｜四个检验是否真正理解的实验

1. **手算**：把第四节的 Query 改为 [0,1]，重新算 QK、Softmax 和输出。
2. **改 Mask**：构建两个 Prefix token、两个 Action token 的 4×4 布尔矩阵，检查被遮蔽位置的 Attention 权重是否近似零。
3. **改头数**：在相同隐藏维度 8 下，从 2 头改成 1 头；对比注意力矩阵的 shape 和总投影参数量。
4. **追源码**：打开 [openpi/pi0.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)，找出图像构造、Suffix 生成、Mask 创建、训练 Loss 和动作采样所在的函数，自己画出数据流。

**达标要求**：你能够解释每个矩阵的输入、输出和维度，不需要背诵答案；还能说清动作 Token 能看观测而观测 Prefix 不依赖随机动作的重要性。

