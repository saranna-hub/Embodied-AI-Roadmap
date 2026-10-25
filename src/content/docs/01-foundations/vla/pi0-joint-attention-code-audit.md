---
title: "π₀ × π₀.₅ 源码深挖：联合注意力、Mask、KV Cache 与 AdaRMS"
description: "逐行对照官方 openpi 的 π₀ / π₀.₅：两专家独立 QKV、分组 Attention Mask、流匹配训练、Prefix KV 缓存、时间调制和实验复现"
sidebar:
  order: 6
---

> **审计基线｜2026-10-25**：本篇针对 Physical Intelligence 的公开 **openpi JAX 实现**，以 [openpi main commit 215abfb](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) 固定源码版本。重点来源：[pi0.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)、[gemma.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py)、[pi0_config.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py)。涉及论文版本与社区移植时须另行比对。下列实验、性能估算是**教学推导**，不是作者真机测量数据。

本文解决五个长期混淆的问题：**拼接 Token 是否等于共享 Transformer 参数？动作 Token 是否因果地逐个生成？双专家 QKV 维度不同如何做同一次注意力？Prefix KV Cache 究竟复用了什么？π₀.₅ 相比 π₀ 具体改了哪段计算？**

## 1. 先看总体数据流：训练和推理不是同一条路径

![π₀ 双专家、联合注意力与 KV 缓存数据流](/Embodied-AI-Roadmap/diagrams/pi0-joint-attention-dataflow.svg)

图注：上方为两套参数分支，中央是一层内部的注意力交互，底部是训练的整序列计算与推理时的 Prefix prefill/多步去噪。图为**对照源码的教学重绘**，不是论文官方图。

**训练**时，将视觉/语言 Prefix 和状态/噪声动作 Suffix 一起输入多层 Gemma 专家。每个 Transformer 层中，各专家首先用**各自参数**计算 Q、K、V，沿序列维拼接后按 Mask 做联合 Attention，接着将注意力输出按序列分开，各自经过输出映射、Norm、FFN 等操作；最后只取动作 Suffix 的后 $H$ 个 Token 预测速度场。

**推理**时，观察不变、带噪动作每轮变化。先只让 Prefix 通过网络，缓存每层的 K/V；后续每个流积分步只计算 Suffix 当前的 Q/K/V，用它访问缓存的 Prefix K/V 与本轮的 Suffix K/V。**上一步 Suffix 的 K/V 不累积到下一步**。

这不是“两套大模型完全独立运行然后拼接结果”，也不是传统 MoE Router 选一个专家。可用的最简理解是**两套投影参数、共享跨 Token 的注意力交互**。

## 2. 源码里真正的张量维度：为什么宽度不同也能交互？

[pi0_config.py 20—31 行](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py#L20-L31) 的 JAX 默认配置与 [gemma.py 69—87 行](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py#L69-L87) 给出以下**实测代码配置**：

| 项目 | VLM/PaliGemma 的语言专家 | Action expert |
| --- | ---: | ---: |
| 配置名 | gemma_2b | gemma_300m |
| Transformer 隐藏宽度 $d_{\rm model}$ | 2048 | 1024 |
| Transformer 层数 | 18 | 18 |
| Query heads | 8 | 8 |
| KV heads | 1 | 1 |
| 每头维度 $d_h$ | 256 | 256 |
| FFN 中间维度 | 16384 | 4096 |
| π₀ 默认动作容器 | — | $H=50,D_a=32$ |

注意：这是 **Gemma 语言/动作专家**的配置，不等于整套网络参数量。还包括视觉编码器 SigLIP 等组件。图像分辨率、实际视觉 Token 数与相机数量受数据适配和配置影响，不能把“有三路标准相机 key”误认为任何机器人永远固定三路。

### 2.1 第一个经常被问错的问题：动作专家宽度 1024，怎么与宽度 2048 的 VLM 做联合 Attention？

因为必须相同的是 **注意力头的输出布局**，不是 Transformer 各专家的 $d_{\rm model}$。

VLM 独立投影：

$$
Q_v=X_vW_Q^v,\qquad K_v=X_vW_K^v,\qquad V_v=X_vW_V^v,
$$

Action expert 独立投影：

$$
Q_a=X_aW_Q^a,\qquad K_a=X_aW_K^a,\qquad V_a=X_aW_V^a.
$$

虽然 $X_v\in\mathbb R^{B\times P\times2048}$、$X_a\in\mathbb R^{B\times S\times1024}$，两侧 Q 的每头维度都是 256、Query head 数都是 8；两侧 K/V 的 head 数都是 1、每头维度 256。于是 K/V 可沿 Token 轴拼接：

$$
K=[K_v;K_a],\quad V=[V_v;V_a],\quad
K,V\in\mathbb R^{B\times(P+S)\times1\times256}.
$$

Q 则保留各自来源、组合为 $8$ 个 Query heads 的共享注意力计算，输出之后再按 Token 来源分段，通过**各自独立的输出投影矩阵**映射回 2048 或 1024。源码 [gemma.py 164—249 行](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py#L164-L249) 直接执行了这种 QKV 独立生成、序列拼接、注意力和分段输出。

**为什么 $d_h=256$，而不是“除以 2048 或 1024”？** 实现中先把 Q 乘 $d_h^{-1/2}$，这里是 $1/16$，因为 $d_h=256$。它让内积尺度随注意力头维度稳定，不是由整个 Transformer 隐藏宽度决定。Softmax 沿 **Key 的序列维**归一化，每个 Query 可以对多少个有效 Key 分配权重，取决于 Attention Mask。

### 2.2 8 个 Query heads、1 个 KV head 的意义

这是 **Grouped-Query Attention (GQA)** 的特例：多个 Query heads 共享较少的 KV heads。其计算逻辑是将 8 个 Query heads 按 1 个 KV head 分组，从同一个 KV head 读取内容，但每个 Query head 的投影不同。它与 8 个独立 KV heads 的完整 MHA 不是同一存储开销。

同样地，**“拼接参与同一个 Attention”不意味着 $W_Q^v=W_Q^a$**。源码创建参数时使用不同后缀命名（例如动作专家带后缀），只有头布局满足合并要求，而不是权重共享。

## 3. 源码审计核心：Attention Mask 根本不是简单下三角

![π₀ Prefix、State 与 Action 的注意力可见矩阵](/Embodied-AI-Roadmap/diagrams/pi0-joint-attention-mask.svg)

图注：8 Token 教学示意，Query 是行，Key 是列。V=视觉、P=语言、S=连续状态、A=未来动作。动作三 Token 互相可见，不是自回归生成。

实际 [pi0.py 19—44 行](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L19-L44) 的 Mask 逻辑可以数学化为：

$$
c_i=\sum_{j\le i}r_j,\quad
M_{ij}=m_im_j\,\mathbf 1[c_j\le c_i].
$$

其中 $m_i$ 是 Token 有效位（Padding 为 0），$r_j$ 是新块起点的布尔标志。**注意 $r_j$ 不是“是否允许读取”的标志，而是“从这里开始新组”的标记。**

若依次构造：

$$
[V_0,V_1,P_0,P_1,S,A_0,A_1,A_2]
$$

则可用的块起点向量和累积编号是：

$$
r=[0,0,0,0,1,1,0,0],
\quad c=[0,0,0,0,1,2,2,2].
$$

对应的可见矩阵（1 可以读取，0 不可）：

| Query \\ Key | V0 | V1 | P0 | P1 | S | A0 | A1 | A2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| V1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| P0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| P1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| S | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| A0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| A1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| A2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

### 3.1 为什么不是动作自回归？

Flow Matching **一次预测整个动作块的速度场**。未来 50 步的动作全部作为带噪的连续向量同时输入；在这块内部双向 Attention，有利于协调整段轨迹。它不要求像语言 Token 那样一位位、一步步自回归输出。

**三个可见性结论**：① Prefix 内部双向可见；② 状态可以看 Prefix，不可看未来动作；③ 动作块可以看 Prefix、状态与全部同组动作 Token。看懂这一点才能理解训练目标和 KV cache，否则会把 π₀ 误讲成“与 GPT 一样的逐动作 Token 生成”。

### 3.2 为什么 Prefix 不读 Suffix？

因为训练时 Suffix 含有由真实未来动作与噪声混合得到的 $x_\tau$。Prefix 若能依赖这些 Token，前缀表示不仅会受本轮噪声污染，还会在每次流积分时改变，无法一次计算后长期复用。当某个 Prefix 位置通过多层间接读取 Suffix 时，同样造成依赖泄漏。这个单向信息结构同时保证训练过程合理和推理缓存可行。

### 3.3 Padding 和位置编号

源码先按有效 Token 生成位置：

$$
p_i=\sum_{j\le i}m_j-1.
$$

Attention 还与 Query、Key 双侧有效性作逻辑与。即使 Padding Token 的位置编号出现重复，它也不应贡献有效注意力。实现验证时不能只打印块 Mask，还必须把 **Padding mask** 一起画出来。

## 4. π₀.₅ 究竟做了哪两处关键结构变化？

最准确的证据是 [pi0_config.py 28—41 行](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py#L28-L41)，这里直接声明 π₀.₅ 相比 π₀ 的两处不同，并切换默认语言 Token 长度。

| 对比 | π₀ | π₀.₅（openpi JAX 分支） |
| --- | --- | --- |
| 自身状态注入 | 连续 state → 线性层 → Suffix 的独立 Token | 状态被编码为离散语言 Token，进入 Prefix |
| 生成时间 $\tau$ | 与每个带噪动作 Token 拼接，再过 action-time MLP | 先 time-MLP，然后通过 AdaRMS 调制动作专家 |
| Suffix 长度（默认 $H=50$） | $1+H=51$ | $H=50$ |
| 语言最大 Token 长度（默认） | 48 | 200 |
| 模型配置开关 | pi05=False | pi05=True |

**特别注意**：这里是官方开放源码版本，不应未经验证就断言论文中所有 π₀.₅ 内部训练策略、数据混合、推理分支与这份代码完全相同。开源仓库公开说明 π₀.₅ 训练/推理当前支持 Flow Matching head；如果问完整论文有无其他头，须回论文、不要用该子集代表全貌。

### 4.1 π₀ 把时间加在哪里？

设动作专家宽度 $d=1024$，每个动作 Token 先投影成 $h_i\in\mathbb R^{1024}$。将时间做正弦余弦编码 $e(\tau)\in\mathbb R^{1024}$，沿特征维拼接：

$$
z_i=[h_i;e(\tau)]\in\mathbb R^{2048},
\qquad h'_i=\mathrm{MLP}_{2048\to1024}(z_i).
$$

注意这不是把 $\tau$ 作为额外一个语言 Token 加在序列末尾，而是逐动作 Token 融合时间信息。代码中涉及 action_time_mlp_in/out。

### 4.2 π₀.₅ 为什么换成 AdaRMS？

π₀.₅ 中动作 Token 保持动作投影本身，时间 embedding 经 time_mlp_in/out 产生条件 $c(\tau)$，进入每层动作专家的 RMSNorm：

$$
\hat h=\frac{h}{\sqrt{\frac1d\sum_{k=1}^d h_k^2+\varepsilon}},
$$

$$
[s(\tau),b(\tau),g(\tau)]=W_c\,c(\tau),
\qquad
\operatorname{AdaRMS}(h,\tau)=\hat h\odot[1+s(\tau)]+b(\tau).
$$

源码 [gemma.py 113—131 行](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py#L113-L131) 还将 $g(\tau)$ 用于残差门控，而不只是标准化。这意味着生成时间可以影响每一层注意力和 FFN 分支，不仅影响输入 Token。**这是从实现推导的结构差异，不等于已经通过严格消融证明“一定更优”。**

注意 AdaRMS 的参数量并不是“整个 PaliGemma 都加了 AdaRMS”。代码通过条件数组把时间送到动作专家侧，VLM 专家侧传入 None。

## 5. 完整一次训练前向：哪个量来自真值、哪个量是网络预测？

代码入口：[pi0.py 189—214 行](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L189-L214)。

示例取默认动作块：

$$
A\in\mathbb R^{B\times50\times32},
\quad\epsilon\sim\mathcal N(0,I).
$$

该实现从 **Beta(1.5,1)** 采样时间，再缩放到 $(0.001,1)$ 的内部采样区间，不是简单均匀采样：

$$
\tau=0.999\cdot\operatorname{Beta}(1.5,1)+0.001.
$$

构造带噪动作与目标：

$$
x_\tau=\tau\epsilon+(1-\tau)A,\qquad u_\tau=\epsilon-A.
$$

输入 $\{I,\ell,s,x_\tau,\tau\}$，经过 Prefix/Suffix 联合 Attention，取动作侧后 50 个位置并执行线性输出：

$$
v_\theta\in\mathbb R^{B\times50\times32}.
$$

源码在动作维取平均：

$$
L_{b,h}=\frac1{32}\sum_{d=1}^{32}
(v_{\theta,b,h,d}-u_{\tau,b,h,d})^2.
$$

**代码返回张量形状是 $[B,50]$，并非这一函数内直接求出的唯一标量**。后续训练循环如何聚合、如何处理动作填充/有效性，应继续沿训练器和数据 transform 查证，不能从这里推断“无效维天然被忽略”。

### 两个极易踩坑的时间问题

**第一**：官方源码明确注释当前实现使用的“$\tau=1$ 是噪声、$\tau=0$ 是目标”与论文中的时间方向**相反**。对着论文手写公式再套代码，流速度方向可能写反。

**第二**：$\tau$ 是去噪路径的插值时间，**不是**机器人采样时刻 $t$。$H=50$ 是真实动作的未来时间长度，不表示 50 次去噪迭代，也不能简单换算控制器频率。

## 6. 推理到底缓存了什么？不是将所有输出永久缓存

[pi0.py 217—279 行](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L217-L279) 的推理：

**阶段 A：Prefix prefill 一次**。当前图像经 SigLIP、语言经 embedding，VLM 专家处理这些条件。每个 Transformer 层保存 **Prefix K/V**。假设前缀长度 $P$、层数 $L=18$、KV heads $N_{kv}=1$、每头 $d_h=256$，每种 K 或 V 的近似存储布局为：

$$
[L,B,P,N_{kv},d_h].
$$

这属于实现级**形状描述**，真实内存还受 dtype、设备分片、框架实现影响。**缓存的是层内 K/V，不是最终直接可执行的动作。**

**阶段 B：Suffix 每一步重新算**。从 $x_1=\epsilon$ 开始。每轮将更新后的当前动作 $x_\tau$ 与时间条件送入动作专家；构造该轮 Suffix 内注意力以及对 Prefix 缓存的读取掩码；计算速度场 $v_\theta$ 并更新：

$$
\Delta\tau=-\frac1K,\qquad
x_{\tau+\Delta\tau}
=x_\tau+\Delta\tau\,v_\theta(x_\tau,\tau,o).
$$

默认 $K=10$ 是可配置的推理积分次数。每一轮 **Suffix 的 K/V 来自当前新动作，而非上一轮缓存的旧动作**。下一次相机观测变了，就需要更新 Prefix，不能把上一时间点的观察 KV 当成永久正确缓存。

### 6.1 为什么比每轮重算全模型更省？

设前缀长度 $P$、Suffix 长度 $S$、去噪步数 $K$。仅从注意力 Query-Key 配对数作**简化估算**（忽略视觉编码、FFN、内存传输与设备 kernel）：

$$
C_{\rm naive}\propto K(P+S)^2,
$$

$$
C_{\rm cache}\propto P^2+K S(P+S).
$$

例如教学取 $P=512,S=51,K=10$，得到约：

| 情形 | 近似 QK 配对数 |
| --- | ---: |
| 每轮重算 Prefix + Suffix | 3,169,690 |
| 一次 Prefix + 十轮 Suffix | 549,274 |

这是**注意力配对量级**的差别，不是“实际推理加速 5.77 倍”的测量！系统延迟还受多次专家 FFN、视觉编码、数据预处理和机器人通讯影响。测试需要分别计 prefill、去噪专家、反归一化和控制发送的耗时。

### 6.2 为什么 Prefix 禁读 Suffix 是 KV Cache 的必要条件？

设 Prefix 表征为 $h_P=f(P,S)$，如果注意力允许前缀读入 Suffix，且 Suffix 在去噪中每轮变化，那么 $h_P$ 与其 K/V 在每一轮都变。直接重用旧值会改变真实全序列前向语义。

当前 Mask 令 $h_P=f(P)$，Suffix 可以读取 $P$ 但不反向修改 $P$，因此 Prefix K/V 可在**同一次观测条件下**跨去噪迭代复用。

## 7. 一份可以直接运行的 Mask 单元测试

以下是根据源码公式独立写的 NumPy 最小实现，用于校验行列约定，不需要下载机器人模型，也不冒充已完成真机复现。

~~~python
import numpy as np

# 4 个 Prefix、1 个 state、3 个 action
tokens = ["V0", "V1", "P0", "P1", "S", "A0", "A1", "A2"]
block_start = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
valid = np.ones(8, dtype=bool)
group = np.cumsum(block_start)

# Query 行 i 只能读到组编号不晚于 i 的 Key j
mask = ((group[None, :] <= group[:, None])
        & valid[:, None] & valid[None, :])

print("groups", group.tolist())
print("Mask (rows=query, columns=key)")
for name, row in zip(tokens, mask, strict=True):
    print(f"{name:>2} :", "".join("1" if x else "0" for x in row))

assert (mask[:4, 4:] == False).all()   # Prefix 不可读 Suffix
assert mask[4, :5].all()               # State 读 Prefix + 自己
assert not mask[4, 5:].any()           # State 不读未来动作
assert mask[5:, :].all()               # Action 可读全部动作块
assert mask[6, 7] and mask[7, 6]       # 动作 Token 互相双向可见

# Padding Query/Key 两端都屏蔽：只测试显式关系，不测试 softmax 全屏蔽行
valid[-1] = False
mask2 = ((group[None, :] <= group[:, None])
         & valid[:, None] & valid[None, :])
assert not mask2[-1, :].any() and not mask2[:, -1].any()
print("PASS")
~~~

**预期输出**：Prefix 的行形如 11110000；State 一行 11111000；三个动作行均为 11111111。如果你看到动作行呈严格下三角，说明你的实现把 Flow Matching 的动作块错误地改成了语言自回归式 Mask。

## 8. 检查 RoPE、Softmax 与多头：三处细节不能猜

**RoPE 在 Q、K 上，不是在原始图像像素上**。源码 [gemma.py 424—440 行](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py#L424-L440) 对两半 head 特征执行旋转，位置编号来自整个有效 Token 序列的逻辑位置。每轮 Suffix 仍需正确的位置偏移，以便对已经缓存的 Prefix K 计算相对位置。

**Softmax 沿 Key 维归一化**，不是沿 Query 维或所有动作维度归一化。可将 logits 理解为 $[B,N_{kv},G,T_{query},T_{key}]$，其中分组 Query heads $G=8$。独立 Query 决定其对每个可见 Key 的权重分布。

**Scale 使用 $1/\sqrt{d_h}$**，本例 $d_h=256$。不能改成 $1/\|q\|\|k\|$ 并声称等价：后者属于按向量范数标准化，会改变 logit 的统计与模型训练目标；也不能改成 $1/\sqrt{2048}$，因为这里点积发生在 head 内的 256 维。

## 9. 如果要把这一模块改造成具身算法研究，怎么做消融？

| 对照 | 唯一改变的变量 | 假设验证 | 必须记录 |
| --- | --- | --- | --- |
| A：正确 Mask / 错误因果动作 Mask | 动作组内部可见性 | chunk 协同是否劣化 | 速度场 MSE、动作平滑度、真机成功率 |
| B：Prefix KV cache 开 / 关 | 是否重复算观测 Prefix | 观察不变时是否省计算 | prefill 时延、单轮 flow 时延、总时延 |
| C：10 / 5 / 2 次积分 | ODE 步数 | 质量与推理延迟权衡 | 闭环成功率、轨迹偏差、GPU 耗时 |
| D：连续状态 Suffix / 离散状态 Prefix | 状态注入方式 | 条件适配和序列长度影响 | prompt 截断、归一化、稳定性 |
| E：π₀ 时间 Token MLP / π₀.₅ AdaRMS | 时间调制位置 | 多层条件注入是否有效 | 相同参数预算、训练曲线、OOD 成功率 |

**重要实验边界**：D/E 本来就可能改变多个参数和预训练权重，仅把 π₀ 换成 π₀.₅ 的 checkpoint 对比，无法单独归因于状态输入方式或 AdaRMS。必须保持初始化、数据、训练预算一致，或严格设计可解释的替代实验。

## 10. 与自动驾驶 BEV Query 的技术类比

在自动驾驶里，BEV Query 从 PV Feature 读取 K/V 是 Cross-Attention：Query 来自 BEV、Key/Value 来自图像；两种特征可以有不同原始隐藏宽度，但投影到兼容头维后才能做点积。

π₀ 联合注意力也类似：动作 Token 查询视觉/语言 Key/Value。**区别是这里还允许动作 Token 之间做同组自注意力**，且每个专家拥有独立的 QKV、输出投影与 FFN。不会因为动作 Query 能读视觉 Key，就断言它和 VLM 的 $W_Q$ 相同。

回到多教师蒸馏场景：如果需要验证“动作监督是否损害视觉语义”，要在共享视觉骨干输出处测梯度夹角，分清哪些梯度确实能回传到视觉特征、哪些因冻结/stop-gradient 而断开。不能只凭“Prefix 不读动作”就得出“动作损失绝不会训练视觉编码器”；Suffix 仍可通过注意力读取 Prefix，并可能把梯度回传到可训练的 Prefix 路径。**注意力可见性、参数共享和梯度冻结，是三个独立问题。**

## 11. 复现验收与源码导航

1. **配置核验**：读取默认模型变体，核对两个专家宽度、head 数、深度、动作容器和 pi05 开关。
2. **结构核验**：沿 gemma.py 的 Attention→Block→Module，确认两专家各自投影、拼接、分段输出、AdaRMS 分支。
3. **训练核验**：打印 $x_\tau,u_\tau,v_\theta$ 的形状，验证 $\tau=1$ 的数学约定与 loss 归约轴。
4. **Mask 核验**：跑上面的 NumPy 测试，再给随机 Padding，保证 Prefix 不能读未来动作。
5. **推理核验**：固定输入和随机种子，对比缓存与不缓存的输出数值和延迟；数值误差应在设定的浮点精度容差内。
6. **机器人核验**：动作需要按数据集配置完成归一化反变换、关节/末端参考系映射、碰撞/限位验证，再用真机做闭环评测。

本篇是**2026-10-25 对官方 JAX 代码的源码审计教程**，已经提供可以独立执行的 Mask 单测，但没有声称训练过同等参数量的 π₀，也没有声称获得真实机器人性能结果。

**全部一手资料**：[π₀ 原论文](https://arxiv.org/abs/2410.24164) · [openpi 固定提交](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479) · [源码 pi0.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py) · [源码 gemma.py](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/gemma.py) · [源码配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py) · [openpi 项目使用边界](https://github.com/Physical-Intelligence/openpi#readme)。
