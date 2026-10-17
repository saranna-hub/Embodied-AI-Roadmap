---
title: 02｜Transformer：从 QKV 到注意力与 KV Cache
sidebar:
  order: 2
---
# Transformer 深度教程

## 1. Transformer 为何适合视觉语言机器人

指令“把红色杯子放进托盘”，需要模型知道“红色”修饰哪件物体、“放进”对应怎样的空间关系。注意力机制让一个 token **按相关性聚合其他 token 的信息**。

![QKV 与注意力计算流程](/Embodied-AI-Roadmap/diagrams/prerequisites-attention.svg)

**Token** 可以是文本符号，也可以是视觉 patch 或动作时间步的特征。Embedding 是 token 的向量表示。

## 2. Q、K、V 用查资料理解

- Query：我在寻找什么？
- Key：每条候选信息提供什么索引？
- Value：真正读取并混合的信息。

同一个输入 X 经过三个不同的线性投影：

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

注意力公式：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

操作顺序：计算 QK 点积得分 → 除以维度缩放因子 → 加 Mask → Softmax 变权重 → 加权求和 V。

## 3. 手算最小例子

设一个 Query 对两个 Key 的缩放后得分是 2 和 0。Softmax：

$$p_1=\frac{e^2}{e^2+1}\approx0.881,\quad p_2\approx0.119.$$

若两个 Value 分别为 [10,0] 与 [0,10]：

$$o=0.881[10,0]+0.119[0,10]=[8.81,1.19].$$

模型不是只选“最相关的那一个”，而是按照权重融合信息。

## 4. Multi-Head、Norm、MLP 与残差

多头注意力让不同的参数投影在不同子空间计算关系，拼接后由输出层融合。典型 Pre-Norm Transformer Block：

$$Y=X+\operatorname{MHA}(\operatorname{Norm}(X)),$$

$$Z=Y+\operatorname{MLP}(\operatorname{Norm}(Y)).$$

现代模型的 Norm、激活和位置编码实现可能不同。位置编码（例如 RoPE）帮助注意力处理 Token 之间的顺序/相对位置。

## 5. Mask 决定谁可以看谁

| 模式 | 规则 |
| --- | --- |
| 双向注意力 | 可读取当前分组内所有有效 Token |
| Causal | 只能读取过去与当前 |
| Padding | 屏蔽无效填充位置 |
| Prefix/Suffix | 按组决定视觉语言和动作之间可见性 |

对 π₀，视觉语言 Prefix 不应依赖每次变化的带噪动作 Suffix；动作侧则可以读取 Prefix 条件。真实规则还包括组边界和 padding，需以 openpi 源码为准。

## 6. KV Cache 的真正收益

固定部分的 K/V 可以缓存，避免重复投影和计算。对 π₀，固定观测的 Prefix 信息可以考虑复用；但动作 Suffix 在每次 Flow Matching 积分中不断变化，**动作网络仍需重复计算**。

## 7. PyTorch 最小演示

~~~python
import torch
q=torch.tensor([[[1.,0.]]])
k=torch.tensor([[[1.,0.],[0.,1.]]])
v=torch.tensor([[[10.,0.],[0.,10.]]])
score=q@k.transpose(-1,-2)/(2**0.5)
prob=torch.softmax(score,dim=-1)
print("attention weights:",prob)
print("output:",prob@v)
~~~

修改 q 为 [0,1]，观察更偏向哪个 Value；将第二个得分设为 -1e9，就相当于屏蔽第二个 Key。

## 8. 与 π₀ 双专家对应

PaliGemma 和 Action Expert 使用不同参数处理不同 Token，通过联合注意力交换条件信息；不是 Router Top-k MoE，也不是 VLM 先生成一段文字让独立控制器执行。

**自测**：Attention 权重是不是训练梯度？不是。前者是 Forward 的信息混合系数，后者是 Loss 的导数。