---
title: 02.2｜QKV 和 Self/Cross-Attention：完整推导
sidebar:
  order: 6
---
# Attention 从输入、权重、形状到输出

## 1. 它为什么被提出？

循环神经网络要按时间依次处理 Token，长距离关系传递路径较长；Self-Attention 让 token 在单层里直接读取序列中其他允许访问的位置。它不是无条件优于 RNN，注意力矩阵的计算量和存储量也随序列长度增长。

## 2. 三份输入分别代表什么？

假设有两个视觉 Token：红色杯子和托盘；一个动作 token 需要判断接下来如何接近杯子。Query 描述**当前需要哪类信息**，Key 描述**各候选能够提供哪些索引**，Value 提供**真正被融合的特征**。

$$
Q=X_qW_Q,\quad K=X_kW_K,\quad V=X_vW_V.
$$

Self-Attention 中通常 $X_q=X_k=X_v=X$，但 WQ、WK、WV 可以不同。Cross-Attention 中 $X_q$ 可以来自动作侧，$X_k,X_v$ 来自视觉侧。**来源相同并不意味着投影矩阵相同。**

## 3. 手算一个完整例子

取 Query q=[1,0]，Key k1=[1,0]、k2=[0,1]，Value v1=[10,0]、v2=[0,10]。

匹配得分 [1,0]。除以 sqrt(2)，得到 [0.7071,0]。Softmax 后约 [0.6698,0.3302]。

最终输出：

$$
o=0.6698[10,0]+0.3302[0,10]=[6.698,3.302].
$$

这里得到的是**混合后的新特征**，不是“第一目标命中概率”。接下来网络还要用 MLP、残差、其他 Transformer 层处理它。

## 4. 逐层检查形状

| 操作 | 输入 | 输出 |
| --- | --- | --- |
| 输入投影 | Xq:[B,Lq,d] | Q:[B,Lq,dk] |
| 键投影 | Xk:[B,Lk,d] | K:[B,Lk,dk] |
| 值投影 | Xv:[B,Lk,d] | V:[B,Lk,dv] |
| 匹配得分 | Q×Kᵀ | [B,Lq,Lk] |
| Softmax | 最后一维 Lk | [B,Lq,Lk] |
| 加权值 | P×V | [B,Lq,dv] |

**为什么 Softmax 沿 Key 维做？** 每一个 Query 需要在可见的全部 Key 中分配权重；沿 Query 维做意味着所有查询之间争夺权重，那不是标准缩放点积注意力。

## 5. 自己写出 Self-Attention

~~~python
import math
import torch
from torch import nn

B,L,D=2,4,8
x=torch.randn(B,L,D)
q=nn.Linear(D,D)(x)
k=nn.Linear(D,D)(x)
v=nn.Linear(D,D)(x)
score=q@k.transpose(-2,-1)/math.sqrt(D)
weight=torch.softmax(score,dim=-1)
out=weight@v
print(score.shape,weight.shape,out.shape)
# [2,4,4] [2,4,4] [2,4,8]
~~~

这只是单头前向演示。完整可训练网络应把三个 Linear 定义在 Module 内，否则每次调用都重建权重。

## 6. Cross-Attention 如何用于 BEV / PV？

例如有 100 个 BEV Query，来自视觉 Backbone 的 PV token 有 400 个。Q 为 [B,100,d]，K/V 为 [B,400,d]，交互矩阵是 [B,100,400]。它说明每个 BEV 查询读取哪些图像特征，不意味着 PV 特征会自动变成 100×400 张物理概率图。

## 7. 为什么多头有价值？

8 个头不是把同一张矩阵复制 8 次，而是使用不同的投影参数，可能学习不同的相关性；“头一管颜色、头二管距离”只能作为直觉，不能当成论文已验证结论。

**下一课**：[Transformer Block 内部结构](./transformer-03-block/)。
