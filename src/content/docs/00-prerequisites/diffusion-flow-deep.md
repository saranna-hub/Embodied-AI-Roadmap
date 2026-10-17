---
title: 05｜Diffusion、Flow Matching、ODE：从零到 π₀
sidebar:
  order: 10
---
# 为什么生成动作不是简单的动作回归？

## 1. 区分概率建模、训练监督和采样算法

一个策略可以使用 Transformer 作为网络结构，使用 Flow Matching 作为训练监督，使用 Euler ODE 积分作为推理算法。三者解决不同问题，不能把 Flow Matching 当作“另一种 Attention”。

## 2. 为什么需要数据分布？

同一个观测下可能有多条抓取路径，生成模型希望建模动作的条件分布 $p(A\mid o)$。单一 MSE 最优常数预测通常是条件均值，可能落在两个操作模式之间，形成不可执行动作。

## 3. 直线概率路径从哪里来？

取真实动作块 $A\in\mathbb R^{H\times D}$ 与独立采样高斯噪声 $\epsilon$，令 t 从 0 到 1：

$$
x_t=(1-t)A+t\epsilon.
$$

当 t=0，$x_t=A$；当 t=1，$x_t=\epsilon$。对 t 求导得到目标条件速度：

$$
u_t=\epsilon-A.
$$

给定混合点、时间和观测条件，训练网络估计速度：

$$
\mathcal L_{\mathrm{FM}}=\mathbb E_{A,\epsilon,t}\left[\|v_\theta(x_t,t,o)-u_t\|^2\right].
$$

注意：样本条件速度 $\epsilon-A$ 和给定 $x_t,t,o$ 后模型学习到的最优平均向量场不必逐样本相同。**不能把“单条直线路径”错误推断成推理轨迹一定是直线。**

## 4. 为什么推理要反向积分？

按上面的时间约定，从 t=1 的噪声走回 t=0 的数据，Euler 更新为：

$$
x_{t-\Delta t}=x_t-\Delta t\,v_\theta(x_t,t,o).
$$

每一步都会重新以更新后的 x 预测速度。积分步 K 与动作长度 H 是不同维度：例如 K=10，每次更新完整 [B,50,32]。

## 5. 和 Diffusion Policy 到底哪里不同？

DDPM 一类 Diffusion 通常通过加噪马尔可夫过程和学习反向去噪过程建模分布；Flow Matching 学习连接噪声分布与数据分布的向量场，可以配合 ODE 数值积分。不能以一句“Flow 更快”代替准确比较；采样器、步数、网络、硬件和任务都影响实际延迟。

## 6. 极简训练构造代码

~~~python
import torch
B,H,D=4,50,32
A=torch.randn(B,H,D)
eps=torch.randn_like(A)
t=torch.rand(B,1,1)
xt=(1-t)*A+t*eps
target=eps-A
assert xt.shape==target.shape==(B,H,D)
~~~

这里构造监督，但没有实现 VLA 条件网络；不能把它叫做 π₀ 复现。

## 7. 把采样和机器人执行分开

动作块输出后需要反归一化、坐标变换和执行端安全校验。离线生成质量不代表碰撞安全。实时部署还要统计感知延迟、模型积分延迟、消息传输和控制器频率。

**自测**：为什么训练不需要沿 ODE 完整积分？因为可以从真值 A 与噪声 ε 直接采样中间点和目标速度。
