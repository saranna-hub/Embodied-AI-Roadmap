---
title: 04｜生成模型、Diffusion 与 Flow Matching
sidebar:
  order: 4
---
# 为什么 π₀ 要从噪声逐步生成动作

## 1. 不止一种正确抓取动作

对同一个杯子，可以从左侧抓也可以从右侧抓。假设动作数值 -1 和 +1 都可行，普通 MSE 回归可能预测平均值 0，恰好对应不可行的中间动作。生成建模希望刻画条件动作分布，而非只给单个均值。

## 2. Diffusion 的直觉

训练时把真实数据加噪，学习去噪；推理时从噪声开始多次迭代得到样本。Diffusion Policy 将这种思想用于机器人动作序列。

## 3. Flow Matching 的核心公式

使用 openpi 常见时间方向：t=0 是数据，t=1 是噪声。

$$
x_t=(1-t)A+t\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

对 t 求导，直线路径的目标速度：

$$
u_t=\frac{dx_t}{dt}=\epsilon-A.
$$

网络在观测 o 条件下预测：

$$
\mathcal L_{\mathrm{FM}}=\mathbb E[\|v_\theta(x_t,t,o)-(\epsilon-A)\|_2^2].
$$

“速度场”在这里是**动作表示空间里的变化方向**，不等同于机器人末端物理速度（m/s）。

## 4. 手算一遍

真实动作 A=2，随机噪声 ε=-2，选择 t=0.25：

$x_t=0.75\times2+0.25\times(-2)=1$。

目标速度 $u=-2-2=-4$。网络要在给定 x_t、t、观测的情况下预测 -4。

## 5. 训练和推理到底有什么区别

训练知道真值 A，可以直接用 A 和随机噪声构造 x_t、u_t，算 MSE，不需要完整迭代。

推理不知道 A，从随机 x₁ 出发，沿负时间方向用 Euler 积分：

$$
x_{t-\Delta t}=x_t-\Delta t\,v_\theta(x_t,t,o).
$$

如果 Action Horizon=50，积分 K=10 次，每次更新的仍是**完整 50 步动作块**，不能把 K 和 H 混淆。

## 6. 代码练习

~~~python
import torch
B,H,D=2,50,32
A=torch.randn(B,H,D)
eps=torch.randn_like(A)
t=torch.rand(B,1,1)
xt=(1-t)*A+t*eps
target=eps-A
print(xt.shape,target.shape)
~~~

真正训练还需网络预测 v_theta；推理要在每个积分步重新调用网络，而不是直接拿 target 当预测结果。

## 7. 与机器人执行的边界

模型输出 [B,H,D] 还要经过反归一化、动作空间映射、机器人控制接口、限位和闭环重规划。EGO 人手视频如果没有经过转换验证的机器人动作标签，不能直接使用标准监督 Flow Matching 训练目标。

**自测**：t=1 的 x_t 是什么？是高斯噪声 ε，而不是机器人真实动作。