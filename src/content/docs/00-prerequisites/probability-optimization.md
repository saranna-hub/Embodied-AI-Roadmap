---
title: 03｜概率、分布、最大似然与 AdamW 优化器
sidebar:
  order: 8
---
# 从概率论和最优化理解机器人生成模型

## 为什么这门课是 VLA 的必修课

给定同一视觉观测，机器人可以采用左侧或右侧抓取。仅预测动作均值无法描述多峰动作分布。Flow Matching、Diffusion、行为克隆、交叉熵都和概率建模有关系。

## 1. 随机变量、期望、方差

设 X 取 -1 或 +1，概率各为 0.5。期望：

$$
\mathbb E[X]=0.5(-1)+0.5(1)=0.
$$

方差：

$$
\operatorname{Var}(X)=\mathbb E[(X-\mathbb E[X])^2]=1.
$$

**关键**：期望 0 不代表实际示范出现 0。多模态动作下，直接做确定性均方误差可能预测折中的无效动作。

## 2. 高斯分布的作用

标准高斯：

$$
\epsilon\sim\mathcal N(0,I).
$$

这里均值为 0、各维协方差由单位矩阵描述。π₀ 训练使用随机噪声与实际动作插值；推理从噪声初始化动作块。它不是“机器人执行时随机给关节施加噪声”。

## 3. 极大似然与交叉熵

若某动作类别正确标签为 y，模型输出概率 p，分类交叉熵：

$$
\mathcal L_{\mathrm{CE}}=-\log p_y.
$$

例如正确类别概率 0.9，则损失约 0.105；概率 0.1 则损失约 2.303。交叉熵常用于离散 Token 分类，但连续动作生成可以使用其他目标，例如高斯负对数似然、Diffusion 噪声回归或 Flow Matching 速度回归。

## 4. SGD 为什么会越过最优点？

梯度下降：

$$
\theta_{k+1}=\theta_k-\eta \nabla_\theta L(\theta_k).
$$

学习率 η 决定步长。对于损失 $L(w)=w^2$、梯度 2w：

如果 w=1、η=0.1，新 w=0.8；如果 η=1.5，新 w=-2，反而离最优值 0 更远。学习率大不必然训练快。

## 5. Adam / AdamW 的内部逻辑

Adam 跟踪梯度的一阶动量和平方梯度的二阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

经偏差修正后按 $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ 更新。AdamW 将权重衰减与梯度更新解耦，通常比直接把 L2 正则加进 Adam 更新更符合明确的 weight decay 设计。

## 6. Warmup、Batch、混合精度与梯度裁剪

学习率 Warmup 通常使优化器早期逐渐增大学习率；扩卡后是否放大学习率，不仅取决于卡数，还取决于全局 Batch、优化器、数据顺序和梯度累积。

混合精度能降低资源开销，但需要注意数值范围；梯度裁剪通过限制梯度范数来缓解过大更新，并不能从根源消除数据错误。

## 7. 可运行代码

~~~python
import torch
w=torch.nn.Parameter(torch.tensor([1.0]))
optimizer=torch.optim.AdamW([w],lr=0.1,weight_decay=0.01)
for i in range(5):
    optimizer.zero_grad()
    loss=w.square().sum()
    loss.backward()
    optimizer.step()
    print(i,round(w.item(),4),round(loss.item(),4))
~~~

**思考**：如果把不同监督的 loss 简单相加，但两项梯度方向几乎相反，Optimizer 能自动解决任务冲突吗？不能保证；需检查权重、梯度以及目标任务收益。
