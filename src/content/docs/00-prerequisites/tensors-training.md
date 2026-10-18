---
title: 01｜张量、线性代数、神经网络和反向传播
sidebar:
  order: 1
---
# 张量和训练到底在做什么

## 1. 从机械专业的直觉开始

机器人有传感器，控制器有输入输出。神经网络同样如此，只不过在输入到输出之间有大量从数据学出的参数。

**Tensor（张量）**可以先视为多维数字容器：

| 数据 | 形状 | 解释 |
| --- | --- | --- |
| 末端位置 | [3] | x、y、z |
| RGB 相机 | [3,224,224] | 通道、高、宽 |
| 2 个样本、3 路相机 | [2,3,3,224,224] | Batch、相机、RGB、高、宽 |
| 未来动作块 | [2,50,32] | Batch、未来动作时刻、动作维度 |

注意：50 是 Action Horizon；32 是通用动作容器维度，不一定是真实关节数量。

## 2. 为什么神经网络离不开矩阵

线性层：

$$
y=Wx+b.
$$

假设输入 x=[1,2,3]，W 的两行是 [1,0,2]、[0,1,-1]，偏置是 [0,1]。那么：

第一维输出 = 1×1+0×2+2×3 = 7。

第二维输出 = 0×1+1×2−1×3+1 = 0。

三维输入就被变成二维输出 [7,0]。训练的关键是**让 W 和 b 根据数据自动调整**。

## 3. Loss 是什么

假设线性模型预测 $\hat y=wx+b$，样本输入 x=2，目标 y=5，而 w=1、b=0。则预测 2，平方误差：

$$
L=(wx+b-y)^2=9.
$$

损失衡量“现在预测有多差”，但不会自己修改参数。

## 4. 为什么 Backward 能更新权重

链式法则：

$$
\frac{\partial L}{\partial w}=2(wx+b-y)x=-12,
$$

$$
\frac{\partial L}{\partial b}=2(wx+b-y)=-6.
$$

学习率 0.1，梯度下降更新：

$$
w'=w-0.1(-12)=2.2,\qquad b'=0.6.
$$

此时预测 = 2.2×2+0.6=5。这是刻意设计的单样本例子，真实深度网络不一定一步收敛。

| 概念 | 作用 |
| --- | --- |
| Forward | 计算预测 |
| Loss | 计算预测与目标误差 |
| Backward | 计算损失对参数的梯度 |
| Optimizer | 根据梯度更新参数 |
| Validation | 检验没见过的数据是否也有效 |

## 5. 为什么需要激活函数和残差

纯线性层连续相乘，本质仍是线性变换。ReLU / GELU / SiLU 让模型具备非线性能力；残差 y=x+F(x) 为信息与梯度提供更直接的通路。

## 6. 用 Python 验证梯度

~~~python
import torch
w=torch.tensor(1.,requires_grad=True)
b=torch.tensor(0.,requires_grad=True)
x=torch.tensor(2.)
y=torch.tensor(5.)
loss=(w*x+b-y)**2
loss.backward()
print("loss=",loss.item())
print("dw=",w.grad.item(),"db=",b.grad.item())
~~~

应输出 loss=9、dw=-12、db=-6。

## 7. 为什么多教师蒸馏会出现梯度拉扯

两项监督损失对同一个 Backbone 特征产生 g1 和 g2：

$$
g=\lambda_1 g_1+\lambda_2 g_2.
$$

余弦相似度：

$$
\cos\theta=\frac{g_1^\top g_2}{\|g_1\|\|g_2\|}.
$$

接近 1 表示方向相似，-1 表示相反，0 表示近似正交。**正交不能直接断言“强冲突”**，还要看模长、变化分布以及主任务效果。

## 8. 广播和 Action Chunk 示例

~~~python
import torch
A=torch.randn(4,50,32)
eps=torch.randn_like(A)
t=torch.rand(4,1,1)
xt=(1-t)*A+t*eps
print(xt.shape)
layer=torch.nn.Linear(32,128)
print(layer(xt).shape) # [4,50,128]
~~~

这里 t 是每个样本的一个标量，广播到 50×32；线性层只改变最后一个特征轴。

**自测**：loss.backward() 会自动改变模型参数吗？不会，通常还需要 optimizer.step()。