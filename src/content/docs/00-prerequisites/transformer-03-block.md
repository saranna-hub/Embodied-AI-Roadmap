---
title: 02.3｜Transformer Block、RoPE、Mask 与梯度
sidebar:
  order: 7
---
# 注意力之后，到底还计算了什么？

## 1. Transformer Block 不能只画一个 Attention 方块

常见的 Pre-Norm 结构：

$$
H=X+\operatorname{MHA}(\operatorname{Norm}(X)),
$$

$$
Y=H+\operatorname{MLP}(\operatorname{Norm}(H)).
$$

第一条残差是注意力分支，第二条残差是 FFN 分支。Attention 实现 token 间交互，MLP 实现每个 token 内部非线性特征变换；归一化调节尺度，残差提供恒等映射路径。

## 2. 多头 Attention 的张量过程

设 B=2、L=32、隐藏维度 D=512、头数 h=8。

1. Q/K/V 投影：三个 [2,32,512]。
2. 拆头为 [2,8,32,64]。
3. QKᵀ 得 [2,8,32,32]。
4. Softmax 后与 V 相乘，得 [2,8,32,64]。
5. 头拼接回 [2,32,512]。
6. 输出投影和残差相加，仍是 [2,32,512]。

注意：[2,8,32,32] 是注意力权重张量，不是输出 Feature，更不是 8×32² 个模型参数。

## 3. FFN 为什么存在？

在典型 MLP 中：

$$
f(x)=W_2\sigma(W_1x+b_1)+b_2.
$$

例如最后一维从 512 扩展到 2048，再缩回 512。没有激活函数，两次线性映射可以合并为一次线性映射；加入非线性才能表达更复杂的变换。现代大模型常采用门控 FFN，例如 SwiGLU。

## 4. LayerNorm 和 RMSNorm

LayerNorm 对每个 token 的特征进行去均值和方差归一化；RMSNorm 主要除以均方根。它们都含可学习缩放参数的常见实现，但不是同一个公式。具体使用哪个，要以模型配置和源码为准。

## 5. RoPE 为什么比单纯记位置更复杂？

Q/K 相对位置可能影响匹配程度。RoPE 对两两配对的维度进行位置相关旋转：

$$
R(\alpha)=\begin{bmatrix}
\cos\alpha&-\sin\alpha\\
\sin\alpha&\cos\alpha
\end{bmatrix}.
$$

对位置 m 的向量旋转 mθ；位置 n 的另一个向量旋转 nθ，二者点积可体现位置差 n−m。真实实现有多组频率，不能用这一个二维矩阵冒充完整 RoPE。

## 6. Mask 与模型执行权限

| Query / Key | P1 | P2 | A1 | A2 |
| --- | --- | --- | --- | --- |
| P1 | 可 | 可 | 禁 | 禁 |
| P2 | 可 | 可 | 禁 | 禁 |
| A1 | 可 | 可 | 可 | 可 |
| A2 | 可 | 可 | 可 | 可 |

这是**教学简化的 Prefix/Suffix Mask**。P 是视觉语言 Prefix，A 是动作 Suffix；具体 π₀ 源码的分组与 padding 规则可能更细。Mask 在分数矩阵中禁止位置加足够大的负数，令 Softmax 权重近零。

如果某个 Query 的所有 Key 都被禁止，可能导致 Softmax 不稳定甚至 NaN，需要特殊处理。

## 7. 残差如何影响梯度？

设 $Y=X+F(X)$，则局部雅可比为：

$$
\frac{\partial Y}{\partial X}=I+\frac{\partial F}{\partial X}.
$$

恒等分支缓解梯度传播困难，但不是所有梯度问题都能解决。Attention 中 Loss→输出→V、Softmax→Q/K→投影矩阵的梯度链条仍存在。

## 8. 简单可运行的 Block

~~~python
import torch
from torch import nn
class Block(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.n1=nn.LayerNorm(d)
        self.attn=nn.MultiheadAttention(d,4,batch_first=True)
        self.n2=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Linear(256,d))
    def forward(self,x):
        z=self.n1(x)
        x=x+self.attn(z,z,z,need_weights=False)[0]
        return x+self.ffn(self.n2(x))
x=torch.randn(2,10,64)
print(Block()(x).shape) # [2,10,64]
~~~

## 9. 对接 π₀：哪些参数共享？

π₀ 视觉语言专家和动作专家可采用不同的 QKV、FFN、Norm 等参数，通过受控联合注意力进行条件交互。**“联合注意力”不等于两侧共用全部投影矩阵，也不是最后多加一层 Cross-Attention。**

**继续**：[π₀ 训练结构](../01-foundations/vla/pi0/)。
