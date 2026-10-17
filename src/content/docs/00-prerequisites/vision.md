---
title: 03｜视觉编码器、ViT、SigLIP 和 VLM
sidebar:
  order: 3
---
# RGB 图像怎么变成语言模型能读的 Token？

## 1. 从像素到 Patch

一张 RGB 图像是三通道数字矩阵。Vision Transformer（ViT）把图像划成小块 Patch，再映射为向量序列。

例如 224×224 图片，Patch Size=16，不计算特殊 Token：

$$N_{\mathrm{patch}}=(224/16)^2=196.$$

每个 Patch 有 16×16×3=768 个数。模型使用线性投影将其映射到隐藏维度，再加位置关系，经 Transformer 处理。

## 2. CNN 与 ViT

CNN 天然适合局部纹理提取，具有卷积的局部归纳偏置。ViT 将 Patch 作为 Token，用 Attention 捕捉长程依赖。不能脱离训练规模、分辨率与任务直接宣布一种方法更优。

## 3. CLIP、SigLIP 为什么懂语义

图文对齐使“红色杯子”的图片特征与对应文本特征接近。CLIP 常用 batch 内 softmax 对比损失，SigLIP 采用成对 sigmoid 损失。它们学习视觉语义**特征**，但不意味着直接输出目标框、物理深度或机器人轨迹。

## 4. PaliGemma 是什么

PaliGemma 结合视觉编码组件与 Gemma 语言 Transformer。图像生成 Vision Tokens，文字生成 Text Tokens，两者提供联合上下文。π₀ 利用该预训练视觉语言能力，让 Action Expert 在视觉语言条件下学习连续动作。

## 5. 视觉 Backbone、蒸馏和任务头

Backbone 输出 Feature，不等于最后的分割/深度。分割头把 Feature 映射到像素类别，深度头映射到几何目标。多教师蒸馏需考虑 Token 数、特征维度、规范化与梯度方向，单纯投影到相同维度不等于学到了同样的语义。

## 6. Python：最简单的 Patch 投影

~~~python
import torch
from torch import nn
image=torch.randn(2,3,224,224)
patch_embed=nn.Conv2d(3,64,kernel_size=16,stride=16)
feature=patch_embed(image)
tokens=feature.flatten(2).transpose(1,2)
print(tokens.shape) # [2,196,64]
~~~

这只是 Patch Embedding 教学，不是完整 SigLIP 网络。

## 7. 联系 π₀

图像 Tokens 与语言 Tokens 形成 Prefix，机器人状态和带噪动作进入 Suffix。视觉语言表示通过受控注意力参与动作生成，并不是先让 VLM 回答中文问题再转换成机器人指令。

**自测**：196 个 Patch Token 是否意味着有 196 个动作自由度？不意味着。视觉序列长度和控制维度是完全不同的量。