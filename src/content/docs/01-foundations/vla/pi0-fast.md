---
title: π₀-FAST：为什么要把连续动作变成离散 Token
sidebar:
  order: 2
description: 离散化难题、DCT 频域压缩、FAST+ 与自回归 VLA
---

**论文**：[FAST: Efficient Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2501.09747)（Pertsch 等，2025） · **官方 Tokenizer**：[physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) · **VLA 代码**：[openpi](https://github.com/Physical-Intelligence/openpi) · **状态**：方法和代码入口解读，未执行本地 tokenizer benchmark。

## 1. π₀ 已经可以生成动作，为什么还要 FAST？

π₀ 采用连续 Flow Matching，推理需要对整个 Action Chunk 进行多次速度场求解。另一条路线是将动作离散成 Token，让 Transformer 像生成文字那样自回归预测动作序列。

但机器人数据的动作通常是高频、连续且多维的。假设动作维度 14、预测 50 步，若对每个维度每个时间点都使用一个离散 Token，单个 chunk 就要处理大量 token；简单量化还会放大微小变化或损失精度。

FAST 的关键选择：**先将一整段动作变成频域中的紧凑表示，再离散化和压缩**，而不是原始逐坐标逐时刻分桶。

## 2. 从时域看频域：为什么 DCT 能压缩动作？

![FAST：连续动作转离散 Token](/Embodied-AI-Roadmap/diagrams/fast.svg)

*DCT 不是图像光流；信息损失主要来自量化与压缩。 图示为教学重绘。*

很多机器人轨迹具有时间连续性：手臂从左移动到右，大部分时间不会每帧随机跳变。这意味着在离散余弦变换（DCT）后，低频系数往往包含较大比例的能量。

对单个动作维度的时序 $a_0,\ldots,a_{H-1}$，DCT-II 的一类标准形式：

$$
C_k=\alpha_k\sum_{n=0}^{H-1}a_n\cos\left[\frac{\pi}{H}\left(n+\frac12\right)k\right].
$$

其中 $k$ 是频率索引，$\alpha_k$ 是归一化系数。缓慢变化的轨迹，常常可以主要用低频表示；快速接触事件需要保留相应高频成分。DCT 是一个**可逆线性变换**；真正的信息丢失来自量化和压缩，而不是仅仅做 DCT。

## 3. Tokenizer 的流程

```text
原始动作块 [H,D]
    ↓ 按训练数据的尺度归一化
DCT 频域变换
    ↓ 量化与序列化
离散动作 Token
    ↓ VLM 自回归预测
离散 Token 序列
    ↓ Tokenizer 解码 + 逆变换
连续动作块 [H,D]
```

它和“关节角每 1° 分一个 token”的简单方案不同：压缩时先利用动作序列自身的时间结构。**Tokenizer 的细节要与官方实现核对**，包括归一化、系数量化、序列编码及解码顺序。

## 4. 自回归 VLA 怎么训练？

设离散 Token 序列为 $z_1,\ldots,z_M$，观察条件为 $o$，通过 teacher forcing 优化：

$$
\mathcal L_{\mathrm{AR}}=-\sum_{i=1}^{M}\log p_\theta(z_i\mid o,z_{<i}).
$$

这与连续 Flow Matching 目标不同：网络预测的是下一个离散 token 的概率分布，损失通常是交叉熵；推理时依次采样/选择 token，最后通过动作 tokenizer 解码回连续轨迹。

**比较性能时要区分：**(1) tokenizer 本身的信息损失；(2) VLA 的 token 预测错误；(3) 生成 token 数与推理延迟。只看动作离散化精度不足以判断闭环机器人成功率。

## 5. FAST+ 有什么用？

作者发布通用 FAST+ Tokenizer，训练数据包含百万级真实机器人动作轨迹。官方 Hugging Face 提供 AutoProcessor 接口和训练自有 Tokenizer 的示例。作者报告在其实验条件下，FAST 路线可以与连续扩散 VLA 性能接近，并降低部分训练设置的耗时；这属于论文实验结论，不意味着任何任务都能降低同样比例成本。

## 6. π₀ 与 π₀-FAST 应该怎么比较？

| 维度 | π₀ | π₀-FAST |
| --- | --- | --- |
| 动作表示 | 连续动作向量 | 频域压缩后的离散 Token |
| 训练目标 | Flow Matching 速度场 | 自回归 Token Cross Entropy |
| 推理方式 | 数值积分多次更新整个动作块 | 自回归预测 Token，再解码动作 |
| 主要成本 | Flow 迭代次数与专家计算 | Token 长度、解码速度及离散化误差 |
| 闭环重点 | 轨迹与采样时延 | 量化精度与 token 推理时延 |

这两个模型不能只根据“单次前向速度”得出谁更好。真正要结合任务动作频率、动作块长度、控制延迟以及真机成功率。

## 7. 最小代码验证建议

先使用 [FAST 官方 Tokenizer](https://huggingface.co/physical-intelligence/fast) 在归一化动作序列上做 encode→decode，测 MAE/RMSE。构造三类输入：平滑轨迹、频繁突变轨迹和接触瞬时跳变轨迹。观察量化误差随时间的分布，再考虑在 VLA 里用生成模型预测这些 Token。

测试时至少报告：每条动作块的 Token 数量、编码/解码耗时、量化误差和动作物理尺度。未经实测不应写“FAST 对接触任务零损失”。

来源：[论文](https://arxiv.org/abs/2501.09747) · [官方 FAST+ 模型](https://huggingface.co/physical-intelligence/fast) · [openpi](https://github.com/Physical-Intelligence/openpi)。
