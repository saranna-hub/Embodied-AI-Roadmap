---
title: ACT：Action Chunking Transformer 与 CVAE
sidebar:
  order: 2
description: 从双臂示范到动作块、潜变量、KL 散度和时序平滑
---

**论文**：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)（Zhao 等，2023） · **项目**：[ALOHA](https://tonyzhaozh.github.io/aloha/) · **代码**：[tonyzhaozh/act](https://github.com/tonyzhaozh/act) · **状态**：论文/源码讲解，未由本站进行独立训练复现。

## 1. 为什么每帧只回归一个动作容易失败？

机器人要把电池插进插槽。若策略每看到新图像就单独决定下一帧关节目标，很容易出现“移动—停顿—修正—移动”的抖动，导致接触任务失败。此外，示范动作的微小误差会在长序列中积累。

ACT 把预测单位从“下一步动作”改为**未来一整段动作**（Action Chunk）。它进一步用 Transformer CVAE 学习示范中的不同动作风格，在部署时基于图像和关节状态生成连续控制序列。

## 2. 训练和推理为什么是两条不同路径？

![ACT：训练与推理共用动作解码器](/Embodied-AI-Roadmap/diagrams/act.svg)

*推理时无真实动作，CVAE Encoder 不运行；z 从先验选择。 图示为教学重绘。*

**训练路径**：相机观测、关节状态和真实未来动作进入模型。CVAE 编码器看到真实动作，形成潜变量 $z$ 的后验分布；Transformer 解码器在视觉和关节条件下预测整段动作。

**推理路径**：不再输入真实未来动作，也不运行训练时的 CVAE 编码器。官方原方案使用先验均值（通常设为零）作为 $z$，由解码器预测动作块。

```text
                  训练时真实动作 A
                         │
当前关节 q ────────→ CVAE Encoder ─→ (μ, σ) ─→ z
     │                                          │
四路 RGB → CNN / 视觉特征 ─→ Transformer Decoder ─→ Action Chunk

推理时：图像 + 当前关节 + z=0 → Transformer Decoder → Action Chunk
```

这与 Diffusion Policy 不同：ACT 不在推理时迭代去噪整段动作，通常一次生成一个 chunk；CVAE 的分布与 KL 正则主要出现在训练中。

## 3. 输入和输出如何组织？

对双臂关节控制可将当前状态写为 $q_t\in\mathbb{R}^{D_s}$，未来 $H$ 步动作块为：

$$
A_t=[a_t,a_{t+1},\ldots,a_{t+H-1}]\in\mathbb{R}^{H\times D_a}.
$$

机器人视觉可以有多路相机，但每个相机的输入语义必须稳定。若训练时左腕/右腕相机交换，视觉特征可能和动作意义不匹配。论文里的 ALOHA 示例使用固定/腕部等多视角；它们在单任务环境中的效果不能直接代表所有机器人系统。

## 4. CVAE 的两个 Loss 是什么？

**动作重建损失**鼓励模型的预测块接近真实示范，**KL 散度**让潜变量后验靠近标准正态先验。常见写法：

$$
\mathcal{L}=\underbrace{\|\hat A-A\|_1}_{\mathrm{reconstruction}}+\beta\underbrace{D_{\mathrm{KL}}(q_\phi(z\mid A,q_t)\|p(z))}_{\mathrm{regularization}}.
$$

这里 $p(z)=\mathcal N(0,I)$，$\beta$ 是 KL 权重；实际实现的 mask、归一化、逐帧聚合方式，应查看源码。为什么需要 KL？因为推理阶段得不到真实未来动作，无法依赖编码器计算特殊后验；让训练中的 $z$ 分布接近先验，才能在推理时用先验生成稳定结果。

## 5. 动作块重叠后怎么办？

即使单次输出一段动作，下一次重规划也会生成另一段，两段对应时刻可能产生不同预测。ACT 使用 **Temporal Ensembling** 思路，对同一实际时刻的多个预测做加权融合，而不是生硬地将两个 chunk 首尾拼接。这样通常能够减轻接缝处的不连续，但也可能引入延迟，需要结合任务速度与接触条件评估。

## 6. 为什么适合新人先学？

它把几个关键概念放进同一个可运行任务：行为克隆、CVAE、Transformer Encoder/Decoder、Action Chunk 和闭环执行。比直接研究十亿参数 VLA 的多阶段训练更容易定位错误。

官方源码建议按顺序看：[`imitate_episodes.py`](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)（数据和训练循环）→ [`policy.py`](https://github.com/tonyzhaozh/act/blob/main/policy.py)（损失和策略接口）→ [`detr`](https://github.com/tonyzhaozh/act/tree/main/detr)（模型）→ [`sim_env.py`](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)（仿真）→ [`visualize_episodes.py`](https://github.com/tonyzhaozh/act/blob/main/visualize_episodes.py)（示范可视化）。

## 7. 最小复现实验如何设计？

官方仓库有模拟 Transfer Cube 和 Bimanual Insertion 任务以及记录示范、训练和评测脚本。不要盲目照搬旧 CUDA/PyTorch 版本；首先阅读官方依赖，并在隔离环境运行。为了理解架构，固定训练集分别实验：chunk size = 1、chunk size = 50、是否启用 temporal ensemble；记录成功率、平均完成时间、关节速度抖动。

只做模型训练 loss 对比还不够，最终要跑 closed-loop rollout。原论文报告了多个精细双臂操作任务的真机成功案例，但其硬件与测试条件跟工业轮式双臂不同。

**与其他模型关系**：[Diffusion Policy](./diffusion-policy/) 建模整个动作分布但采样方式不同；[π₀](../vla/pi0/) 将生成式连续动作头与预训练 VLM 结合。把 ACT 理解为“动作块与 CVAE”的组合，而不是一个 Diffusion Policy 版本。

来源：[论文](https://arxiv.org/abs/2304.13705) · [官方项目](https://tonyzhaozh.github.io/aloha/) · [官方代码](https://github.com/tonyzhaozh/act)。
