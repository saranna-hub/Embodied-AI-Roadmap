---
title: RDT-1B：面向双臂操作的 Diffusion Transformer
sidebar:
  order: 4
description: 1B 级机器人扩散模型、多视角条件、动作块与跨本体数据
---

**论文/项目**：[RDT-1B](https://rdt-robotics.github.io/rdt-robotics/) · **官方实现**：[thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) · **状态**：论文与官方 README 核验，未本地复现。

## 1. RDT 解决的不是“能否抓起来”，而是大量机器人数据如何共享

单任务行为克隆可能只学会一个固定工位的示范，难以复用其他机器人和操作任务的数据。RDT-1B 使用大容量 Diffusion Transformer，在异构机器人示范中学习通用动作先验，再针对具体机器人/任务微调。官方仓库介绍模型规模约 1B 参数，预训练涉及百万级多机器人 episodes，并可输入语言和最多三路 RGB，生成未来 **64 步动作**。这些是官方技术披露，不是本站实测的吞吐和泛化保证。

## 2. 模型数据流

![RDT-1B：多模态条件到动作](/Embodied-AI-Roadmap/diagrams/rdt.svg)

*预测动作步数和扩散采样迭代次数不是同一数值。 图示为教学重绘。*

```text
语言指令 ──────────┐
Camera 1 / 2 / 3 ─┤── 条件编码
Robot proprioception┘       │
                            ▼
随机动作序列 + diffusion time
             │
   Diffusion Transformer
             │
        未来动作块
             │
      控制器 / 重规划
```

具体特征编码器、条件融合、动作投影和训练噪声目标需以论文与官方配置逐项核对。可以先把目标抽象成条件分布 $p_\theta(A_t\mid o_t)$，其中 $A_t\in\mathbb{R}^{H\times D_a}$ 是连续动作块；官方仓库所指 64 步，是模型的时间预测长度，不等于扩散去噪步数。

## 3. 多本体“统一”究竟统一了什么？

来自 ALOHA、单臂机械臂、轮式移动机械臂的动作空间可能不同，状态甚至包含不同维度和参考坐标。为利用异构数据，需要**动作语义映射、维度掩码与归一化**。统一的张量容器只是数学实现；真正的机器人可执行性还取决于数据集怎样定义控制量、末端工具坐标与控制接口。

假设左/右臂输出分别为 $a_t^L,a_t^R$，双臂动作可抽象为：

$$
a_t=\operatorname{concat}(a_t^L,a_t^R,a_t^{\mathrm{base}}).
$$

这只是便于理解的动作接口设计范例，不代表 RDT 原论文对所有机器人都强制采用这样的分量顺序。

## 4. 扩散动作生成与 π₀ 的差别

RDT-1B 使用 Diffusion Transformer 预测高维时序动作；π₀ 使用预训练 VLM + Action Expert，并以 Flow Matching 学习连续动作生成。二者都能表达多模态分布，但网络内部信息交换、训练目标和推理求解器不同。实际比较时应固定任务、数据、动作空间和推理预算，而不是比较论文报告的不同任务成功率。

## 5. 源码和工程实践

官方仓库包含训练/采样脚本、DeepSpeed 训练工具、ManiSkill 仿真评测及 ALOHA 部署示例。[官方 README](https://github.com/thu-ml/RoboticsDiffusionTransformer) 对数据与依赖有详细说明。建议按“数据生成器→特征编码→主模型 forward→噪声调度→训练循环→推理接口”阅读，再调试某个 batch 的完整形状。

实验建议：同一操作任务上，固定训练数据和机器人本体，分别调整 action horizon、扩散采样次数与不同相机缺失模式，记录闭环成功率、推理耗时、动作平滑性。未实际执行的实验只标为建议。

## 6. 已知局限

大容量模型本身不能保证解决视觉遮挡或接触力控制，且训练在不同数据集上的时钟、相机布置与动作语义不一致会引入系统性噪声。研究轮式双臂时，首先验证能否映射到实际控制器动作空间，再考虑迁移规模化策略。

来源：[官方项目/论文入口](https://rdt-robotics.github.io/rdt-robotics/) · [官方代码与权重](https://github.com/thu-ml/RoboticsDiffusionTransformer)。
