---
title: GR00T N1：双系统 VLA 怎样生成动作
sidebar:
  order: 3
description: Vision-Language 系统、Diffusion Transformer、异构训练数据与动作接口
---

**原始论文**：[GR00T N1: An Open Foundation Model for Generalist Humanoid Robots](https://arxiv.org/abs/2503.14734)（NVIDIA，2025） · [NVIDIA 官方研究介绍](https://research.nvidia.com/labs/lpr/publication/gr00tn1_2025/) · [Isaac-GR00T 代码](https://github.com/NVIDIA/Isaac-GR00T) · **状态**：论文原理解读，未完成本地训练验证。

## 1. 问题：一个模型如何理解指令又输出高频动作？

机器人要理解“收拾桌面”的目标，也要连续控制几十个自由度。让大型语言模型一帧一帧直接预测关节可能存在推理吞吐、连续性和动作精度问题。GR00T N1 把这两类处理分成 **System 2（视觉语言理解）** 与 **System 1（生成式动作模型）**，并端到端联合训练。注意：这不是把自然语言“思考结果”手工写成规则后再传给一个传统控制器；两个神经网络模块通过学习的表征配合。

## 2. 整体结构

![GR00T N1：双系统架构](/Embodied-AI-Roadmap/diagrams/gr00t.svg)

*示意图；不同 GR00T 版本可能有不同网络与训练配置。 图示为教学重绘。*

```text
多路视觉 + 指令 + 本体状态
          │
    视觉语言模块 System 2
          │ 条件特征
          ▼
 Diffusion Transformer System 1 ← 带噪动作 / 时间
          │ 去噪 / 连续动作分布
          ▼
     Action Chunk
          ▼
  实机控制器 + 安全约束
```

把 System 2 看作条件编码器而不只是“文字输出器”，System 1 看作动作生成专家而不只是末端关节回归头。虽然都可能使用视觉语言和生成模型，GR00T N1 与 π₀ 的具体 Transformer 参数耦合、训练配方和数据规范并不相同。

## 3. 张量如何进入模型？

一般将机器人状态写为 $s_t\in\mathbb R^{D_s}$、动作块写为 $A_t\in\mathbb R^{H\times D_a}$，图像和语言转换成条件 token。实际 $H,D_s,D_a$ 受所选 embodiment 和 checkpoint 配置控制，不存在“一套维度覆盖所有真实机器人”的自动保证。

机器人本体差异非常关键：轮式双臂的底盘速度、两手末端 SE(3)、夹爪角度，与人形机器人关节目标不是同一个动作语义。要验证模型能否迁移，先检查 action adapter、state normalization、训练与部署的坐标系以及帧率，而不只是能否加载网络权重。

## 4. 为什么使用 Diffusion Transformer？

动作块的真实分布可能是多模态的：抓杯子既可从左侧，也可从右侧绕开障碍。扩散模型通过随机噪声逐步产生条件轨迹，可表达多个合理解，而不必回归平均动作。GR00T N1 使用生成式扩散 Transformer 建模动作；公式上可以先理解常见噪声预测目标：

$$
\mathcal L=\mathbb E\left[\|\epsilon-\epsilon_\theta(A_k,k,c)\|^2\right],
$$

其中 $c$ 是视觉语言及本体状态条件。这是理解扩散模型的通用写法，**不能当作 NVIDIA GR00T N1 在所有版本代码里精确使用的完整损失函数**。阅读官方代码时须核对实际噪声路径、目标速度/噪声参数化及调度器。

## 5. 训练数据是模型结构的一部分

论文训练混合真实机器人轨迹、人类视频和合成数据。它们提供的监督并不相同：机器人数据天然带有本体动作；人类视频有丰富场景经验但不一定有 robot action；模拟数据能够提供精确状态却存在 Sim2Real 差异。因此“模型吃过人类视频”不等于“人类视频直接提供了可执行机器人 action”。数据中各模态的配比、缺失掩码与动作空间适配都影响最终结果。

论文作者在多机器人仿真 benchmark 和真实 Fourier GR-1 双臂任务上展示效果。这支持模型可以在作者设置下利用异构数据，但不能推导出工业柔性装配已经解决。

## 6. 读源码与实验怎么入手

先看 [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) 的 README 和模型/训练目录：一是确认 System 1/2 接口；二是找 action token 与噪声时间如何进入网络；三是找真实训练样本如何完成预处理和 normalization；四是查输出动作如何对应真实机器人。

**建议验证**：在已支持的示例数据上跑一个 checkpoint 推理，打印观察张量、动作块维度和数据归一化前后范围；再对相机缺失/动作归一化错误做故障注入，比较输出异常。本站尚未运行该实验。

## 7. 与 π₀ 的阅读顺序

先理解 [π₀](./pi0/) 的视觉语言专家、动作专家与连续生成，再比较 GR00T N1 的双系统结构、扩散 Transformer、异构数据策略与部署软件栈。不要把两个模型都简化成“一个 VLM 加一个头”；架构相似不代表优化与推理方式一致。

来源：[论文](https://arxiv.org/abs/2503.14734) · [官方介绍](https://research.nvidia.com/labs/lpr/publication/gr00tn1_2025/) · [代码](https://github.com/NVIDIA/Isaac-GR00T)。
