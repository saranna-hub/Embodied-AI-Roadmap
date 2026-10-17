---
title: HumanEgo：从人手轨迹学物体中心的交互策略
sidebar:
  order: 4
description: Interaction-centric token、EGO 到机器人表示与少样本 Flow Matching
---

**论文**：[HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos](https://arxiv.org/abs/2605.24934)（Wang 等，2026） · **项目**：[HumanEgo](https://humanego-ai.github.io/) · **官方代码**：[TX-Leo/HumanEgo](https://github.com/TX-Leo/HumanEgo) · **状态**：论文原理和可公开的代码入口解读，未完成复现。

## 1. 为什么模仿人的绝对手轨迹可能没有意义？

人高矮不同、机器人臂长不同、相机视角不同。人示范“把杯子放到托盘里”，重要的往往不是手腕在世界坐标中的绝对轨迹，而是**手相对杯子在哪里、抓住后杯子相对托盘如何运动、何时松手**。

HumanEgo 尝试用物体/交互中心的表示减少 Human–Robot embodiment gap：先把示范提取为手、物体和参考坐标之间的关系，再训练策略生成符合这种关系的机器人动作。

## 2. 数据与表示

![HumanEgo：以物体为参考](/Embodied-AI-Roadmap/diagrams/humanego.svg)

*物体中心表征不等于任何机器人都能直接执行。 图示为教学重绘。*

原始输入是 EGO 视频，论文系统依赖同步视觉、相机运动与三维手部位置等结构化信息。Project Aria 及其处理输出可提供重要的轨迹与标定信息；它不是“普通手机随便拍一段视频就能得到完美三维动作”的方法。

**Interaction-Centric Token** 可以理解为表示任务相关的相对几何，而不是固定机器人各关节值：

$$
T_{OH}(t)=T_{WO}(t)^{-1}T_{WH}(t),
$$

其中 $W$ 是世界系、$O$ 是物体系、$H$ 是人手参考系。若新机器人执行时物体移动了，就可以在新的 $T_{WO}^{\mathrm{new}}$ 下重建期望交互位置，而不必直接重放原先的人手世界坐标。

注意：这只是 Object-centric coordinate 表示的教学公式，论文真实 Token 的组成和学习流程，应查看原始方法，不要把简化的 6DoF 相对位姿误认为完整 ICT 实现。

## 3. 模型如何把人类交互变成机器人策略？

```text
EGO 示范视频
    ↓ 相机/手/物体关联
Interaction-centric Representation
    ↓ 视觉域处理与机器人条件
Flow Matching Robot Policy
    ↓ 连续 Action Chunk
机器人真机执行
```

模型的关键不只是某个手模型精度，而是**提供跨本体相对稳定的动作监督接口**。作者还加入未来物体状态/轨迹相关辅助目标，增加单条示范的监督信息量，使少数据条件下训练具备更多物理约束线索。

## 4. Flow Matching 为什么能用于少样本策略？

给定当前观察 $o$ 和未来机器人动作块 $A$，用随机噪声与目标动作混合，训练条件速度场。常见线性插值写法：

$$
x_t=(1-t)A+t\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

模型学习条件速度后，推理时从随机动作积分生成动作块。这里的核心并不是“参数更少”，而是对复杂多模态动作分布有较自然的生成式建模能力。但如果人类数据没有可靠的 robot-compatible target，单靠 Flow Matching 本身无法消除本体差异。

## 5. 论文实验告诉我们什么？

作者报告每任务约 30 分钟的人类视频，在 4 个真实任务中平均成功率 **92.5%**；使用约 15 分钟示范时为 **75%**。这是作者选择任务与硬件条件下的实验结果，不代表任意工业装配只要半小时 EGO 就能达到 92.5%。

对高精度插接、柔性线束和高速接触操作，物体相对位姿可能仍不足以描述接触、力与状态转换。即使 HumanEgo 的少样本转移有效，也不代表精细操控的物理误差可以忽略。

## 6. 与 Ego2Robot 有什么关键区别？

**Ego2Robot** 倾向于显式生成 robot-format 数据（动作标签+机器人视觉）；**HumanEgo** 更强调交互关系本身是跨本体的重要监督，减少对人的绝对运动轨迹的依赖。这两种路线可以组合：高质量相对交互特征作为 Robotization 的约束，最后仍用真实机器人数据验证可执行性。

## 7. 工厂数据采集应增加什么字段？

除了 RGB/Hand Pose，优先保存：物体的 ID 与位姿、物体坐标中的交互位置、抓握/释放事件、子任务状态与时间戳。对于数据手套，接触区域和触觉变化可作为补充标签，但不能未经校准直接换算成机器人力控指令。

**最小验证**：同一插接件在不同位置、方向的 30 条示范，分别训练绝对世界坐标轨迹基线和物体相对轨迹基线，固定训练预算和评测姿态扰动；统计成功率、对位误差与失败恢复率。没有实测前，这只是实验设计。

来源：[论文](https://arxiv.org/abs/2605.24934) · [项目页](https://humanego-ai.github.io/) · [代码](https://github.com/TX-Leo/HumanEgo)。
